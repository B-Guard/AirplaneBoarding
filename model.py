"""
Assumptions:
- Single center aisle, single entry at the front (row 1).
- Aisle divided into row segments, each with capacity=2 (one "serving" at row + one waiting behind).
- No overtakes (enforced by downstream capacity + entry order gating).
- Each passenger has an assigned seat (row, seat) and optional luggage.
- Seating service time depends on how many "blocker" seats (between aisle and target) are already occupied.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Literal
import math
import random

import simpy

SeatId = Literal["A","B","C","D","E","F"]
SeatDist = Literal["deterministic","lognormal","triangular","exponential"]

SEATS_3X3: List[SeatId] = ["A","B","C","D","E","F"]


def blockers_for(seat: SeatId) -> List[SeatId]:
    """Return seat(s) between aisle and target seat on that side."""
    if seat in ("C","D"):     # aisle seats
        return []
    if seat == "B":           # left middle
        return ["C"]
    if seat == "A":           # left window
        return ["B","C"]
    if seat == "E":           # right middle
        return ["D"]
    if seat == "F":           # right window
        return ["E","D"]
    raise ValueError(f"Unknown seat {seat}")


def seat_group(seat: SeatId) -> str:
    if seat in ("A","F"):
        # return "window"
        return 1
    if seat in ("B","E"):
        # return "middle"
        return 2
    if seat in ("C","D"):
        # return "aisle"
        return 3
    # return "other"
    return 4


def draw_service_time(mean: float, rng: random.Random, dist: SeatDist) -> float:
    """Draw a positive service time with the requested distribution."""
    if mean <= 0.0:
        return 0.0
    if dist == "deterministic":
        return float(mean)
    if dist == "lognormal":
        sigma = 0.25
        mu = math.log(mean) - 0.5 * sigma * sigma
        return float(rng.lognormvariate(mu, sigma))
    if dist == "triangular":
        low = 0.7 * mean
        high = 1.3 * mean
        return float(rng.triangular(low, high, mean))
    if dist == "exponential":
        return float(rng.expovariate(1.0 / mean))
    raise ValueError(f"Unknown dist: {dist}")


@dataclass(frozen=True)
class Passenger:
    pid: int
    row: int
    seat: SeatId
    has_luggage: bool


@dataclass(frozen=True)
class AirplaneConfig:
    rows: int = 30
    seat_ids: Tuple[SeatId, ...] = tuple(SEATS_3X3)
    aisle_capacity_per_row: int = 2


@dataclass(frozen=True)
class TimeConfig:
    walk_time_per_row: float = 1.5
    base_seat_time: float = 8.0 # Time it takes to seat with no blockers and no luggage
    interference_penalty: float = 6.0 # Additional time per added blocker
    luggage_add_time: float = 8.0 # Additional time if the passenger has luggage
    seat_time_dist: SeatDist = "deterministic"


@dataclass
class PassengerTrace:
    pid: int
    seat_row: int
    seat_id: str
    blockers_taken: int
    has_luggage: bool
    enter_t: float
    reach_row_t: float
    seat_start_t: float
    seat_end_t: float
    seat_service_time: float


class BoardingModel:
    def __init__(
        self,
        env: simpy.Environment,
        rng: random.Random,
        airplane: AirplaneConfig,
        times: TimeConfig,
        manifest: List[Passenger],
        trace_first_n: int = 0,
        tick: float = 0.05,
    ):
        self.env = env
        self.rng = rng
        self.airplane = airplane
        self.times = times
        self.manifest = manifest
        self.trace_first_n = trace_first_n
        self.tick = tick

        R = airplane.rows
        self.aisle_occ: List[int] = [0] * (R + 1)  # 1..R used
        self.seat_occ: Dict[Tuple[int, str], int] = {(r, s): 0 for r in range(1, R+1) for s in airplane.seat_ids}

        self._next_to_enter = 0
        self._seated = 0
        self.finish_time: Optional[float] = None
        self.trace: List[PassengerTrace] = []

    def can_enter(self) -> bool:
        return self.aisle_occ[1] < self.airplane.aisle_capacity_per_row

    def can_move_into(self, r_next: int) -> bool:
        return self.aisle_occ[r_next] < self.airplane.aisle_capacity_per_row

    def seating_mean_time(self, p: Passenger) -> Tuple[float, int]:
        blockers = blockers_for(p.seat)
        k = sum(self.seat_occ[(p.row, b)] for b in blockers)
        mean = self.times.base_seat_time + self.times.interference_penalty * k
        if p.has_luggage:
            mean += self.times.luggage_add_time
        return mean, k

    def passenger_process(self, order_index: int, p: Passenger):
        # Outside queue (infinite): wait until it's your turn in the boarding sequence
        while order_index != self._next_to_enter:
            yield self.env.timeout(self.tick)

        # Then wait until there is space at entry (row 1)
        while not self.can_enter():
            yield self.env.timeout(self.tick)

        enter_t = self.env.now
        self.aisle_occ[1] += 1
        self._next_to_enter += 1  # allow next person to start attempting entry

        # Walk forward row by row (no overtakes)
        pos = 1
        while pos < p.row:
            while not self.can_move_into(pos + 1):
                yield self.env.timeout(self.tick)
            yield self.env.timeout(self.times.walk_time_per_row)
            self.aisle_occ[pos] -= 1
            pos += 1
            self.aisle_occ[pos] += 1

        reach_row_t = self.env.now
        seat_start_t = self.env.now

        mean, k = self.seating_mean_time(p)
        st = draw_service_time(mean, self.rng, self.times.seat_time_dist)
        yield self.env.timeout(st)

        seat_end_t = self.env.now

        # Take seat and free aisle segment at that row
        self.seat_occ[(p.row, p.seat)] = 1
        self.aisle_occ[pos] -= 1

        # Trace for face validation
        if order_index < self.trace_first_n:
            self.trace.append(PassengerTrace(
                pid=p.pid, seat_row=p.row, seat_id=p.seat, blockers_taken=k, has_luggage=p.has_luggage,
                enter_t=float(enter_t), reach_row_t=float(reach_row_t),
                seat_start_t=float(seat_start_t), seat_end_t=float(seat_end_t),
                seat_service_time=float(st),
            ))

        self._seated += 1
        if self._seated == len(self.manifest):
            self.finish_time = float(self.env.now)


def simulate(
    manifest: List[Passenger],
    seed: int = 0,
    airplane: AirplaneConfig = AirplaneConfig(),
    times: TimeConfig = TimeConfig(),
    trace_first_n: int = 0,) -> Tuple[float, List[PassengerTrace]]:

    rng = random.Random(seed)
    env = simpy.Environment()
    model = BoardingModel(env, rng, airplane, times, manifest, trace_first_n=trace_first_n)
    for i, p in enumerate(manifest):
        env.process(model.passenger_process(i, p))
    env.run()
    assert model.finish_time is not None
    return model.finish_time, model.trace
