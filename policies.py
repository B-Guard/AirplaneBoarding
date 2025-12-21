"""
Seat assignment + boarding-order strategies.
"""

from __future__ import annotations

from typing import List, Tuple, Callable, Optional, Literal
import random

from model import Passenger, SeatId, SEATS_3X3, seat_group

PolicyName = Literal["random", "back_to_front", "wilma", "blocks"]


def all_seats(rows: int, seat_ids: Tuple[SeatId, ...] = tuple(SEATS_3X3)) -> List[Tuple[int, SeatId]]:
    return [(r, s) for r in range(1, rows + 1) for s in seat_ids]


def make_manifest_from_ordered_seats(
    ordered_seats: List[Tuple[int, SeatId]],
    rng: random.Random,
    p_luggage: float = 0.5,
) -> List[Passenger]:
    manifest: List[Passenger] = []
    for pid, (r, s) in enumerate(ordered_seats):
        has_luggage = rng.random() < p_luggage
        manifest.append(Passenger(pid=pid, row=r, seat=s, has_luggage=has_luggage))
    return manifest


def policy_random(rows: int, rng: random.Random, p_luggage: float = 0.5,
                  seat_ids: Tuple[SeatId, ...] = tuple(SEATS_3X3)) -> List[Passenger]:
    seats = all_seats(rows, seat_ids)
    rng.shuffle(seats)
    return make_manifest_from_ordered_seats(seats, rng, p_luggage)


def get_policy(name: PolicyName) -> Callable[..., List[Passenger]]:
    if name == "random":
        return policy_random
    raise ValueError(f"Unknown policy: {name}")
