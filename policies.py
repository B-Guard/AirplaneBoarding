"""
Seat assignment + boarding-order strategies.
"""

from __future__ import annotations

from typing import List, Tuple, Callable, Optional, Literal
import random

from model import Passenger, SeatId, SEATS_3X3, seat_group

PolicyName = Literal["random", "back_to_front", "wilma", "steffen_boarding"]


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

def back_to_front(rows: int, rng: random.Random, p_luggage: float = 0.5,
                  seat_ids: Tuple[SeatId, ...] = tuple(SEATS_3X3)) -> List[Passenger]:
    seats = all_seats(rows, seat_ids)
    # Sort by seat group (window, middle, aisle), then by row descending
    seats.sort(key=lambda x: seat_group(x[1]))
    seats.sort(key=lambda x: x[0], reverse=True)
    return make_manifest_from_ordered_seats(seats, rng, p_luggage)

def wilma(rows: int, rng: random.Random, p_luggage: float = 0.5,
          seat_ids: Tuple[SeatId, ...] = tuple(SEATS_3X3)) -> List[Passenger]:
    seats = all_seats(rows, seat_ids)
    # Sort by seat group (window, middle, aisle)
    seats.sort(key=lambda x: seat_group(x[1]))
    return make_manifest_from_ordered_seats(seats, rng, p_luggage)

def policy_random(rows: int, rng: random.Random, p_luggage: float = 0.5,
                  seat_ids: Tuple[SeatId, ...] = tuple(SEATS_3X3)) -> List[Passenger]:
    seats = all_seats(rows, seat_ids)
    rng.shuffle(seats)
    return make_manifest_from_ordered_seats(seats, rng, p_luggage)


def steffen_boarding(rows: int, rng: random.Random, p_luggage: float = 0.5,
                     seat_ids: Tuple[SeatId, ...] = tuple(SEATS_3X3)) -> List[Passenger]:

    seats = all_seats(rows, seat_ids)

    def sorting_steffen_method(seat):
        row, seat_id = seat

        #to return for sorting sequence: seat_id e.g A,B,.., even rows and then odd, back to front 
        return (seat_group(seat_id), row % 2, -row)  

    seats.sort(key=sorting_steffen_method)
    return make_manifest_from_ordered_seats(seats, rng, p_luggage)



def with_business_class(manifest: List[Passenger],
                        business_rows: int,
                        ) -> List[Passenger]:
    """Reorder manifest to have business class (first N rows) first."""
    manifest.sort(key=lambda p: p.seat_row <= business_rows, reverse=True)
    return manifest

def get_policy(name: PolicyName) -> Callable[..., List[Passenger]]:
    if name == "random":
        return policy_random
    elif name == "back_to_front":
        return back_to_front
    elif name == "wilma":
        return wilma
    elif name == "steffen_boarding":
        return steffen_boarding
    raise ValueError(f"Unknown policy: {name}")
