"""
Outputs CSVs into ./results/ by default.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import List, Dict, Any, Optional, Sequence, Tuple
import os
import csv
import json
import math
import statistics as stats
import random
from itertools import product
import tqdm

from model import simulate, AirplaneConfig, TimeConfig, PassengerTrace
from policies import get_policy, PolicyName


def mean_ci95(xs: Sequence[float]) -> Tuple[float, float, float]:
    """Return (mean, sd, halfwidth_95_ci) using normal approximation."""
    xs = list(xs)
    m = stats.mean(xs)
    sd = stats.pstdev(xs) if len(xs) > 1 else 0.0
    hw = 1.96 * (sd / math.sqrt(len(xs))) if len(xs) > 1 else 0.0
    return m, sd, hw


def write_csv(
    path: str, rows: List[Dict[str, Any]], fieldnames: Optional[List[str]] = None
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def run_replications(
    policy: PolicyName,
    airplane: AirplaneConfig,
    times: TimeConfig,
    reps: int = 50,
    seed0: int = 0,
    p_luggage: float = 0.5,
    # policy kwargs:
    policy_kwargs: Optional[Dict[str, Any]] = None,
    trace_first_n: int = 0,
) -> Dict[str, Any]:
    """
    Build a manifest from the policy
    """
    policy_fn = get_policy(policy)
    policy_kwargs = policy_kwargs or {}

    finish_times: List[float] = []
    sample_trace: List[PassengerTrace] = []

    for r in range(reps):
        rng = random.Random(seed0 + r)
        manifest = policy_fn(
            airplane.rows,
            rng=rng,
            p_luggage=p_luggage,
            seat_ids=airplane.seat_ids,
            **policy_kwargs,
        )
        t, tr = simulate(
            manifest,
            seed=seed0 + r,
            airplane=airplane,
            times=times,
            trace_first_n=trace_first_n,
        )
        finish_times.append(t)
        if r == 0:
            sample_trace = tr

    m, sd, hw = mean_ci95(finish_times)
    return {
        "policy": policy,
        "reps": reps,
        "mean_time_s": m,
        "sd_s": sd,
        "ci95_halfwidth_s": hw,
        "min_s": min(finish_times),
        "max_s": max(finish_times),
        "times": asdict(times),
        "airplane": asdict(airplane),
        "p_luggage": p_luggage,
        "policy_kwargs": policy_kwargs,
        "sample_trace": [asdict(x) for x in sample_trace],
    }


def task2_face_validation(out_dir: str = "results") -> Dict[str, Any]:
    airplane = AirplaneConfig(rows=30)
    base = TimeConfig(
        walk_time_per_row=1.5,
        base_seat_time=8.0,
        interference_penalty=6.0,
        luggage_add_time=8.0,
        seat_time_dist="deterministic",
    )

    scenarios = [
        (
            "all_zero_seating",
            dict(
                times=TimeConfig(
                    walk_time_per_row=1.5,
                    base_seat_time=0.0,
                    interference_penalty=0.0,
                    luggage_add_time=0.0,
                ),
                p_luggage=0.0,
            ),
        ),
        ("no_luggage", dict(times=base, p_luggage=0.0)),
        ("all_luggage", dict(times=base, p_luggage=1.0)),
        (
            "high_interference",
            dict(
                times=TimeConfig(
                    walk_time_per_row=1.5,
                    base_seat_time=8.0,
                    interference_penalty=12.0,
                    luggage_add_time=8.0,
                ),
                p_luggage=0.5,
            ),
        ),
    ]

    rows = []
    traces = []

    for name, cfg in scenarios:
        res = run_replications(
            policy="random",
            airplane=airplane,
            times=cfg["times"],
            reps=20,
            seed0=1000,
            p_luggage=cfg["p_luggage"],
            trace_first_n=5,
        )
        rows.append(
            {
                "scenario": name,
                "mean_time_s": res["mean_time_s"],
                "sd_s": res["sd_s"],
                "ci95_halfwidth_s": res["ci95_halfwidth_s"],
                "min_s": res["min_s"],
                "max_s": res["max_s"],
                "p_luggage": cfg["p_luggage"],
                "base_seat_time": cfg["times"].base_seat_time,
                "interference_penalty": cfg["times"].interference_penalty,
                "luggage_add_time": cfg["times"].luggage_add_time,
            }
        )
        for tr in res["sample_trace"]:
            tr2 = dict(tr)
            tr2["scenario"] = name
            traces.append(tr2)

    summary_csv = os.path.join(out_dir, "task2_face_validation_summary.csv")
    trace_csv = os.path.join(out_dir, "task2_face_validation_trace_first_run.csv")
    write_csv(summary_csv, rows)
    write_csv(trace_csv, traces)
    return {"summary_csv": summary_csv, "trace_csv": trace_csv}


def task2_sweep_seat_times(out_dir: str = "results") -> Dict[str, Any]:
    airplane = AirplaneConfig(rows=30)
    reps = 50

    base_seat_values = [4, 6, 8, 10, 12]
    interference_values = [1, 3, 8, 12]
    luggage_add_values = [1, 8, 16]
    p_luggage_values = [0, 0.5, 0.7, 1]
    strategy_values = ["random", "back_to_front", "wilma"]

    param_grid = list(
        product(
            base_seat_values,
            interference_values,
            luggage_add_values,
            p_luggage_values,
            strategy_values,
        )
    )
    rows = []
    for base_seat, interference, luggage_add, p_luggage, strategy in tqdm.tqdm(
        param_grid
    ):
        times = TimeConfig(
            walk_time_per_row=1.5,
            base_seat_time=base_seat,
            interference_penalty=interference,
            luggage_add_time=luggage_add,
            seat_time_dist="deterministic",
        )
        res = run_replications(
            policy=strategy,
            airplane=airplane,
            times=times,
            reps=reps,
            seed0=2000,
            p_luggage=p_luggage,
        )
        rows.append(
            {
                "base_seat_time": base_seat,
                "interference_penalty": interference,
                "luggage_add_time": luggage_add,
                "p_luggage": p_luggage,
                "reps": reps,
                "mean_time_s": res["mean_time_s"],
                "sd_s": res["sd_s"],
                "ci95_halfwidth_s": res["ci95_halfwidth_s"],
            }
        )

    path = os.path.join(out_dir, "task2_sweep_seat_times.csv")
    write_csv(path, rows)
    return {"csv": path}


def test_seat_time_distributions(
    strategy: PolicyName = "random",
    airplane_rows: int = 30,
    walk_time_per_row: float = 1.5,
    base_seat_time: float = 6.0,
    interference_penalty: float = 7.0,
    luggage_add_time: float = 7.0,
    p_luggage: float = 0.6,
    reps: int = 30,
    out_dir: str = "results",
    out_json: str = "seat_time_distributions.json",
    seat_time_dists: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """
    For each seat_time_dist in seat_time_dists, run several replications and collect per-passenger seat times for all runs.
    Save all data in a JSON file.
    """
    if seat_time_dists is None:
        seat_time_dists = ["lognormal", "triangular", "exponential"]

    airplane = AirplaneConfig(rows=airplane_rows)
    all_data = []

    for dist in seat_time_dists:
        times = TimeConfig(
            walk_time_per_row=walk_time_per_row,
            base_seat_time=base_seat_time,
            interference_penalty=interference_penalty,
            luggage_add_time=luggage_add_time,
            seat_time_dist=dist,
        )
        for r in range(reps):
            seed = 3000 + r
            policy_fn = get_policy(strategy)
            rng = random.Random(seed)
            manifest = policy_fn(
                airplane.rows,
                rng=rng,
                p_luggage=p_luggage,
                seat_ids=airplane.seat_ids,
            )
            # trace_first_n = all passengers
            _, trace = simulate(
                manifest,
                seed=seed,
                airplane=airplane,
                times=times,
                trace_first_n=airplane.rows * len(airplane.seat_ids[0]),
            )
            for passenger in trace:
                entry = asdict(passenger)
                entry.update(
                    {"replication": r, "seat_time_dist": dist, "strategy": strategy}
                )
                all_data.append(entry)

    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, out_json)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=2)
    return {"json": json_path, "n_records": len(all_data)}


def task3_many_boarding_strategies(
    out_dir: str = os.path.join("results", "json"),
) -> Dict[str, Any]:
    strategies = ["random", "back_to_front", "wilma", "steffen_boarding"]
    seat_time_distributions = ["lognormal", "triangular", "exponential"]

    param_grid = list(product(strategies, seat_time_distributions))

    for strategy, seat_time_dist in tqdm.tqdm(param_grid):
        test_seat_time_distributions(
            strategy=strategy,
            seat_time_dists=[seat_time_dist],
            out_dir=out_dir,
            out_json=f"{strategy}_{seat_time_dist}.json",
        )
    return {"n_tests": len(param_grid), "out_dir": out_dir}
