"""
Single entry-point script.

Examples:
  python run.py task2_face_validation
  python run.py task2_sweep
"""

import argparse
from experiments import (
    task2_face_validation,
    task2_sweep_seat_times,
    task3_many_boarding_strategies,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=[
        "task2_face_validation",
        "task2_sweep",
        "task3_seat_time_dist",
    ])
    parser.add_argument("--two_entries", action='store_true', help="Pass this flag to use two entry doors instead of one, if not specified, single entry is used.")
    parser.add_argument("--out", default="results", help="Output directory for CSV results (default: results)")
    args = parser.parse_args()

    two_entries = not args.two_entries

    if two_entries:
        print("Simulation with single entry door.")
    else:
        print("Simulation with two entry doors.")

    if args.task == "task2_face_validation":
        out = task2_face_validation(out_dir=args.out, single_entry=two_entries)
    elif args.task == "task2_sweep":
        out = task2_sweep_seat_times(out_dir=args.out, single_entry=two_entries)
    elif args.task == "task3_seat_time_dist":
        out = task3_many_boarding_strategies(single_entry=two_entries, out_dir=args.out)
    else:
        raise ValueError(args.task)

    print("Done.")
    for k, v in out.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
