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
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=[
        "task2_face_validation",
        "task2_sweep",
    ])
    parser.add_argument("--out", default="results", help="Output directory for CSV results (default: results)")
    args = parser.parse_args()

    if args.task == "task2_face_validation":
        out = task2_face_validation(out_dir=args.out)
    elif args.task == "task2_sweep":
        out = task2_sweep_seat_times(out_dir=args.out)
    else:
        raise ValueError(args.task)

    print("Done.")
    for k, v in out.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
