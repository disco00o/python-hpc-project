#!/usr/bin/env python
"""Reference simulation script (assignment baseline)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wall_heating import (
    DEFAULT_ABS_TOL,
    DEFAULT_LOAD_DIR,
    DEFAULT_MAX_ITER,
    estimate_full_runtime,
    jacobi_reference,
    load_building_ids,
    run_buildings_serial,
    write_stats_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("N", type=int, nargs="?", default=1, help="number of buildings to process")
    parser.add_argument("--load-dir", default=str(DEFAULT_LOAD_DIR))
    parser.add_argument("--max-iter", type=int, default=DEFAULT_MAX_ITER)
    parser.add_argument("--atol", type=float, default=DEFAULT_ABS_TOL)
    parser.add_argument("--output-csv", default="results/reference_stats.csv")
    parser.add_argument(
        "--print-csv",
        action="store_true",
        help="print CSV rows to stdout in addition to writing file",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    building_ids = load_building_ids(args.load_dir)
    selected = building_ids[: args.N]

    rows, elapsed = run_buildings_serial(
        selected,
        load_dir=args.load_dir,
        solver=jacobi_reference,
        max_iter=args.max_iter,
        atol=args.atol,
    )
    write_stats_csv(rows, args.output_csv)

    if args.print_csv:
        with open(args.output_csv, "r", encoding="utf-8") as f:
            print(f.read(), end="")

    est_full_s = estimate_full_runtime(elapsed, len(selected), len(building_ids))
    print(f"Processed {len(selected)} buildings in {elapsed:.3f} s")
    print(f"Estimated full runtime ({len(building_ids)} buildings): {est_full_s/3600:.2f} h")
    print(f"Wrote: {args.output_csv}")


if __name__ == "__main__":
    main()
