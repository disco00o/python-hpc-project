#!/usr/bin/env python
"""Run any solver variant serially or with static/dynamic multi-process scheduling."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wall_heating import (
    DEFAULT_ABS_TOL,
    DEFAULT_LOAD_DIR,
    DEFAULT_MAX_ITER,
    available_solver_names,
    estimate_full_runtime,
    get_solver,
    load_building_ids,
    run_buildings_serial,
    run_parallel_dynamic,
    run_parallel_static,
    write_stats_csv,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("N", type=int, nargs="?", default=10)
    p.add_argument("--load-dir", default=str(DEFAULT_LOAD_DIR))
    p.add_argument("--solver", default="reference", choices=available_solver_names())
    p.add_argument("--max-iter", type=int, default=DEFAULT_MAX_ITER)
    p.add_argument("--atol", type=float, default=DEFAULT_ABS_TOL)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--schedule", choices=("static", "dynamic"), default="static")
    p.add_argument("--chunksize", type=int, default=1, help="used for dynamic scheduling")
    p.add_argument("--output-csv", default="results/solver_stats.csv")
    p.add_argument("--metadata-json", default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    building_ids = load_building_ids(args.load_dir)
    selected = building_ids[: args.N]

    if args.workers == 1:
        solver = get_solver(args.solver)
        rows, elapsed = run_buildings_serial(
            selected,
            load_dir=args.load_dir,
            solver=solver,
            max_iter=args.max_iter,
            atol=args.atol,
        )
    elif args.schedule == "static":
        rows, elapsed = run_parallel_static(
            selected,
            load_dir=args.load_dir,
            solver_name=args.solver,
            max_iter=args.max_iter,
            atol=args.atol,
            workers=args.workers,
        )
    else:
        rows, elapsed = run_parallel_dynamic(
            selected,
            load_dir=args.load_dir,
            solver_name=args.solver,
            max_iter=args.max_iter,
            atol=args.atol,
            workers=args.workers,
            chunksize=args.chunksize,
        )

    write_stats_csv(rows, args.output_csv)

    est_full_s = estimate_full_runtime(elapsed, len(selected), len(building_ids))
    print(
        f"solver={args.solver} schedule={args.schedule} workers={args.workers} "
        f"N={len(selected)} elapsed={elapsed:.3f}s est_full={est_full_s/3600:.2f}h"
    )
    print(f"Wrote: {args.output_csv}")

    if args.metadata_json:
        meta = {
            "solver": args.solver,
            "schedule": args.schedule,
            "workers": args.workers,
            "N": len(selected),
            "max_iter": args.max_iter,
            "atol": args.atol,
            "elapsed_s": elapsed,
            "est_full_runtime_s": est_full_s,
        }
        out = Path(args.metadata_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"Wrote: {out}")


if __name__ == "__main__":
    main()
