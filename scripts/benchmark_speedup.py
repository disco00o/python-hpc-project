#!/usr/bin/env python
"""Benchmark parallel speedup and fit Amdahl's law."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from statistics import median

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wall_heating import (
    DEFAULT_ABS_TOL,
    DEFAULT_LOAD_DIR,
    DEFAULT_MAX_ITER,
    get_solver,
    load_building_ids,
    run_buildings_serial,
    run_parallel_dynamic,
    run_parallel_static,
)


def parse_workers(workers_arg: str) -> list[int]:
    workers = sorted({int(w) for w in workers_arg.split(",") if w.strip()})
    if not workers or workers[0] != 1:
        raise ValueError("workers list must include 1 for baseline speedup")
    return workers


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=40, help="number of buildings for the experiment")
    p.add_argument("--workers", default="1,2,4,8")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--schedule", choices=("static", "dynamic"), default="static")
    p.add_argument("--chunksize", type=int, default=1)
    p.add_argument("--solver", default="reference")
    p.add_argument("--max-iter", type=int, default=DEFAULT_MAX_ITER)
    p.add_argument("--atol", type=float, default=DEFAULT_ABS_TOL)
    p.add_argument("--load-dir", default=str(DEFAULT_LOAD_DIR))
    p.add_argument("--output-csv", default="results/speedup.csv")
    p.add_argument("--plot", default="figures/speedup.png")
    return p.parse_args()


def _run_once(args: argparse.Namespace, ids: list[str], workers: int) -> float:
    if workers == 1:
        solver = get_solver(args.solver)
        _, elapsed = run_buildings_serial(
            ids,
            load_dir=args.load_dir,
            solver=solver,
            max_iter=args.max_iter,
            atol=args.atol,
        )
        return elapsed

    if args.schedule == "static":
        _, elapsed = run_parallel_static(
            ids,
            load_dir=args.load_dir,
            solver_name=args.solver,
            max_iter=args.max_iter,
            atol=args.atol,
            workers=workers,
        )
    else:
        _, elapsed = run_parallel_dynamic(
            ids,
            load_dir=args.load_dir,
            solver_name=args.solver,
            max_iter=args.max_iter,
            atol=args.atol,
            workers=workers,
            chunksize=args.chunksize,
        )
    return elapsed


def main() -> None:
    args = parse_args()
    workers_list = parse_workers(args.workers)
    ids = load_building_ids(args.load_dir)[: args.n]

    rows: list[dict[str, float]] = []

    # Warm-up for JIT solvers so compile cost does not pollute repeated timings too much.
    if args.solver in {"numba-cpu", "numba-cuda", "cupy"}:
        _ = _run_once(args, ids[:1], workers=1)

    for workers in workers_list:
        samples = []
        for rep in range(args.repeats):
            elapsed = _run_once(args, ids, workers)
            samples.append(elapsed)
            print(f"workers={workers} rep={rep+1}/{args.repeats} elapsed={elapsed:.3f}s")

        med = median(samples)
        rows.append(
            {
                "workers": workers,
                "time_s_median": med,
                "time_s_min": min(samples),
                "time_s_max": max(samples),
            }
        )

    df = pd.DataFrame(rows).sort_values("workers")
    t1 = float(df.loc[df["workers"] == 1, "time_s_median"].iloc[0])
    df["speedup"] = t1 / df["time_s_median"]

    def parallel_fraction(row: pd.Series) -> float:
        n = int(row["workers"])
        s = float(row["speedup"])
        if n == 1:
            return float("nan")
        return (1.0 - 1.0 / s) / (1.0 - 1.0 / n)

    df["p_est"] = df.apply(parallel_fraction, axis=1)
    p_vals = df["p_est"].dropna().clip(lower=0.0, upper=1.0)
    p_hat = float(p_vals.median()) if len(p_vals) else 0.0
    s_max = float("inf") if p_hat >= 0.999999 else 1.0 / (1.0 - p_hat)

    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}")
    print(f"Estimated parallel fraction p ~= {p_hat:.4f}")
    print(f"Theoretical max speedup from Amdahl ~= {s_max:.2f}")

    try:
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
        import matplotlib.pyplot as plt

        Path(args.plot).parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        workers_np = df["workers"].to_numpy()
        speedup_np = df["speedup"].to_numpy()
        ax.plot(workers_np, speedup_np, marker="o", label="Measured")

        if p_hat > 0:
            model_workers = workers_np
            model_speedup = 1.0 / ((1.0 - p_hat) + p_hat / model_workers)
            ax.plot(model_workers, model_speedup, linestyle="--", label=f"Amdahl fit p={p_hat:.3f}")

        ax.set_xlabel("Workers")
        ax.set_ylabel("Speedup")
        ax.set_title(f"{args.schedule.capitalize()} scheduling speedup ({args.solver})")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.plot, dpi=160)
        plt.close(fig)
        print(f"Wrote {args.plot}")
    except Exception as exc:  # pragma: no cover
        print(f"Skipping plot generation: {exc}")


if __name__ == "__main__":
    main()
