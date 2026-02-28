#!/usr/bin/env python
"""Validate a solver against the reference on a small set of buildings."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wall_heating import DEFAULT_ABS_TOL, DEFAULT_LOAD_DIR, DEFAULT_MAX_ITER, get_solver, load_building_ids, load_data
from wall_heating.core import jacobi_reference


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--solver", default="numpy")
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--load-dir", default=str(DEFAULT_LOAD_DIR))
    p.add_argument("--max-iter", type=int, default=DEFAULT_MAX_ITER)
    p.add_argument("--atol", type=float, default=DEFAULT_ABS_TOL)
    p.add_argument("--rtol", type=float, default=1e-4)
    p.add_argument("--atol-compare", type=float, default=1e-3)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ids = load_building_ids(args.load_dir)[: args.n]
    solver = get_solver(args.solver)

    worst_abs = 0.0
    worst_rel = 0.0

    for bid in ids:
        u0, mask = load_data(args.load_dir, bid)
        ref = jacobi_reference(u0, mask, args.max_iter, args.atol)
        test = solver(u0, mask, args.max_iter, args.atol)

        diff = np.abs(ref - test)
        max_abs = float(diff.max())
        rel = diff / np.maximum(np.abs(ref), 1e-8)
        max_rel = float(rel.max())

        worst_abs = max(worst_abs, max_abs)
        worst_rel = max(worst_rel, max_rel)
        print(f"{bid}: max_abs={max_abs:.6e} max_rel={max_rel:.6e}")

    print(f"Worst max_abs: {worst_abs:.6e}")
    print(f"Worst max_rel: {worst_rel:.6e}")

    if worst_abs > args.atol_compare and worst_rel > args.rtol:
        raise SystemExit("Validation failed: differences are above tolerances")


if __name__ == "__main__":
    main()
