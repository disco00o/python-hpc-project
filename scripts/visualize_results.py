#!/usr/bin/env python
"""Task 3: visualize simulation outputs for selected floorplans."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wall_heating import (
    DEFAULT_ABS_TOL,
    DEFAULT_LOAD_DIR,
    DEFAULT_MAX_ITER,
    jacobi_pingpong_numpy,
    load_building_ids,
    load_data,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--load-dir", default=str(DEFAULT_LOAD_DIR))
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--ids", nargs="*", default=None)
    p.add_argument("--max-iter", type=int, default=DEFAULT_MAX_ITER)
    p.add_argument("--atol", type=float, default=DEFAULT_ABS_TOL)
    p.add_argument("--output-dir", default="figures")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.ids:
        ids = args.ids
    else:
        ids = load_building_ids(args.load_dir)[: args.n]

    for bid in ids:
        u0, mask = load_data(args.load_dir, bid)
        u = jacobi_pingpong_numpy(u0, mask, args.max_iter, args.atol)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        im0 = axes[0].imshow(u0[1:-1, 1:-1], cmap="coolwarm", vmin=5, vmax=25)
        axes[0].set_title(f"{bid} initial")
        axes[0].axis("off")
        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

        im1 = axes[1].imshow(u[1:-1, 1:-1], cmap="inferno", vmin=5, vmax=25)
        axes[1].set_title(f"{bid} solved")
        axes[1].axis("off")
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        fig.tight_layout()
        out_path = out_dir / f"{bid}_solution.png"
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
