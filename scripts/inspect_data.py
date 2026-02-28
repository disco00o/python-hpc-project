#!/usr/bin/env python
"""Task 1: visualize raw input domain/mask grids for selected buildings."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wall_heating import DEFAULT_LOAD_DIR, load_building_ids


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--load-dir", default=str(DEFAULT_LOAD_DIR))
    p.add_argument("--n", type=int, default=4, help="number of buildings to plot")
    p.add_argument("--ids", nargs="*", default=None, help="specific building IDs (overrides --n)")
    p.add_argument("--output-dir", default="figures")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    load_dir = Path(args.load_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.ids:
        ids = args.ids
    else:
        ids = load_building_ids(load_dir)[: args.n]

    for bid in ids:
        domain = np.load(load_dir / f"{bid}_domain.npy")
        interior = np.load(load_dir / f"{bid}_interior.npy")

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        im0 = axes[0].imshow(domain, cmap="coolwarm", vmin=5, vmax=25)
        axes[0].set_title(f"{bid} domain")
        axes[0].axis("off")
        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

        axes[1].imshow(interior, cmap="gray")
        axes[1].set_title(f"{bid} interior mask")
        axes[1].axis("off")

        fig.tight_layout()
        out_path = out_dir / f"{bid}_input.png"
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
