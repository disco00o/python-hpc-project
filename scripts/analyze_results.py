#!/usr/bin/env python
"""Task 12 analysis over final CSV results."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("csv", help="CSV produced by simulation script")
    p.add_argument("--output-dir", default="figures")
    p.add_argument("--summary-txt", default="results/final_summary.txt")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.csv)

    mean_temp_avg = float(df["mean_temp"].mean())
    std_temp_avg = float(df["std_temp"].mean())
    n_above_18 = int((df["pct_above_18"] >= 50.0).sum())
    n_below_15 = int((df["pct_below_15"] >= 50.0).sum())

    lines = [
        f"Rows: {len(df)}",
        f"Average mean temperature: {mean_temp_avg:.4f}",
        f"Average std temperature: {std_temp_avg:.4f}",
        f"Buildings with >=50% area above 18C: {n_above_18}",
        f"Buildings with >=50% area below 15C: {n_below_15}",
    ]
    summary = "\n".join(lines)
    print(summary)

    summary_path = Path(args.summary_txt)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(summary + "\n", encoding="utf-8")
    print(f"Wrote {summary_path}")

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib.pyplot as plt

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(df["mean_temp"], bins=40, color="#d95f02", alpha=0.85)
    axes[0].set_title("Mean temperature distribution")
    axes[0].set_xlabel("Mean interior temperature [C]")
    axes[0].set_ylabel("Count")

    axes[1].hist(df["std_temp"], bins=40, color="#1b9e77", alpha=0.85)
    axes[1].set_title("Std-dev distribution")
    axes[1].set_xlabel("Temperature std-dev [C]")
    axes[1].set_ylabel("Count")

    fig.tight_layout()
    out_hist = out_dir / "final_distributions.png"
    fig.savefig(out_hist, dpi=160)
    plt.close(fig)
    print(f"Wrote {out_hist}")


if __name__ == "__main__":
    main()
