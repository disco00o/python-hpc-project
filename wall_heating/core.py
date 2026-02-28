"""Data loading, baseline Jacobi solver, and summary metrics."""

from __future__ import annotations

from os.path import join
from pathlib import Path
from time import perf_counter
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import numpy as np

from .constants import DEFAULT_LOAD_DIR, GRID_SIZE, PADDED_SIZE, STATS_KEYS

StatsDict = Dict[str, float]
SolverFn = Callable[[np.ndarray, np.ndarray, int, float], np.ndarray]


def load_building_ids(load_dir: Path | str = DEFAULT_LOAD_DIR) -> List[str]:
    """Load all available building IDs."""
    load_dir = Path(load_dir)
    with open(load_dir / "building_ids.txt", "r", encoding="utf-8") as f:
        return f.read().splitlines()


def load_data(load_dir: Path | str, bid: str, *, dtype=np.float64) -> Tuple[np.ndarray, np.ndarray]:
    """Load one building's padded temperature grid and interior mask."""
    load_dir = Path(load_dir)
    u = np.zeros((PADDED_SIZE, PADDED_SIZE), dtype=dtype)
    u[1:-1, 1:-1] = np.load(join(load_dir, f"{bid}_domain.npy"))
    interior_mask = np.load(join(load_dir, f"{bid}_interior.npy")).astype(bool, copy=False)
    return u, interior_mask


def jacobi_reference(u: np.ndarray, interior_mask: np.ndarray, max_iter: int, atol: float = 1e-6) -> np.ndarray:
    """Reference Jacobi implementation from the assignment description."""
    u = np.copy(u)

    for _ in range(max_iter):
        u_new = 0.25 * (u[1:-1, :-2] + u[1:-1, 2:] + u[:-2, 1:-1] + u[2:, 1:-1])
        u_new_interior = u_new[interior_mask]
        delta = np.abs(u[1:-1, 1:-1][interior_mask] - u_new_interior).max()
        u[1:-1, 1:-1][interior_mask] = u_new_interior

        if delta < atol:
            break
    return u


def jacobi_pingpong_numpy(
    u: np.ndarray,
    interior_mask: np.ndarray,
    max_iter: int,
    atol: float = 1e-6,
) -> np.ndarray:
    """NumPy Jacobi using ping-pong buffers and reduced temporary allocations."""
    current = np.array(u, copy=True)
    nxt = np.array(u, copy=True)

    core = np.s_[1:-1, 1:-1]
    valid = interior_mask

    for _ in range(max_iter):
        nxt[core] = 0.25 * (
            current[1:-1, :-2]
            + current[1:-1, 2:]
            + current[:-2, 1:-1]
            + current[2:, 1:-1]
        )

        # Keep non-interior points fixed.
        nxt_inner = nxt[core]
        cur_inner = current[core]
        nxt_inner[~valid] = cur_inner[~valid]

        delta = np.max(np.abs(cur_inner[valid] - nxt_inner[valid]))
        current, nxt = nxt, current

        if delta < atol:
            break

    return current


def summary_stats(u: np.ndarray, interior_mask: np.ndarray) -> StatsDict:
    """Compute assignment metrics for interior points."""
    u_interior = u[1:-1, 1:-1][interior_mask]
    mean_temp = float(u_interior.mean())
    std_temp = float(u_interior.std())
    pct_above_18 = float(np.mean(u_interior > 18.0) * 100.0)
    pct_below_15 = float(np.mean(u_interior < 15.0) * 100.0)
    return {
        "mean_temp": mean_temp,
        "std_temp": std_temp,
        "pct_above_18": pct_above_18,
        "pct_below_15": pct_below_15,
    }


def run_buildings_serial(
    building_ids: Sequence[str],
    *,
    load_dir: Path | str = DEFAULT_LOAD_DIR,
    solver: SolverFn = jacobi_reference,
    max_iter: int,
    atol: float,
    dtype=np.float64,
) -> Tuple[List[Tuple[str, StatsDict]], float]:
    """Run one solver across multiple buildings serially."""
    t0 = perf_counter()
    rows: List[Tuple[str, StatsDict]] = []
    for bid in building_ids:
        u0, mask = load_data(load_dir, bid, dtype=dtype)
        u = solver(u0, mask, max_iter, atol)
        rows.append((bid, summary_stats(u, mask)))
    elapsed = perf_counter() - t0
    return rows, elapsed


def write_stats_csv(rows: Iterable[Tuple[str, StatsDict]], output_path: Path | str) -> None:
    """Write rows to CSV with assignment-compatible headers."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("building_id," + ",".join(STATS_KEYS) + "\n")
        for bid, stats in rows:
            f.write(
                f"{bid},"
                + ",".join(str(stats[k]) for k in STATS_KEYS)
                + "\n"
            )


def estimate_full_runtime(sample_seconds: float, sample_n: int, total_n: int) -> float:
    """Linear extrapolation helper."""
    if sample_n <= 0:
        raise ValueError("sample_n must be > 0")
    return sample_seconds * (total_n / sample_n)
