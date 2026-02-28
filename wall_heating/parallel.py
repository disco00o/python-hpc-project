"""Parallel execution helpers for static and dynamic floorplan scheduling."""

from __future__ import annotations

from multiprocessing import Pool
from pathlib import Path
from time import perf_counter
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from .core import load_data, summary_stats
from .solvers import get_solver

# Worker-local config set by _init_worker.
_WORKER_CFG: Dict[str, object] = {}


def _init_worker(load_dir: str, solver_name: str, max_iter: int, atol: float) -> None:
    _WORKER_CFG["load_dir"] = load_dir
    _WORKER_CFG["solver"] = get_solver(solver_name)
    _WORKER_CFG["max_iter"] = max_iter
    _WORKER_CFG["atol"] = atol


def _solve_one(bid: str) -> Tuple[str, Dict[str, float]]:
    load_dir = _WORKER_CFG["load_dir"]
    solver = _WORKER_CFG["solver"]
    max_iter = _WORKER_CFG["max_iter"]
    atol = _WORKER_CFG["atol"]

    u0, interior_mask = load_data(load_dir, bid)
    u = solver(u0, interior_mask, max_iter, atol)
    return bid, summary_stats(u, interior_mask)


def _solve_chunk(chunk: Sequence[str]) -> List[Tuple[str, Dict[str, float]]]:
    return [_solve_one(bid) for bid in chunk]


def run_parallel_static(
    building_ids: Sequence[str],
    *,
    load_dir: Path | str,
    solver_name: str,
    max_iter: int,
    atol: float,
    workers: int,
) -> Tuple[List[Tuple[str, Dict[str, float]]], float]:
    """Static scheduling: each worker gets one equally-sized chunk."""
    if workers < 1:
        raise ValueError("workers must be >= 1")
    if not building_ids:
        return [], 0.0

    chunks = [chunk.tolist() for chunk in np.array_split(np.array(building_ids), workers) if len(chunk)]

    t0 = perf_counter()
    with Pool(
        processes=workers,
        initializer=_init_worker,
        initargs=(str(load_dir), solver_name, max_iter, atol),
    ) as pool:
        chunk_results = pool.map(_solve_chunk, chunks)

    rows: List[Tuple[str, Dict[str, float]]] = []
    for group in chunk_results:
        rows.extend(group)
    elapsed = perf_counter() - t0

    return rows, elapsed


def run_parallel_dynamic(
    building_ids: Sequence[str],
    *,
    load_dir: Path | str,
    solver_name: str,
    max_iter: int,
    atol: float,
    workers: int,
    chunksize: int = 1,
) -> Tuple[List[Tuple[str, Dict[str, float]]], float]:
    """Dynamic scheduling: workers pull floorplans from a shared task queue."""
    if workers < 1:
        raise ValueError("workers must be >= 1")
    if not building_ids:
        return [], 0.0

    rank = {bid: i for i, bid in enumerate(building_ids)}

    t0 = perf_counter()
    with Pool(
        processes=workers,
        initializer=_init_worker,
        initargs=(str(load_dir), solver_name, max_iter, atol),
    ) as pool:
        rows = list(pool.imap_unordered(_solve_one, building_ids, chunksize=chunksize))

    rows.sort(key=lambda x: rank[x[0]])
    elapsed = perf_counter() - t0
    return rows, elapsed
