"""Solver registry for selecting implementation variants by name."""

from __future__ import annotations

from typing import Callable

import numpy as np

from .core import jacobi_pingpong_numpy, jacobi_reference

SolverFn = Callable[[np.ndarray, np.ndarray, int, float], np.ndarray]


def available_solver_names() -> tuple[str, ...]:
    return (
        "reference",
        "numpy",
        "numba-cpu",
        "numba-cuda",
        "cupy",
    )


def get_solver(name: str) -> SolverFn:
    name = name.lower()

    if name == "reference":
        return jacobi_reference
    if name == "numpy":
        return jacobi_pingpong_numpy
    if name == "numba-cpu":
        from .numba_impl import jacobi_numba_cpu

        return jacobi_numba_cpu
    if name == "numba-cuda":
        from .numba_impl import jacobi_numba_cuda

        return jacobi_numba_cuda
    if name == "cupy":
        from .cupy_impl import jacobi_cupy

        return jacobi_cupy

    raise ValueError(f"Unknown solver '{name}'. Choices: {available_solver_names()}")
