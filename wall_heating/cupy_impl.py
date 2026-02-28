"""CuPy-based Jacobi solver."""

from __future__ import annotations

import numpy as np

try:
    import cupy as cp
except ImportError as exc:  # pragma: no cover
    cp = None
    _CUPY_IMPORT_ERROR = exc
else:
    _CUPY_IMPORT_ERROR = None


def _require_cupy() -> None:
    if _CUPY_IMPORT_ERROR is not None:
        raise RuntimeError(
            "CuPy is not installed. Install a CUDA-matched wheel, e.g. 'pip install cupy-cuda12x'."
        ) from _CUPY_IMPORT_ERROR


def jacobi_cupy(
    u: np.ndarray,
    interior_mask: np.ndarray,
    max_iter: int,
    atol: float = 1e-6,
) -> np.ndarray:
    """Jacobi solver on GPU using vectorized CuPy array operations."""
    _require_cupy()

    current = cp.asarray(u)
    nxt = cp.asarray(u)
    mask = cp.asarray(interior_mask, dtype=current.dtype)

    for _ in range(max_iter):
        prev_inner = current[1:-1, 1:-1]
        new_inner = 0.25 * (
            current[1:-1, :-2]
            + current[1:-1, 2:]
            + current[:-2, 1:-1]
            + current[2:, 1:-1]
        )
        delta = cp.max(cp.abs(new_inner - prev_inner) * mask)

        nxt[1:-1, 1:-1] = prev_inner + (new_inner - prev_inner) * mask
        current, nxt = nxt, current

        if float(delta.get()) < atol:
            break

    cp.cuda.Stream.null.synchronize()
    return cp.asnumpy(current)
