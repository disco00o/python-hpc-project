"""Numba-based solvers (CPU JIT and custom CUDA kernel)."""

from __future__ import annotations

import numpy as np

try:
    from numba import cuda, njit
except ImportError as exc:  # pragma: no cover
    cuda = None
    njit = None
    _NUMBA_IMPORT_ERROR = exc
else:
    _NUMBA_IMPORT_ERROR = None


if njit is not None:

    @njit(cache=True)
    def _jacobi_numba_cpu_kernel(
        u0: np.ndarray,
        interior_mask: np.ndarray,
        max_iter: int,
        atol: float,
    ) -> np.ndarray:
        """CPU kernel with row-major memory access for better cache usage."""
        current = u0.copy()
        nxt = u0.copy()
        rows, cols = current.shape

        for _ in range(max_iter):
            delta = 0.0
            for i in range(1, rows - 1):
                for j in range(1, cols - 1):
                    if interior_mask[i - 1, j - 1]:
                        new_val = 0.25 * (
                            current[i, j - 1]
                            + current[i, j + 1]
                            + current[i - 1, j]
                            + current[i + 1, j]
                        )
                        diff = abs(new_val - current[i, j])
                        if diff > delta:
                            delta = diff
                        nxt[i, j] = new_val
                    else:
                        nxt[i, j] = current[i, j]

            current, nxt = nxt, current
            if delta < atol:
                break

        return current


    @cuda.jit
    def _jacobi_cuda_step(current: np.ndarray, nxt: np.ndarray, interior_mask: np.ndarray) -> None:
        """One Jacobi iteration for interior region (synchronization happens between launches)."""
        i, j = cuda.grid(2)
        rows, cols = current.shape

        if 1 <= i < rows - 1 and 1 <= j < cols - 1:
            if interior_mask[i - 1, j - 1]:
                nxt[i, j] = 0.25 * (
                    current[i, j - 1]
                    + current[i, j + 1]
                    + current[i - 1, j]
                    + current[i + 1, j]
                )
            else:
                nxt[i, j] = current[i, j]


def _require_numba() -> None:
    if _NUMBA_IMPORT_ERROR is not None:
        raise RuntimeError(
            "Numba is not installed. Install with 'pip install numba' in your HPC environment."
        ) from _NUMBA_IMPORT_ERROR


def jacobi_numba_cpu(
    u: np.ndarray,
    interior_mask: np.ndarray,
    max_iter: int,
    atol: float = 1e-6,
) -> np.ndarray:
    """Drop-in replacement for the reference Jacobi function using CPU JIT."""
    _require_numba()
    return _jacobi_numba_cpu_kernel(u, interior_mask, max_iter, atol)


def jacobi_numba_cuda(
    u: np.ndarray,
    interior_mask: np.ndarray,
    max_iter: int,
    atol: float = 0.0,  # ignored to keep same signature as other solvers
) -> np.ndarray:
    """Run fixed-iteration Jacobi on GPU with a custom Numba CUDA kernel."""
    _require_numba()
    if cuda is None:
        raise RuntimeError("Numba CUDA support is unavailable.")

    current_d = cuda.to_device(np.asarray(u))
    nxt_d = cuda.to_device(np.asarray(u))
    mask_d = cuda.to_device(np.asarray(interior_mask, dtype=np.bool_))

    threads_per_block = (16, 16)
    blocks_per_grid = (
        (u.shape[0] + threads_per_block[0] - 1) // threads_per_block[0],
        (u.shape[1] + threads_per_block[1] - 1) // threads_per_block[1],
    )

    for _ in range(max_iter):
        _jacobi_cuda_step[blocks_per_grid, threads_per_block](current_d, nxt_d, mask_d)
        current_d, nxt_d = nxt_d, current_d

    cuda.synchronize()
    return current_d.copy_to_host()
