# python-hpc-project

Project scaffold for the 02613 *Wall Heating* mini-project.

The dataset path used by all scripts is:
`/dtu/projects/02613_2025/data/modified_swiss_dwellings`

## What is included

- `wall_heating/`: reusable code for loading data, running Jacobi solvers, metrics, and parallel scheduling
- `scripts/`: runnable scripts for each assignment task
- `jobs/`: IBM LSF job scripts for reproducible timing experiments
- `results/` and `figures/`: default output folders

## Quick start

Run from repo root.

```bash
make install
make install-gpu
make run-ref N=20
make inspect N=4
make visualize N=4 MAX_ITER=2000
```

## Main experiment runner

Use one script for most variants:

```bash
python scripts/run_solver.py 40 --solver reference --workers 1
python scripts/run_solver.py 40 --solver reference --workers 8 --schedule static
python scripts/run_solver.py 40 --solver reference --workers 8 --schedule dynamic
python scripts/run_solver.py 40 --solver numba-cpu --workers 1
python scripts/run_solver.py 40 --solver numba-cuda --workers 1 --max-iter 2000
python scripts/run_solver.py 40 --solver cupy --workers 1
```

Solver options:
- `reference`: direct baseline from assignment text
- `numpy`: optimized NumPy ping-pong version
- `numba-cpu`: CPU JIT (needs `numba`)
- `numba-cuda`: custom CUDA kernel, fixed iterations (needs `numba` + CUDA)
- `cupy`: GPU vectorized implementation (needs `cupy`)

## Task-specific scripts

- Task 4 profile: `kernprof -l -v scripts/profile_jacobi.py`
- Task 5/6 speedup + Amdahl: `python scripts/benchmark_speedup.py --schedule static` and `--schedule dynamic`
- Task 10 CuPy nsys: `bash scripts/profile_cupy_nsys.sh 20`
- Task 12 final analysis: `python scripts/analyze_results.py results/all_buildings_stats.csv`

## IBM LSF jobs

Created LSF job scripts for your queues (`hpc` and `gpua100`) under `jobs/`:

- `jobs/lsf_reference_hpc.lsf`
- `jobs/lsf_speedup_static_hpc.lsf`
- `jobs/lsf_speedup_dynamic_hpc.lsf`
- `jobs/lsf_numba_cpu_hpc.lsf`
- `jobs/lsf_numba_cuda_gpua100.lsf`
- `jobs/lsf_cupy_gpua100.lsf`
- `jobs/lsf_cupy_nsys_gpua100.lsf`
- `jobs/lsf_full_run_hpc.lsf`

Submit examples:

```bash
make submit-ref N=20
make submit-static N=80
make submit-numba-cuda N=40 MAX_ITER=2000
make submit-cupy N=40
```

The LSF scripts now use the project virtualenv via `VENV_PATH` (default: repo-local `.venv`).

## Notes

- This environment currently has `numpy`, `matplotlib`, `pandas`; `numba`, `cupy`, and `line_profiler` are not installed.
- For GPU tasks on HPC, install matching versions in your job environment before running GPU scripts.
- For your GPU driver/CUDA (`Driver 580.126.20`, `CUDA 13.0`), `make install-gpu` defaults to `cupy-cuda13x`.
