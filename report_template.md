# 02613 Mini-Project Report Template

## 1. Data familiarization

- Show a few input floorplans (`scripts/inspect_data.py` outputs).
- Briefly describe what `domain` and `interior_mask` represent.

## 2. Reference timing

- Command used:
  - `python scripts/simulate_reference.py <N> --output-csv ...`
- LSF job script used:
  - `jobs/lsf_reference_hpc.lsf`
- Measured runtime for `N=...`:
- Extrapolated runtime for all 4571 buildings:

## 3. Simulation visualizations

- Show solved temperature maps (`scripts/visualize_results.py` outputs).
- Brief interpretation.

## 4. Line profiling of reference Jacobi

- Command:
  - `kernprof -l -v scripts/profile_jacobi.py --building-id ...`
- Include the profiler table and explain top time-consuming lines.

## 5. Parallel over floorplans (static scheduling)

- Implementation: `scripts/run_parallel_static.py` (wrapper over `run_solver.py`).
- Speedup experiments: `scripts/benchmark_speedup.py --schedule static`
- Include speedup plot and table.

### 5a. Speedup vs workers

### 5b. Amdahl parallel fraction

Use:
\[
S(N) = \frac{1}{(1-p) + p/N}
\]

Estimated from measured data:
\[
p = \frac{1 - 1/S(N)}{1 - 1/N}
\]

### 5c. Theoretical max speedup and achieved fraction

\[
S_{max} = \frac{1}{1-p}
\]

### 5d. Estimated time for all buildings

## 6. Parallel over floorplans (dynamic scheduling)

- Implementation: `scripts/run_parallel_dynamic.py`
- Compare against static from task 5.

## 7. Numba CPU JIT

- Implementation: `wall_heating/numba_impl.py::jacobi_numba_cpu`
- Runner: `scripts/run_numba_cpu.py`

### 7a. Performance vs reference

### 7b. Access pattern and cache discussion

### 7c. Estimated full dataset runtime

## 8. Numba custom CUDA kernel

- Implementation: `wall_heating/numba_impl.py::jacobi_numba_cuda`
- Kernel: one Jacobi iteration per launch, fixed iterations.

### 8a. Kernel/helper structure

### 8b. Performance vs reference

### 8c. Estimated full dataset runtime

## 9. CuPy implementation

- Implementation: `wall_heating/cupy_impl.py::jacobi_cupy`
- Runner: `scripts/run_cupy.py`

### 9a. Performance vs reference

### 9b. Estimated full dataset runtime

### 9c. Surprises / observations

## 10. nsys profile of CuPy solution

- Script: `scripts/profile_cupy_nsys.sh`
- Include nsys findings and the main bottleneck.
- Describe fix and impact on runtime.

## 11. Optional extra optimization

- Describe additional optimization and results.

## 12. Full dataset analysis

- Run full processing and generate CSV.
- Analyze with `scripts/analyze_results.py`.

### 12a. Mean-temperature histogram

### 12b. Average mean temperature

### 12c. Average temperature std-dev

### 12d. Buildings with >=50% area above 18C

### 12e. Buildings with >=50% area below 15C

## Appendix

- Environment details (CPU/GPU, Python, package versions)
- Command log and LSF job IDs
