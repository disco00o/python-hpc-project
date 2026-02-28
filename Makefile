SHELL := /bin/bash

VENV ?= .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

N ?= 20
MAX_ITER ?= 20000
ATOL ?= 1e-4
WORKERS ?= 12
CHUNKSIZE ?= 1
SOLVER ?= reference
SCHEDULE ?= dynamic
CSV ?= results/solver_stats.csv
CUPY_PKG ?= cupy-cuda13x

HPC_MODEL_RES := select[model == XeonE5_2650v4] span[hosts=1] affinity[socket(1)]

.PHONY: help venv install install-gpu clean run-ref run-solver inspect visualize analyze \
        submit-ref submit-static submit-dynamic submit-numba-cpu submit-numba-cuda \
        submit-cupy submit-cupy-nsys submit-full

help:
	@echo "Targets:"
	@echo "  make venv                      # create project virtualenv (.venv)"
	@echo "  make install                   # install base dependencies"
	@echo "  make install-gpu               # install optional GPU/profiling deps (default: cupy-cuda13x)"
	@echo "  make run-ref N=20              # local reference run"
	@echo "  make run-solver N=40 SOLVER=reference WORKERS=8 SCHEDULE=dynamic"
	@echo "  make inspect N=4               # save input visualizations"
	@echo "  make visualize N=4             # save solved visualizations"
	@echo "  make analyze CSV=results/all_buildings_stats.csv"
	@echo "  make submit-ref N=20           # submit LSF CPU job"
	@echo "  make submit-static N=80        # submit LSF static speedup"
	@echo "  make submit-dynamic N=80       # submit LSF dynamic speedup"
	@echo "  make submit-numba-cpu N=40     # submit LSF Numba CPU"
	@echo "  make submit-numba-cuda N=40 MAX_ITER=2000"
	@echo "  make submit-cupy N=40"
	@echo "  make submit-cupy-nsys N=20"
	@echo "  make submit-full               # submit full dataset run"

$(PYTHON):
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip setuptools wheel

venv: $(PYTHON)

install: venv
	@$(PIP) install -q -r requirements.txt

install-gpu: install
	@echo "Installing optional GPU/profiling deps (CuPy package: $(CUPY_PKG))"
	$(PIP) install numba line_profiler $(CUPY_PKG)

run-ref: install
	@mkdir -p results
	$(PYTHON) scripts/simulate_reference.py $(N) --max-iter $(MAX_ITER) --atol $(ATOL) --output-csv results/reference_stats_$(N).csv

run-solver: install
	@mkdir -p results
	$(PYTHON) scripts/run_solver.py $(N) --solver $(SOLVER) --workers $(WORKERS) --schedule $(SCHEDULE) --chunksize $(CHUNKSIZE) --max-iter $(MAX_ITER) --atol $(ATOL) --output-csv $(CSV)

inspect: install
	@mkdir -p figures
	$(PYTHON) scripts/inspect_data.py --n $(N) --output-dir figures

visualize: install
	@mkdir -p figures
	$(PYTHON) scripts/visualize_results.py --n $(N) --max-iter $(MAX_ITER) --atol $(ATOL) --output-dir figures

analyze: install
	@mkdir -p results figures
	$(PYTHON) scripts/analyze_results.py $(CSV) --summary-txt results/final_summary.txt --output-dir figures

submit-ref: venv
	@mkdir -p results figures
	N=$(N) VENV_PATH=$$(realpath $(VENV)) bsub -env "all" < jobs/lsf_reference_hpc.lsf

submit-static: venv
	@mkdir -p results figures
	N=$(N) VENV_PATH=$$(realpath $(VENV)) bsub -env "all" < jobs/lsf_speedup_static_hpc.lsf

submit-dynamic: venv
	@mkdir -p results figures
	N=$(N) VENV_PATH=$$(realpath $(VENV)) bsub -env "all" < jobs/lsf_speedup_dynamic_hpc.lsf

submit-numba-cpu: venv
	@mkdir -p results figures
	N=$(N) VENV_PATH=$$(realpath $(VENV)) bsub -env "all" < jobs/lsf_numba_cpu_hpc.lsf

submit-numba-cuda: venv
	@mkdir -p results figures
	N=$(N) MAX_ITER=$(MAX_ITER) VENV_PATH=$$(realpath $(VENV)) bsub -env "all" < jobs/lsf_numba_cuda_gpua100.lsf

submit-cupy: venv
	@mkdir -p results figures
	N=$(N) VENV_PATH=$$(realpath $(VENV)) bsub -env "all" < jobs/lsf_cupy_gpua100.lsf

submit-cupy-nsys: venv
	@mkdir -p results figures
	N=$(N) VENV_PATH=$$(realpath $(VENV)) bsub -env "all" < jobs/lsf_cupy_nsys_gpua100.lsf

submit-full: venv
	@mkdir -p results figures
	VENV_PATH=$$(realpath $(VENV)) bsub -env "all" < jobs/lsf_full_run_hpc.lsf

clean:
	rm -rf __pycache__ */__pycache__ .pytest_cache
