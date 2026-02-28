#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/profile_cupy_nsys.sh 20
# Produces an nsys profile under results/nsys_cupy.

N="${1:-20}"
OUT="results/nsys_cupy"
mkdir -p "$(dirname "$OUT")"

nsys profile \
  --stats=true \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --output="$OUT" \
  python scripts/run_cupy.py "$N" --max-iter 2000 --output-csv results/cupy_profile.csv

echo "Wrote ${OUT}.qdrep and ${OUT}.nsys-rep"
