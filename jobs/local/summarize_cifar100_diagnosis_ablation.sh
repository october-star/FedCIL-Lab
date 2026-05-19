#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

python scripts/summarize_results.py \
  --results_dir outputs/results \
  --pattern "diag_cifar100_*.json" \
  --output_csv outputs/results/diag_cifar100_summary.csv
