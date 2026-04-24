#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

python scripts/summarize_results.py \
  --results_dir outputs/results \
  --output_csv outputs/results/summary.csv
