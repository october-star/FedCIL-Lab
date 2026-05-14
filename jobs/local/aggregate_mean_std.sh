#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

python scripts/aggregate_mean_std.py \
  --results_dir outputs/results \
  --output_csv outputs/results/mean_std_summary.csv
