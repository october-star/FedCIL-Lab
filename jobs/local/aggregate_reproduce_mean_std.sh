#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

python scripts/aggregate_mean_std.py \
  --results_dir outputs/results \
  --pattern "reproduce_core_*.json" \
  --output_csv outputs/results/reproduce_core_mean_std_summary.csv
