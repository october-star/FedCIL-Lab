#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

for SEED in 1 2 3; do
  python scripts/build_splits.py \
    --dataset cifar100 \
    --num_tasks 10 \
    --seed "$SEED"

  python scripts/build_federated_partitions.py \
    --dataset cifar100 \
    --num_tasks 10 \
    --num_clients 5 \
    --beta 0.1 \
    --seed "$SEED"

  python scripts/build_federated_partitions.py \
    --dataset cifar100 \
    --num_tasks 10 \
    --num_clients 5 \
    --beta 0.5 \
    --seed "$SEED"
done
