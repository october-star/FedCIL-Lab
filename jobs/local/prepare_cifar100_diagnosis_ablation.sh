#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

SEEDS=(${SEEDS:-1 2 3})
NUM_CLIENTS=${NUM_CLIENTS:-5}
BETA_VALUES=(${BETA_VALUES:-0.5 1.0 10.0})
TASK_COUNTS=(${TASK_COUNTS:-5 10})

echo "==> Preparing CIFAR-100 diagnosis ablation splits"

for SEED in "${SEEDS[@]}"; do
  for NUM_TASKS in "${TASK_COUNTS[@]}"; do
    echo "==> Split: cifar100 ${NUM_TASKS}task seed=${SEED}"
    python scripts/build_splits.py \
      --dataset cifar100 \
      --num_tasks "${NUM_TASKS}" \
      --seed "${SEED}"

    for BETA in "${BETA_VALUES[@]}"; do
      echo "==> Partition: cifar100 ${NUM_TASKS}task beta=${BETA} seed=${SEED}"
      python scripts/build_federated_partitions.py \
        --dataset cifar100 \
        --num_tasks "${NUM_TASKS}" \
        --num_clients "${NUM_CLIENTS}" \
        --beta "${BETA}" \
        --seed "${SEED}"
    done
  done
done

echo "==> CIFAR-100 diagnosis ablation splits prepared"
