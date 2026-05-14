#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

for SEED in 1 2 3; do
  TASK_SPLIT="data/processed/task_splits/cifar100_10task_seed${SEED}.json"
  PARTITION="data/processed/federated_partitions/cifar100_10task_5clients_beta01_seed${SEED}.json"

  python scripts/train.py \
    --method local_replay_gdr_paper \
    --dataset cifar100 \
    --task_split_path "$TASK_SPLIT" \
    --partition_path "$PARTITION" \
    --num_clients 5 \
    --batch_size 128 \
    --local_epochs 2 \
    --rounds 100 \
    --buffer_size 500 \
    --gdr_rank 8 \
    --no-gdr_class_wise \
    --seed "$SEED" \
    --run_name "cifar100_beta01_replay_gdr_paper_buf500_seed${SEED}" \
    --no_download
done
