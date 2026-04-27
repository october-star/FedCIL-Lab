#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

for SEED in 1 2 3; do
  TASK_SPLIT="data/processed/task_splits/cifar100_10task_seed${SEED}.json"
  PARTITION="data/processed/federated_partitions/cifar100_10task_5clients_beta05_seed${SEED}.json"

  python scripts/train.py \
    --method local_replay \
    --dataset cifar100 \
    --task_split_path "$TASK_SPLIT" \
    --partition_path "$PARTITION" \
    --num_clients 5 \
    --batch_size 128 \
    --rounds 100 \
    --buffer_size 500 \
    --seed "$SEED" \
    --run_name "cifar100_beta05_replay_buf500_seed${SEED}" \
    --no_download

  python scripts/train.py \
    --method local_replay_tts \
    --dataset cifar100 \
    --task_split_path "$TASK_SPLIT" \
    --partition_path "$PARTITION" \
    --num_clients 5 \
    --batch_size 128 \
    --rounds 100 \
    --buffer_size 500 \
    --seed "$SEED" \
    --tts_old_temp 2.0 \
    --tts_new_temp 1.0 \
    --tts_old_weight 1.5 \
    --tts_new_weight 1.0 \
    --run_name "cifar100_beta05_replay_tts_buf500_seed${SEED}" \
    --no_download

  python scripts/train.py \
    --method local_replay_gdr \
    --dataset cifar100 \
    --task_split_path "$TASK_SPLIT" \
    --partition_path "$PARTITION" \
    --num_clients 5 \
    --batch_size 128 \
    --rounds 100 \
    --buffer_size 500 \
    --samples_per_task 50 \
    --gdr_rank 8 \
    --seed "$SEED" \
    --run_name "cifar100_beta05_replay_gdr_backbonegdr_buf500_seed${SEED}" \
    --no_download

  python scripts/train.py \
    --method local_replay_gdr_tts \
    --dataset cifar100 \
    --task_split_path "$TASK_SPLIT" \
    --partition_path "$PARTITION" \
    --num_clients 5 \
    --batch_size 128 \
    --rounds 100 \
    --buffer_size 500 \
    --samples_per_task 50 \
    --gdr_rank 8 \
    --seed "$SEED" \
    --tts_old_temp 2.0 \
    --tts_new_temp 1.0 \
    --tts_old_weight 1.5 \
    --tts_new_weight 1.0 \
    --run_name "cifar100_beta05_replay_gdr_tts_backbonegdr_buf500_seed${SEED}" \
    --no_download
done
