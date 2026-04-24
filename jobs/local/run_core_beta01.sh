#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

TASK_SPLIT="data/processed/task_splits/cifar100_10task_seed1.json"
PARTITION="data/processed/federated_partitions/cifar100_10task_5clients_beta01_seed1.json"

COMMON_ARGS=(
  --dataset cifar100
  --task_split_path "$TASK_SPLIT"
  --partition_path "$PARTITION"
  --num_clients 5
  --batch_size 128
  --rounds 100
  --buffer_size 500
  --no_download
)

python scripts/train.py \
  --method local_replay \
  "${COMMON_ARGS[@]}" \
  --run_name cifar100_beta01_replay_buf500_seed1

python scripts/train.py \
  --method local_replay_tts \
  "${COMMON_ARGS[@]}" \
  --tts_old_temp 2.0 \
  --tts_new_temp 1.0 \
  --tts_old_weight 1.5 \
  --tts_new_weight 1.0 \
  --run_name cifar100_beta01_replay_tts_buf500_seed1

python scripts/train.py \
  --method local_replay_gdr \
  "${COMMON_ARGS[@]}" \
  --samples_per_task 50 \
  --gdr_rank 8 \
  --run_name cifar100_beta01_replay_gdr_buf500_seed1

python scripts/train.py \
  --method local_replay_gdr_tts \
  "${COMMON_ARGS[@]}" \
  --samples_per_task 50 \
  --gdr_rank 8 \
  --tts_old_temp 2.0 \
  --tts_new_temp 1.0 \
  --tts_old_weight 1.5 \
  --tts_new_weight 1.0 \
  --run_name cifar100_beta01_replay_gdr_tts_buf500_seed1
