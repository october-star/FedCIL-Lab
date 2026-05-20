#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

mkdir -p logs

for SEED in 1 2 3; do

  TASK_SPLIT="data/processed/task_splits/cifar100_10task_seed${SEED}.json"
  PARTITION="data/processed/federated_partitions/cifar100_10task_5clients_beta01_seed${SEED}.json"

  echo "===================================================="
  echo "Running seed ${SEED}"
  echo "===================================================="

  ########################################
  # Replay baseline
  ########################################
  python scripts/train.py \
    --method local_replay \
    --dataset cifar100 \
    --task_split_path "$TASK_SPLIT" \
    --partition_path "$PARTITION" \
    --num_clients 5 \
    --rounds 100 \
    --local_epochs 1 \
    --batch_size 128 \
    --buffer_size 500 \
    --seed "$SEED" \
    --run_name "cifar100_beta01_replay_seed${SEED}" \
    --no_download \
    > "logs/replay_seed${SEED}.log" 2>&1

  ########################################
  # Replay + TTS
  ########################################
  python scripts/train.py \
    --method local_replay_tts \
    --dataset cifar100 \
    --task_split_path "$TASK_SPLIT" \
    --partition_path "$PARTITION" \
    --num_clients 5 \
    --rounds 100 \
    --local_epochs 1 \
    --batch_size 128 \
    --buffer_size 500 \
    --tts_old_temp 2.0 \
    --tts_new_temp 1.0 \
    --tts_old_weight 1.5 \
    --tts_new_weight 1.0 \
    --seed "$SEED" \
    --run_name "cifar100_beta01_replay_tts_seed${SEED}" \
    --no_download \
    > "logs/replay_tts_seed${SEED}.log" 2>&1

  ########################################
  # Replay + GDR (paper)
  ########################################
  python scripts/train.py \
    --method local_replay_gdr_paper \
    --dataset cifar100 \
    --task_split_path "$TASK_SPLIT" \
    --partition_path "$PARTITION" \
    --num_clients 5 \
    --rounds 100 \
    --local_epochs 1 \
    --batch_size 128 \
    --buffer_size 500 \
    --samples_per_task 50 \
    --gdr_rank 32 \
    --seed "$SEED" \
    --run_name "cifar100_beta01_gdr_paper_seed${SEED}" \
    --no_download \
    > "logs/gdr_paper_seed${SEED}.log" 2>&1

  ########################################
  # Replay + GDR + TTS (paper)
  ########################################
  python scripts/train.py \
    --method local_replay_gdr_tts_paper \
    --dataset cifar100 \
    --task_split_path "$TASK_SPLIT" \
    --partition_path "$PARTITION" \
    --num_clients 5 \
    --rounds 100 \
    --local_epochs 1 \
    --batch_size 128 \
    --buffer_size 500 \
    --samples_per_task 50 \
    --gdr_rank 32 \
    --tts_old_temp 2.0 \
    --tts_new_temp 1.0 \
    --tts_old_weight 1.5 \
    --tts_new_weight 1.0 \
    --seed "$SEED" \
    --run_name "cifar100_beta01_gdr_tts_paper_seed${SEED}" \
    --no_download \
    > "logs/gdr_tts_paper_seed${SEED}.log" 2>&1

done

echo "All runs finished."