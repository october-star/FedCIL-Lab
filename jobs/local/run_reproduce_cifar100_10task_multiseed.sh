#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

for SEED in 1 ; do
  TASK_SPLIT="data/processed/task_splits/cifar100_10task_seed${SEED}.json"
  PARTITION="data/processed/federated_partitions/cifar100_10task_5clients_beta05_seed${SEED}.json"

  # python scripts/train.py \
  #   --method finetune \
  #   --dataset cifar100 \
  #   --task_split_path "$TASK_SPLIT" \
  #   --partition_path "$PARTITION" \
  #   --num_clients 5 \
  #   --batch_size 128 \
  #   --local_epochs 2 \
  #   --rounds 100 \
  #   --seed "$SEED" \
  #   --run_name "reproduce_core_cifar100_10task_beta05_finetune_seed${SEED}" \
  #   --no_download

  # python scripts/train.py \
  #   --method local_replay \
  #   --dataset cifar100 \
  #   --task_split_path "$TASK_SPLIT" \
  #   --partition_path "$PARTITION" \
  #   --num_clients 5 \
  #   --batch_size 128 \
  #   --local_epochs 2 \
  #   --rounds 100 \
  #   --buffer_size 500 \
  #   --samples_per_task 50 \
  #   --seed "$SEED" \
  #   --run_name "reproduce_core_cifar100_10task_beta05_replay_buf500_seed${SEED}" \
  #   --no_download

  # python scripts/train.py \
  #   --method local_replay_tts \
  #   --dataset cifar100 \
  #   --task_split_path "$TASK_SPLIT" \
  #   --partition_path "$PARTITION" \
  #   --num_clients 5 \
  #   --batch_size 128 \
  #   --local_epochs 2 \
  #   --rounds 100 \
  #   --buffer_size 500 \
  #   --samples_per_task 50 \
  #   --seed "$SEED" \
  #   --tts_old_temp 0.9 \
  #   --tts_new_temp 1.1 \
  #   --tts_old_weight 1.1 \
  #   --tts_new_weight 0.9 \
  #   --run_name "reproduce_core_cifar100_10task_beta05_replay_tts_buf500_seed${SEED}" \
  #   --no_download

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
    --samples_per_task 100 \
    --gdr_rank 8 \
    --seed "$SEED" \
    --run_name "0524-3-reproduce_core_cifar100_10task_beta05_replay_gd_kd_paper_buf500_seed${SEED}" \
    --kd_lambda: 1.0 \
    --kd_temperature: 2.0 \
    --no_download


  # python scripts/train.py \
  #   --method local_replay_gdr_tts_paper \
  #   --dataset cifar100 \
  #   --task_split_path "$TASK_SPLIT" \
  #   --partition_path "$PARTITION" \
  #   --num_clients 5 \
  #   --batch_size 128 \
  #   --local_epochs 2 \
  #   --rounds 100 \
  #   --buffer_size 500 \
  #   --samples_per_task 200 \
  #   --gdr_rank 8 \
  #   --seed "$SEED" \
  #   --tts_old_temp 0.9 \
  #   --tts_new_temp 1.1 \
  #   --tts_old_weight 1.1 \
  #   --tts_new_weight 0.9 \
  #   --run_name "0523-2-reproduce_core_cifar100_10task_beta05_replay_gdr_tts_paper_buf500_seed${SEED}" \
  #   --no_download
done
