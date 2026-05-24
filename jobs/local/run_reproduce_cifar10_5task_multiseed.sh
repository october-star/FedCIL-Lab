#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

for SEED in 1 ; do
  TASK_SPLIT="data/processed/task_splits/cifar10_5task_seed${SEED}.json"
  PARTITION="data/processed/federated_partitions/cifar10_5task_5clients_beta05_seed${SEED}.json"

  # python scripts/train.py \
  #   --method finetune \
  #   --dataset cifar10 \
  #   --task_split_path "$TASK_SPLIT" \
  #   --partition_path "$PARTITION" \
  #   --num_clients 5 \
  #   --batch_size 128 \
  #   --local_epochs 2 \
  #   --rounds 100 \
  #   --seed "$SEED" \
  #   --run_name "reproduce_core_cifar10_5task_beta05_finetune_seed${SEED}" \
  #   --no_download

  # python scripts/train.py \
  #   --method local_replay \
  #   --dataset cifar10 \
  #   --task_split_path "$TASK_SPLIT" \
  #   --partition_path "$PARTITION" \
  #   --num_clients 5 \
  #   --batch_size 128 \
  #   --local_epochs 2 \
  #   --rounds 100 \
  #   --buffer_size 300 \
  #   --samples_per_task 60 \
  #   --seed "$SEED" \
  #   --run_name "reproduce_core_cifar10_5task_beta05_replay_buf300_seed${SEED}" \
  #   --no_download

  # python scripts/train.py \
  #   --method local_replay_tts \
  #   --dataset cifar10 \
  #   --task_split_path "$TASK_SPLIT" \
  #   --partition_path "$PARTITION" \
  #   --num_clients 5 \
  #   --batch_size 128 \
  #   --local_epochs 2 \
  #   --rounds 100 \
  #   --buffer_size 300 \
  #   --samples_per_task 60 \
  #   --seed "$SEED" \
  #   --tts_old_temp 0.9 \
  #   --tts_new_temp 1.1 \
  #   --tts_old_weight 1.1 \
  #   --tts_new_weight 0.9 \
  #   --run_name "reproduce_core_cifar10_5task_beta05_replay_tts_buf300_seed${SEED}" \
  #   --no_download

  python scripts/train.py \
  --method local_replay_gdr_paper \
  --dataset cifar10 \
  --task_split_path "$TASK_SPLIT" \
  --partition_path "$PARTITION" \
  --num_clients 5 \
  --batch_size 128 \
  --local_epochs 2 \
  --rounds 100 \
  --buffer_size 450 \
  --samples_per_task 90 \
  --gdr_rank 8 \
  --seed "$SEED" \
  --run_name "0524-1-reproduce_core_cifar10_5task_beta05_replay_gdr_paper_buf450_seed${SEED}" \
  --no_download


  # python scripts/train.py \
  #   --method local_replay_gdr_tts_paper \
  #   --dataset cifar10 \
  #   --task_split_path "$TASK_SPLIT" \
  #   --partition_path "$PARTITION" \
  #   --num_clients 5 \
  #   --batch_size 128 \
  #   --local_epochs 2 \
  #   --rounds 100 \
  #   --buffer_size 450 \
  #   --samples_per_task 90 \
  #   --gdr_rank 8 \
  #   --seed "$SEED" \
  #   --tts_old_temp 0.9 \
  #   --tts_new_temp 1.1 \
  #   --tts_old_weight 1.1 \
  #   --tts_new_weight 0.9 \
  #   --run_name "0523-2-reproduce_core_cifar10_5task_beta05_replay_gdr_tts_paper_buf450_seed${SEED}" \
  #   --no_download
done
