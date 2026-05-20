#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

SEED=1

TASK_SPLIT="data/processed/task_splits/cifar10_5task_seed${SEED}.json"
PARTITION="data/processed/federated_partitions/cifar10_5task_5clients_beta05_seed${SEED}.json"

echo "==== Replay ===="
python scripts/train.py \
  --method local_replay \
  --dataset cifar10 \
  --task_split_path "$TASK_SPLIT" \
  --partition_path "$PARTITION" \
  --num_clients 5 \
  --rounds 5 \
  --local_epochs 1 \
  --batch_size 128 \
  --buffer_size 100 \
  --seed "$SEED" \
  --run_name "smoke_replay_seed${SEED}" \
  --no_download

echo "==== GDR Paper ===="
python scripts/train.py \
  --method local_replay_gdr_paper \
  --dataset cifar10 \
  --task_split_path "$TASK_SPLIT" \
  --partition_path "$PARTITION" \
  --num_clients 5 \
  --rounds 5 \
  --local_epochs 1 \
  --batch_size 128 \
  --buffer_size 100 \
  --samples_per_task 20 \
  --gdr_rank 8 \
  --seed "$SEED" \
  --run_name "smoke_gdr_paper_seed${SEED}" \
  --no_download

echo "==== GDR + TTS Paper ===="
python scripts/train.py \
  --method local_replay_gdr_tts_paper \
  --dataset cifar10 \
  --task_split_path "$TASK_SPLIT" \
  --partition_path "$PARTITION" \
  --num_clients 5 \
  --rounds 5 \
  --local_epochs 1 \
  --batch_size 128 \
  --buffer_size 100 \
  --samples_per_task 20 \
  --gdr_rank 8 \
  --tts_old_temp 2.0 \
  --tts_new_temp 1.0 \
  --tts_old_weight 1.5 \
  --tts_new_weight 1.0 \
  --seed "$SEED" \
  --run_name "smoke_gdr_tts_paper_seed${SEED}" \
  --no_download

echo "Done."