#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

TASK_SPLIT="data/processed/task_splits/cifar100_10task_seed1.json"
PARTITION="data/processed/federated_partitions/cifar100_10task_5clients_beta05_seed1.json"

COMMON_ARGS=(
  --dataset cifar100
  --task_split_path "$TASK_SPLIT"
  --partition_path "$PARTITION"
  --num_clients 5
  --batch_size 128
  --rounds 100
  --buffer_size 500
  --seed 1
  --no_download
)

echo "[1/2] Running Replay + GDR with backbone features..."
python scripts/train.py \
  --method local_replay_gdr \
  "${COMMON_ARGS[@]}" \
  --samples_per_task 50 \
  --gdr_rank 8 \
  --run_name cifar100_beta05_replay_gdr_backbonecheck_buf500_seed1

echo "[2/2] Running Replay + GDR + TTS with backbone features..."
python scripts/train.py \
  --method local_replay_gdr_tts \
  "${COMMON_ARGS[@]}" \
  --samples_per_task 50 \
  --gdr_rank 8 \
  --tts_old_temp 2.0 \
  --tts_new_temp 1.0 \
  --tts_old_weight 1.5 \
  --tts_new_weight 1.0 \
  --run_name cifar100_beta05_replay_gdr_tts_backbonecheck_buf500_seed1

echo "Done. Compare against the old seed-1 beta=0.5 references:"
echo "  local_replay_gdr      : 0.1174"
echo "  local_replay_gdr_tts  : 0.2044"
