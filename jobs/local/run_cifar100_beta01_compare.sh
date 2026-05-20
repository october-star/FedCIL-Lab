#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

mkdir -p logs

SEED=1
DATASET="cifar100"
NUM_TASKS=10
NUM_CLIENTS=5
BETA="01"

ROUNDS=30
BUFFER_SIZE=500
SAMPLES_PER_TASK=50
GDR_RANK=8
BATCH_SIZE=128

TASK_SPLIT="data/processed/task_splits/${DATASET}_${NUM_TASKS}task_seed${SEED}.json"

PARTITION="data/processed/federated_partitions/${DATASET}_${NUM_TASKS}task_${NUM_CLIENTS}clients_beta${BETA}_seed${SEED}.json"

COMMON_ARGS=(
  --dataset "$DATASET"
  --task_split_path "$TASK_SPLIT"
  --partition_path "$PARTITION"
  --num_clients "$NUM_CLIENTS"
  --batch_size "$BATCH_SIZE"
  --rounds "$ROUNDS"
  --local_epochs 1
  --buffer_size "$BUFFER_SIZE"
  --samples_per_task "$SAMPLES_PER_TASK"
  --seed "$SEED"
  --no_download
)

echo "======================================================"
echo "CIFAR100 beta=0.1 seed=${SEED}"
echo "======================================================"

echo "===== Local Replay ====="
python scripts/train.py \
  --method local_replay \
  "${COMMON_ARGS[@]}" \
  --run_name "compare_cifar100_beta01_replay_buf500_seed${SEED}" \
  2>&1 | tee "logs/compare_cifar100_beta01_replay_seed${SEED}.log"

echo "===== Local Replay + GDR ====="
python scripts/train.py \
  --method local_replay_gdr \
  "${COMMON_ARGS[@]}" \
  --gdr_rank "$GDR_RANK" \
  --run_name "compare_cifar100_beta01_gdr_buf500_seed${SEED}" \
  2>&1 | tee "logs/compare_cifar100_beta01_gdr_seed${SEED}.log"

echo "======================================================"
echo "Finished."
echo "======================================================"

echo "Replay result:"
echo "outputs/results/compare_cifar100_beta01_replay_buf500_seed${SEED}.json"

echo "GDR result:"
echo "outputs/results/compare_cifar100_beta01_gdr_buf500_seed${SEED}.json"