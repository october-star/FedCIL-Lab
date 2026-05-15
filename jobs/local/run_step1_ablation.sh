#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

DATASET="cifar10"
NUM_TASKS=5
NUM_CLIENTS=5
BETA=0.5
SEED=1
BUFFER_SIZE=300
ROUNDS=100
LOCAL_EPOCHS=2
BATCH_SIZE=128

TASK_SPLIT="data/processed/task_splits/${DATASET}_${NUM_TASKS}task_seed${SEED}.json"
PARTITION="data/processed/federated_partitions/${DATASET}_${NUM_TASKS}task_${NUM_CLIENTS}clients_beta05_seed${SEED}.json"

echo "==> Preparing ${DATASET} data"
python scripts/prepare_data.py --dataset "$DATASET"

echo "==> Building task split"
python scripts/build_splits.py \
  --dataset "$DATASET" \
  --num_tasks "$NUM_TASKS" \
  --seed "$SEED"

echo "==> Building federated partition"
python scripts/build_federated_partitions.py \
  --dataset "$DATASET" \
  --num_tasks "$NUM_TASKS" \
  --num_clients "$NUM_CLIENTS" \
  --beta "$BETA" \
  --seed "$SEED"

COMMON_ARGS=(
  --dataset "$DATASET"
  --task_split_path "$TASK_SPLIT"
  --partition_path "$PARTITION"
  --num_clients "$NUM_CLIENTS"
  --batch_size "$BATCH_SIZE"
  --local_epochs "$LOCAL_EPOCHS"
  --rounds "$ROUNDS"
  --seed "$SEED"
  --no_download
)

# echo "==> [A1] Finetune"
# python scripts/train.py \
#   --method finetune \
#   "${COMMON_ARGS[@]}" \
#   --run_name A1_finetune_cifar10_seed1_beta05

# echo "==> [A2] Local Replay"
# python scripts/train.py \
#   --method local_replay \
#   "${COMMON_ARGS[@]}" \
#   --buffer_size "$BUFFER_SIZE" \
#   --run_name A2_local_replay_cifar10_seed1_beta05

# echo "==> [A3] Local Replay + GDR"
# python scripts/train.py \
#   --method local_replay_gdr \
#   "${COMMON_ARGS[@]}" \
#   --buffer_size "$BUFFER_SIZE" \
#   --run_name A3_local_replay_gdr_cifar10_seed1_beta05

# echo "==> [A4] Local Replay + TTS"
# python scripts/train.py \
#   --method local_replay_tts \
#   "${COMMON_ARGS[@]}" \
#   --buffer_size "$BUFFER_SIZE" \
#   --run_name A4_local_replay_tts_cifar10_seed1_beta05

# echo "==> [A5] FedCBDR (GDR + TTS)"
# python scripts/train.py \
#   --method local_replay_gdr_tts \
#   "${COMMON_ARGS[@]}" \
#   --buffer_size "$BUFFER_SIZE" \
#   --run_name A5_fedcbdr_cifar10_seed1_beta05

echo "==> [A8] FedCBDR (GDR + TTS + class_balanced)"
python scripts/train.py \
  --method local_replay_gdr_tts_paper \
  "${COMMON_ARGS[@]}" \
  --buffer_size "$BUFFER_SIZE" \
  --run_name A8_fedcbdr_paper_cifar10_seed1_beta05

echo "==> Step 1 ablation finished"
echo "==> Summarizing results"
python scripts/summarize_results.py
