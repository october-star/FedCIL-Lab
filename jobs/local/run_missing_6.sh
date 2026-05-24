#!/bin/bash
set -euo pipefail

ROOT_DIR="/storage/homefs/cl25n064/FedCIL-Lab"
cd "$ROOT_DIR"
source /storage/homefs/cl25n064/venvs/fedcil_fixed/bin/activate

NUM_CLIENTS=5
ROUNDS=100
LOCAL_EPOCHS=2
BATCH_SIZE=128
BUFFER_SIZE=500
SAMPLES_PER_TASK=50
GDR_RANK=8

mkdir -p outputs/logs outputs/results

run_one() {
  local DATASET="$1"
  local TASKS="$2"
  local BETA_TAG="$3"
  local METHOD="$4"
  local SEED="$5"
  local TASK_SPLIT="data/processed/task_splits/${DATASET}_${TASKS}task_seed${SEED}.json"
  local PARTITION="data/processed/federated_partitions/${DATASET}_${TASKS}task_${NUM_CLIENTS}clients_${BETA_TAG}_seed${SEED}.json"
  local RUN_NAME="paper_${DATASET}_${TASKS}task_${BETA_TAG}_${METHOD}_seed${SEED}"
  local RESULT_PATH="outputs/results/${RUN_NAME}.json"
  if [[ -f "$RESULT_PATH" ]]; then
    echo "[SKIP] $RESULT_PATH"
    return
  fi
  echo "=================================================="
  echo "Running $RUN_NAME"
  echo "=================================================="
  python scripts/train.py \
    --method "$METHOD" \
    --dataset "$DATASET" \
    --task_split_path "$TASK_SPLIT" \
    --partition_path "$PARTITION" \
    --num_clients "$NUM_CLIENTS" \
    --rounds "$ROUNDS" \
    --local_epochs "$LOCAL_EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --buffer_size "$BUFFER_SIZE" \
    --samples_per_task "$SAMPLES_PER_TASK" \
    --gdr_rank "$GDR_RANK" \
    --tts_old_temp 2.0 \
    --tts_new_temp 1.0 \
    --tts_old_weight 1.5 \
    --tts_new_weight 1.0 \
    --seed "$SEED" \
    --run_name "$RUN_NAME" \
    --no_download
}

run_one cifar100 10 beta05 local_replay 2
run_one cifar100 10 beta05 local_replay 3
run_one cifar100 10 beta10 finetune     2
run_one cifar100 10 beta10 finetune     3
run_one cifar100 10 beta10 local_replay 2
run_one cifar100 10 beta10 local_replay 3

echo "Done."
