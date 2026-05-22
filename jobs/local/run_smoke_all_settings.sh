#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

SEED=1
NUM_CLIENTS=5

ROUNDS=5
LOCAL_EPOCHS=1
BATCH_SIZE=128
BUFFER_SIZE=100
SAMPLES_PER_TASK=20
GDR_RANK=8

mkdir -p outputs/logs outputs/results outputs/analysis/smoke

METHODS=(
  finetune
  local_replay
#  local_replay_gdr_paper
#  local_replay_tts
#  local_replay_gdr_tts_paper
)

prepare_setting() {
  local DATASET="$1"
  local TASKS="$2"
  local BETA="$3"
  local BETA_TAG="$4"

  local TASK_SPLIT="data/processed/task_splits/${DATASET}_${TASKS}task_seed${SEED}.json"
  local PARTITION="data/processed/federated_partitions/${DATASET}_${TASKS}task_${NUM_CLIENTS}clients_${BETA_TAG}_seed${SEED}.json"

  # build task split if missing
  if [[ ! -f "$TASK_SPLIT" ]]; then
    echo "==> Building split: ${DATASET} ${TASKS}task seed=${SEED}"

    python scripts/build_splits.py \
      --dataset "$DATASET" \
      --num_tasks "$TASKS" \
      --seed "$SEED"
  fi

  # build partition if missing
  if [[ ! -f "$PARTITION" ]]; then
    echo "==> Building partition: ${DATASET} ${TASKS}task beta=${BETA} seed=${SEED}"

    python scripts/build_federated_partitions.py \
      --dataset "$DATASET" \
      --num_tasks "$TASKS" \
      --num_clients "$NUM_CLIENTS" \
      --beta "$BETA" \
      --seed "$SEED"
  fi
}

run_one() {
  local DATASET="$1"
  local TASKS="$2"
  local BETA_TAG="$3"
  local METHOD="$4"

  local TASK_SPLIT="data/processed/task_splits/${DATASET}_${TASKS}task_seed${SEED}.json"
  local PARTITION="data/processed/federated_partitions/${DATASET}_${TASKS}task_${NUM_CLIENTS}clients_${BETA_TAG}_seed${SEED}.json"

  local RUN_NAME="smoke_${DATASET}_${TASKS}task_${BETA_TAG}_${METHOD}_seed${SEED}"

  if [[ -f "outputs/results/${RUN_NAME}.json" ]]; then
    echo "[SKIP] already exists: ${RUN_NAME}"
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

run_setting() {
  local DATASET="$1"
  local TASKS="$2"
  local BETA="$3"
  local BETA_TAG="$4"

  echo ""
  echo "##################################################"
  echo "Setting: dataset=${DATASET}, tasks=${TASKS}, beta=${BETA}"
  echo "##################################################"

  prepare_setting "$DATASET" "$TASKS" "$BETA" "$BETA_TAG"

  for METHOD in "${METHODS[@]}"; do
    run_one "$DATASET" "$TASKS" "$BETA_TAG" "$METHOD"
  done
}

# ---------------- CIFAR10 ----------------
for TASKS in 3 5; do
  run_setting cifar10 "$TASKS" 0.5 beta05
  run_setting cifar10 "$TASKS" 1.0 beta10
done

# ---------------- CIFAR100 ----------------
#for TASKS in 5 10; do
#  run_setting cifar100 "$TASKS" 0.1 beta01
#  run_setting cifar100 "$TASKS" 0.5 beta05
#  run_setting cifar100 "$TASKS" 1.0 beta10
#done

python scripts/analysis/make_cbd_reproduction_outputs.py \
  --results_dir outputs/results \
  --prefix "smoke_" \
  --output_dir outputs/analysis/smoke

echo "Done."