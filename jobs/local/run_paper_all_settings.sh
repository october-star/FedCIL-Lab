#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

SEEDS=(1)
NUM_CLIENTS=5

ROUNDS=100
LOCAL_EPOCHS=2
BATCH_SIZE=128
BUFFER_SIZE=450
GDR_RANK=8

#declare -A SAMPLES_PER_TASK_MAP=(
#  ["cifar10_3"]=90
#  ["cifar10_5"]=60
#  ["cifar100_5"]=200
#  ["cifar100_10"]=100
#)

mkdir -p outputs/logs outputs/results outputs/analysis/final

METHODS=(
#  finetune
#  local_replay
  local_replay_gdr_paper
#  local_replay_tts
#  local_replay_gdr_tts_paper
  cbdr_adaptive_reply
)

prepare_setting() {
  local DATASET="$1"
  local TASKS="$2"
  local BETA="$3"
  local BETA_TAG="$4"
  local SEED="$5"

  local TASK_SPLIT="data/processed/task_splits/${DATASET}_${TASKS}task_seed${SEED}.json"
  local PARTITION="data/processed/federated_partitions/${DATASET}_${TASKS}task_${NUM_CLIENTS}clients_${BETA_TAG}_seed${SEED}.json"

  if [[ ! -f "$TASK_SPLIT" ]]; then
    echo "==> Building split: ${DATASET} ${TASKS}task seed=${SEED}"
    python scripts/build_splits.py \
      --dataset "$DATASET" \
      --num_tasks "$TASKS" \
      --seed "$SEED"
  fi

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
  local SEED="$5"

  local TASK_SPLIT="data/processed/task_splits/${DATASET}_${TASKS}task_seed${SEED}.json"
  local PARTITION="data/processed/federated_partitions/${DATASET}_${TASKS}task_${NUM_CLIENTS}clients_${BETA_TAG}_seed${SEED}.json"
  local RUN_NAME="paper_${DATASET}_${TASKS}task_${BETA_TAG}_${METHOD}_seed${SEED}"
  local RESULT_PATH="outputs/results/${RUN_NAME}.json"

#  if [[ -f "$RESULT_PATH" ]]; then
#    echo "[SKIP] already exists: $RESULT_PATH"
#    return
#  fi

  local KEY="${DATASET}_${TASKS}"
  local SAMPLES_PER_TASK="${SAMPLES_PER_TASK_MAP[$KEY]:-}"

  if [[ -z "$SAMPLES_PER_TASK" ]]; then
    echo "[ERROR] Missing samples_per_task for ${KEY}"
    exit 1
  fi

  echo "=================================================="
  echo "Running $RUN_NAME"
  echo "samples_per_task=${SAMPLES_PER_TASK}"
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
    --tts_old_temp 0.9 \
    --tts_new_temp 1.1 \
    --tts_old_weight 1.1 \
    --tts_new_weight 0.9 \
    --candidate_pool_multiplier 2.0 \
    --adaptive_replay_gamma 0.5 \
    --adaptive_replay_min_weight 0.8 \
    --adaptive_replay_max_weight 1.2 \
    --kl_temperature 2.0 \
    --kl_max_samples_per_class 100 \
    --seed "$SEED" \
    --run_name "$RUN_NAME" \
    --no_download
}

run_setting() {
  local DATASET="$1"
  local TASKS="$2"
  local BETA="$3"
  local BETA_TAG="$4"
  local SEED="$5"

  echo ""
  echo "##################################################"
  echo "Setting: dataset=${DATASET}, tasks=${TASKS}, beta=${BETA}, seed=${SEED}"
  echo "##################################################"

  prepare_setting "$DATASET" "$TASKS" "$BETA" "$BETA_TAG" "$SEED"

  for METHOD in "${METHODS[@]}"; do
    run_one "$DATASET" "$TASKS" "$BETA_TAG" "$METHOD" "$SEED"
  done
}

for SEED in "${SEEDS[@]}"; do
  echo ""
  echo "=================================================="
  echo "Running seed=${SEED}"
  echo "=================================================="

#  for TASKS in 3 5; do
#    run_setting cifar10 "$TASKS" 0.5 beta05 "$SEED"
#    run_setting cifar10 "$TASKS" 1.0 beta10 "$SEED"
#  done
#
#  for TASKS in 5 10; do
#    run_setting cifar100 "$TASKS" 0.1 beta01 "$SEED"
#    run_setting cifar100 "$TASKS" 0.5 beta05 "$SEED"
#    run_setting cifar100 "$TASKS" 1.0 beta10 "$SEED"
#  done
done

python scripts/analysis/make_cbd_reproduction_outputs.py \
  --results_dir outputs/results \
  --prefix "paper_" \
  --output_dir outputs/analysis/final

echo "Done."
