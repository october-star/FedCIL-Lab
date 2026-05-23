#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

SEED=1
DATASET=cifar10
TASKS=5
NUM_CLIENTS=5
BETA=0.5
BETA_TAG=beta05

ROUNDS=20
LOCAL_EPOCHS=2
BATCH_SIZE=128
BUFFER_SIZE=450
GDR_RANK=8
SAMPLES_PER_TASK=90

TASK_SPLIT="data/processed/task_splits/${DATASET}_${TASKS}task_seed${SEED}.json"
PARTITION="data/processed/federated_partitions/${DATASET}_${TASKS}task_${NUM_CLIENTS}clients_${BETA_TAG}_seed${SEED}.json"

mkdir -p outputs/logs outputs/results outputs/analysis/kl_smoke

if [[ ! -f "$TASK_SPLIT" ]]; then
  python scripts/build_splits.py \
    --dataset "$DATASET" \
    --num_tasks "$TASKS" \
    --seed "$SEED"
fi

if [[ ! -f "$PARTITION" ]]; then
  python scripts/build_federated_partitions.py \
    --dataset "$DATASET" \
    --num_tasks "$TASKS" \
    --num_clients "$NUM_CLIENTS" \
    --beta "$BETA" \
    --seed "$SEED"
fi

run_one() {
  local METHOD="$1"
  shift || true
  local EXTRA_ARGS=("$@")

  local RUN_NAME="klsmoke_${DATASET}_${TASKS}task_${BETA_TAG}_${METHOD}_seed${SEED}"

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
    --tts_old_temp 0.9 \
    --tts_new_temp 1.1 \
    --tts_old_weight 1.1 \
    --tts_new_weight 0.9 \
    --seed "$SEED" \
    --run_name "$RUN_NAME" \
    --no_download \
    "${EXTRA_ARGS[@]}"
}

#run_one local_replay_gdr_tts_paper

run_one cbdr_adaptive_reply \
  --adaptive_replay \
  --adaptive_replay_gamma 2.0 \
  --adaptive_replay_min_weight 0.5 \
  --adaptive_replay_max_weight 2.0 \
  --replay_sampling_mass 0.5 \
  --kl_temperature 2.0 \
  --kl_max_samples_per_class 100

#python scripts/analysis/make_cbd_reproduction_outputs.py \
#  --results_dir outputs/results \
#  --prefix "klsmoke_${DATASET}_${TASKS}task_${BETA_TAG}_" \
#  --output_dir outputs/analysis/kl_smoke

echo "Done."