#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

SEEDS=(${SEEDS:-1 2 3})
NUM_CLIENTS=${NUM_CLIENTS:-5}
ROUNDS=${ROUNDS:-100}
LOCAL_EPOCHS=${LOCAL_EPOCHS:-2}
BATCH_SIZE=${BATCH_SIZE:-128}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/results}

beta_tag() {
  case "$1" in
    0.1) echo "01" ;;
    0.5) echo "05" ;;
    1.0) echo "10" ;;
    10.0) echo "100" ;;
    *)
      echo "Unsupported beta value: $1" >&2
      exit 1
      ;;
  esac
}

run_train() {
  local run_name="$1"
  shift

  echo "==> ${run_name}"
  python scripts/train.py \
    --dataset cifar100 \
    --num_clients "${NUM_CLIENTS}" \
    --batch_size "${BATCH_SIZE}" \
    --local_epochs "${LOCAL_EPOCHS}" \
    --rounds "${ROUNDS}" \
    --output_dir "${OUTPUT_DIR}" \
    --no_download \
    "$@" \
    --run_name "${run_name}"
}

# echo "==> Phase 1: task-count diagnosis (5 task vs 10 task)"
# for SEED in "${SEEDS[@]}"; do
#   for NUM_TASKS in 5 10; do
#     TASK_SPLIT="data/processed/task_splits/cifar100_${NUM_TASKS}task_seed${SEED}.json"
#     PARTITION="data/processed/federated_partitions/cifar100_${NUM_TASKS}task_${NUM_CLIENTS}clients_beta05_seed${SEED}.json"

#     run_train \
#       "diag_cifar100_taskcount_${NUM_TASKS}task_replay_buf500_spt50_seed${SEED}" \
#       --method local_replay \
#       --task_split_path "${TASK_SPLIT}" \
#       --partition_path "${PARTITION}" \
#       --seed "${SEED}" \
#       --buffer_size 500 \
#       --samples_per_task 50
#   done
# done

# echo "==> Phase 2: replay-capacity diagnosis"
# for SEED in "${SEEDS[@]}"; do
#   TASK_SPLIT="data/processed/task_splits/cifar100_10task_seed${SEED}.json"
#   PARTITION="data/processed/federated_partitions/cifar100_10task_${NUM_CLIENTS}clients_beta05_seed${SEED}.json"

#   run_train \
#     "diag_cifar100_replay_buf500_spt50_seed${SEED}" \
#     --method local_replay \
#     --task_split_path "${TASK_SPLIT}" \
#     --partition_path "${PARTITION}" \
#     --seed "${SEED}" \
#     --buffer_size 500 \
#     --samples_per_task 50

#   run_train \
#     "diag_cifar100_replay_buf1000_spt100_seed${SEED}" \
#     --method local_replay \
#     --task_split_path "${TASK_SPLIT}" \
#     --partition_path "${PARTITION}" \
#     --seed "${SEED}" \
#     --buffer_size 1000 \
#     --samples_per_task 100

#   run_train \
#     "diag_cifar100_replay_buf2000_spt200_seed${SEED}" \
#     --method local_replay \
#     --task_split_path "${TASK_SPLIT}" \
#     --partition_path "${PARTITION}" \
#     --seed "${SEED}" \
#     --buffer_size 2000 \
#     --samples_per_task 200
# done

echo "==> Phase 3: non-IID diagnosis (beta sweep)"
for SEED in "${SEEDS[@]}"; do
  TASK_SPLIT="data/processed/task_splits/cifar100_10task_seed${SEED}.json"

  for BETA in 0.5 1.0 10.0; do
    BETA_STR="$(beta_tag "${BETA}")"
    PARTITION="data/processed/federated_partitions/cifar100_10task_${NUM_CLIENTS}clients_beta${BETA_STR}_seed${SEED}.json"

    run_train \
      "diag_cifar100_beta${BETA_STR}_replay_buf500_spt50_seed${SEED}" \
      --method local_replay \
      --task_split_path "${TASK_SPLIT}" \
      --partition_path "${PARTITION}" \
      --seed "${SEED}" \
      --buffer_size 500 \
      --samples_per_task 50
  done
done

echo "==> Phase 4: TTS diagnosis"
TTS_BUFFER_SIZE=${TTS_BUFFER_SIZE:-1000}
TTS_SAMPLES_PER_TASK=${TTS_SAMPLES_PER_TASK:-100}

for SEED in "${SEEDS[@]}"; do
  TASK_SPLIT="data/processed/task_splits/cifar100_10task_seed${SEED}.json"
  PARTITION="data/processed/federated_partitions/cifar100_10task_${NUM_CLIENTS}clients_beta05_seed${SEED}.json"

  run_train \
    "diag_cifar100_tts_baseline_buf${TTS_BUFFER_SIZE}_spt${TTS_SAMPLES_PER_TASK}_seed${SEED}" \
    --method local_replay \
    --task_split_path "${TASK_SPLIT}" \
    --partition_path "${PARTITION}" \
    --seed "${SEED}" \
    --buffer_size "${TTS_BUFFER_SIZE}" \
    --samples_per_task "${TTS_SAMPLES_PER_TASK}"

  run_train \
    "diag_cifar100_tts_default_buf${TTS_BUFFER_SIZE}_spt${TTS_SAMPLES_PER_TASK}_seed${SEED}" \
    --method local_replay_tts \
    --task_split_path "${TASK_SPLIT}" \
    --partition_path "${PARTITION}" \
    --seed "${SEED}" \
    --buffer_size "${TTS_BUFFER_SIZE}" \
    --samples_per_task "${TTS_SAMPLES_PER_TASK}" \
    --tts_old_temp 0.9 \
    --tts_new_temp 1.1 \
    --tts_old_weight 1.1 \
    --tts_new_weight 0.9

  run_train \
    "diag_cifar100_tts_neutral_buf${TTS_BUFFER_SIZE}_spt${TTS_SAMPLES_PER_TASK}_seed${SEED}" \
    --method local_replay_tts \
    --task_split_path "${TASK_SPLIT}" \
    --partition_path "${PARTITION}" \
    --seed "${SEED}" \
    --buffer_size "${TTS_BUFFER_SIZE}" \
    --samples_per_task "${TTS_SAMPLES_PER_TASK}" \
    --tts_old_temp 1.0 \
    --tts_new_temp 1.0 \
    --tts_old_weight 1.0 \
    --tts_new_weight 1.0
done

echo "==> Diagnosis ablations finished"
echo "==> Suggested summary commands:"
echo "python scripts/summarize_results.py --pattern 'diag_cifar100_*.json' --output_csv outputs/results/diag_cifar100_summary.csv"
echo "python scripts/aggregate_mean_std.py --pattern 'diag_cifar100_*.json' --output_csv outputs/results/diag_cifar100_mean_std.csv"
