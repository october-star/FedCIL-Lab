# FedCIL-Lab

Federated Class-Incremental Learning playground built around CIFAR, federated client partitions, replay-based continual learning baselines, GDR sample selection, and TTS stabilization.

## Overview

This project studies class-incremental learning under a federated setting:

- data is split into incremental tasks by class order
- each task is partitioned across clients
- a shared incremental classifier is trained with FedAvg
- replay, GDR, and TTS are added on top as ablations and main methods

Current implemented stages:

1. `finetune`: federated finetune baseline
2. `local_replay`: per-client replay buffer baseline
3. `local_replay_gdr`: replay + GDR sample selection
4. `local_replay_tts`: replay + TTS
5. `local_replay_gdr_tts`: replay + GDR + TTS

## Project Layout

```text
src/
  data/         CIFAR loading, task split building, federated dataset manager
  federated/    client training and FedAvg aggregation
  gdr/          pseudo features, server SVD, leverage score, plotting
  methods/      training methods and ablations
  models/       incremental backbone + expandable classifier head
  replay/       replay buffer
  tts/          task-wise temperature scaling loss

scripts/
  prepare_data.py                dataset sanity check
  build_splits.py                build class-incremental task splits
  build_federated_partitions.py  build per-client partitions
  visualize_partition.py         visualize class distribution
  train.py                       main training entry
```

## Environment

Recommended: Python 3.11 with a virtual environment.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Device selection in training:

- CUDA if available
- otherwise MPS on Apple Silicon
- otherwise CPU

## Data Preparation

### 1. Download and inspect CIFAR

```bash
python scripts/prepare_data.py
```

This downloads CIFAR into `data/raw/` and prints basic label statistics.

### 2. Build class-incremental task splits

Example for CIFAR-10 with 5 tasks and seed 1:

```bash
python scripts/build_splits.py --dataset cifar10 --num_tasks 5 --seed 1
```

This creates:

- `data/processed/class_orders/...`
- `data/processed/task_splits/...`

### 3. Build federated client partitions

Example:

```bash
python scripts/build_federated_partitions.py \
  --dataset cifar10 \
  --num_tasks 5 \
  --num_clients 5 \
  --beta 0.5 \
  --seed 1
```

This creates:

- `data/processed/federated_partitions/...`

### 4. Optional: visualize class distribution

```bash
python scripts/visualize_partition.py \
  --dataset cifar10 \
  --num_tasks 5 \
  --beta 0.5 \
  --seed 1
```

Plots are saved under `outputs/figures/data_distribution/`.

## Training

Main entry:

```bash
python scripts/train.py --method <method_name>
```

Common arguments:

```text
--dataset
--task_split_path
--partition_path
--num_clients
--rounds
--local_epochs
--batch_size
--lr
--buffer_size
--samples_per_task
--seed
--run_name
--save_checkpoint
--no_download
```

### 1. Finetune baseline

```bash
python scripts/train.py \
  --method finetune \
  --run_name finetune_seed1_beta05 \
  --no_download
```

### 2. Replay baseline

```bash
python scripts/train.py \
  --method local_replay \
  --buffer_size 200 \
  --samples_per_task 50 \
  --run_name local_replay_seed1_beta05 \
  --no_download
```

### 3. Replay + GDR

```bash
python scripts/train.py \
  --method local_replay_gdr \
  --buffer_size 200 \
  --samples_per_task 50 \
  --gdr_rank 8 \
  --gdr_feature_samples 500 \
  --run_name local_replay_gdr_backbonegdr_seed1_beta05 \
  --no_download
```

Additional GDR arguments:

```text
--gdr_rank
--gdr_feature_samples
--figure_dir
```

`local_replay_gdr` now extracts GDR scores from backbone features by default. Use a
distinct `run_name` such as `backbonegdr` if you want to keep older pseudo-feature
results for comparison.

### 4. Replay + TTS

```bash
python scripts/train.py \
  --method local_replay_tts \
  --buffer_size 200 \
  --samples_per_task 50 \
  --tts_old_temp 2.0 \
  --tts_new_temp 1.0 \
  --tts_old_weight 1.5 \
  --tts_new_weight 1.0 \
  --run_name local_replay_tts_seed1_beta05 \
  --no_download
```

### 5. Replay + GDR + TTS

```bash
python scripts/train.py \
  --method local_replay_gdr_tts \
  --buffer_size 200 \
  --samples_per_task 50 \
  --gdr_rank 8 \
  --tts_old_temp 2.0 \
  --tts_new_temp 1.0 \
  --tts_old_weight 1.5 \
  --tts_new_weight 1.0 \
  --run_name local_replay_gdr_tts_backbonegdr_seed1_beta05 \
  --no_download
```

Additional TTS arguments:

```text
--tts_old_temp
--tts_new_temp
--tts_old_weight
--tts_new_weight
```

## Implemented Methods

### Finetune

- expands classifier head when new task classes arrive
- trains only on current-task client data
- evaluates on all seen classes

### Replay

- each client owns a replay buffer
- current task data and local replay data are concatenated for training
- buffer class counts are tracked in training results

### GDR

- constructs lightweight pseudo features from raw images
- server aggregates client features and runs SVD
- leverage scores are computed per sample
- replay insertion uses class-balanced, leverage-guided selection
- plots are saved for leverage scores and final buffer distribution

### TTS

- logits are split into old-class and new-class groups
- each group is temperature-scaled independently
- sample losses for old and new classes are reweighted separately

## Outputs

### Result JSON

Training results are saved to:

```text
outputs/results/<run_name>.json
```

Each result file stores:

- run name
- training config
- device
- per-task metrics
- final buffer statistics
- GDR metadata and figure paths when applicable

### Model checkpoint

If `--save_checkpoint` is enabled:

```text
outputs/results/<run_name>.pt
```

### Figures

GDR-related plots are saved to:

```text
outputs/figures/gdr/
```

Typical files:

- `..._task0_leverage.png`
- `..._task0_buffer_distribution.png`

### Matplotlib cache

The project sets Matplotlib's cache/config directory to:

```text
outputs/.matplotlib/
```

This is only runtime cache, not experiment output.

## Important Implementation Notes

### Label remapping

Task splits keep original CIFAR labels, but training uses remapped incremental labels so the expandable classifier head always sees contiguous targets:

- task 0 classes `[6, 8]` become labels `[0, 1]`
- task 1 old/new seen classes continue as contiguous indices

This logic lives in [src/data/federated_dataset.py](/Users/octobercity/Desktop/project/FedCIL-Lab/src/data/federated_dataset.py).

### BatchNorm stability

Client training uses `drop_last=True` and skips subsets smaller than 2 samples to avoid BatchNorm failures on singleton batches.

### Replay buffer contents

Replay buffers store:

- image tensor
- remapped label
- metadata such as task id, client id, dataset index

For GDR, metadata also includes leverage scores.

## Typical Workflow

```bash
python scripts/prepare_data.py
python scripts/build_splits.py --dataset cifar10 --num_tasks 5 --seed 1
python scripts/build_federated_partitions.py --dataset cifar10 --num_tasks 5 --num_clients 5 --beta 0.5 --seed 1
python scripts/train.py --method local_replay --run_name replay_seed1_beta05 --no_download
python scripts/train.py --method local_replay_tts --run_name replay_tts_seed1_beta05 --no_download
python scripts/train.py --method local_replay_gdr --run_name replay_gdr_backbonegdr_seed1_beta05 --no_download
python scripts/train.py --method local_replay_gdr_tts --run_name replay_gdr_tts_backbonegdr_seed1_beta05 --no_download
```

## Sequential Execution —— Core Experiment Table

For the full CIFAR-100 core experiment pipeline with 3 seeds, run the following steps in order:

### 1. Prepare multi-seed task splits and federated partitions

```bash
bash jobs/local/prepare_multiseed_cifar100.sh
```

This generates:

- CIFAR-100 task splits for `seed = 1, 2, 3`
- federated partitions for `beta = 0.1` and `beta = 0.5`

### 2. Run all core experiments

```bash
bash jobs/local/run_core_all_multiseed.sh
```

This runs the 8 core configurations for each seed:

- `local_replay`
- `local_replay_tts`
- `local_replay_gdr` (saved with `backbonegdr` in the run name)
- `local_replay_gdr_tts` (saved with `backbonegdr` in the run name)

under:

- `beta = 0.1`
- `beta = 0.5`

with:

- `buffer_size = 500`
- `rounds = 100`

### 3. Summarize all result JSON files

```bash
bash jobs/local/summarize_results.sh
```

This collects `outputs/results/*.json` into:

```text
outputs/results/summary.csv
```

### Recommended full order

```bash
bash jobs/local/prepare_multiseed_cifar100.sh
bash jobs/local/run_core_all_multiseed.sh
bash jobs/local/summarize_results.sh
```

## Ablation Suggestions

To compare stage-by-stage effects, run:

1. `finetune`
2. `local_replay`
3. `local_replay_tts`
4. `local_replay_gdr`
5. `local_replay_gdr_tts`

This gives a clean view of:

- replay gain over finetune
- TTS gain over replay
- GDR gain over replay
- combined gain of GDR + TTS

## Status

Implemented and smoke-tested:

- finetune baseline
- replay baseline
- replay + GDR
- replay + TTS
- replay + GDR + TTS

For full experiments, use consistent seeds, save all result JSON files, and compare the final seen accuracy and buffer statistics across runs.
