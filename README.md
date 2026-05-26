# FedCIL-Lab

FedCIL-Lab is a research sandbox for federated class-incremental learning (FCIL) on CIFAR-10 and CIFAR-100. It combines class-incremental task construction, federated Dirichlet client partitions, FedAvg training, replay-based continual learning baselines, GDR-style sample selection, TTS stabilization, and an adaptive replay variant for forgetting-aware experiments.

The repository is designed to make it easy to:

- build task-wise class splits for CIFAR benchmarks
- partition each task across federated clients
- train an incremental classifier with multiple replay variants
- save per-task experiment histories as JSON
- aggregate multi-seed results into summary CSV files

## What Is In This Repo

- `scripts/prepare_data.py`: download and sanity-check CIFAR datasets
- `scripts/build_splits.py`: create fixed class orders and class-incremental task splits
- `scripts/build_federated_partitions.py`: create per-task client partitions with a Dirichlet distribution
- `scripts/train.py`: main training entrypoint
- `scripts/summarize_results.py`: flatten result JSON files into a CSV summary
- `scripts/aggregate_mean_std.py`: compute grouped mean/std summaries across runs
- `src/models/incremental_model.py`: ResNet18-based expandable classifier
- `src/methods/`: FCIL training methods and research variants

## Method IDs

The source of truth for supported methods is `scripts/train.py`.

| CLI value | Description |
| --- | --- |
| `finetune` | No replay, trains only on the current task |
| `local_replay` | Per-client replay buffer baseline |
| `local_replay_gdr_paper` | Paper-style GDR replay variant |
| `local_replay_tts` | Replay with task-wise temperature scaling |
| `local_replay_gdr_tts_paper` | Paper-style GDR + TTS variant |
| `cbdr_adaptive_reply` | Adaptive replay variant with KL-forgetting logic |

Notes:

- `cbdr_adaptive_reply` is spelled `reply` in the current CLI and code; use that exact string when launching runs.
- Some research scripts currently focus on the `*_paper` and adaptive variants rather than every baseline.

## Repository Layout

```text
src/
  data/         CIFAR loading, task splits, federated partitions, dataset wrappers
  federated/    local client training and FedAvg aggregation
  gdr/          feature extraction, server-side leverage computation, visualization
  kd/           knowledge distillation loss helpers
  methods/      finetune, replay, GDR, TTS, paper variants, adaptive replay
  models/       incremental backbone with expandable classifier head
  replay/       replay buffer and selection datasets
  tts/          task-wise temperature scaling losses
  utils/        KL-forgetting utilities

scripts/
  prepare_data.py
  build_splits.py
  build_federated_partitions.py
  visualize_partition.py
  train.py
  summarize_results.py
  aggregate_mean_std.py

jobs/
  local/        local experiment launchers
  ubelix/       cluster launchers
```

## Environment Setup

Recommended environment: Python 3.11 in a virtual environment.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you need a specific CUDA build of PyTorch, install the matching wheel before or instead of the default `requirements.txt` flow.

Device selection during training is automatic:

- CUDA if available
- otherwise MPS on Apple Silicon
- otherwise CPU

## Data Preparation

### 1. Download CIFAR and verify the files

```bash
python scripts/prepare_data.py --dataset all
```

This downloads CIFAR into `data/raw/` and prints dataset statistics.

### 2. Build class-incremental task splits

Example: CIFAR-10, 5 tasks, seed 1.

```bash
python scripts/build_splits.py \
  --dataset cifar10 \
  --num_tasks 5 \
  --seed 1
```

Artifacts are written to:

- `data/processed/class_orders/`
- `data/processed/task_splits/`

### 3. Build federated client partitions

Example: 5 clients with Dirichlet `beta=0.5`.

```bash
python scripts/build_federated_partitions.py \
  --dataset cifar10 \
  --num_tasks 5 \
  --num_clients 5 \
  --beta 0.5 \
  --seed 1
```

Artifacts are written to:

- `data/processed/federated_partitions/`

### 4. Optional: visualize class distributions

```bash
python scripts/visualize_partition.py \
  --dataset cifar10 \
  --num_tasks 5 \
  --beta 0.5 \
  --seed 1
```

Plots are saved under `outputs/figures/data_distribution/`.

## Quick Start

The following example uses:

- dataset: `cifar10`
- tasks: `5`
- clients: `5`
- beta: `0.5`
- seed: `1`

First prepare the split and partition:

```bash
python scripts/build_splits.py --dataset cifar10 --num_tasks 5 --seed 1
python scripts/build_federated_partitions.py --dataset cifar10 --num_tasks 5 --num_clients 5 --beta 0.5 --seed 1
```

Then launch a run:

```bash
python scripts/train.py \
  --method local_replay \
  --dataset cifar10 \
  --task_split_path data/processed/task_splits/cifar10_5task_seed1.json \
  --partition_path data/processed/federated_partitions/cifar10_5task_5clients_beta05_seed1.json \
  --num_clients 5 \
  --rounds 100 \
  --local_epochs 2 \
  --batch_size 128 \
  --buffer_size 300 \
  --samples_per_task 60 \
  --seed 1 \
  --run_name demo_cifar10_replay \
  --no_download
```

## Training Recipes

### Finetune baseline

```bash
python scripts/train.py \
  --method finetune \
  --dataset cifar10 \
  --task_split_path data/processed/task_splits/cifar10_5task_seed1.json \
  --partition_path data/processed/federated_partitions/cifar10_5task_5clients_beta05_seed1.json \
  --num_clients 5 \
  --rounds 100 \
  --local_epochs 2 \
  --batch_size 128 \
  --seed 1 \
  --run_name demo_finetune \
  --no_download
```

### Replay baseline

```bash
python scripts/train.py \
  --method local_replay \
  --dataset cifar10 \
  --task_split_path data/processed/task_splits/cifar10_5task_seed1.json \
  --partition_path data/processed/federated_partitions/cifar10_5task_5clients_beta05_seed1.json \
  --num_clients 5 \
  --rounds 100 \
  --local_epochs 2 \
  --batch_size 128 \
  --buffer_size 300 \
  --samples_per_task 60 \
  --seed 1 \
  --run_name demo_replay \
  --no_download
```

### Replay + GDR

```bash
python scripts/train.py \
  --method local_replay_gdr \
  --dataset cifar10 \
  --task_split_path data/processed/task_splits/cifar10_5task_seed1.json \
  --partition_path data/processed/federated_partitions/cifar10_5task_5clients_beta05_seed1.json \
  --num_clients 5 \
  --rounds 100 \
  --local_epochs 2 \
  --batch_size 128 \
  --buffer_size 300 \
  --samples_per_task 60 \
  --gdr_rank 8 \
  --seed 1 \
  --run_name demo_replay_gdr \
  --no_download
```

### Replay + TTS

```bash
python scripts/train.py \
  --method local_replay_tts \
  --dataset cifar10 \
  --task_split_path data/processed/task_splits/cifar10_5task_seed1.json \
  --partition_path data/processed/federated_partitions/cifar10_5task_5clients_beta05_seed1.json \
  --num_clients 5 \
  --rounds 100 \
  --local_epochs 2 \
  --batch_size 128 \
  --buffer_size 300 \
  --samples_per_task 60 \
  --tts_old_temp 0.9 \
  --tts_new_temp 1.1 \
  --tts_old_weight 1.1 \
  --tts_new_weight 0.9 \
  --seed 1 \
  --run_name demo_replay_tts \
  --no_download
```

### Paper-style GDR with KD enabled

```bash
python scripts/train.py \
  --method local_replay_gdr_paper \
  --dataset cifar10 \
  --task_split_path data/processed/task_splits/cifar10_5task_seed1.json \
  --partition_path data/processed/federated_partitions/cifar10_5task_5clients_beta05_seed1.json \
  --num_clients 5 \
  --rounds 100 \
  --local_epochs 2 \
  --batch_size 128 \
  --buffer_size 450 \
  --samples_per_task 90 \
  --gdr_rank 8 \
  --kd_lambda 1.0 \
  --kd_temperature 2.0 \
  --seed 1 \
  --run_name demo_replay_gdr_kd_paper \
  --no_download
```

### Adaptive replay smoke run

```bash
python scripts/train.py \
  --method cbdr_adaptive_reply \
  --dataset cifar10 \
  --task_split_path data/processed/task_splits/cifar10_5task_seed1.json \
  --partition_path data/processed/federated_partitions/cifar10_5task_5clients_beta05_seed1.json \
  --num_clients 5 \
  --rounds 20 \
  --local_epochs 2 \
  --batch_size 128 \
  --buffer_size 450 \
  --samples_per_task 90 \
  --gdr_rank 8 \
  --adaptive_replay \
  --adaptive_replay_gamma 2.0 \
  --adaptive_replay_min_weight 0.5 \
  --adaptive_replay_max_weight 2.0 \
  --replay_sampling_mass 0.5 \
  --kl_temperature 2.0 \
  --kl_max_samples_per_class 100 \
  --seed 1 \
  --run_name demo_cbdr_smoke \
  --no_download
```

## Important Training Arguments

Common arguments:

- `--dataset`: `cifar10` or `cifar100`
- `--task_split_path`: JSON produced by `scripts/build_splits.py`
- `--partition_path`: JSON produced by `scripts/build_federated_partitions.py`
- `--num_clients`: number of federated clients
- `--rounds`: FedAvg rounds per task
- `--local_epochs`: local client epochs per round
- `--batch_size`
- `--lr`
- `--run_name`
- `--save_checkpoint`
- `--no_download`

Replay-related arguments:

- `--buffer_size`
- `--samples_per_task`

GDR-related arguments:

- `--gdr_rank`
- `--gdr_feature_samples`
- `--gdr_class_wise` or `--no-gdr_class_wise`
- `--figure_dir`

TTS-related arguments:

- `--tts_old_temp`
- `--tts_new_temp`
- `--tts_old_weight`
- `--tts_new_weight`

KD and adaptive replay arguments:

- `--kd_lambda`
- `--kd_temperature`
- `--use_global_view`
- `--adaptive_replay`
- `--adaptive_replay_gamma`
- `--adaptive_replay_min_weight`
- `--adaptive_replay_max_weight`
- `--replay_sampling_mass`
- `--kl_temperature`
- `--kl_max_samples_per_class`

Current backbone support is intentionally narrow:

- `--backbone resnet18`

## Outputs

### Result JSON

Each run writes a JSON payload to:

```text
outputs/results/<run_name>.json
```

The payload stores:

- run name
- training config
- selected device
- per-task and per-round histories
- replay, GDR, or TTS metadata when applicable

### Checkpoints

If `--save_checkpoint` is enabled:

```text
outputs/results/<run_name>.pt
```

### Figures

GDR-related visualizations are typically written to:

```text
outputs/figures/gdr/
```

### Summary CSV files

Direct script usage:

```bash
python scripts/summarize_results.py
python scripts/aggregate_mean_std.py
```

Default outputs:

- `outputs/results/summary.csv`
- `outputs/results/mean_std_summary.csv`

The local wrapper scripts currently use:

- `jobs/local/summarize_results.sh` -> `outputs/results/summary_two.csv`
- `jobs/local/aggregate_reproduce_mean_std.sh` -> `outputs/results/reproduce_core_mean_std_summary.csv`

### Standalone plotting helper

`plot_fedcil_results.py` is a standalone figure script with baked-in numbers. It writes PNG files into `output/`, not `outputs/`.

## Experiment Scripts

### Reproduction preparation

Build common multi-seed assets for CIFAR-10 5-task and CIFAR-100 10-task setups:

```bash
bash jobs/local/prepare_reproduce_multiseed.sh
```

### Current reproduction launchers

```bash
bash jobs/local/run_reproduce_cifar10_5task_KD_multiseed.sh
bash jobs/local/run_reproduce_cifar100_10task_KD_multiseed.sh
```

These scripts currently target the paper-style GDR branch with KD enabled.

### Aggregate reproduction summaries

```bash
bash jobs/local/aggregate_reproduce_mean_std.sh
```

### Adaptive replay smoke test

```bash
bash jobs/local/run_kl_adaptive_smoke.sh
```

### Paper sweep helpers

The `jobs/local/run_paper_all_settings*.sh` scripts are research-oriented launchers for broader sweeps. They contain commented sections and tunable constants, so inspect them before starting long runs.

## Implementation Notes

### Label remapping

Task splits keep the original CIFAR labels, but training remaps all seen classes to contiguous indices for the expandable head. This logic lives in `src/data/federated_dataset.py`.

### Incremental classifier growth

The model starts without a classifier head and expands the output layer whenever a new task arrives. This logic lives in `src/models/incremental_model.py`.

### Small-client stability

Client subsets that are too small for stable BatchNorm behavior are skipped during training in the replay pipelines.

### Research-script drift

This repository is an active research workspace. A few helper scripts are older than the newest method naming or experiment focus. In particular, prefer the explicit `run_reproduce_*_KD_multiseed.sh` launchers over the older wrapper scripts when you want the current reproduction flow.

## Suggested Workflow

For a clean end-to-end run:

1. Create the environment and install dependencies.
2. Download CIFAR with `scripts/prepare_data.py`.
3. Build a task split with `scripts/build_splits.py`.
4. Build a federated partition with `scripts/build_federated_partitions.py`.
5. Train one or more methods with `scripts/train.py`.
6. Summarize outputs with `scripts/summarize_results.py` and `scripts/aggregate_mean_std.py`.

That gives you a minimal but complete FCIL experiment loop from raw CIFAR data to summary tables.
