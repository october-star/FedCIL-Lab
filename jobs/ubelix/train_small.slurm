#!/bin/bash
#SBATCH --job-name=fedcil_small
#SBATCH --output=$SCRATCH/FedCIL-Lab/outputs/logs/%x_%j.out
#SBATCH --error=$SCRATCH/FedCIL-Lab/outputs/logs/%x_%j.err

#SBATCH --account=gratis
#SBATCH --partition=gpu
#SBATCH --qos=job_gratis
#SBATCH --gres=gpu:rtx4090:1

#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

module purge
module load Python/3.11
source ~/venvs/myddlenv/bin/activate

cd $SCRATCH/FedCIL-Lab

mkdir -p outputs/logs
mkdir -p outputs/results
mkdir -p outputs/figures
mkdir -p data/raw

echo "Running on node: $(hostname)"
echo "Python: $(which python)"
echo "Start time: $(date)"

# Step 1: prepare CIFAR data locally on the cluster node if needed.
python scripts/prepare_data.py

# Step 2: build a small CIFAR-10 incremental setup.
python scripts/build_splits.py \
  --dataset cifar10 \
  --num_tasks 5 \
  --seed 1

python scripts/build_federated_partitions.py \
  --dataset cifar10 \
  --num_tasks 5 \
  --num_clients 5 \
  --beta 0.5 \
  --seed 1

# Step 3: run a short sanity experiment.
python scripts/train.py \
  --method local_replay_gdr_tts \
  --dataset cifar10 \
  --task_split_path data/processed/task_splits/cifar10_5task_seed1.json \
  --partition_path data/processed/federated_partitions/cifar10_5task_5clients_beta05_seed1.json \
  --num_clients 5 \
  --batch_size 128 \
  --rounds 5 \
  --buffer_size 200 \
  --samples_per_task 20 \
  --gdr_rank 8 \
  --tts_old_temp 2.0 \
  --tts_new_temp 1.0 \
  --tts_old_weight 1.5 \
  --tts_new_weight 1.0 \
  --run_name ubelix_cifar10_small_gdr_tts_seed1_beta05 \
  --save_checkpoint

echo "Finished time: $(date)"
