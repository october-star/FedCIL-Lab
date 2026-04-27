#!/bin/bash
#SBATCH --job-name=fedcil_c100_small
#SBATCH --output=outputs/logs/%x_%j.out
#SBATCH --error=outputs/logs/%x_%j.err

#SBATCH --account=gratis
#SBATCH --partition=gpu
#SBATCH --qos=job_gratis
#SBATCH --gres=gpu:rtx4090:1

#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G

module purge
module load Python/3.11
source ~/venvs/myddlenv/bin/activate

echo "Running on node: $(hostname)"
echo "Python: $(which python)"
echo "Start time: $(date)"

# Step 1: ensure CIFAR-100 is available.
python scripts/build_splits.py \
  --dataset cifar100 \
  --num_tasks 10 \
  --seed 1

python scripts/build_federated_partitions.py \
  --dataset cifar100 \
  --num_tasks 10 \
  --num_clients 5 \
  --beta 0.5 \
  --seed 1

# Step 2: short CIFAR-100 sanity experiment on the target setting.
python scripts/train.py \
  --method local_replay_gdr_tts \
  --dataset cifar100 \
  --task_split_path data/processed/task_splits/cifar100_10task_seed1.json \
  --partition_path data/processed/federated_partitions/cifar100_10task_5clients_beta05_seed1.json \
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
  --run_name ubelix_cifar100_small_gdr_tts_seed1_beta05 \
  --save_checkpoint

echo "Finished time: $(date)"
