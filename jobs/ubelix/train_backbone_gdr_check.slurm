#!/bin/bash
#SBATCH --job-name=fedcil_gdr_check
#SBATCH --output=outputs/logs/%x_%j.out
#SBATCH --error=outputs/logs/%x_%j.err

#SBATCH --account=gratis
#SBATCH --partition=gpu
#SBATCH --qos=job_gratis
#SBATCH --gres=gpu:rtx4090:1

#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

module purge
module load Python/3.11
source ~/venvs/myddlenv/bin/activate

echo "Running on node: $(hostname)"
echo "Python: $(which python)"
echo "Start time: $(date)"

# Ensure the target split/partition exists.
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

bash jobs/local/run_backbone_gdr_check.sh

echo "Finished time: $(date)"
