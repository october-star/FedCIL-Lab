#!/bin/bash
#SBATCH --job-name=fedcil_b05_ms
#SBATCH --output=outputs/logs/%x_%j.out
#SBATCH --error=outputs/logs/%x_%j.err

#SBATCH --account=gratis
#SBATCH --partition=gpu
#SBATCH --qos=job_gratis
#SBATCH --gres=gpu:rtx4090:1

#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

module purge
module load Python/3.11
source ~/venvs/myddlenv/bin/activate

echo "Running on node: $(hostname)"
echo "Python: $(which python)"
echo "Start time: $(date)"

# Prepare multi-seed CIFAR-100 splits and partitions.
bash jobs/local/prepare_multiseed_cifar100.sh

# Run the beta=0.5 core multiseed experiments.
bash jobs/local/run_core_beta05_multiseed.sh

echo "Finished time: $(date)"
