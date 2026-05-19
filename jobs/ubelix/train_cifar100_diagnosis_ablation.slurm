#!/bin/bash
#SBATCH --job-name=diag_c100_ab
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

cd $SCRATCH/FedCIL-Lab

mkdir -p outputs/logs
mkdir -p outputs/results
mkdir -p outputs/figures
mkdir -p data/raw
mkdir -p data/processed

echo "Running on node: $(hostname)"
echo "Python: $(which python)"
echo "Start time: $(date)"

bash jobs/local/prepare_cifar100_diagnosis_ablation.sh
bash jobs/local/run_cifar100_diagnosis_ablation.sh

echo "Finished time: $(date)"
