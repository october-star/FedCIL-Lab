#!/bin/bash
#SBATCH --job-name=diag_c100_sum
#SBATCH --output=outputs/logs/%x_%j.out
#SBATCH --error=outputs/logs/%x_%j.err

#SBATCH --account=gratis
#SBATCH --partition=gpu
#SBATCH --qos=job_gratis
#SBATCH --gres=gpu:rtx4090:1

#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G

module purge
module load Python/3.11
source ~/venvs/myddlenv/bin/activate

cd $SCRATCH/FedCIL-Lab

mkdir -p outputs/logs
mkdir -p outputs/results

echo "Running on node: $(hostname)"
echo "Python: $(which python)"
echo "Start time: $(date)"

bash jobs/local/summarize_cifar100_diagnosis_ablation.sh
bash jobs/local/aggregate_cifar100_diagnosis_ablation.sh

echo "Finished time: $(date)"
