#!/bin/bash
#SBATCH --job-name=fedcil_sum
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

echo "Running on node: $(hostname)"
echo "Python: $(which python)"
echo "Start time: $(date)"

bash jobs/local/summarize_results.sh

echo "Finished time: $(date)"
