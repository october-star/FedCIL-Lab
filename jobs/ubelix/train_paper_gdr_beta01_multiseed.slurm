#!/bin/bash
#SBATCH --job-name=papgdr_b01_ms
#SBATCH --output=outputs/logs/%x_%j.out
#SBATCH --error=outputs/logs/%x_%j.err

#SBATCH --account=gratis
#SBATCH --partition=gpu
#SBATCH --qos=job_gratis
#SBATCH --gres=gpu:rtx4090:1

#SBATCH --time=12:00:00
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

bash jobs/local/prepare_multiseed_cifar100.sh
bash jobs/local/run_paper_gdr_beta01_multiseed.sh

echo "Finished time: $(date)"
