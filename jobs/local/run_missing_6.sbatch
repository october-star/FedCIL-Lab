#!/bin/bash
#SBATCH --job-name=fedcil_missing6
#SBATCH --partition=gpu
#SBATCH --qos=job_gratis
#SBATCH --nodes=1
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/storage/homefs/cl25n064/FedCIL-Lab/outputs/logs/missing6_%j.out
#SBATCH --error=/storage/homefs/cl25n064/FedCIL-Lab/outputs/logs/missing6_%j.err

module load Python/3.11.3-GCCcore-12.3.0
source /storage/homefs/cl25n064/venvs/fedcil_clean/bin/activate
cd /storage/homefs/cl25n064/FedCIL-Lab
bash /storage/homefs/cl25n064/FedCIL-Lab/jobs/local/run_missing_6.sh
