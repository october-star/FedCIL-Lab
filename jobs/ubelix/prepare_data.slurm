#!/bin/bash
#SBATCH --job-name=prepare_data
#SBATCH --output=$SCRATCH/FedCIL-Lab/outputs/logs/%x_%j.out
#SBATCH --error=$SCRATCH/FedCIL-Lab/outputs/logs/%x_%j.err

#SBATCH --account=gratis
#SBATCH --partition=gpu
#SBATCH --qos=job_gratis
#SBATCH --gres=gpu:rtx4090:1

#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G

module purge
module load Python/3.11
source ~/venvs/myddlenv/bin/activate

cd $SCRATCH/FedCIL-Lab

mkdir -p outputs/logs
mkdir -p data/raw

echo "Running on node: $(hostname)"
echo "Python: $(which python)"

python scripts/prepare_data.py --dataset all

