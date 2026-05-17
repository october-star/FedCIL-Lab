#!/bin/bash
#SBATCH --job-name=repr_core_ms
#SBATCH --output=$SCRATCH/FedCIL-Lab/outputs/logs/%x_%j.out
#SBATCH --error=$SCRATCH/FedCIL-Lab/outputs/logs/%x_%j.err

#SBATCH --account=gratis
#SBATCH --partition=gpu
#SBATCH --qos=job_gratis
#SBATCH --gres=gpu:rtx4090:1

#SBATCH --time=48:00:00
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

bash jobs/local/prepare_reproduce_multiseed.sh
bash jobs/local/run_reproduce_core_multiseed.sh

echo "Finished time: $(date)"
