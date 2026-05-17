#!/bin/bash
#SBATCH --job-name=repr_c100_10t
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

for SEED in 1 2 3; do
  python scripts/build_splits.py \
    --dataset cifar100 \
    --num_tasks 10 \
    --seed "$SEED"

  python scripts/build_federated_partitions.py \
    --dataset cifar100 \
    --num_tasks 10 \
    --num_clients 5 \
    --beta 0.5 \
    --seed "$SEED"
done

bash jobs/local/run_reproduce_cifar100_10task_multiseed.sh

echo "Finished time: $(date)"
