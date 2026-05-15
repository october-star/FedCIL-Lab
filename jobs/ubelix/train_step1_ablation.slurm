#!/bin/bash
#SBATCH --job-name=fedcil_step1
#SBATCH --output=outputs/logs/%x_%j.out
#SBATCH --error=outputs/logs/%x_%j.err

#SBATCH --account=gratis
#SBATCH --partition=gpu
#SBATCH --qos=job_gratis
#SBATCH --gres=gpu:rtx4090:1

#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

module purge
module load Python/3.11
source ~/venvs/myddlenv/bin/activate

cd $SCRATCH/FedCIL-LAB

mkdir -p outputs/logs
mkdir -p outputs/results
mkdir -p outputs/figures
mkdir -p data/raw
mkdir -p data/processed

echo "Running on node: $(hostname)"
echo "Python: $(which python)"
echo "Start time: $(date)"

# Run Step 1 CIFAR-10 ablation suite:
# A1 finetune
# A2 local replay
# A3 local replay + GDR
# A4 local replay + TTS
# A5 FedCBDR (GDR + TTS)
bash jobs/local/run_step1_ablation.sh

echo "Finished time: $(date)"
