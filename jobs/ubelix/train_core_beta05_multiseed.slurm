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
module load Python/3.10

# 删除旧的虚拟环境
# rm -rf ~/venvs/myddlenv

# 用 Python 3.11 重新创建
# python -m venv ~/venvs/myddlenv

# source ~/venvs/myddlenv/bin/activate

VENV_PATH=~/venvs/myddlenv

# 激活虚拟环境
source $VENV_PATH/bin/activate

# 确认使用的是虚拟环境的 Python
echo "Python path: $(which python)"
echo "Pip path: $(which pip)"

# 升级 pip（可选，但推荐）
# pip install --upgrade pip

# pip install -r requirements.txt


echo "Running on node: $(hostname)"
echo "Python: $(which python)"
echo "Start time: $(date)"

# Prepare multi-seed CIFAR-100 splits and partitions.
bash jobs/local/prepare_multiseed_cifar100.sh

# Run the beta=0.5 core multiseed experiments.
bash jobs/local/run_core_beta05_multiseed.sh

echo "Finished time: $(date)"
