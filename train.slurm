#!/bin/bash
#SBATCH --job-name=spanish-ocr-train
#SBATCH --partition=coc-gpu
#SBATCH --account=coc
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=slurm-%j.out

echo "Running on node: $(hostname)"
echo "Starting at: $(date)"

# Change to your project directory
cd /storage/ice1/6/7/kdaga7/Spanish-Portuguese-Transfer-Learning

# Activate the pre-built virtual environment
source .venv/bin/activate

# Check that GPU is visible
nvidia-smi

# jupyter nbconvert --execute --to notebook --inplace model.ipynb --ExecutePreprocessor.timeout=-1
python3 train_spanish.py

echo "Finished at: $(date)"