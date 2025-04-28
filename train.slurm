#!/bin/bash
#SBATCH --job-name=spanish-ocr-train
#SBATCH --partition=coc-gpu
#SBATCH --account=coc
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=slurm-%j.out

echo "---------------------------------------"
echo "Running on node: $(hostname)"
echo "Starting at: $(date)"
echo "---------------------------------------"

# Navigate to your project directory
cd /storage/ice1/6/7/kdaga7/Spanish-Portuguese-Transfer-Learning

# Activate virtual environment
source .venv/bin/activate

# Show which GPU you're running on
nvidia-smi

# Step 1: Train the model
echo "Starting training..."
python3 train_spanish.py

# Step 2: Evaluate the model immediately after training
echo "Training complete. Starting evaluation..."
python3 evaluate_spanish.py

# Step 3: Save model and evaluation outputs
echo "Zipping model directory..."
tar -czvf spanish_ocr_model.tar.gz spanish_ocr_model/

echo "Finished at: $(date)"
echo "---------------------------------------"