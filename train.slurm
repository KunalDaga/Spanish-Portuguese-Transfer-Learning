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

cd /storage/ice1/6/7/kdaga7/Spanish-Portuguese-Transfer-Learning

source .venv/bin/activate

nvidia-smi

python3 train_spanish.py

echo "Zipping model directory..."
tar -czvf spanish_ocr_model.tar.gz spanish_ocr_model/

echo "Copying model back to laptop..."
scp spanish_ocr_model.tar.gz Kunal@ipsec-10-2-192-69.local:~/Downloads/

echo "Finished at: $(date)"
echo "---------------------------------------"