# MLHUM Project
 
 - `data/` contains data used
 - `model.ipynb` used to train and infer


 # Steps
 1. ssh into pace-ice cluster
 2. cd ~/scratc
 3. if first session, clone repo; otherwise cd into it and then pull for latest update
 4. source .venv/bin/activate
 5. pip install transformers scikit-learn pandas pillow datasets jupyter accelerate tqdm evaluate jiwer torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
 6. sbatch train.slurm
 7. cat slurm-<jobid>.out