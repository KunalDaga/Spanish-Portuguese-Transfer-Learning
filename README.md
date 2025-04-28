# MLHUM Project
 
 - `data/` contains data used
 - `model.ipynb` used to train and infer

 # Steps
 1. ssh into pace-ice cluster
 2. cd ~/scratch
 3. if first session, clone repo; otherwise cd into it and then pull for latest update
 4. source .venv/bin/activate
 5. pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && pip install transformers scikit-learn pandas pillow datasets jupyter accelerate tqdm evaluate jiwer
 6. Schedule job `sbatch train.slurm`
 7. look at queue using `squeue -u <gt username>`
 7. cat slurm-<jobid>.out