# MLHUM Project
 
 - `data/` contains data used
 - `model.ipynb` used to train and infer


 # Steps
 1. ssh into pace-ice cluster
 2. cd ~/scratc
 3. if first session, clone repo; otherwise cd into it and then pull for latest update
 4. list partitions with `sinfo` and then request a gpu `salloc --partition=<name> --gres=gpu:1 --time=02:00:00 --mem=32G`
 5. source .venv/bin/activate
 6. pip install transformers scikit-learn pandas pillow datasets jupyter accelerate>=0.26.0 tqdm evaluate jiwer
 7. pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
 8. jupyter nbconvert --execute --to notebook --inplace model.ipynb



 