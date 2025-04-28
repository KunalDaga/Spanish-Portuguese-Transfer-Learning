#!/usr/bin/env python3
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import VisionEncoderDecoderModel, TrOCRProcessor
import evaluate
from tqdm import tqdm

# ─── CONFIG ──────────────────────────────────────────────────────────────────────

MODEL_DIR   = "./spanish_ocr_model"
CSV_PATH    = "data/spanish_data.csv"
IMAGES_DIR  = "data/images/"
TEST_SIZE   = 0.2
RANDOM_SEED = 42
BATCH_SIZE  = 8

# Generation tweaks
GEN_KWARGS = dict(
    num_beams=5,
    early_stopping=True,
    do_sample=False,
    max_new_tokens=128,
    length_penalty=0.8,
    no_repeat_ngram_size=2,
)

# ─── DATASET ─────────────────────────────────────────────────────────────────────

class SpanishEvalDataset(Dataset):
    def __init__(self, dataframe, images_dir, processor):
        self.df = dataframe.reset_index(drop=True)
        self.images_dir = images_dir
        self.processor = processor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # full path to image
        img_path = os.path.join(self.images_dir, row["image_path"])
        text    = row["text"]

        # load & preprocess
        image = Image.open(img_path).convert("RGB")
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.squeeze(0)
        return {"pixel_values": pixel_values, "text": text}

# ─── NORMALIZATION ──────────────────────────────────────────────────────────────

def normalize(s: str) -> str:
    # strip, collapse whitespace
    return " ".join(s.strip().split())

# ─── MAIN ────────────────────────────────────────────────────────────────────────

def main():
    # 1. load model + processor
    processor = TrOCRProcessor.from_pretrained(MODEL_DIR)
    model     = VisionEncoderDecoderModel.from_pretrained(MODEL_DIR)
    model.eval()

    # push to GPU/CPU
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps"   if torch.backends.mps.is_available() else
                          "cpu")
    model.to(device)

    # 2. load & split dataframe
    df = pd.read_csv(CSV_PATH)
    _, val_df = \
      __import__("sklearn.model_selection").model_selection.train_test_split(
        df, test_size=TEST_SIZE, random_state=RANDOM_SEED)

    # 3. build evaluation DataLoader
    val_ds = SpanishEvalDataset(val_df, IMAGES_DIR, processor)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # 4. prepare metrics
    cer_metric = evaluate.load("cer")
    wer_metric = evaluate.load("wer")

    # 5. run inference
    for batch in tqdm(val_loader, desc="Evaluating"):
        pixel_values = batch["pixel_values"].to(device)
        texts        = batch["text"]

        with torch.no_grad():
            generated_ids = model.generate(pixel_values, **GEN_KWARGS)

        # decode & normalize
        preds = processor.batch_decode(generated_ids, skip_special_tokens=True)
        preds = [normalize(p) for p in preds]
        refs  = [normalize(t) for t in texts]

        # accumulate
        cer_metric.add_batch(predictions=preds, references=refs)
        wer_metric.add_batch(predictions=preds, references=refs)

    # 6. compute & print
    cer = cer_metric.compute()
    wer = wer_metric.compute()
    print(f"\n→ Final CER: {cer:.4f}")
    print(f"→ Final WER: {wer:.4f}")

if __name__ == "__main__":
    main()