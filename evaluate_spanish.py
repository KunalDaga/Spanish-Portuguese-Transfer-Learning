import os
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm
import pandas as pd
import evaluate

class SpanishOCRDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, images_dir, processor):
        self.data = dataframe.reset_index(drop=True)
        self.images_dir = images_dir
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.images_dir, row['image_path'])
        text = row['text']

        image = Image.open(img_path).convert("RGB")
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.squeeze(0)
        return {"pixel_values": pixel_values, "text": text}

def evaluate_model(model_dir="spanish_ocr_model", csv_path="data/spanish_data.csv", images_dir="data/images/"):
    processor = TrOCRProcessor.from_pretrained(model_dir)
    model = VisionEncoderDecoderModel.from_pretrained(model_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    model.eval()

    df = pd.read_csv(csv_path)
    dataset = SpanishOCRDataset(df, images_dir, processor)
    dataloader = DataLoader(dataset, batch_size=1)

    preds, refs = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["text"]

            generated_ids = model.generate(pixel_values.unsqueeze(0)) if pixel_values.ndim == 3 else model.generate(pixel_values)
            pred_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            preds.append(pred_text)
            refs.append(labels[0])

    cer_metric = evaluate.load("cer")
    wer_metric = evaluate.load("wer")

    cer = cer_metric.compute(predictions=preds, references=refs)
    wer = wer_metric.compute(predictions=preds, references=refs)

    print(f"→ Final CER: {cer:.4f}")
    print(f"→ Final WER: {wer:.4f}")

    # Save metrics into a file
    with open("evaluation_metrics.txt", "w") as f:
        f.write(f"Final CER: {cer:.4f}\n")
        f.write(f"Final WER: {wer:.4f}\n")


if __name__ == "__main__":
    evaluate_model()