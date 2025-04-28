import os
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
from transformers import (
    VisionEncoderDecoderModel,
    TrOCRProcessor,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)
import evaluate
from tqdm import tqdm
from torchvision import transforms as T


class SpanishOCRDataset(Dataset):
    def __init__(self, dataframe, images_dir, processor, train=False, max_target_length=128):
        self.data = dataframe.reset_index(drop=True)
        self.images_dir = images_dir
        self.processor = processor
        self.train = train
        self.max_target_length = max_target_length

        if self.train:
            self.augmentations = T.Compose([
                T.RandomRotation(degrees=2),
                T.ColorJitter(brightness=0.1, contrast=0.1),
            ])
        else:
            self.augmentations = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.images_dir, row['image_path'])
        text = row['text']

        image = Image.open(img_path).convert("RGB")
        if self.augmentations:
            image = self.augmentations(image)

        # Processor will handle resizing to 384×384 internally
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.squeeze(0)
        labels = self.processor.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_target_length,
            truncation=True,
            return_tensors="pt"
        ).input_ids.squeeze(0)

        # Replace padding token id's of the labels by -100 so they are ignored by the loss
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {
            "pixel_values": pixel_values,
            "labels": labels,
        }


class TQDMProgressBar(TrainerCallback):
    def __init__(self):
        self.progress_bar = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.progress_bar = tqdm(total=state.max_steps, desc="Training Progress", dynamic_ncols=True)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if self.progress_bar is not None and logs is not None:
            loss = logs.get('loss', None)
            desc = f"Epoch {int(state.epoch):>2} | Step {state.global_step:>5}"
            if loss is not None:
                desc += f" | Loss {loss:.4f}"
            if 'eval_cer' in logs and 'eval_wer' in logs:
                desc += f" | CER {logs['eval_cer']:.4f} | WER {logs['eval_wer']:.4f}"
            self.progress_bar.set_description(desc)

    def on_step_end(self, args, state, control, **kwargs):
        if self.progress_bar:
            self.progress_bar.update(1)

    def on_train_end(self, args, state, control, **kwargs):
        if self.progress_bar:
            self.progress_bar.close()


def build_model():
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model.to(device)

    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id

    return processor, model, device


def build_datasets(processor, test_size=0.2):
    df = pd.read_csv("data/spanish_data.csv")
    train_df, val_df = train_test_split(df, test_size=test_size, random_state=42)

    train_dataset = SpanishOCRDataset(train_df, "data/images/", processor, train=True)
    val_dataset = SpanishOCRDataset(val_df, "data/images/", processor, train=False)
    return train_dataset, val_dataset


def build_compute_metrics(processor):
    cer_metric = evaluate.load("cer")
    wer_metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        if pred_ids.ndim == 3:  # logits
            pred_ids = pred_ids.argmax(axis=-1)
        label_ids = pred.label_ids

        label_ids = torch.tensor(label_ids)
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

        cer = cer_metric.compute(predictions=pred_str, references=label_str)
        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        return {"cer": cer, "wer": wer}

    return compute_metrics


def main():
    # Training arguments with improved settings
    training_args = TrainingArguments(
        output_dir="./spanish_ocr_model",
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir="./logs",
        save_total_limit=2,
        learning_rate=1e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=50,
        weight_decay=0.01,
        warmup_steps=500,
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        report_to="none",
        predict_with_generate=True,
        generation_max_length=128,
        generation_num_beams=5,
        generation_early_stopping=True,
    )

    processor, model, device = build_model()
    train_dataset, val_dataset = build_datasets(processor)
    compute_metrics = build_compute_metrics(processor)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[TQDMProgressBar()],
        tokenizer=processor.feature_extractor,  # ensures correct preprocessing for generation
    )

    # Optionally, resume from last checkpoint automatically:
    trainer.train(resume_from_checkpoint=True)

    # Save the best model & processor
    model.save_pretrained("./spanish_ocr_model")
    processor.save_pretrained("./spanish_ocr_model")


if __name__ == "__main__":
    main()