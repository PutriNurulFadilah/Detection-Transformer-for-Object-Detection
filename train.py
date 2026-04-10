import os
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from roboflow import Roboflow
from transformers import DetrImageProcessor, DetrForObjectDetection
from pytorch_lightning.loggers import TensorBoardLogger
import torch
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt  # Import matplotlib untuk visualisasi

from config import *
from data.dataset import CocoDetection, collate_fn

class Detr(pl.LightningModule):
    def __init__(self, num_labels=91, lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4):
        super().__init__()
        self.model = DetrForObjectDetection.from_pretrained(
            pretrained_model_name_or_path=CHECKPOINT,
            num_labels=num_labels,
            ignore_mismatched_sizes=True
        )
        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay

        # Inisialisasi tracking loss
        self.train_losses = []

        # Membuat direktori untuk menyimpan log
        self.log_dir = Path('training_logs') / datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created log directory: {self.log_dir}")

    def training_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        pixel_mask = batch["pixel_mask"]
        labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]

        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)

        # Log loss ke dalam list
        loss_dict = {
            'epoch': self.current_epoch,
            'step': self.global_step,
            'total_loss': outputs.loss.item(),
            'loss_ce': outputs.loss_dict['loss_ce'].item(),
            'loss_bbox': outputs.loss_dict['loss_bbox'].item(),
            'loss_giou': outputs.loss_dict['loss_giou'].item()
        }
        self.train_losses.append(loss_dict)

        # Log ke TensorBoard atau logger lainnya
        for name, value in outputs.loss_dict.items():
            self.log(f"train_{name}", value.item())

        return outputs.loss

    def on_train_epoch_end(self):
        # Ubah list loss menjadi DataFrame
        df = pd.DataFrame(self.train_losses)

        # Simpan loss ke file CSV
        csv_path = self.log_dir / 'training_losses.csv'
        df.to_csv(csv_path, index=False)

        # Hitung dan cetak statistik epoch
        epoch_stats = df[df['epoch'] == self.current_epoch].mean()
        print(f"\nEpoch {self.current_epoch} statistics:")
        print(f"Total Loss: {epoch_stats['total_loss']:.4f}")
        print(f"Classification Loss: {epoch_stats['loss_ce']:.4f}")
        print(f"Bbox Loss: {epoch_stats['loss_bbox']:.4f}")
        print(f"GIoU Loss: {epoch_stats['loss_giou']:.4f}")

        # Visualisasi loss
        plt.figure(figsize=(10, 5))
        plt.plot(df['step'], df['total_loss'], label="Total Loss", color='blue')
        plt.plot(df['step'], df['loss_ce'], label="Classification Loss", color='red')
        plt.plot(df['step'], df['loss_bbox'], label="Bbox Loss", color='green')
        plt.plot(df['step'], df['loss_giou'], label="GIoU Loss", color='purple')

        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.title(f"Training Loss (Epoch {self.current_epoch})")
        plt.legend()
        plt.grid(True)

        plot_path = self.log_dir / f'loss_epoch_{self.current_epoch}.png'
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved loss plot: {plot_path}")

    def configure_optimizers(self):
        param_dicts = [
            {
                "params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad],
                "weight_decay": 1e-4,  # Weight decay untuk detektor
            },
            {
                "params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": self.lr_backbone,
            },
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=self.lr)

        # Scheduler dengan warmup
        def lr_lambda(epoch):
            if epoch < 10:  # Warmup selama 10 epoch
                return epoch / 10
            else:
                return 0.1 ** ((epoch - 10) // 30)  # Decay learning rate setiap 30 epoch

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return [optimizer], [scheduler]

    @classmethod
    def from_pretrained(cls, model_path, num_labels=91):
        model = cls(num_labels=num_labels)
        model.model = DetrForObjectDetection.from_pretrained(
            model_path,
            num_labels=num_labels,
            ignore_mismatched_sizes=True
        )
        return model

def setup_data():
    # Download dataset dari Roboflow
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    project = rf.workspace("your workspace name").project("your project name")
    version = project.version(8) #your version project
    dataset = version.download("coco")

    # Setup direktori dataset
    train_dir = os.path.join(dataset.location, "train")
    val_dir = os.path.join(dataset.location, "valid")
    test_dir = os.path.join(dataset.location, "test")

    image_processor = DetrImageProcessor.from_pretrained(CHECKPOINT)

    # Buat dataset
    train_dataset = CocoDetection(train_dir, image_processor, train=True)
    val_dataset = CocoDetection(val_dir, image_processor, train=False)
    test_dataset = CocoDetection(test_dir, image_processor, train=False)

    # Buat DataLoader
    train_dataloader = DataLoader(
        train_dataset,
        collate_fn=lambda b: collate_fn(b, image_processor),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )
    val_dataloader = DataLoader(
        val_dataset,
        collate_fn=lambda b: collate_fn(b, image_processor),
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )
    test_dataloader = DataLoader(
        test_dataset,
        collate_fn=lambda b: collate_fn(b, image_processor),
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )

    return train_dataloader, val_dataloader, test_dataloader, train_dataset.coco.cats

def main():
    # Inisialisasi TensorBoard logger
    logger = TensorBoardLogger(
        save_dir="logs",
        name="detr_training",
        default_hp_metric=False
    )

    # Setup data
    train_dataloader, val_dataloader, test_dataloader, categories = setup_data()
    id2label = {k: v['name'] for k, v in categories.items()}

    # Inisialisasi model
    model = Detr(
        num_labels=len(id2label),
        lr=1e-4,
        lr_backbone=1e-5,
        weight_decay=1e-4
    )

    # Konfigurasi trainer dengan logger
    trainer = pl.Trainer(
        devices=1,
        accelerator="gpu",
        max_epochs=MAX_EPOCHS,
        gradient_clip_val=0.1,
        accumulate_grad_batches=8,
        log_every_n_steps=5,
        logger=logger
    )
    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader
    )
    # Simpan model
    model.model.save_pretrained(MODEL_PATH)

if __name__ == "__main__":
    main()
