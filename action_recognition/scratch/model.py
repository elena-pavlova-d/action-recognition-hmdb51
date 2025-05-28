from pathlib import Path
from typing import Optional, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchmetrics import Accuracy

from data import HumanActionDataset


class VideoActionModel(pl.LightningModule):
    def __init__(self, num_classes=51, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()

        # 3D CNN архитектура
        self.conv1 = nn.Conv3d(3, 16, kernel_size=(3, 3, 3), padding=1)
        self.pool1 = nn.MaxPool3d((1, 2, 2))

        self.conv2 = nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=1)
        self.pool2 = nn.MaxPool3d((1, 2, 2))

        self.conv3 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=1)
        self.pool3 = nn.MaxPool3d((1, 2, 2))

        self.gap = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(64, num_classes)

        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.lr = learning_rate

    def forward(self, x):
        # x: (B, T, C, H, W) -> (B, C, T, H, W)
        x = x.permute(0, 2, 1, 3, 4)

        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        x = F.relu(self.conv3(x))
        x = self.pool3(x)

        x = self.gap(x)
        x = x.view(x.size(0), -1)  # Выравниваем для полносвязного слоя
        return self.fc(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.accuracy(logits, y), prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.accuracy(logits, y), prog_bar=True)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)


def create_dataloaders(
    dataset_path: Union[str, Path],
    batch_size: int = 8,
    num_workers: int = 4,
    seq_len: int = 16,
    num_classes: Optional[int] = None,
    target_size: tuple = (128, 128),
):
    dataset = HumanActionDataset(
        video_dir=dataset_path,
        seq_len=seq_len,
        num_classes=num_classes if num_classes else 51,
        target_size=target_size,
    )

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_set, val_set = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )

    return train_loader, val_loader
