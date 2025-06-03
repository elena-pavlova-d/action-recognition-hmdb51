from typing import Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torchmetrics import Accuracy


class VideoActionModel(pl.LightningModule):
    def __init__(self, num_classes: int = 51, learning_rate: float = 1e-3):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1, 3, 4)
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.accuracy(logits, y), prog_bar=True)
        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.accuracy(logits, y), prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = self.accuracy(
            logits, y
        )  # тут меняю self.test_accuracy(logits, y) на self.accuracy(logits, y)
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)
