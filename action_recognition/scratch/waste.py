import os
import sys
from timeit import default_timer as timer

import pytorch_lightning as L
import torch
import torchmetrics
from torch.utils.data import DataLoader, random_split
from torchinfo import summary
from torchvision.models.video import MViT_V2_S_Weights, mvit_v2_s

from action_recognition.scratch.data import HumanActionDataset
from action_recognition.scratch.utils import set_seed

# from scratch.data import HumanActionDataset


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# ----------------------- Константы -----------------------

DATA_PATH = "./data/data"
SEQUENCE_LENGTH = 16
NUM_CLASSES = 2
BATCH_SIZE = 16
NUM_EPOCHS = 3
RESULTS_DIR = "results"
WORKERS = os.cpu_count()
SEED = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------- Модель -----------------------


def create_model(num_classes: int, device: torch.device, seed: int = 42):
    weights = MViT_V2_S_Weights.DEFAULT
    transforms = weights.transforms()
    model = mvit_v2_s(weights=weights)

    for params in model.parameters():
        params.requires_grad = False

    set_seed(seed)
    dropout_layer = model.head[0]
    in_features = model.head[1].in_features
    model.head = torch.nn.Sequential(
        dropout_layer,
        torch.nn.Linear(
            in_features=in_features, out_features=num_classes, bias=True, device=device
        ),
    )
    return model.to(device), transforms


# ----------------------- Dataloaders -----------------------


def create_dataloaders(dataset, batch, shuffle, workers):
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        drop_last=True,
    )
    print(f"[INFO] Dataloader created with {len(dataloader)} batches of size {batch}.")
    return dataloader


# ----------------------- Lightning Модель -----------------------


class PyLightHMDB51(L.LightningModule):
    def __init__(self, model, loss_fn, optimizer, num_classes):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.num_classes = num_classes
        self.train_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=self.num_classes
        )
        self.test_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=self.num_classes
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        X, y = batch
        y_preds = self(X)
        loss = self.loss_fn(y_preds, y)
        self.log("train_loss", loss, prog_bar=True)
        preds = torch.argmax(torch.softmax(y_preds, dim=1), dim=1)
        self.train_acc.update(preds, y)
        self.log("train_acc", self.train_acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        y_preds = self(X)
        loss = self.loss_fn(y_preds, y)
        self.log("val_loss", loss, prog_bar=True)
        preds = torch.argmax(torch.softmax(y_preds, dim=1), dim=1)
        self.test_acc.update(preds, y)
        self.log("val_acc", self.test_acc, prog_bar=True)

    def configure_optimizers(self):
        return self.optimizer


# ----------------------- Обучение -----------------------


def main():
    set_seed(SEED)

    if not os.path.exists(RESULTS_DIR):
        os.mkdir(RESULTS_DIR)
        print(f"[INFO] Created results dir at: {RESULTS_DIR}")
    else:
        print(f"[INFO] Using existing results dir: {RESULTS_DIR}")

    dataset = HumanActionDataset(
        video_dir=DATA_PATH, seq_len=SEQUENCE_LENGTH, num_classes=NUM_CLASSES
    )
    print(f"[INFO] Dataset loaded with {len(dataset)} samples.")

    model, transforms = create_model(num_classes=len(dataset.classes), device=device)
    summary(model, input_size=(1, 3, SEQUENCE_LENGTH, 224, 224))

    dataset = HumanActionDataset(
        video_dir=DATA_PATH,
        seq_len=SEQUENCE_LENGTH,
        num_classes=NUM_CLASSES,
        transform=transforms,
    )

    train_size = int(0.75 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(
        dataset, [train_size, test_size], generator=torch.Generator().manual_seed(SEED)
    )

    train_dataloader = create_dataloaders(
        train_dataset, batch=BATCH_SIZE, shuffle=True, workers=WORKERS
    )
    test_dataloader = create_dataloaders(
        test_dataset, batch=BATCH_SIZE, shuffle=False, workers=WORKERS
    )

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0001)

    lightning_model = PyLightHMDB51(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        num_classes=len(dataset.classes),
    )

    trainer = L.Trainer(
        max_epochs=NUM_EPOCHS,
        logger=L.pytorch.loggers.CSVLogger(
            save_dir=RESULTS_DIR, name="pytorch_lightning"
        ),
        accelerator="auto",
    )

    start = timer()
    trainer.fit(
        model=lightning_model,
        train_dataloaders=train_dataloader,
        val_dataloaders=test_dataloader,
    )
    end = timer()

    print(f"[INFO] Training completed in {(end - start)/60:.2f} minutes.")


if __name__ == "__main__":
    main()
