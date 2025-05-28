from pathlib import Path

import pytorch_lightning as pl
from model import VideoActionModel, create_dataloaders
from pytorch_lightning.callbacks import ModelCheckpoint


def train():
    # Путь к данным
    data_path = Path(
        "C:/Users/pelen/OneDrive/Рабочий стол/лена/мага/mlops/action_recognition/data/data"
    )

    # Создаем даталоадеры
    train_loader, val_loader = create_dataloaders(
        dataset_path=data_path,
        batch_size=8,
        num_workers=4,
        seq_len=16,
        target_size=(128, 128),
    )

    # Инициализируем модель
    model = VideoActionModel(num_classes=51)

    # Callback для сохранения лучших моделей
    checkpoint_callback = ModelCheckpoint(
        monitor="val_acc",
        mode="max",
        save_top_k=1,
        filename="best-{epoch}-{val_acc:.2f}",
    )

    # Тренер
    trainer = pl.Trainer(
        max_epochs=20,
        callbacks=[checkpoint_callback],
        accelerator="auto",
        devices=1,
        logger=True,
        enable_progress_bar=True,
    )

    # Запуск обучения
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    train()
