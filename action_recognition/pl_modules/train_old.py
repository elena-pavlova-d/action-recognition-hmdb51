from pathlib import Path

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint

from .data import HumanActionDataModule
from .model import VideoActionModel


@hydra.main(config_path="../../conf", config_name="config", version_base="1.3")
def train(cfg: DictConfig):
    # Конфигурация
    data_dir = Path("data/data")

    # Инициализация модуля данных
    datamodule = HumanActionDataModule(
        data_dir=data_dir,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        seq_len=cfg.training.seq_len,
        target_size=tuple(cfg.training.target_size),
    )

    # Инициализация модели
    model = VideoActionModel(
        num_classes=cfg.model.num_classes,
        learning_rate=cfg.training.learning_rate,
    )

    # Callback для сохранения моделей
    checkpoint_callback = ModelCheckpoint(
        monitor=cfg.model.monitor,
        mode=cfg.model.mode,
        save_top_k=cfg.model.save_top_k,
        filename=cfg.model.checkpoint_filename,
    )

    # Тренер
    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        callbacks=[checkpoint_callback],
        accelerator=cfg.training.accelerator,
        enable_progress_bar=cfg.training.enable_progress_bar,
    )

    # Запуск обучения
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    train()
