from pathlib import Path

import hydra
import pytorch_lightning as pl
from omegaconf import (  # from pytorch_lightning.callbacks import ModelCheckpoint
    DictConfig,
)

from action_recognition.pl_modules.data import HumanActionDataModule
from action_recognition.pl_modules.model import VideoActionModel


@hydra.main(
    config_path="../../conf/inference", config_name="inference", version_base="1.3"
)
def infer(cfg: DictConfig):
    # Данные
    data_dir = Path(cfg.data_dir)

    datamodule = HumanActionDataModule(
        data_dir=data_dir,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        seq_len=cfg.seq_len,
        target_size=tuple(cfg.target_size),
        num_classes=cfg.num_classes,
    )
    datamodule.setup(stage="test")

    # Загружаем модель из чекпоинта
    model = VideoActionModel.load_from_checkpoint(cfg.ckpt_path)

    # Создаем тренера без обучения, только для теста
    trainer = pl.Trainer(accelerator="auto")

    # Запускаем тестирование
    trainer.test(model=model, datamodule=datamodule)


if __name__ == "__main__":
    infer()
