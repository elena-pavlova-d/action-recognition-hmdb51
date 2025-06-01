from datetime import datetime
from pathlib import Path

import git
import hydra
import mlflow
import pytorch_lightning as pl
from omegaconf import DictConfig

from action_recognition.pl_modules.data import HumanActionDataModule
from action_recognition.pl_modules.model import VideoActionModel


class MLflowLogger(pl.Callback):
    def __init__(self, cfg):
        self.cfg = cfg
        self.run_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.run_active = False

    def setup(self, trainer, pl_module, stage: str):  # <-- исправлено здесь
        if not self.run_active:
            mlflow.set_tracking_uri(self.cfg.mlflow.tracking_uri)
            mlflow.set_experiment(self.cfg.mlflow.experiment_name)
            mlflow.start_run(run_name=self.run_name)
            self.run_active = True

            mlflow.log_params(
                {
                    "batch_size": self.cfg.training.batch_size,
                    "learning_rate": self.cfg.training.learning_rate,
                    "seq_len": self.cfg.training.seq_len,
                    "target_size": f"{self.cfg.training.target_size[0]}x{self.cfg.training.target_size[1]}",
                    "num_classes": self.cfg.model.num_classes,
                    "max_epochs": self.cfg.training.max_epochs,
                    "num_workers": self.cfg.training.num_workers,
                }
            )

            try:
                repo = git.Repo(search_parent_directories=True)
                mlflow.log_param("git_commit", repo.head.commit.hexsha)
            except Exception as e:
                print(f"Could not log git commit: {e}")

    def on_test_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        mlflow.log_metrics(
            {
                "test_loss": metrics.get("test_loss", float("nan")),
                "test_acc": metrics.get("test_acc", float("nan")),
            },
            step=trainer.current_epoch,
        )

    def teardown(self, trainer, pl_module, stage: str):
        if self.run_active:
            mlflow.end_run()
            self.run_active = False


@hydra.main(
    config_path="../../conf",
    config_name="config",  # Используем основной конфиг для доступа к настройкам MLflow
    version_base="1.3",
)
def infer(cfg: DictConfig):
    # Инициализация данных
    data_dir = Path(cfg.inference.data_dir)

    datamodule = HumanActionDataModule(
        data_dir=data_dir,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        seq_len=cfg.training.seq_len,
        target_size=tuple(cfg.training.target_size),
        num_classes=cfg.model.num_classes,
    )
    datamodule.setup(stage="test")

    # Загрузка модели
    model = VideoActionModel.load_from_checkpoint(cfg.inference.ckpt_path)

    # Инициализация MLflow callback
    mlflow_callback = MLflowLogger(cfg)

    # Настройка Trainer с MLflow callback
    trainer = pl.Trainer(
        accelerator="auto",
        callbacks=[mlflow_callback],
        logger=True,  # Отключаем стандартные логгеры, т.к. используем MLflow
    )

    # Запуск тестирования с логированием
    trainer.test(model=model, datamodule=datamodule)


if __name__ == "__main__":
    infer()
