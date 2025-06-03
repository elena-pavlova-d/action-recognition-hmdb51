from datetime import datetime
from pathlib import Path

import git
import hydra
import mlflow
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint

from action_recognition.pl_modules.data import HumanActionDataModule
from action_recognition.pl_modules.model import VideoActionModel


class MLflowLogger(pl.Callback):
    def __init__(self, cfg):
        self.cfg = cfg
        self.run_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def setup(self, trainer, pl_module, stage=None):
        if stage != "fit":
            return

        mlflow.set_tracking_uri(self.cfg.mlflow.tracking_uri)
        mlflow.set_experiment(self.cfg.mlflow.experiment_name)
        mlflow.start_run(run_name=self.run_name)

        params = {
            "batch_size": self.cfg.training.batch_size,
            "learning_rate": self.cfg.training.learning_rate,
            "seq_len": self.cfg.training.seq_len,
            "target_size": f"{self.cfg.training.target_size[0]}x{self.cfg.training.target_size[1]}",
            "num_classes": self.cfg.model.num_classes,
            "max_epochs": self.cfg.training.max_epochs,
        }
        mlflow.log_params(params)

        try:
            repo = git.Repo(search_parent_directories=True)
            mlflow.log_param("git_commit", repo.head.commit.hexsha)
        except Exception as e:
            print(f"Could not log git commit: {e}")

    def on_train_epoch_end(self, trainer, pl_module, outputs):
        metrics = trainer.callback_metrics
        mlflow.log_metrics(
            {
                "train_loss": metrics.get("train_loss", float("nan")),
                "train_acc": metrics.get("train_acc", float("nan")),
            },
            step=trainer.current_epoch,
        )

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        mlflow.log_metrics(
            {
                "val_loss": metrics.get("val_loss", float("nan")),
                "val_acc": metrics.get("val_acc", float("nan")),
            },
            step=trainer.current_epoch,
        )

    def teardown(self, trainer, pl_module, stage=None):
        if stage == "fit":
            mlflow.end_run()


@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def train(cfg: DictConfig):
    datamodule = HumanActionDataModule(
        data_dir=Path("data"),
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        seq_len=cfg.training.seq_len,
        num_classes=cfg.model.num_classes,
        target_size=tuple(cfg.training.target_size),
    )

    model = VideoActionModel(
        num_classes=cfg.model.num_classes,
        learning_rate=cfg.training.learning_rate,
    )

    checkpoint_callback = ModelCheckpoint(
        monitor=cfg.model.monitor,
        mode=cfg.model.mode,
        save_top_k=cfg.model.save_top_k,
        filename=cfg.model.checkpoint_filename,
    )

    mlflow_callback = MLflowLogger(cfg)

    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        callbacks=[checkpoint_callback, mlflow_callback],
        accelerator=cfg.training.accelerator,
        enable_progress_bar=cfg.training.enable_progress_bar,
    )

    trainer.fit(model, datamodule=datamodule)

    """# Логирование тестовых метрик
    if hasattr(datamodule, "test_dataloader"):
        test_results = trainer.test(model, datamodule=datamodule)
        with mlflow.start_run(run_id=mlflow_callback.run_id):
            mlflow.log_metrics(
                {
                    "test_loss": test_results[0].get("test_loss", float("nan")),
                    "test_acc": test_results[0].get("test_acc", float("nan")),
                }
            )"""


if __name__ == "__main__":
    train()
