from datetime import datetime
from pathlib import Path

import git
import hydra
import mlflow
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint

from action_recognition.pl_modules.data import HumanActionDataModule
from action_recognition.pl_modules.model import VideoActionModel
from scripts.dvc_helper import fetch_data_with_dvc


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

    def on_train_epoch_end(self, trainer, pl_module):
        """Логирование метрик после каждой тренировочной эпохи"""
        metrics = trainer.callback_metrics
        mlflow.log_metrics(
            {
                "train_loss": metrics.get("train_loss", float("nan")),
                "train_acc": metrics.get("train_acc", float("nan")),
            },
            step=trainer.current_epoch,
        )

    def on_validation_epoch_end(self, trainer, pl_module):
        """Логирование метрик после каждой валидационной эпохи"""
        metrics = trainer.callback_metrics
        mlflow.log_metrics(
            {
                "val_loss": metrics.get("val_loss", float("nan")),
                "val_acc": metrics.get("val_acc", float("nan")),
            },
            step=trainer.current_epoch,
        )

        checkpoint_callback = next(
            (cb for cb in trainer.callbacks if isinstance(cb, ModelCheckpoint)), None
        )
        if checkpoint_callback and checkpoint_callback.best_model_path:
            mlflow.log_artifact(checkpoint_callback.best_model_path)

    def teardown(self, trainer, pl_module, stage=None):
        """Завершение работы MLflow в конце обучения"""
        """if stage == "fit":
            mlflow.end_run()"""
        pass


@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def train(cfg: DictConfig):
    fetch_data_with_dvc(data_path="data")
    Path("plots/checkpoints").mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    with mlflow.start_run(run_name=f"train-{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        try:
            mlflow.log_params(
                {
                    "batch_size": cfg.training.batch_size,
                    "learning_rate": cfg.training.learning_rate,
                    "seq_len": cfg.training.seq_len,
                    "target_size": f"{cfg.training.target_size[0]}x{cfg.training.target_size[1]}",
                    "num_classes": cfg.model.num_classes,
                    "max_epochs": cfg.training.max_epochs,
                }
            )

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
                dirpath=Path(cfg.mlflow.tracking_uri.replace("file:", "")).joinpath(
                    "checkpoints"
                ),
                monitor=cfg.model.monitor,
                mode=cfg.model.mode,
                save_top_k=cfg.model.save_top_k,
                filename=cfg.model.checkpoint_filename,
            )

            class MetricsLogger(pl.Callback):
                def on_train_epoch_end(self, trainer, pl_module):
                    metrics = trainer.callback_metrics
                    mlflow.log_metrics(
                        {
                            "train_loss": metrics.get("train_loss"),
                            "train_acc": metrics.get("train_acc"),
                        },
                        step=trainer.current_epoch,
                    )

                def on_validation_epoch_end(self, trainer, pl_module):
                    metrics = trainer.callback_metrics
                    mlflow.log_metrics(
                        {
                            "val_loss": metrics.get("val_loss"),
                            "val_acc": metrics.get("val_acc"),
                        },
                        step=trainer.current_epoch,
                    )

            trainer = pl.Trainer(
                max_epochs=cfg.training.max_epochs,
                callbacks=[checkpoint_callback, MetricsLogger()],
                accelerator=cfg.training.accelerator,
                enable_progress_bar=cfg.training.enable_progress_bar,
            )

            trainer.fit(model, datamodule=datamodule)

            print("Начало регистрации модели в MLflow...")

            val_loader = datamodule.val_dataloader()
            val_batch = next(iter(val_loader))
            input_example = val_batch[0][0].unsqueeze(0).numpy()  # [1, C, T, H, W]
            print(f"Форма input_example: {input_example.shape}")

            mlflow.pytorch.log_model(
                pytorch_model=model,
                artifact_path="model",
                registered_model_name=cfg.mlflow.registered_model_name,
                input_example=input_example,
                pip_requirements=["torch", "pytorch-lightning"],
                signature=mlflow.models.infer_signature(
                    input_example,
                    model(torch.from_numpy(input_example)).detach().numpy(),
                ),
            )

            if checkpoint_callback.best_model_path:
                mlflow.log_artifact(checkpoint_callback.best_model_path, "checkpoints")
                print(f"Чекпоинт сохранён: {checkpoint_callback.best_model_path}")

            client = mlflow.tracking.MlflowClient()
            model_versions = client.search_model_versions(
                f"name='{cfg.mlflow.registered_model_name}'"
            )

            if model_versions:
                latest_version = model_versions[-1].version
                print(f"Найдена версия модели: {latest_version}")

                client.transition_model_version_stage(
                    name=cfg.mlflow.registered_model_name,
                    version=latest_version,
                    stage=cfg.mlflow.model_stage,
                )
                print(f"Модель переведена в {cfg.mlflow.model_stage} stage")
            else:
                print("Не удалось найти зарегистрированные версии модели")

        except Exception as e:
            print(f"Ошибка во время обучения: {str(e)}")
            mlflow.log_text(str(e), "error_log.txt")
            raise

    print("Обучение завершено, MLflow run закрыт")


if __name__ == "__main__":
    train()
