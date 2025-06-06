import subprocess

import typer

app = typer.Typer()


@app.command()
def download_data():
    subprocess.run(["poetry", "run", "python", "scripts/download_data.py"], check=True)


@app.command()
def export_onnx():
    """Экспортировать модель в ONNX"""
    subprocess.run(
        ["poetry", "run", "python", "-m", "scripts.export_to_onnx"], check=True
    )


@app.command()
def convert_trt(
    onnx_path: str = "video_action_model.onnx",
    engine_path: str = "video_action_model.engine",
    precision: str = "fp16",
):
    """Конвертировать ONNX-модель в TensorRT"""
    subprocess.run(
        [
            "poetry",
            "run",
            "python",
            "scripts/convert_to_trt.py",
            onnx_path,
            engine_path,
            precision,
        ],
        check=True,
    )


@app.command()
def train_ml_flow_dvc():
    """Запустить обучение с MLflow и DVC"""
    subprocess.run(
        ["poetry", "run", "python", "-m", "action_recognition.train_ml_flow_dvc"],
        check=True,
    )


@app.command()
def train_ml_flow():
    """Запустить обучение с MLflow (без DVC)"""
    subprocess.run(
        ["poetry", "run", "python", "-m", "action_recognition.train_ml_flow"],
        check=True,
    )


@app.command()
def train_ml_flow_server_dvc():
    """Запустить обучение с для Mlflow Server"""
    subprocess.run(
        [
            "poetry",
            "run",
            "python",
            "-m",
            "action_recognition.train_ml_flow_server_dvc",
        ],
        check=True,
    )


@app.command()
def infer_from_onnx():
    """Запустить inference из ONNX модели"""
    subprocess.run(
        [
            "poetry",
            "run",
            "python",
            "-m",
            "action_recognition.pl_modules.infer_from_onnx",
        ],
        check=True,
    )


@app.command()
def infer_ml_flow():
    """Запустить inference с использованием MLflow"""
    subprocess.run(
        [
            "poetry",
            "run",
            "python",
            "-m",
            "action_recognition.pl_modules.infer_ml_flow",
        ],
        check=True,
    )


@app.command()
def infer():
    """Запустить inference с использованием MLflow"""
    subprocess.run(
        ["poetry", "run", "python", "-m", "action_recognition.pl_modules.infer"],
        check=True,
    )


if __name__ == "__main__":
    app()
