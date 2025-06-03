# Action Recognition with Neural Networks

Проект для распознавания действий на видео с помощью нейронных сетей. Включает
загрузку датасета HMDB51, предобработку данных и обучение модели.

## 🗂️ Датасет

Используется HMDB51:
<http://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/>

## 📁 Структура проекта

```bash
action_recognition/
├── data/                  # Папка с распакованными видеофайлами
├── scripts/               # Скрипты (например, загрузка данных)
│   └── download_data.py
├── README.md              # Описание проекта
├── pyproject.toml         # Файл зависимостей Poetry

```

## 🚀 Быстрый старт

1. Установите Poetry (если не установлен):
   <https://python-poetry.org/docs/#installation>

2. Клонируйте репозиторий и установите зависимости:

```bash
poetry install
```

3. Активируйте виртуальное окружение:

```bash
poetry shell
```

4. Запустите загрузку данных:

```bash
poetry run python scripts\download_data.py
poetry run python commands.py download-data # новый вариант через commands
```

5. Если хотите запустить обучение с ml flow (без dvc):

```bash
poetry run python -m action_recognition.train_ml_flow_dvc
poetry run python commands.py train-ml-flow # новый вариант через commands
```

5. Если хотите запустить обучение с ml flow и dvc:

```bash
poetry run python -m action_recognition.train_ml_flow_dvc
poetry run python commands.py train-ml-flow-dvc # новый вариант через commands
```

При запуске ml_flow создастся папка в корне проекта, куда будут записываться
графики и логи 5. Запустите mlflow для просмотра графиков и логов:

```bash
poetry run mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 8080
poetry run mlflow ui --backend-store-uri ./plots/mlruns --port 8080 #новый вариант
```

После завершения обучения можно модель перевести в onnx формат. Для этого
пропишите явно ваш путь к ckpt файлу в conf\export\export.yaml. Чекпоинты
обучения лежат в папке lightning_logs.

6. Посла этого для перевода модели в формат onnx выполните:

```bash
poetry run python -m scripts.export_to_onnx.py
poetry run python commands.py export-onnx # новый вариант через commands
```

Теперь в основной директории должен появиться файл video_action_model.onnx.

7. Можно перевести этот файл в TensorRT:

```bash
poetry run python scripts/convert_to_trt.py video_action_model.onnx video_action_model.engine fp16
poetry run python commands.py convert-trt # новый вариант через commands
```

6. Чтобы запустить inference из ONNX модели выполните:

```bash
poetry run python commands.py infer-from-onnx
```

6. Чтобы запустить inference на тестовом датасете с логированием в mlflow:

```bash
poetry run python commands.py infer-ml-flow
```

6. Запустить inference на тестовом датасете без логирования в mlflow:

```bash
poetry run python commands.py infer
```
