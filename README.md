# Human Action Recognition on HMDB51 with Neural Networks

## Описание проекта

Этот проект направлен на создание системы глубокого обучения для распознавания
действий человека на видео с использованием датасета HMDB51. Цель — разработать
модель, которая может автоматически классифицировать видеоклипы по типу
выполняемого действия, таких как бег, прыжок, подъем, махи руками и др. Всего
предусмотрено 51 возможное действие.

## Датасет

Используется датасет HMDB51, доступный для скачивания из открытых источников:
<http://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/>

## Архитектура модели

Модель основана на трёхслойной 3D-сверточной нейронной сети (3D CNN),
построенной с использованием PyTorch Lightning. Входное видео преобразуется в
тензор с размерностью [batch, channels, frames, height, width], после чего
проходит через несколько 3D свёрток и слоёв подвыборки.

Модель реализована как класс VideoActionModel, унаследованный от
pl.LightningModule с поддержкой training_step, validation_step, test_step и
оптимизации с помощью Adam.

## Инференс и экспорт модели

Модель поддерживает двойной режим инференса:

Из PyTorch .ckpt файла — для тестирования на тестовом датасете.

Из ONNX файла — для ускоренного инференса и продакшен-интеграции.

Проект содержит специальную команду для экспорта модели в ONNX и TensorRT, а
также скрипт для запуска инференса на произвольных видеофайлах.

## Структура проекта

```bash
action_recognition/
├── action_recognition/     # Основной пакет с кодом
├── conf/                   # Конфигурационные файлы
├── scripts/                # Скрипты для различных операций
├── data/                   # Папка с данными (управляется через DVC)
├── data.dvc                # DVC файл для отслеживания данных
├── .dvc/                   # Конфигурация DVC
├── .dvcignore              # Игнорируемые DVC файлы
├── commands.py             # Основные команды проекта
├── pyproject.toml          # Файл зависимостей Poetry
├── poetry.lock             # Зафиксированные версии зависимостей
├── .pre-commit-config.yaml # Конфигурация pre-commit хуков
└── README.md               # Описание проекта
```

## Setup

Установите Poetry (если не установлен):
<https://python-poetry.org/docs/#installation>

Клонируйте репозиторий и установите зависимости:

```bash
poetry install
```

Активируйте виртуальное окружение:

```bash
poetry shell
```

## Загрузка данных

Для начала работы необходимо загрузить датасет HMDB51. Проект включает скрипт,
который автоматически скачивает и распаковывает данные с официального сайта.
После выполнения команды в корне проекта появится папка data/ с данными,
разделенными на val, train и test.

Запустите загрузку данных:

```bash
poetry run python commands.py download-data
```

## Обучение

Если хотите запустить обучение с ml flow (без dvc):

```bash
poetry run python commands.py train-ml-flow
```

Если хотите запустить обучение с ml flow и dvc:

```bash
poetry run python commands.py train-ml-flow-dvc
```

При запуске ml_flow создастся папка plots в корне проекта, куда будут
записываться графики и логи.

Запустите mlflow для просмотра графиков и логов:

```bash
poetry run mlflow ui --backend-store-uri ./plots/mlruns --port 8080
```

После завершения обучения можно модель перевести в onnx формат. Для этого
пропишите явно ваш путь к ckpt файлу в conf\export\export.yaml. Чекпоинты
обучения лежат в папке lightning_logs.

Посла этого для перевода модели в формат onnx выполните:

```bash
poetry run python commands.py export-onnx
```

Теперь в основной директории должен появиться файл video_action_model.onnx.

Далее можно перевести этот файл в TensorRT:

```bash
poetry run python commands.py convert-trt
```

## Inference

Для начала запустите код для создания csv файла с соответствием индекса классу

```bash
poetry run python commands.py index-to-class
```

Чтобы запустить inference из ONNX модели выполните:

```bash
poetry run python commands.py infer-from-onnx
```

Чтобы запустить inference на тестовом датасете с логированием в mlflow:

```bash
poetry run python commands.py infer-ml-flow
```

Запустить inference на тестовом датасете без логирования в mlflow:

```bash
poetry run python commands.py infer
```

Запустить train для MLflow Serving

```bash
poetry run python commands.py train-ml-flow-server-dvc
```

Запустить MLflow Serving. Для запуска надо после вписать Run ID:

```bash
poetry run mlflow models serve -m "plots/mlruns/435846018931546351/57d9bfa626ca48ddbb5aa3323700bf69/artifacts/model" --host 127.0.0.1 --port 5001 --env-manager local
```
