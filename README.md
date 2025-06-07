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

## Git Hooks & Code Quality

В этом проекте используется pre-commit — инструмент для автоматического запуска
проверок и автоформатирования кода при коммите.

Настроенные хуки:

- Yaml/JSON checks:

  - Проверка синтаксиса YAML (check-yaml)

  - Проверка синтаксиса JSON (check-json)

- Файлы:

  - Проверка на слишком большие добавленные файлы (check-added-large-files)

  - Автоматическое исправление окончания файла (end-of-file-fixer)

  - Удаление лишних пробелов (trailing-whitespace)

  - Проверка на конфликт регистра в названиях файлов (check-case-conflict)

  - Проверка на смешанные окончания строк (mixed-line-ending)

- Форматирование Python:

  - Автоматическое форматирование с помощью Black

  - Сортировка импортов с помощью isort с профилем Black

- Статический анализ Python:

  - Проверка стиля кода с помощью Flake8 + flake8-bugbear, Flake8-pyproject

- Форматирование других файлов:

  - Форматирование Markdown, YAML, TOML, JSON, Dockerfile и shell-скриптов с
    помощью Prettier

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

В проекте предусмотрена единая точка входа — файл commands.py, реализованный с
помощью Typer.

Через этот интерфейс удобно запускать основные команды проекта из консоли, такие
как:

- Загрузка данных

- Обучение модели

- Экспорт модели в ONNX / TensorRT

- Inference

- Логирование с MLflow

- Поддержка DVC

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
записываться графики и логи. Chekpoints модели буду лежать в папке
lightning_logs.

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

Для начала запустите код для создания csv файла с соответствием индекса классу:

```bash
poetry run python commands.py index-to-class
```

Чтобы запустить inference из ONNX модели в файле
action-recognition-hmdb51\action_recognition\pl_modules\infer_from_onnx.py
пропиши путь к видео, для которого хотите узнать класс и выполните:

```bash
poetry run python commands.py infer-from-onnx
```

Поле этого в консоли будет выведен класс и название класса, которое
предсказывает модель.

Чтобы запустить inference на тестовом датасете для просмотра accuracy и лосса с
логированием в mlflow:

```bash
poetry run python commands.py infer-ml-flow
```

ЧТобы запустить inference на тестовом датасете без логирования в mlflow сначала
пропишите путь к chekpoint в папке conf\inference\inference.yaml, а потом
запустите:

```bash
poetry run python commands.py infer
```

Запустить train для MLflow Serving

```bash
poetry run python commands.py train-ml-flow-server-dvc
```

Запустить MLflow Serving. Для запуска надо прописать путь к модели из папки
plots, которую хотим запустить:

```bash
poetry run mlflow models serve -m "plots/mlruns/435846018931546351/57d9bfa626ca48ddbb5aa3323700bf69/artifacts/model" --host 127.0.0.1 --port 5001 --env-manager local
```
