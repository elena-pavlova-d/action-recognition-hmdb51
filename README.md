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
```
