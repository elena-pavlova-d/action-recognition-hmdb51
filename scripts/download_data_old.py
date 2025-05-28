import random
import shutil
from pathlib import Path

import rarfile
import requests
from tqdm import tqdm

# Пути проекта
BASE_DIR = Path(__file__).parent.parent.resolve()
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw_data"
EXTRACTED_DATA_DIR = DATA_DIR / "data"
HMDB_RAR_PATH = BASE_DIR / "hmdb51_org.rar"
TARGET_DATA_DIR = BASE_DIR / "data"

# Ссылка на архив с данными
DATA_URL = "http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_org.rar"


def download_data():
    if HMDB_RAR_PATH.is_file():
        print("[INFO] Архив уже загружен.")
    else:
        print("[INFO] Загружаем архив с данными...")
        with requests.get(DATA_URL, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get("content-length", 0))
            with open(HMDB_RAR_PATH, "wb") as f, tqdm(
                total=total_size, unit="B", unit_scale=True, desc="[INFO] Загрузка"
            ) as bar:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    bar.update(len(chunk))
        print("[INFO] Загрузка завершена.")


def extract_raw_data():
    if RAW_DATA_DIR.exists():
        print("[INFO] Папка raw_data уже существует.")
    else:
        print("[INFO] Распаковываем основной архив...")
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        with rarfile.RarFile(HMDB_RAR_PATH) as rf:
            rf.extractall(RAW_DATA_DIR)
        print("[INFO] Основной архив распакован.")


def extract_all_data():
    # Проверяем, есть ли уже данные в EXTRACTED_DATA_DIR (data/data/)
    if EXTRACTED_DATA_DIR.exists() and any(EXTRACTED_DATA_DIR.iterdir()):
        print("[INFO] Папка data/data уже существует и не пуста.")
    else:
        print("[INFO] Распаковываем классы из .rar файлов...")
        EXTRACTED_DATA_DIR.mkdir(parents=True, exist_ok=True)  # Создаём data/data/
        for file_path in RAW_DATA_DIR.iterdir():
            if file_path.suffix == ".rar":
                with rarfile.RarFile(file_path) as rf:
                    rf.extractall(EXTRACTED_DATA_DIR)  # <-- Извлекаем в data/data/
                print(f"[INFO] Распакован: {file_path.name}")
        print("[INFO] Удаляем временные файлы...")
        if HMDB_RAR_PATH.exists():
            HMDB_RAR_PATH.unlink()
        if RAW_DATA_DIR.exists():
            shutil.rmtree(RAW_DATA_DIR)
        print("[INFO] Очистка завершена.")


def split_data_into_train_val_test(
    source_dir: Path,
    target_dir: Path,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    seed=42,
):
    random.seed(seed)

    # Проверим, что сумма долей == 1
    assert (
        abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    ), "Ratios must sum to 1."

    classes = [d for d in source_dir.iterdir() if d.is_dir()]
    print(f"[INFO] Найдено классов: {len(classes)}")

    for cls_dir in classes:
        video_files = [
            f
            for f in cls_dir.iterdir()
            if f.is_file() and f.suffix.lower() in [".avi", ".mp4"]
        ]
        random.shuffle(video_files)

        n_total = len(video_files)
        n_train = int(train_ratio * n_total)
        n_val = int(val_ratio * n_total)
        n_test = n_total - n_train - n_val

        print(f"[INFO] Класс '{cls_dir.name}': всего {n_total} видео")
        print(f"       train: {n_train}, val: {n_val}, test: {n_test}")

        # Создаём папки для класса в train/val/test
        for split in ["train", "val", "test"]:
            (target_dir / split / cls_dir.name).mkdir(parents=True, exist_ok=True)

        # Копируем файлы в нужные папки
        for i, video_file in enumerate(video_files):
            if i < n_train:
                split = "train"
            elif i < n_train + n_val:
                split = "val"
            else:
                split = "test"

            dest = target_dir / split / cls_dir.name / video_file.name
            shutil.copy2(video_file, dest)

    print("[INFO] Разделение данных завершено.")


if __name__ == "__main__":
    download_data()
    extract_raw_data()
    extract_all_data()
    split_data_into_train_val_test(EXTRACTED_DATA_DIR, TARGET_DATA_DIR)
