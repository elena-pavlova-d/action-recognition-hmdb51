import random
import shutil
from pathlib import Path

import hydra
import rarfile
import requests
from omegaconf import DictConfig
from tqdm import tqdm


def download_data(rar_path: Path, download_url: str):
    if rar_path.is_file():
        print("[INFO] Архив уже загружен.")
    else:
        print("[INFO] Загружаем архив с данными...")
        with requests.get(download_url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get("content-length", 0))
            with open(rar_path, "wb") as f, tqdm(
                total=total_size, unit="B", unit_scale=True, desc="[INFO] Загрузка"
            ) as bar:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    bar.update(len(chunk))
        print("[INFO] Загрузка завершена.")


def extract_raw_data(rar_path: Path, raw_data_dir: Path):
    if raw_data_dir.exists():
        print("[INFO] Папка raw_data уже существует.")
    else:
        print("[INFO] Распаковываем основной архив...")
        raw_data_dir.mkdir(parents=True, exist_ok=True)
        with rarfile.RarFile(rar_path) as rf:
            rf.extractall(raw_data_dir)
        print("[INFO] Основной архив распакован.")


def extract_all_data(raw_data_dir: Path, extracted_data_dir: Path, rar_path: Path):
    if extracted_data_dir.exists() and any(extracted_data_dir.iterdir()):
        print("[INFO] Папка с извлечёнными данными уже существует.")
    else:
        print("[INFO] Распаковываем классы из .rar файлов...")
        extracted_data_dir.mkdir(parents=True, exist_ok=True)
        for file_path in raw_data_dir.iterdir():
            if file_path.suffix == ".rar":
                with rarfile.RarFile(file_path) as rf:
                    rf.extractall(extracted_data_dir)
                print(f"[INFO] Распакован: {file_path.name}")
        print("[INFO] Удаляем временные файлы...")
        if rar_path.exists():
            rar_path.unlink()
        if raw_data_dir.exists():
            shutil.rmtree(raw_data_dir)
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

        for split in ["train", "val", "test"]:
            (target_dir / split / cls_dir.name).mkdir(parents=True, exist_ok=True)

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


@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    print("Loaded config keys:", list(cfg.keys()))
    download_data(Path(cfg.rar_path), cfg.download_url)
    extract_raw_data(Path(cfg.rar_path), Path(cfg.raw_data_dir))
    extract_all_data(
        Path(cfg.raw_data_dir), Path(cfg.extracted_data_dir), Path(cfg.rar_path)
    )
    split_data_into_train_val_test(
        Path(cfg.extracted_data_dir),
        Path(cfg.data_dir),
        cfg.split.train_ratio,
        cfg.split.val_ratio,
        cfg.split.test_ratio,
        cfg.split.seed,
    )


if __name__ == "__main__":
    main()
