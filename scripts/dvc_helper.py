# scripts/dvc_helper.py
from pathlib import Path

from dvc.repo import Repo


def fetch_data_with_dvc(data_path="data"):
    print(f"[INFO] Проверка данных в '{data_path}'...")
    if not Path(data_path).exists():
        print("[INFO] Данные не найдены. Пытаемся получить через DVC...")

        try:
            repo = Repo(".")
            repo.pull(targets=[f"{data_path}.dvc"], remote="localstorage")
            print("[INFO] DVC pull завершён успешно.")
        except Exception as e:
            print(f"[ERROR] DVC pull завершился с ошибкой: {e}")
            raise
    else:
        print("[INFO] Данные уже существуют. Пропускаем DVC pull.")
