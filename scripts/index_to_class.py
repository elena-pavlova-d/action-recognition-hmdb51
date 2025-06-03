import json
from pathlib import Path
from pprint import pprint
from typing import Dict

from action_recognition.pl_modules.data import HumanActionDataset


def create_and_save_class_mapping(
    data_dir: str = "data",
    split: str = "train",
    output_path: str = "index_to_class.json",
    verbose: bool = True,
) -> Dict[int, str]:
    """
    Создает и сохраняет словарь соответствия индексов классов их названиям.
    """
    try:
        if not Path(data_dir).exists():
            raise ValueError(f"Директория {data_dir} не существует")

        dataset = HumanActionDataset(root_dir=data_dir, split=split)
        index_to_class = {
            idx: class_name for idx, class_name in enumerate(dataset.classes)
        }

        if verbose:
            print("Создан словарь соответствия классов:")
            pprint(index_to_class)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(index_to_class, f, ensure_ascii=False, indent=4)

        if verbose:
            print(f"\nСловарь сохранен в файл: {output_path}")

        return index_to_class

    except Exception as e:
        if verbose:
            print(f"Ошибка при создании маппинга классов: {str(e)}")
        raise


if __name__ == "__main__":
    class_mapping = create_and_save_class_mapping(
        data_dir="data", split="train", output_path="index_to_class.json", verbose=True
    )
