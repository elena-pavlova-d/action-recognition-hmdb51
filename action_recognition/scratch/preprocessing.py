from pathlib import Path

from torchvision.io import read_video


def remove_short_videos(data_path: str | Path, sequence_length: int = 16) -> None:
    data_path = Path(data_path)
    remove_list = []

    for video_file in data_path.glob("*/*"):
        if not video_file.is_file():
            continue

        try:
            vframes, _, _ = read_video(
                str(video_file), pts_unit="sec", output_format="TCHW"
            )
            if len(vframes) <= sequence_length:
                print(
                    f"[WARNING] Недостаточно кадров: {video_file} ({len(vframes)} кадров)"
                )
                remove_list.append(video_file)
        except Exception as e:
            print(f"[ERROR] Ошибка при чтении {video_file}: {e}")
            remove_list.append(video_file)

    for file_path in remove_list:
        try:
            file_path.unlink()
            print(f"[INFO] Удалено: {file_path}")
        except FileNotFoundError:
            print(f"[WARNING] Файл уже удалён: {file_path}")

    print(
        "[INFO] Очистка завершена."
        if remove_list
        else "[INFO] Все видео удовлетворяют требованиям."
    )
