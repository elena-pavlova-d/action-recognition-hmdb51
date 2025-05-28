import random
from pathlib import Path
from typing import Callable, Optional, Tuple, Union

import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
from torchvision.io import read_video


class HumanActionDataset(Dataset):
    def __init__(
        self,
        video_dir: Union[str, Path],
        seq_len: int = 16,
        num_classes: int = 51,
        transform: Optional[Callable] = None,
        target_size: Tuple[int, int] = (128, 128),
    ):
        self.video_dir = Path(video_dir)
        self.seq_len = seq_len
        self.num_classes = min(num_classes, 51)
        self.transform = transform
        self.target_size = target_size
        self.resize = T.Resize(target_size)

        random.seed(42)

        # Получаем список классов (поддиректорий)
        self.classes = sorted(
            [cls.name for cls in self.video_dir.iterdir() if cls.is_dir()]
        )[: self.num_classes]

        # Собираем все видеофайлы
        self.files_list = []
        for cls in self.classes:
            for video_file in (self.video_dir / cls).iterdir():
                if video_file.is_file() and video_file.suffix.lower() in [
                    ".avi",
                    ".mp4",
                ]:
                    self.files_list.append(video_file)

        if not self.files_list:
            raise RuntimeError(f"No supported video files found in {video_dir}")

    def __len__(self) -> int:
        return len(self.files_list)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        video_path = self.files_list[index]
        video_str_path = str(video_path)  # Явное преобразование в строку

        try:
            # Чтение видео без параметра backend
            vframes, _, _ = read_video(
                video_str_path, pts_unit="sec", output_format="TCHW"
            )

            # Нормализация и изменение размера
            vframes = vframes.float() / 255.0
            vframes = torch.stack([self.resize(frame) for frame in vframes])

            # Выбор кадров
            total_frames = len(vframes)
            if total_frames < self.seq_len:
                # Дублируем последний кадр, если кадров недостаточно
                padding = self.seq_len - total_frames
                last_frame = vframes[-1].unsqueeze(0)
                vframes = torch.cat([vframes, last_frame.repeat(padding, 1, 1, 1)])
            else:
                indices = torch.linspace(0, total_frames - 1, steps=self.seq_len).long()
                vframes = vframes[indices]

            label_name = video_path.parent.name
            label_idx = self.classes.index(label_name)

            if self.transform:
                vframes = self.transform(vframes)

            return vframes, torch.tensor(label_idx)

        except Exception as e:
            print(f"Error loading video {video_str_path}: {str(e)}")
            # Возвращаем нулевой тензор вместо рекурсии
            return (torch.zeros((self.seq_len, 3, *self.target_size)), torch.tensor(0))
