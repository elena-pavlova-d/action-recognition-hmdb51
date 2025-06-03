from pathlib import Path
from typing import Callable, Optional, Tuple, Union

import pytorch_lightning as pl
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_video


class HumanActionDataset(Dataset):
    def __init__(
        self,
        root_dir: Union[str, Path],
        split: str = "train",
        seq_len: int = 16,
        num_classes: int = 51,
        transform: Optional[Callable] = None,
        target_size: Tuple[int, int] = (128, 128),
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.seq_len = seq_len
        self.num_classes = min(num_classes, 51)
        self.transform = transform
        self.target_size = target_size
        self.resize = T.Resize(target_size)

        split_dir = self.root_dir / split
        self.classes = sorted(
            [cls.name for cls in split_dir.iterdir() if cls.is_dir()]
        )[: self.num_classes]

        self.files_list = []
        for cls in self.classes:
            cls_dir = split_dir / cls
            for video_file in cls_dir.iterdir():
                if video_file.is_file() and video_file.suffix.lower() in [
                    ".avi",
                    ".mp4",
                ]:
                    self.files_list.append(video_file)

        if not self.files_list:
            raise RuntimeError(f"No supported video files found in {split_dir}")

    def __len__(self) -> int:
        return len(self.files_list)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        video_path = self.files_list[index]
        video_str_path = str(video_path)

        try:
            vframes, _, _ = read_video(
                video_str_path, pts_unit="sec", output_format="TCHW"
            )

            vframes = vframes.float() / 255.0
            vframes = torch.stack([self.resize(frame) for frame in vframes])

            total_frames = len(vframes)
            if total_frames < self.seq_len:
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
            return (torch.zeros((self.seq_len, 3, *self.target_size)), torch.tensor(0))


class HumanActionDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: Union[str, Path],
        batch_size: int = 8,
        num_workers: int = 0,
        seq_len: int = 16,
        num_classes: Optional[int] = None,
        target_size: Tuple[int, int] = (128, 128),
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seq_len = seq_len
        self.num_classes = num_classes if num_classes else 51
        self.target_size = target_size

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        for split in ["train", "val", "test"]:
            split_dir = self.data_dir / split
            if not split_dir.exists():
                raise FileNotFoundError(
                    f"Directory {split_dir} not found. "
                    "Please ensure data is properly split into train/val/test folders."
                )

    def setup(self, stage: Optional[str] = None):
        if stage in (None, "fit"):
            self.train_dataset = HumanActionDataset(
                root_dir=self.data_dir,
                split="train",
                seq_len=self.seq_len,
                num_classes=self.num_classes,
                target_size=self.target_size,
            )
            self.val_dataset = HumanActionDataset(
                root_dir=self.data_dir,
                split="val",
                seq_len=self.seq_len,
                num_classes=self.num_classes,
                target_size=self.target_size,
            )

        if stage in (None, "test"):
            self.test_dataset = HumanActionDataset(
                root_dir=self.data_dir,
                split="test",
                seq_len=self.seq_len,
                num_classes=self.num_classes,
                target_size=self.target_size,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
