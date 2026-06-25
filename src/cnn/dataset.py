"""
PyTorch Dataset for ELA-based forgery detection training.

Expected directory layout:
    <data_dir>/
        genuine/   ← ELA images of unmanipulated documents  (label 1.0)
        forged/    ← ELA images of manipulated documents    (label 0.0)

Generate these with:
    python -m src.cnn.generate_data
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]
IMG_SIZE = (224, 224)


class ELADataset(Dataset):
    """
    Loads ELA images from genuine/ and forged/ subdirectories.

    Args:
        data_dir  : Root containing 'genuine' and 'forged' sub-folders.
        split     : 'train' or 'val'.
        val_split : Fraction reserved for validation (default 0.2).
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: Literal["train", "val"] = "train",
        val_split: float = 0.2,
    ) -> None:
        data_dir = Path(data_dir)
        genuine = sorted((data_dir / "genuine").glob("*.png"))
        forged  = sorted((data_dir / "forged").glob("*.png"))

        samples: list[tuple[Path, float]] = (
            [(p, 1.0) for p in genuine] + [(p, 0.0) for p in forged]
        )

        n_val = int(len(samples) * val_split)
        self.samples = samples[:n_val] if split == "val" else samples[n_val:]

        if split == "train":
            self.tf = transforms.Compose([
                transforms.Resize(IMG_SIZE),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.1),
                transforms.ToTensor(),
                transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
            ])
        else:
            self.tf = transforms.Compose([
                transforms.Resize(IMG_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
            ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.tf(img), torch.tensor(label, dtype=torch.float32)
