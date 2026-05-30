"""
EarVN1.0 Dataset class.
28,412 ear images, 164 subjects.
Source: https://data.mendeley.com/datasets/yws3v3mwx3/4
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
from pathlib import Path
from sklearn.model_selection import StratifiedShuffleSplit
from typing import Tuple, Optional
import numpy as np

from .augmentation import get_train_transforms, get_val_transforms


class EarVN10Dataset(Dataset):
    """
    EarVN1.0 ear recognition dataset.

    Expected directory structure:
        root/
        ├── 001/
        │   ├── img001.jpg
        │   ├── img002.jpg
        │   └── ...
        ├── 002/
        └── ...  (164 subject folders)
    """

    def __init__(self, root: str, transform: Optional[transforms.Compose] = None):
        self.root = Path(root)
        self.transform = transform
        self.samples: list[Tuple[str, int]] = []
        self.class_to_idx: dict[str, int] = {}

        self._load_dataset()

    def _load_dataset(self):
        """Walk directory structure and collect (path, label) pairs."""
        if not self.root.exists():
            raise FileNotFoundError(
                f"Dataset not found at {self.root}. "
                f"Download from: https://data.mendeley.com/datasets/yws3v3mwx3/4"
            )

        # Sort subject folders for deterministic class indices
        subject_dirs = sorted([
            d for d in self.root.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ])

        for idx, subject_dir in enumerate(subject_dirs):
            self.class_to_idx[subject_dir.name] = idx
            # Collect all image files for this subject
            for img_path in sorted(subject_dir.iterdir()):
                if img_path.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
                    self.samples.append((str(img_path), idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[index]
        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    @property
    def num_classes(self) -> int:
        return len(self.class_to_idx)

    @property
    def labels(self) -> np.ndarray:
        return np.array([s[1] for s in self.samples])


def build_dataset(cfg: dict) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Build train/val/test datasets with stratified splits.

    Args:
        cfg: Data configuration dict from YAML

    Returns:
        (train_dataset, val_dataset, test_dataset)
    """
    root = cfg["root"]
    image_size = cfg.get("image_size", 224)
    train_split = cfg.get("train_split", 0.70)
    val_split = cfg.get("val_split", 0.15)
    aug_cfg = cfg.get("augmentation", {})

    # Create full dataset (no transforms yet)
    full_dataset = EarVN10Dataset(root, transform=None)
    labels = full_dataset.labels
    num_samples = len(full_dataset)
    indices = np.arange(num_samples)

    # First split: separate test set
    test_ratio = 1.0 - train_split - val_split
    splitter1 = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=42)
    train_val_idx, test_idx = next(splitter1.split(indices, labels))

    # Second split: separate train and val from remaining
    val_ratio_adjusted = val_split / (train_split + val_split)
    splitter2 = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio_adjusted, random_state=42)
    train_val_labels = labels[train_val_idx]
    train_idx_rel, val_idx_rel = next(splitter2.split(train_val_idx, train_val_labels))
    train_idx = train_val_idx[train_idx_rel]
    val_idx = train_val_idx[val_idx_rel]

    # Create subset datasets with appropriate transforms
    train_transform = get_train_transforms(image_size, aug_cfg)
    val_transform = get_val_transforms(image_size)

    train_dataset = TransformedSubset(full_dataset, train_idx, train_transform)
    val_dataset = TransformedSubset(full_dataset, val_idx, val_transform)
    test_dataset = TransformedSubset(full_dataset, test_idx, val_transform)

    print(f"Dataset splits - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    print(f"Number of classes: {full_dataset.num_classes}")

    return train_dataset, val_dataset, test_dataset


class TransformedSubset(Dataset):
    """Dataset subset with its own transform."""

    def __init__(self, dataset: EarVN10Dataset, indices: np.ndarray,
                 transform: transforms.Compose):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        real_idx = self.indices[index]
        img_path, label = self.dataset.samples[real_idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label
