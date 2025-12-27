"""Dataset classes for gaze estimation.

Supports:
- MPIIFaceGaze: 15-fold leave-one-subject-out cross-validation
- Gaze360: Large-scale dataset with train/val/test splits
"""

import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class MPIIGazeDataset(Dataset):
    """MPIIFaceGaze dataset with leave-one-subject-out evaluation.

    Args:
        data_root: Root directory containing Image/ and Label/ folders.
        transform: Image transforms to apply.
        fold: Fold index for leave-one-out (0-14). Default: 0.
        train: If True, use all folds except `fold` for training.
        angle_limit: Maximum gaze angle in degrees. Default: 42.

    Example:
        >>> dataset = MPIIGazeDataset("data/MPIIFaceGaze", fold=0, train=True)
        >>> img, label = dataset[0]
    """

    def __init__(
        self,
        data_root: str,
        transform: transforms.Compose | None = None,
        fold: int = 0,
        train: bool = True,
        angle_limit: float = 42.0,
    ):
        self.data_root = Path(data_root)
        self.transform = transform or self._default_transform()
        self.angle_limit = angle_limit

        # Find label files
        label_dir = self.data_root / "Label"
        label_files = sorted(label_dir.glob("*.label"))

        if not label_files:
            raise FileNotFoundError(f"No label files found in {label_dir}")

        # Select folds
        if train:
            selected_files = [f for i, f in enumerate(label_files) if i != fold]
        else:
            selected_files = [label_files[fold]]

        # Load samples
        self.samples = []
        for label_file in selected_files:
            self._load_samples(label_file)

    def _default_transform(self) -> transforms.Compose:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def _load_samples(self, label_file: Path) -> None:
        """Load samples from a label file."""
        with open(label_file) as f:
            lines = f.readlines()[1:]  # Skip header

        for line in lines:
            parts = line.strip().split()
            if len(parts) < 8:
                continue

            face_path = parts[0]
            gaze_str = parts[7]

            # Parse gaze angles (radians -> degrees)
            gaze = np.array(gaze_str.split(","), dtype=float)
            pitch_deg = gaze[0] * 180 / np.pi
            yaw_deg = gaze[1] * 180 / np.pi

            # Filter by angle limit
            if abs(pitch_deg) <= self.angle_limit and abs(yaw_deg) <= self.angle_limit:
                self.samples.append({
                    "face": face_path,
                    "pitch": pitch_deg,
                    "yaw": yaw_deg,
                })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]

        # Load image
        img_path = self.data_root / "Image" / sample["face"]
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        label = torch.tensor([sample["pitch"], sample["yaw"]], dtype=torch.float32)
        return img, label


class Gaze360Dataset(Dataset):
    """Gaze360 dataset.

    Args:
        data_root: Root directory containing Image/ and Label/ folders.
        label_file: Path to label file (train.label, val.label, or test.label).
        transform: Image transforms to apply.
        angle_limit: Maximum gaze angle in degrees. Default: 60.

    Example:
        >>> dataset = Gaze360Dataset("data/Gaze360", "Label/train.label")
        >>> img, label = dataset[0]
    """

    def __init__(
        self,
        data_root: str,
        label_file: str = "Label/train.label",
        transform: transforms.Compose | None = None,
        angle_limit: float = 60.0,
    ):
        self.data_root = Path(data_root)
        self.transform = transform or self._default_transform()
        self.angle_limit = angle_limit

        # Load samples
        label_path = self.data_root / label_file
        self.samples = self._load_samples(label_path)

    def _default_transform(self) -> transforms.Compose:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def _load_samples(self, label_path: Path) -> list[dict]:
        samples = []
        with open(label_path) as f:
            lines = f.readlines()[1:]  # Skip header

        for line in lines:
            parts = line.strip().split()
            if len(parts) < 6:
                continue

            face_path = parts[0]
            gaze_str = parts[5]

            gaze = np.array(gaze_str.split(","), dtype=float)
            pitch_deg = gaze[0] * 180 / np.pi
            yaw_deg = gaze[1] * 180 / np.pi

            if abs(pitch_deg) <= self.angle_limit and abs(yaw_deg) <= self.angle_limit:
                samples.append({
                    "face": face_path,
                    "pitch": pitch_deg,
                    "yaw": yaw_deg,
                })

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]

        img_path = self.data_root / "Image" / sample["face"]
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        label = torch.tensor([sample["pitch"], sample["yaw"]], dtype=torch.float32)
        return img, label


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    prefetch_factor: int = 4,
) -> DataLoader:
    """Create a DataLoader with sensible defaults.

    Args:
        dataset: PyTorch Dataset instance.
        batch_size: Batch size. Default: 32.
        shuffle: Whether to shuffle. Default: True.
        num_workers: Number of worker processes. Default: 4.
        pin_memory: Pin memory for faster GPU transfer. Default: True.
        prefetch_factor: Number of batches to prefetch per worker. Default: 4.

    Returns:
        DataLoader instance.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        drop_last=shuffle,  # Drop last incomplete batch for training
    )
