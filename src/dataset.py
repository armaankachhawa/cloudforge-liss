"""PyTorch datasets for synthetic and manually labelled LISS-IV patches."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class SyntheticLISSDataset(Dataset):
    """Load compressed patches created by ``create_synthetic_clouds.py``."""

    def __init__(
        self,
        manifest: str | Path,
        split: str,
        *,
        augment: bool = False,
        root: str | Path = ".",
    ) -> None:
        frame = pd.read_csv(manifest)
        if "split" not in frame:
            raise ValueError("Patch manifest must contain a scene-level split column")
        self.frame = frame.loc[frame["split"] == split].reset_index(drop=True)
        if self.frame.empty:
            raise ValueError(f"No samples assigned to split {split!r}")
        self.augment = augment
        self.root = Path(root)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        row = self.frame.iloc[index]
        path = Path(row["path"])
        if not path.is_absolute():
            path = self.root / path
        with np.load(path) as sample:
            cloudy = sample["cloudy"].astype(np.float32)
            target = sample["target"].astype(np.float32)
            labels = sample["mask"].astype(np.uint8)
        if self.augment:
            cloudy, target, labels = random_flip_rotate(cloudy, target, labels)
        cloud = (labels == 1).astype(np.float32)[None]
        shadow = (labels == 2).astype(np.float32)[None]
        inputs = np.concatenate([cloudy, cloud, shadow], axis=0)
        contaminated = (labels > 0).astype(np.float32)[None]
        return {
            "input": torch.from_numpy(np.ascontiguousarray(inputs)),
            "target": torch.from_numpy(np.ascontiguousarray(target)),
            "mask": torch.from_numpy(np.ascontiguousarray(contaminated)),
            "sample_id": str(row["sample_id"]),
        }


def random_flip_rotate(
    cloudy: np.ndarray, target: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply geometry-only augmentation identically to input, target and mask."""
    if random.random() < 0.5:
        cloudy, target, labels = cloudy[:, :, ::-1], target[:, :, ::-1], labels[:, ::-1]
    if random.random() < 0.5:
        cloudy, target, labels = cloudy[:, ::-1, :], target[:, ::-1, :], labels[::-1, :]
    rotations = random.randint(0, 3)
    if rotations:
        cloudy = np.rot90(cloudy, rotations, axes=(1, 2))
        target = np.rot90(target, rotations, axes=(1, 2))
        labels = np.rot90(labels, rotations, axes=(0, 1))
    return cloudy, target, labels


class MaskPatchDataset(Dataset):
    """Dataset for manually corrected three-class mask patches stored as NPZ."""

    def __init__(self, paths: list[Path], augment: bool = False) -> None:
        self.paths = paths
        self.augment = augment

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        with np.load(self.paths[index]) as sample:
            image = sample["image"].astype(np.float32)
            labels = sample["mask"].astype(np.uint8)
        if self.augment:
            image, _, labels = random_flip_rotate(image, image.copy(), labels)
        return torch.from_numpy(np.ascontiguousarray(image)), torch.from_numpy(
            np.ascontiguousarray(labels)
        ).long()
