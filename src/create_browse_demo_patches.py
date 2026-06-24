"""Create a tiny demo patch set from product browse JPEGs.

This is for end-to-end smoke testing on low-disk CPU machines. It proves the
CloudForge-LISS training/inference plumbing but is not a replacement for final
training on the full BAND2/BAND3/BAND4 GeoTIFF products.
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import cv2
import numpy as np

from src.bhoonidhi_zip import read_browse_jpeg
from src.create_masks import CLOUD, SHADOW, rule_based_mask
from src.create_synthetic_clouds import corrupt_clear_patch


def read_browse_array(zip_path: Path) -> np.ndarray:
    data = read_browse_jpeg(zip_path)
    if data is None:
        raise ValueError(f"No browse JPEG found in {zip_path.name}")
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not decode browse JPEG from {zip_path.name}")
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    valid = rgb.sum(axis=2) > 0.03
    rows, cols = np.where(valid)
    if rows.size == 0:
        raise ValueError(f"Browse JPEG {zip_path.name} has no valid scene pixels")
    cropped = rgb[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]
    return np.moveaxis(cropped, -1, 0)


def sample_valid_patch(array: np.ndarray, size: int, rng: random.Random) -> np.ndarray:
    if array.shape[1] < size or array.shape[2] < size:
        array = np.stack(
            [
                cv2.resize(band, (max(size, array.shape[2]), max(size, array.shape[1])))
                for band in array
            ],
            axis=0,
        )
    for _ in range(100):
        row = rng.randint(0, array.shape[1] - size)
        col = rng.randint(0, array.shape[2] - size)
        patch = array[:, row : row + size, col : col + size]
        if patch.mean() > 0.02:
            return patch
    return array[:, :size, :size]


def create_demo_patches(
    clear_dir: Path,
    cloudy_dir: Path,
    output_dir: Path,
    count: int,
    patch_size: int,
    seed: int,
) -> Path:
    rng = random.Random(seed)
    clear = [(path, read_browse_array(path)) for path in sorted(clear_dir.glob("*.zip"))]
    cloudy = [(path, read_browse_array(path)) for path in sorted(cloudy_dir.glob("*.zip"))]
    if len(clear) < 2 or not cloudy:
        raise ValueError("Need at least two clear ZIPs and one cloudy ZIP for the demo split")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = output_dir.parent / "manifests" / "patch_manifest.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str | float]] = []
    for index in range(count):
        clear_path, clear_array = clear[index % len(clear)]
        cloudy_path, cloudy_array = rng.choice(cloudy)
        target = sample_valid_patch(clear_array, patch_size, rng)
        texture = sample_valid_patch(cloudy_array, patch_size, rng)
        labels = rule_based_mask(texture)
        if np.mean(labels > 0) < 0.05:
            labels[: patch_size // 2, : patch_size // 2] = CLOUD
            labels[patch_size // 2 :, : patch_size // 2] = SHADOW
        synthetic = corrupt_clear_patch(target, texture, labels)
        sample_path = output_dir / f"browse_demo_{index:04d}.npz"
        np.savez_compressed(
            sample_path,
            cloudy=synthetic.astype(np.float32),
            target=target.astype(np.float32),
            mask=labels.astype(np.uint8),
        )
        split = "train" if clear_path == clear[0][0] else "validation"
        rows.append(
            {
                "sample_id": sample_path.stem,
                "path": sample_path.as_posix(),
                "clear_scene": clear_path.stem,
                "cloud_source": cloudy_path.stem,
                "masked_fraction": float(np.mean(labels > 0)),
                "split": split,
            }
        )
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Create browse-image demo patches")
    parser.add_argument("--clear-dir", type=Path, default=Path("data/raw/liss_clear"))
    parser.add_argument("--cloudy-dir", type=Path, default=Path("data/raw/liss_cloudy"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/patches"))
    parser.add_argument("--count", type=int, default=40)
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(
        create_demo_patches(
            args.clear_dir,
            args.cloudy_dir,
            args.output_dir,
            args.count,
            args.patch_size,
            args.seed,
        )
    )


if __name__ == "__main__":
    main()
