"""Create supervised reconstruction patches using real cloud texture/masks."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np

from src.bhoonidhi_zip import product_shape, read_liss_window
from src.create_masks import CLOUD, SHADOW, rule_based_mask
from src.geo_utils import normalize_percentile, read_raster


def corrupt_clear_patch(
    clear: np.ndarray,
    cloud_texture: np.ndarray,
    labels: np.ndarray,
    *,
    shadow_strength: float = 0.45,
) -> np.ndarray:
    """Apply real cloud pixels and textured shadow attenuation to a clear target."""
    if clear.shape != cloud_texture.shape or clear.shape[1:] != labels.shape:
        raise ValueError("clear, cloud_texture and labels must share spatial dimensions")
    synthetic = clear.copy()
    cloud = labels == CLOUD
    shadow = labels == SHADOW
    synthetic[:, cloud] = cloud_texture[:, cloud]
    if np.any(shadow):
        texture = 0.85 + 0.15 * cloud_texture.mean(axis=0)
        factor = np.clip(shadow_strength * texture, 0.2, 0.7)
        synthetic[:, shadow] = clear[:, shadow] * factor[shadow]
    return np.clip(synthetic, 0.0, 1.0)


def sample_patch(array: np.ndarray, size: int, rng: random.Random) -> np.ndarray:
    if array.shape[1] < size or array.shape[2] < size:
        raise ValueError(f"Raster {array.shape[1:]} is smaller than patch size {size}")
    row = rng.randint(0, array.shape[1] - size)
    col = rng.randint(0, array.shape[2] - size)
    return array[:, row : row + size, col : col + size]


def load_scene(path: Path) -> tuple[Path, np.ndarray | None]:
    """Load ordinary GeoTIFF scenes eagerly; read ZIP products lazily."""
    if path.suffix.lower() == ".zip":
        return path, None
    return path, normalize_percentile(read_raster(path).array)[0]


def sample_scene_patch(
    scene: tuple[Path, np.ndarray | None], size: int, rng: random.Random
) -> np.ndarray:
    path, cached = scene
    if cached is not None:
        return sample_patch(cached, size, rng)
    height, width = product_shape(path)
    if height < size or width < size:
        raise ValueError(f"Product {path.name} is smaller than patch size {size}")
    row = rng.randint(0, height - size)
    col = rng.randint(0, width - size)
    patch = read_liss_window(path, row, col, size)
    return normalize_percentile(patch)[0]


def generate_dataset(
    clear_paths: list[Path],
    cloudy_paths: list[Path],
    output_dir: Path,
    count: int,
    patch_size: int,
    seed: int,
) -> Path:
    if not clear_paths or not cloudy_paths:
        raise ValueError("At least one clear and one cloudy raster are required")
    rng = random.Random(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    clear_cache = [load_scene(path) for path in clear_paths]
    cloudy_cache = [load_scene(path) for path in cloudy_paths]
    manifest_path = output_dir.parent / "manifests" / "patch_manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str | int | float]] = []
    for index in range(count):
        clear_path, clear_scene = rng.choice(clear_cache)
        cloudy_path, cloudy_scene = rng.choice(cloudy_cache)
        target = sample_scene_patch((clear_path, clear_scene), patch_size, rng)
        texture = sample_scene_patch((cloudy_path, cloudy_scene), patch_size, rng)
        labels = rule_based_mask(texture)
        if not np.any(labels > 0):
            continue
        synthetic = corrupt_clear_patch(target, texture, labels)
        sample_path = output_dir / f"sample_{index:06d}.npz"
        np.savez_compressed(
            sample_path,
            cloudy=synthetic.astype(np.float32),
            target=target.astype(np.float32),
            mask=labels.astype(np.uint8),
        )
        rows.append(
            {
                "sample_id": sample_path.stem,
                "path": sample_path.as_posix(),
                "clear_scene": clear_path.stem,
                "cloud_source": cloudy_path.stem,
                "masked_fraction": float(np.mean(labels > 0)),
                "split": "unassigned",
            }
        )
    if rows:
        assign_scene_splits(rows, seed)
    with manifest_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["sample_id", "path"])
        writer.writeheader()
        writer.writerows(rows)
    return manifest_path


def assign_scene_splits(rows: list[dict[str, str | int | float]], seed: int) -> None:
    """Assign whole clear scenes to splits to prevent spatial/temporal leakage."""
    scenes = sorted({str(row["clear_scene"]) for row in rows})
    if len(scenes) < 2:
        raise ValueError(
            "At least two independent clear scenes are required for a train/validation split"
        )
    random.Random(seed).shuffle(scenes)
    train_count = min(max(round(len(scenes) * 0.70), 1), len(scenes) - 1)
    remaining = len(scenes) - train_count
    validation_count = max(1, remaining // 2)
    assignment = {scene: "train" for scene in scenes[:train_count]}
    assignment.update(
        {scene: "validation" for scene in scenes[train_count : train_count + validation_count]}
    )
    assignment.update({scene: "test" for scene in scenes[train_count + validation_count :]})
    for row in rows:
        row["split"] = assignment[str(row["clear_scene"])]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic cloudy LISS-IV training patches"
    )
    parser.add_argument("--clear-dir", type=Path, default=Path("data/raw/liss_clear"))
    parser.add_argument("--cloudy-dir", type=Path, default=Path("data/raw/liss_cloudy"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/patches"))
    parser.add_argument("--count", type=int, default=3000)
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    clear = (
        sorted(args.clear_dir.glob("*.tif"))
        + sorted(args.clear_dir.glob("*.tiff"))
        + sorted(args.clear_dir.glob("*.zip"))
    )
    cloudy = (
        sorted(args.cloudy_dir.glob("*.tif"))
        + sorted(args.cloudy_dir.glob("*.tiff"))
        + sorted(args.cloudy_dir.glob("*.zip"))
    )
    manifest = generate_dataset(
        clear, cloudy, args.output_dir, args.count, args.patch_size, args.seed
    )
    print(f"Synthetic patch manifest written to {manifest}")


if __name__ == "__main__":
    main()
