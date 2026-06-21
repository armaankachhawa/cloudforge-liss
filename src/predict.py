"""Operational tiled GeoTIFF inference with clear-pixel preservation."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from src.create_masks import rule_based_mask
from src.geo_utils import (
    blend_weight,
    compose_preserving_clear,
    denormalize_percentile,
    iter_windows,
    normalize_percentile,
    read_raster,
    save_json,
    save_preview,
    write_raster,
)
from src.models import CloudMaskUNet, build_model
from src.train import resolve_device


def load_reconstruction_model(
    checkpoint_path: Path, device: torch.device
) -> tuple[torch.nn.Module, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model_config = config.get("model", {})
    model = build_model(
        model_config.get("name", "attention_resunet"),
        int(config["data"]["input_channels"]),
        int(config["data"]["output_channels"]),
        int(model_config.get("base_channels", 32)),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    return model, config


def _pad_tile(tile: np.ndarray, size: int) -> tuple[np.ndarray, tuple[int, int]]:
    height, width = tile.shape[-2:]
    pad_height, pad_width = size - height, size - width
    if pad_height or pad_width:
        tile = np.pad(tile, ((0, 0), (0, pad_height), (0, pad_width)), mode="reflect")
    return tile, (height, width)


def tiled_prediction(
    model: torch.nn.Module,
    normalized: np.ndarray,
    labels: np.ndarray,
    device: torch.device,
    *,
    tile_size: int = 256,
    overlap: int = 32,
    test_time_augmentation: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    cloud = (labels == 1).astype(np.float32)[None]
    shadow = (labels == 2).astype(np.float32)[None]
    model_input = np.concatenate([normalized, cloud, shadow], axis=0)
    output = np.zeros_like(normalized, dtype=np.float32)
    variance = np.zeros_like(normalized, dtype=np.float32)
    weights = np.zeros(labels.shape, dtype=np.float32)
    with torch.inference_mode():
        for window in iter_windows(labels.shape[0], labels.shape[1], tile_size, overlap):
            row, col = int(window.row_off), int(window.col_off)
            height, width = int(window.height), int(window.width)
            tile = model_input[:, row : row + height, col : col + width]
            tile, original_shape = _pad_tile(tile, tile_size)
            tensor = torch.from_numpy(tile[None]).to(device)
            prediction = model(tensor)
            if test_time_augmentation:
                flipped = torch.flip(tensor, dims=[3])
                flipped_prediction = torch.flip(model(flipped), dims=[3])
                tile_variance = torch.var(
                    torch.stack([prediction, flipped_prediction]), dim=0, unbiased=False
                )
                prediction = (prediction + flipped_prediction) / 2
            else:
                tile_variance = torch.zeros_like(prediction)
            prediction_np = prediction[0, :, : original_shape[0], : original_shape[1]].cpu().numpy()
            variance_np = (
                tile_variance[0, :, : original_shape[0], : original_shape[1]].cpu().numpy()
            )
            weight = blend_weight(height, width)
            output[:, row : row + height, col : col + width] += prediction_np * weight
            variance[:, row : row + height, col : col + width] += variance_np * weight
            weights[row : row + height, col : col + width] += weight
    output /= np.maximum(weights[None], 1e-6)
    variance /= np.maximum(weights[None], 1e-6)
    return output, variance.mean(axis=0)


def predict_mask_tiled(
    normalized: np.ndarray,
    checkpoint_path: Path,
    device: torch.device,
    *,
    tile_size: int = 256,
    overlap: int = 32,
) -> np.ndarray:
    """Run the trained three-class mask model over an arbitrary-size scene."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = CloudMaskUNet(in_channels=normalized.shape[0], classes=3)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    probabilities = np.zeros((3, *normalized.shape[1:]), dtype=np.float32)
    weights = np.zeros(normalized.shape[1:], dtype=np.float32)
    with torch.inference_mode():
        for window in iter_windows(*normalized.shape[1:], tile_size, overlap):
            row, col = int(window.row_off), int(window.col_off)
            height, width = int(window.height), int(window.width)
            tile = normalized[:, row : row + height, col : col + width]
            tile, original_shape = _pad_tile(tile, tile_size)
            tensor = torch.from_numpy(tile[None]).to(device)
            prediction = (
                torch.softmax(model(tensor), dim=1)[0, :, : original_shape[0], : original_shape[1]]
                .cpu()
                .numpy()
            )
            weight = blend_weight(height, width)
            probabilities[:, row : row + height, col : col + width] += prediction * weight
            weights[row : row + height, col : col + width] += weight
    probabilities /= np.maximum(weights[None], 1e-6)
    return probabilities.argmax(axis=0).astype(np.uint8)


def confidence_map(labels: np.ndarray, variance: np.ndarray, tile_size: int) -> np.ndarray:
    contaminated = (labels > 0).astype(np.uint8)
    distance = cv2.distanceTransform(contaminated, cv2.DIST_L2, 5)
    evidence = np.exp(-distance / max(tile_size / 3, 1))
    agreement = np.exp(-20 * np.clip(variance, 0, None))
    confidence = np.clip(evidence * agreement, 0, 1).astype(np.float32)
    component_count, components, component_stats, _ = cv2.connectedComponentsWithStats(
        contaminated, connectivity=8
    )
    for component in range(1, component_count):
        area = component_stats[component, cv2.CC_STAT_AREA]
        area_penalty = 1.0 / (1.0 + area / max(2 * tile_size**2, 1))
        confidence[components == component] *= area_penalty
    confidence[labels == 0] = 1.0
    return confidence


def predict_scene(
    input_path: Path,
    checkpoint_path: Path,
    output_dir: Path,
    *,
    mask_path: Path | None = None,
    mask_checkpoint_path: Path | None = None,
    device_name: str = "auto",
    tile_size: int = 256,
    overlap: int = 32,
) -> dict[str, Path]:
    raster = read_raster(input_path)
    image = raster.array
    normalized, stats = normalize_percentile(image)
    device = resolve_device(device_name)
    if mask_path:
        labels = read_raster(mask_path, dtype="uint8").array[0]
        if labels.shape != image.shape[1:]:
            raise ValueError("Mask must be aligned to the input LISS-IV grid")
        mask_source = "provided"
    elif mask_checkpoint_path:
        labels = predict_mask_tiled(
            normalized,
            mask_checkpoint_path,
            device,
            tile_size=tile_size,
            overlap=overlap,
        )
        mask_source = "cloud_mask_unet"
    else:
        labels = rule_based_mask(image)
        mask_source = "rule_based_seed"
    model, _ = load_reconstruction_model(checkpoint_path, device)
    generated_normalized, variance = tiled_prediction(
        model, normalized, labels, device, tile_size=tile_size, overlap=overlap
    )
    generated = denormalize_percentile(generated_normalized, stats)
    reconstructed = compose_preserving_clear(image, generated, labels)
    confidence = confidence_map(labels, variance, tile_size)
    output_dir.mkdir(parents=True, exist_ok=True)
    tags = raster.tags | {
        "PROCESSOR": "CloudForge-LISS",
        "AI_MODIFIED_PIXELS": str(int(np.sum(labels > 0))),
        "CLEAR_PIXELS_PRESERVED": "true",
    }
    paths = {
        "reconstructed": write_raster(
            output_dir / "reconstructed_liss.tif",
            reconstructed,
            raster.profile,
            dtype=raster.profile["dtype"],
            nodata=raster.profile.get("nodata"),
            descriptions=raster.descriptions,
            tags=tags,
        ),
        "mask": write_raster(
            output_dir / "cloud_shadow_mask.tif", labels, raster.profile, dtype="uint8", nodata=255
        ),
        "confidence": write_raster(
            output_dir / "confidence_map.tif",
            confidence,
            raster.profile,
            dtype="float32",
            nodata=-1,
        ),
    }
    paths["preview"] = save_preview(output_dir / "preview_before_after.png", image, reconstructed)
    contaminated = labels > 0
    report = {
        "input": str(input_path),
        "checkpoint": str(checkpoint_path),
        "device": str(device),
        "cloud_shadow_fraction": float(contaminated.mean()),
        "mean_reconstruction_confidence": float(confidence[contaminated].mean())
        if contaminated.any()
        else 1.0,
        "clear_pixels_preserved": True,
        "mask_source": mask_source,
        "warning": "Reconstructed pixels are model estimates, not guaranteed observations.",
    }
    paths["report"] = save_json(output_dir / "quality_report.json", report)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="CloudForge-LISS operational reconstruction")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--mask", type=Path)
    parser.add_argument("--mask-checkpoint", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--overlap", type=int, default=32)
    args = parser.parse_args()
    products = predict_scene(
        args.input,
        args.checkpoint,
        args.output,
        mask_path=args.mask,
        mask_checkpoint_path=args.mask_checkpoint,
        device_name=args.device,
        tile_size=args.tile_size,
        overlap=args.overlap,
    )
    for name, path in products.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
