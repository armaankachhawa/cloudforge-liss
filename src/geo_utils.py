"""Geospatial I/O, alignment, tiling, normalization and preview utilities."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from rasterio.windows import Window


@dataclass(frozen=True)
class RasterData:
    """A channel-first raster and the profile needed to reproduce it."""

    array: np.ndarray
    profile: dict
    descriptions: tuple[str | None, ...]
    tags: dict[str, str]


def read_raster(path: str | Path, dtype: str = "float32") -> RasterData:
    """Read every band without discarding geospatial metadata."""
    with rasterio.open(path) as src:
        array = src.read().astype(dtype, copy=False)
        profile = src.profile.copy()
        descriptions = tuple(src.descriptions)
        tags = src.tags().copy()
    return RasterData(array, profile, descriptions, tags)


def valid_data_mask(array: np.ndarray, nodata: float | int | None) -> np.ndarray:
    """Return a 2-D mask where all bands contain finite, non-nodata values."""
    valid = np.isfinite(array).all(axis=0)
    if nodata is not None:
        valid &= np.all(array != nodata, axis=0)
    return valid


def write_raster(
    path: str | Path,
    array: np.ndarray,
    reference_profile: dict,
    *,
    dtype: str | None = None,
    nodata: float | int | None = None,
    descriptions: tuple[str | None, ...] | None = None,
    tags: dict[str, str] | None = None,
) -> Path:
    """Write a channel-first GeoTIFF on the reference grid."""
    if array.ndim == 2:
        array = array[None, ...]
    if array.ndim != 3:
        raise ValueError(f"Expected (bands, height, width), received {array.shape}")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    target_dtype = dtype or str(array.dtype)
    profile = reference_profile.copy()
    profile.update(
        driver="GTiff",
        count=array.shape[0],
        height=array.shape[1],
        width=array.shape[2],
        dtype=target_dtype,
        nodata=nodata,
        compress="deflate",
        predictor=2 if np.issubdtype(np.dtype(target_dtype), np.integer) else 3,
        tiled=True,
        BIGTIFF="IF_SAFER",
    )
    output_array = array
    target_type = np.dtype(target_dtype)
    if np.issubdtype(target_type, np.integer):
        limits = np.iinfo(target_type)
        output_array = np.rint(np.clip(array, limits.min, limits.max))
    with rasterio.open(output, "w", **profile) as dst:
        dst.write(output_array.astype(target_dtype, copy=False))
        if descriptions:
            for index, description in enumerate(descriptions[: array.shape[0]], start=1):
                if description:
                    dst.set_band_description(index, description)
        if tags:
            dst.update_tags(**tags)
    return output


def align_to_reference(
    source_path: str | Path,
    reference_path: str | Path,
    output_path: str | Path,
    *,
    categorical: bool = False,
) -> Path:
    """Reproject/resample a raster exactly onto a reference raster grid."""
    with rasterio.open(reference_path) as ref, rasterio.open(source_path) as src:
        destination = np.zeros((src.count, ref.height, ref.width), dtype=src.dtypes[0])
        method = Resampling.nearest if categorical else Resampling.bilinear
        for band_index in range(1, src.count + 1):
            reproject(
                source=rasterio.band(src, band_index),
                destination=destination[band_index - 1],
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=src.nodata,
                dst_transform=ref.transform,
                dst_crs=ref.crs,
                dst_nodata=src.nodata,
                resampling=method,
            )
        profile = ref.profile.copy()
        profile.update(count=src.count, dtype=src.dtypes[0], nodata=src.nodata)
        descriptions = tuple(src.descriptions)
        tags = src.tags().copy()
    return write_raster(
        output_path,
        destination,
        profile,
        dtype=str(destination.dtype),
        nodata=profile.get("nodata"),
        descriptions=descriptions,
        tags=tags,
    )


def robust_band_stats(array: np.ndarray, valid: np.ndarray | None = None) -> dict:
    """Compute per-band percentiles suitable for stable satellite normalization."""
    if valid is None:
        valid = np.isfinite(array).all(axis=0)
    stats: list[dict[str, float]] = []
    for band in array:
        pixels = band[valid & np.isfinite(band)]
        if pixels.size == 0:
            raise ValueError("Cannot compute statistics from an empty valid region")
        p02, p50, p98 = np.percentile(pixels, [2, 50, 98])
        mean = float(pixels.mean())
        std = float(pixels.std())
        stats.append(
            {
                "p02": float(p02),
                "p50": float(p50),
                "p98": float(p98),
                "mean": mean,
                "std": max(std, 1e-6),
            }
        )
    return {"bands": stats}


def normalize_percentile(array: np.ndarray, stats: dict | None = None) -> tuple[np.ndarray, dict]:
    """Scale each band to [0, 1] using training-compatible 2–98% bounds."""
    stats = stats or robust_band_stats(array)
    normalized = np.empty_like(array, dtype=np.float32)
    for index, band_stats in enumerate(stats["bands"]):
        low, high = band_stats["p02"], band_stats["p98"]
        normalized[index] = np.clip((array[index] - low) / max(high - low, 1e-6), 0.0, 1.0)
    return normalized, stats


def denormalize_percentile(array: np.ndarray, stats: dict) -> np.ndarray:
    """Invert :func:`normalize_percentile`."""
    restored = np.empty_like(array, dtype=np.float32)
    for index, band_stats in enumerate(stats["bands"]):
        low, high = band_stats["p02"], band_stats["p98"]
        restored[index] = array[index] * (high - low) + low
    return restored


def iter_windows(height: int, width: int, size: int, overlap: int = 32) -> Iterator[Window]:
    """Yield full-coverage inference windows, including right/bottom edges."""
    if not 0 <= overlap < size:
        raise ValueError("overlap must satisfy 0 <= overlap < size")
    stride = size - overlap
    rows = list(range(0, max(height - size, 0) + 1, stride))
    cols = list(range(0, max(width - size, 0) + 1, stride))
    final_row, final_col = max(height - size, 0), max(width - size, 0)
    if not rows or rows[-1] != final_row:
        rows.append(final_row)
    if not cols or cols[-1] != final_col:
        cols.append(final_col)
    for row in rows:
        for col in cols:
            yield Window(col, row, min(size, width - col), min(size, height - row))


def blend_weight(height: int, width: int, minimum: float = 0.05) -> np.ndarray:
    """Cosine-like center weighting for seam-free overlapping tiles."""
    y = np.hanning(max(height, 3))[:height]
    x = np.hanning(max(width, 3))[:width]
    weight = np.outer(y, x)
    if weight.max() > 0:
        weight /= weight.max()
    return np.maximum(weight, minimum).astype(np.float32)


def compose_preserving_clear(
    original: np.ndarray, generated: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    """Replace only cloud/shadow pixels; clear measurements remain bit-identical."""
    if original.shape != generated.shape:
        raise ValueError("original and generated arrays must have identical shapes")
    contaminated = mask > 0
    result = original.copy()
    result[:, contaminated] = generated[:, contaminated]
    return result


def save_preview(path: str | Path, cloudy: np.ndarray, reconstructed: np.ndarray) -> Path:
    """Save a side-by-side false-colour preview without altering raster products."""

    def display(array: np.ndarray) -> np.ndarray:
        selected = array[[2, 1, 0]] if array.shape[0] >= 3 else np.repeat(array[:1], 3, axis=0)
        scaled, _ = normalize_percentile(selected)
        return np.moveaxis((scaled * 255).astype(np.uint8), 0, -1)

    left, right = display(cloudy), display(reconstructed)
    canvas = np.concatenate([left, right], axis=1)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    return output


def save_json(path: str | Path, payload: dict) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output
