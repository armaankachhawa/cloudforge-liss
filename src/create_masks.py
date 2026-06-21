"""Rule-based masks and label utilities for the cloud-mask training stage."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from src.geo_utils import normalize_percentile, read_raster, valid_data_mask, write_raster

CLEAR, CLOUD, SHADOW = 0, 1, 2


def rule_based_mask(
    image: np.ndarray,
    *,
    cloud_quantile: float = 0.88,
    shadow_quantile: float = 0.18,
    dilation_pixels: int = 15,
) -> np.ndarray:
    """Create conservative rough labels for subsequent manual correction.

    LISS-IV lacks a blue/SWIR band, so this is deliberately a seed mask rather
    than a claim of production-quality cloud detection.
    """
    normalized, _ = normalize_percentile(image)
    brightness = normalized.mean(axis=0)
    spectral_flatness = normalized.std(axis=0)
    valid = np.isfinite(image).all(axis=0)
    bright_threshold = float(np.quantile(brightness[valid], cloud_quantile))
    cloud = valid & (brightness >= bright_threshold) & (spectral_flatness < 0.24)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cloud = cv2.morphologyEx(cloud.astype(np.uint8), cv2.MORPH_CLOSE, kernel) > 0
    neighbourhood_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * dilation_pixels + 1, 2 * dilation_pixels + 1)
    )
    near_cloud = cv2.dilate(cloud.astype(np.uint8), neighbourhood_kernel) > 0

    dark_threshold = float(np.quantile(brightness[valid], shadow_quantile))
    # Band convention is confirmed per product. For common LISS-IV ordering,
    # green is index 0 and NIR is index 2; NDWI helps avoid labeling water.
    if image.shape[0] >= 3:
        green, nir = normalized[0], normalized[2]
        ndwi = (green - nir) / (green + nir + 1e-6)
        likely_water = ndwi > 0.15
    else:
        likely_water = np.zeros_like(brightness, dtype=bool)
    shadow = valid & near_cloud & ~cloud & ~likely_water & (brightness <= dark_threshold)
    shadow = cv2.morphologyEx(shadow.astype(np.uint8), cv2.MORPH_OPEN, kernel) > 0

    labels = np.full(brightness.shape, CLEAR, dtype=np.uint8)
    labels[shadow] = SHADOW
    labels[cloud] = CLOUD
    return labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a rough LISS-IV cloud/shadow mask")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    raster = read_raster(args.input)
    labels = rule_based_mask(raster.array)
    valid = valid_data_mask(raster.array, raster.profile.get("nodata"))
    labels[~valid] = 255
    write_raster(args.output, labels, raster.profile, dtype="uint8", nodata=255)
    print(f"Wrote rough mask to {args.output}")


if __name__ == "__main__":
    main()
