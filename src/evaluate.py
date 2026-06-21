"""Masked reference metrics for synthetic-cloud validation scenes."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from skimage.metrics import structural_similarity

from src.geo_utils import read_raster


def masked_metrics(
    prediction: np.ndarray, target: np.ndarray, mask: np.ndarray
) -> dict[str, float]:
    contaminated = mask > 0
    if not np.any(contaminated):
        raise ValueError("Evaluation mask contains no reconstructed pixels")
    difference = prediction[:, contaminated].astype(np.float64) - target[:, contaminated].astype(
        np.float64
    )
    mae = float(np.mean(np.abs(difference)))
    rmse = float(np.sqrt(np.mean(difference**2)))
    values = target[:, contaminated]
    data_range = float(values.max() - values.min())
    psnr = float("inf") if rmse == 0 else 20 * math.log10(max(data_range, 1e-6) / rmse)
    pred_vectors = prediction[:, contaminated].T.astype(np.float64)
    target_vectors = target[:, contaminated].T.astype(np.float64)
    cosine = np.sum(pred_vectors * target_vectors, axis=1) / (
        np.linalg.norm(pred_vectors, axis=1) * np.linalg.norm(target_vectors, axis=1) + 1e-12
    )
    sam = float(np.mean(np.arccos(np.clip(cosine, -1, 1))))
    # SSIM needs neighbourhoods, so calculate the full map and average only the mask.
    ssim_maps = []
    for pred_band, target_band in zip(prediction, target, strict=True):
        band_range = float(target_band.max() - target_band.min())
        _, score_map = structural_similarity(
            target_band.astype(np.float64),
            pred_band.astype(np.float64),
            data_range=max(band_range, 1e-6),
            full=True,
        )
        ssim_maps.append(score_map)
    ssim = float(np.mean(np.stack(ssim_maps)[:, contaminated]))
    metrics = {"mae": mae, "rmse": rmse, "psnr_db": psnr, "ssim": ssim, "sam_radians": sam}
    if prediction.shape[0] >= 3:
        target_ndvi = (target[2] - target[1]) / (target[2] + target[1] + 1e-6)
        prediction_ndvi = (prediction[2] - prediction[1]) / (prediction[2] + prediction[1] + 1e-6)
        metrics["ndvi_mae"] = float(
            np.mean(np.abs(target_ndvi[contaminated] - prediction_ndvi[contaminated]))
        )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a reconstructed GeoTIFF inside its cloud mask"
    )
    parser.add_argument("--prediction", required=True, type=Path)
    parser.add_argument("--target", required=True, type=Path)
    parser.add_argument("--mask", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    prediction = read_raster(args.prediction).array
    target = read_raster(args.target).array
    mask = read_raster(args.mask).array[0]
    if prediction.shape != target.shape or prediction.shape[1:] != mask.shape:
        raise ValueError("Prediction, target and mask must use the same grid")
    result = masked_metrics(prediction, target, mask)
    payload = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
