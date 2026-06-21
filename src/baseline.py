"""No-reconstruction baseline: preserve clear pixels and set contamination to nodata."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.geo_utils import read_raster, write_raster


def masked_blank(image: np.ndarray, labels: np.ndarray, fill_value: float | int = 0) -> np.ndarray:
    result = image.copy()
    result[:, labels > 0] = fill_value
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the no-AI cloud-masking baseline")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--mask", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--fill", type=float, default=0)
    args = parser.parse_args()
    raster = read_raster(args.input)
    labels = read_raster(args.mask, dtype="uint8").array[0]
    if labels.shape != raster.array.shape[1:]:
        raise ValueError("Mask and input must use the same grid")
    output = masked_blank(raster.array, labels, args.fill)
    write_raster(
        args.output,
        output,
        raster.profile,
        dtype=raster.profile["dtype"],
        nodata=raster.profile.get("nodata"),
        descriptions=raster.descriptions,
        tags=raster.tags | {"PROCESSOR": "CloudForge-LISS no-AI baseline"},
    )


if __name__ == "__main__":
    main()
