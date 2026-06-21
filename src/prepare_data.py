"""Audit raw products and align auxiliary rasters to LISS-IV reference grids."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd
import rasterio

from src.geo_utils import align_to_reference

MANIFEST_COLUMNS = [
    "scene_id",
    "aoi",
    "date",
    "type",
    "cloud_percent",
    "crs",
    "resolution",
    "bands",
    "dtype",
    "nodata",
    "width",
    "height",
    "path",
    "sha256",
]


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def audit_raster(path: Path, root: Path) -> dict:
    with rasterio.open(path) as src:
        return {
            "scene_id": path.stem,
            "aoi": "",
            "date": "",
            "type": "clear"
            if "liss_clear" in path.parts
            else "cloudy"
            if "liss_cloudy" in path.parts
            else "auxiliary",
            "cloud_percent": "",
            "crs": src.crs.to_string() if src.crs else "",
            "resolution": abs(src.res[0]),
            "bands": src.count,
            "dtype": ";".join(src.dtypes),
            "nodata": src.nodata,
            "width": src.width,
            "height": src.height,
            "path": path.relative_to(root).as_posix(),
            "sha256": sha256(path),
        }


def build_manifest(raw_dir: Path, output: Path, root: Path) -> pd.DataFrame:
    rasters = sorted({*raw_dir.rglob("*.tif"), *raw_dir.rglob("*.tiff")})
    frame = pd.DataFrame((audit_raster(path, root) for path in rasters), columns=MANIFEST_COLUMNS)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit or align CloudForge-LISS data")
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit = subparsers.add_parser("audit")
    audit.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    audit.add_argument(
        "--output", type=Path, default=Path("data/processed/manifests/scene_manifest.csv")
    )
    align = subparsers.add_parser("align")
    align.add_argument("--source", required=True, type=Path)
    align.add_argument("--reference", required=True, type=Path)
    align.add_argument("--output", required=True, type=Path)
    align.add_argument("--categorical", action="store_true")
    args = parser.parse_args()
    if args.command == "audit":
        frame = build_manifest(args.raw_dir, args.output, Path.cwd())
        print(f"Audited {len(frame)} rasters into {args.output}")
    else:
        align_to_reference(args.source, args.reference, args.output, categorical=args.categorical)
        print(f"Aligned raster written to {args.output}")


if __name__ == "__main__":
    main()
