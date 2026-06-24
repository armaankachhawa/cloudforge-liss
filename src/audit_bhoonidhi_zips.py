"""Build a manifest and browse previews from downloaded Bhoonidhi ZIP products."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from src.bhoonidhi_zip import (
    estimate_browse_cloudiness,
    read_browse_jpeg,
    read_product_meta,
    write_browse_jpeg,
)

FIELDS = [
    "scene_id",
    "role",
    "satellite",
    "sensor",
    "path",
    "row",
    "date",
    "bands",
    "band_numbers",
    "product_type",
    "image_format",
    "processing_level",
    "input_resolution",
    "output_resolution",
    "projection",
    "zone",
    "center_lat",
    "center_lon",
    "cloud_percent_meta",
    "browse_bright_fraction",
    "zip_path",
]


def role_from_path(path: Path) -> str:
    text = path.as_posix().lower()
    if "liss_clear" in text:
        return "clear"
    if "liss_cloudy" in text:
        return "cloudy"
    return "unknown"


def record(zip_path: Path, previews_dir: Path) -> dict[str, str | float | None]:
    meta = read_product_meta(zip_path)
    scene_id = meta.get("OTSProductID", zip_path.stem)
    jpeg = read_browse_jpeg(zip_path)
    write_browse_jpeg(zip_path, previews_dir / f"{scene_id}.jpg")
    return {
        "scene_id": scene_id,
        "role": role_from_path(zip_path),
        "satellite": meta.get("SatID", ""),
        "sensor": meta.get("Sensor", ""),
        "path": meta.get("Path", ""),
        "row": meta.get("Row", ""),
        "date": meta.get("DateOfPass", ""),
        "bands": meta.get("NoOfBands", ""),
        "band_numbers": meta.get("BandNumbers", ""),
        "product_type": meta.get("ProdType", ""),
        "image_format": meta.get("ImageFormat", ""),
        "processing_level": meta.get("ProcessingLevel", ""),
        "input_resolution": meta.get("InputResolutionAlong", ""),
        "output_resolution": meta.get("OutputResolutionAlong", ""),
        "projection": meta.get("MapProjection", ""),
        "zone": meta.get("ZoneNo", ""),
        "center_lat": meta.get("SceneCenterLat", ""),
        "center_lon": meta.get("SceneCenterLon", ""),
        "cloud_percent_meta": meta.get("CloudPercent", ""),
        "browse_bright_fraction": estimate_browse_cloudiness(jpeg),
        "zip_path": zip_path.as_posix(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit downloaded Bhoonidhi LISS-IV ZIPs")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/manifests/bhoonidhi_zip_manifest.csv"),
    )
    parser.add_argument("--previews-dir", type=Path, default=Path("outputs/browse_previews"))
    args = parser.parse_args()
    zips = sorted(args.raw_dir.rglob("*.zip"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.previews_dir.mkdir(parents=True, exist_ok=True)
    rows = [record(path, args.previews_dir) for path in zips]
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Audited {len(rows)} ZIP products into {args.output}")


if __name__ == "__main__":
    main()
