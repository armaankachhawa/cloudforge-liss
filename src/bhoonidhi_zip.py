"""Read Bhoonidhi LISS-IV ZIP products without extracting full scenes."""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window


def _vsizip_path(zip_path: Path, member: str) -> str:
    safe_zip = zip_path.resolve().as_posix()
    return f"/vsizip/{safe_zip}/{member}"


def product_members(zip_path: str | Path) -> dict[str, str]:
    """Return important member paths from a Bhoonidhi product ZIP."""
    zip_path = Path(zip_path)
    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
    members: dict[str, str] = {}
    for name in names:
        upper = name.upper()
        if upper.endswith("BAND2.TIF"):
            members["BAND2"] = name
        elif upper.endswith("BAND3.TIF"):
            members["BAND3"] = name
        elif upper.endswith("BAND4.TIF"):
            members["BAND4"] = name
        elif upper.endswith(".META"):
            members["META"] = name
        elif upper.endswith(".JPG") or upper.endswith(".JPEG"):
            members["JPG"] = name
    missing = {"BAND2", "BAND3", "BAND4", "META"} - members.keys()
    if missing:
        raise ValueError(f"{zip_path.name} is missing expected members: {sorted(missing)}")
    return members


def read_product_meta(zip_path: str | Path) -> dict[str, str]:
    """Parse the product .meta text file into a dictionary."""
    zip_path = Path(zip_path)
    members = product_members(zip_path)
    with zipfile.ZipFile(zip_path) as archive:
        raw = archive.read(members["META"])
    text = raw.decode("utf-8", errors="replace")
    parsed: dict[str, str] = {}
    for line in text.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            parsed[key.strip()] = value.strip()
    parsed["zip_path"] = zip_path.as_posix()
    parsed["zip_name"] = zip_path.name
    return parsed


def open_band(zip_path: str | Path, band_name: str):
    """Open BAND2/BAND3/BAND4 from inside a ZIP through GDAL's /vsizip."""
    zip_path = Path(zip_path)
    members = product_members(zip_path)
    return rasterio.open(_vsizip_path(zip_path, members[band_name]))


def product_shape(zip_path: str | Path) -> tuple[int, int]:
    with open_band(zip_path, "BAND2") as src:
        return src.height, src.width


def read_liss_window(zip_path: str | Path, row: int, col: int, size: int) -> np.ndarray:
    """Read a channel-first BAND2/BAND3/BAND4 patch from a ZIP product."""
    window = Window(col, row, size, size)
    bands = []
    for band_name in ("BAND2", "BAND3", "BAND4"):
        with open_band(zip_path, band_name) as src:
            bands.append(src.read(1, window=window).astype(np.float32))
    return np.stack(bands, axis=0)


def read_browse_jpeg(zip_path: str | Path) -> bytes | None:
    zip_path = Path(zip_path)
    members = product_members(zip_path)
    member = members.get("JPG")
    if member is None:
        return None
    with zipfile.ZipFile(zip_path) as archive:
        return archive.read(member)


def estimate_browse_cloudiness(jpeg_bytes: bytes | None) -> float | None:
    """Crude browse-level brightness estimate used only for manifest hints."""
    if jpeg_bytes is None:
        return None
    try:
        import cv2

        data = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if image is None:
            return None
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        valid = gray > 3
        if not np.any(valid):
            return None
        return float(np.mean(gray[valid] > 180))
    except Exception:
        return None


def write_browse_jpeg(zip_path: str | Path, output_path: str | Path) -> Path | None:
    data = read_browse_jpeg(zip_path)
    if data is None:
        return None
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(data)
    return output
