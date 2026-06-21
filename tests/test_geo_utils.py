from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from src.geo_utils import (
    align_to_reference,
    compose_preserving_clear,
    iter_windows,
    read_raster,
    write_raster,
)


def profile(width: int = 32, height: int = 24) -> dict:
    return {
        "driver": "GTiff",
        "width": width,
        "height": height,
        "count": 3,
        "dtype": "uint16",
        "crs": "EPSG:32646",
        "transform": from_origin(500000, 3000000, 5.8, 5.8),
        "nodata": 0,
    }


def test_geotiff_roundtrip_preserves_grid(tmp_path: Path) -> None:
    array = np.arange(3 * 24 * 32, dtype=np.uint16).reshape(3, 24, 32) + 1
    path = write_raster(
        tmp_path / "scene.tif",
        array,
        profile(),
        descriptions=("green", "red", "nir"),
        tags={"SCENE_ID": "TEST"},
    )
    loaded = read_raster(path, dtype="uint16")
    np.testing.assert_array_equal(loaded.array, array)
    assert loaded.profile["crs"].to_string() == "EPSG:32646"
    assert loaded.profile["transform"] == profile()["transform"]
    assert loaded.descriptions == ("green", "red", "nir")
    assert loaded.tags["SCENE_ID"] == "TEST"


def test_compose_preserves_every_clear_value() -> None:
    original = np.arange(3 * 4 * 5).reshape(3, 4, 5)
    generated = np.full_like(original, 999)
    mask = np.zeros((4, 5), dtype=np.uint8)
    mask[1:3, 2:4] = 1
    result = compose_preserving_clear(original, generated, mask)
    np.testing.assert_array_equal(result[:, mask == 0], original[:, mask == 0])
    np.testing.assert_array_equal(result[:, mask > 0], generated[:, mask > 0])


def test_windows_cover_entire_scene() -> None:
    covered = np.zeros((517, 601), dtype=bool)
    for window in iter_windows(517, 601, 256, 32):
        row, col = int(window.row_off), int(window.col_off)
        covered[row : row + int(window.height), col : col + int(window.width)] = True
    assert covered.all()


def test_align_to_reference_uses_exact_reference_grid(tmp_path: Path) -> None:
    reference = tmp_path / "reference.tif"
    source = tmp_path / "source.tif"
    output = tmp_path / "aligned.tif"
    write_raster(reference, np.ones((3, 24, 32), dtype=np.uint16), profile())
    source_profile = profile(16, 12)
    source_profile["transform"] = from_origin(500000, 3000000, 11.6, 11.6)
    write_raster(source, np.ones((2, 12, 16), dtype=np.uint16), source_profile)
    align_to_reference(source, reference, output)
    with rasterio.open(reference) as ref, rasterio.open(output) as aligned:
        assert (aligned.width, aligned.height) == (ref.width, ref.height)
        assert aligned.transform == ref.transform
        assert aligned.crs == ref.crs
        assert aligned.count == 2
