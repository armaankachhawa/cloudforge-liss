import numpy as np
import torch
from rasterio.transform import from_origin

from src.create_masks import rule_based_mask
from src.create_synthetic_clouds import CLOUD, SHADOW, corrupt_clear_patch
from src.evaluate import masked_metrics
from src.geo_utils import read_raster, write_raster
from src.models import AttentionResUNet
from src.predict import confidence_map, predict_scene, tiled_prediction


class CopyVisibleBands(torch.nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs[:, :3]


def test_rule_mask_has_supported_classes() -> None:
    rng = np.random.default_rng(2)
    image = rng.normal(1000, 100, size=(3, 96, 96)).astype(np.float32)
    image[:, 20:45, 20:45] = 4000
    labels = rule_based_mask(image)
    assert set(np.unique(labels)).issubset({0, 1, 2})
    assert np.any(labels == 1)


def test_real_texture_corruption_changes_only_masked_pixels() -> None:
    clear = np.full((3, 16, 16), 0.4, dtype=np.float32)
    texture = np.full((3, 16, 16), 0.9, dtype=np.float32)
    labels = np.zeros((16, 16), dtype=np.uint8)
    labels[2:6, 2:6] = CLOUD
    labels[9:13, 9:13] = SHADOW
    synthetic = corrupt_clear_patch(clear, texture, labels)
    np.testing.assert_array_equal(synthetic[:, labels == 0], clear[:, labels == 0])
    assert np.all(synthetic[:, labels == CLOUD] == 0.9)
    assert np.all(synthetic[:, labels == SHADOW] < clear[:, labels == SHADOW])


def test_tiled_prediction_covers_odd_scene_dimensions() -> None:
    image = np.random.default_rng(3).random((3, 141, 173), dtype=np.float32)
    labels = np.zeros((141, 173), dtype=np.uint8)
    labels[30:100, 40:130] = 1
    prediction, variance = tiled_prediction(
        CopyVisibleBands(), image, labels, torch.device("cpu"), tile_size=64, overlap=16
    )
    np.testing.assert_allclose(prediction, image, atol=1e-6)
    assert variance.shape == labels.shape


def test_perfect_reconstruction_metrics() -> None:
    target = np.random.default_rng(4).random((3, 32, 32), dtype=np.float32)
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[5:25, 6:26] = 1
    metrics = masked_metrics(target.copy(), target, mask)
    assert metrics["mae"] == 0
    assert metrics["rmse"] == 0
    assert metrics["ssim"] == 1
    assert metrics["ndvi_mae"] == 0


def test_confidence_is_one_on_clear_pixels() -> None:
    labels = np.zeros((32, 32), dtype=np.uint8)
    labels[8:24, 8:24] = 1
    confidence = confidence_map(labels, np.zeros_like(labels, dtype=np.float32), 32)
    assert np.all(confidence[labels == 0] == 1)
    assert np.all((confidence >= 0) & (confidence <= 1))


def test_operational_inference_writes_analysis_ready_products(tmp_path) -> None:
    image = np.random.default_rng(5).integers(100, 4000, (3, 32, 32), dtype=np.uint16)
    labels = np.zeros((32, 32), dtype=np.uint8)
    labels[8:24, 8:24] = 1
    raster_profile = {
        "driver": "GTiff",
        "width": 32,
        "height": 32,
        "count": 3,
        "dtype": "uint16",
        "crs": "EPSG:32646",
        "transform": from_origin(500000, 3000000, 5.8, 5.8),
        "nodata": 0,
    }
    input_path = write_raster(tmp_path / "input.tif", image, raster_profile)
    mask_path = write_raster(tmp_path / "mask.tif", labels, raster_profile, dtype="uint8")
    model = AttentionResUNet(base=8)
    config = {
        "data": {"input_channels": 5, "output_channels": 3},
        "model": {"name": "attention_resunet", "base_channels": 8},
    }
    checkpoint = tmp_path / "model.pt"
    torch.save({"model_state": model.state_dict(), "config": config}, checkpoint)
    products = predict_scene(
        input_path,
        checkpoint,
        tmp_path / "products",
        mask_path=mask_path,
        device_name="cpu",
        tile_size=32,
        overlap=8,
    )
    assert set(products) == {"reconstructed", "mask", "confidence", "preview", "report"}
    reconstructed = read_raster(products["reconstructed"], dtype="uint16")
    np.testing.assert_array_equal(reconstructed.array[:, labels == 0], image[:, labels == 0])
    assert reconstructed.profile["crs"].to_string() == "EPSG:32646"
    assert reconstructed.profile["transform"] == raster_profile["transform"]
