import torch

from src.losses import CloudReconstructionLoss
from src.models import AttentionResUNet, CloudMaskUNet


def test_attention_resunet_shape_and_range() -> None:
    model = AttentionResUNet(in_channels=5, out_channels=3, base=8)
    output = model(torch.rand(2, 5, 64, 64))
    assert output.shape == (2, 3, 64, 64)
    assert torch.all((0 <= output) & (output <= 1))


def test_cloud_mask_unet_shape() -> None:
    model = CloudMaskUNet(in_channels=3, classes=3, base=8)
    assert model(torch.rand(2, 3, 64, 64)).shape == (2, 3, 64, 64)


def test_composite_loss_is_finite_and_differentiable() -> None:
    prediction = torch.rand(2, 3, 32, 32, requires_grad=True)
    target = torch.rand(2, 3, 32, 32)
    mask = torch.zeros(2, 1, 32, 32)
    mask[:, :, 8:24, 8:24] = 1
    loss, components = CloudReconstructionLoss()(prediction, target, mask)
    assert torch.isfinite(loss)
    assert set(components) == {"masked_l1", "masked_ssim", "spectral_angle", "edge"}
    loss.backward()
    assert prediction.grad is not None
