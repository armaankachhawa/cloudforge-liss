"""Masked spatial, structural, spectral and edge reconstruction losses."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _masked_mean(values: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    expanded = mask.expand_as(values)
    return (values * expanded).sum() / expanded.sum().clamp_min(eps)


def masked_l1(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return _masked_mean(torch.abs(prediction - target), mask)


def masked_ssim_loss(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, window: int = 7
) -> torch.Tensor:
    padding = window // 2
    mean_x = F.avg_pool2d(prediction, window, stride=1, padding=padding)
    mean_y = F.avg_pool2d(target, window, stride=1, padding=padding)
    var_x = F.avg_pool2d(prediction.square(), window, 1, padding) - mean_x.square()
    var_y = F.avg_pool2d(target.square(), window, 1, padding) - mean_y.square()
    covariance = F.avg_pool2d(prediction * target, window, 1, padding) - mean_x * mean_y
    c1, c2 = 0.01**2, 0.03**2
    ssim = ((2 * mean_x * mean_y + c1) * (2 * covariance + c2)) / (
        (mean_x.square() + mean_y.square() + c1) * (var_x + var_y + c2) + 1e-6
    )
    return _masked_mean(1.0 - ssim.clamp(-1, 1), mask)


def spectral_angle_loss(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    dot = (prediction * target).sum(dim=1, keepdim=True)
    norm = (
        prediction.square().sum(dim=1, keepdim=True).sqrt()
        * target.square().sum(dim=1, keepdim=True).sqrt()
    )
    cosine = (dot / norm.clamp_min(1e-6)).clamp(-1 + 1e-6, 1 - 1e-6)
    return _masked_mean(torch.acos(cosine), mask)


def _gradients(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    kernel_x = image.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 8
    kernel_y = kernel_x.t()
    channels = image.shape[1]
    kernel_x = kernel_x.expand(channels, 1, 3, 3)
    kernel_y = kernel_y.expand(channels, 1, 3, 3)
    return (
        F.conv2d(image, kernel_x, padding=1, groups=channels),
        F.conv2d(image, kernel_y, padding=1, groups=channels),
    )


def edge_loss(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    pred_x, pred_y = _gradients(prediction)
    target_x, target_y = _gradients(target)
    difference = torch.abs(pred_x - target_x) + torch.abs(pred_y - target_y)
    return _masked_mean(difference, mask)


class CloudReconstructionLoss(nn.Module):
    def __init__(
        self,
        l1_weight: float = 0.60,
        ssim_weight: float = 0.20,
        sam_weight: float = 0.15,
        edge_weight: float = 0.05,
    ) -> None:
        super().__init__()
        self.weights = {
            "masked_l1": l1_weight,
            "masked_ssim": ssim_weight,
            "spectral_angle": sam_weight,
            "edge": edge_weight,
        }

    def forward(
        self, prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        components = {
            "masked_l1": masked_l1(prediction, target, mask),
            "masked_ssim": masked_ssim_loss(prediction, target, mask),
            "spectral_angle": spectral_angle_loss(prediction, target, mask),
            "edge": edge_loss(prediction, target, mask),
        }
        total = sum(self.weights[name] * value for name, value in components.items())
        return total, components
