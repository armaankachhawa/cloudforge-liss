"""Cloud-mask and mask-guided reconstruction networks."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(out_channels), out_channels),
        )
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1)
        )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(self.body(inputs) + self.skip(inputs))


def _groups(channels: int) -> int:
    for value in (8, 4, 2):
        if channels % value == 0:
            return value
    return 1


class AttentionGate(nn.Module):
    """Suppress irrelevant encoder features using the coarser decoder context."""

    def __init__(self, skip_channels: int, gate_channels: int) -> None:
        super().__init__()
        hidden = max(skip_channels // 2, 8)
        self.skip_projection = nn.Conv2d(skip_channels, hidden, 1, bias=False)
        self.gate_projection = nn.Conv2d(gate_channels, hidden, 1, bias=False)
        self.score = nn.Sequential(nn.SiLU(inplace=True), nn.Conv2d(hidden, 1, 1), nn.Sigmoid())

    def forward(self, skip: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        gate = F.interpolate(gate, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        attention = self.score(self.skip_projection(skip) + self.gate_projection(gate))
        return skip * attention


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.attention = AttentionGate(skip_channels, in_channels)
        self.block = ResidualBlock(in_channels + skip_channels, out_channels)

    def forward(self, inputs: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        inputs = F.interpolate(inputs, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        attended = self.attention(skip, inputs)
        return self.block(torch.cat([inputs, attended], dim=1))


class AttentionResUNet(nn.Module):
    """Compact 5-channel LISS + masks reconstruction network."""

    def __init__(self, in_channels: int = 5, out_channels: int = 3, base: int = 32) -> None:
        super().__init__()
        channels = [base, base * 2, base * 4, base * 8]
        self.encoder1 = ResidualBlock(in_channels, channels[0])
        self.encoder2 = ResidualBlock(channels[0], channels[1])
        self.encoder3 = ResidualBlock(channels[1], channels[2])
        self.encoder4 = ResidualBlock(channels[2], channels[3])
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ResidualBlock(channels[3], channels[3] * 2)
        self.decoder4 = DecoderBlock(channels[3] * 2, channels[3], channels[3])
        self.decoder3 = DecoderBlock(channels[3], channels[2], channels[2])
        self.decoder2 = DecoderBlock(channels[2], channels[1], channels[1])
        self.decoder1 = DecoderBlock(channels[1], channels[0], channels[0])
        self.head = nn.Sequential(nn.Conv2d(channels[0], out_channels, 1), nn.Sigmoid())

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        e1 = self.encoder1(inputs)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))
        e4 = self.encoder4(self.pool(e3))
        center = self.bottleneck(self.pool(e4))
        d4 = self.decoder4(center, e4)
        d3 = self.decoder3(d4, e3)
        d2 = self.decoder2(d3, e2)
        d1 = self.decoder1(d2, e1)
        return self.head(d1)


class CloudMaskUNet(nn.Module):
    """Small three-class U-Net for clear/cloud/cloud-shadow segmentation."""

    def __init__(self, in_channels: int = 3, classes: int = 3, base: int = 24) -> None:
        super().__init__()
        self.e1 = ResidualBlock(in_channels, base)
        self.e2 = ResidualBlock(base, base * 2)
        self.e3 = ResidualBlock(base * 2, base * 4)
        self.pool = nn.MaxPool2d(2)
        self.center = ResidualBlock(base * 4, base * 8)
        self.d3 = DecoderBlock(base * 8, base * 4, base * 4)
        self.d2 = DecoderBlock(base * 4, base * 2, base * 2)
        self.d1 = DecoderBlock(base * 2, base, base)
        self.head = nn.Conv2d(base, classes, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        e1 = self.e1(inputs)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        center = self.center(self.pool(e3))
        return self.head(self.d1(self.d2(self.d3(center, e3), e2), e1))


class AuxiliaryFusionResUNet(AttentionResUNet):
    """Optional enhanced mode for LISS + masks + Sentinel-1/temporal channels.

    The channel count is explicit so unavailable auxiliary data never becomes a
    hidden requirement of the core five-channel model.
    """

    def __init__(self, in_channels: int = 7, out_channels: int = 3, base: int = 32) -> None:
        super().__init__(in_channels=in_channels, out_channels=out_channels, base=base)


def build_model(name: str, in_channels: int, out_channels: int, base: int = 32) -> nn.Module:
    models = {
        "attention_resunet": AttentionResUNet,
        "auxiliary_fusion": AuxiliaryFusionResUNet,
        "cloud_mask_unet": CloudMaskUNet,
    }
    if name not in models:
        raise ValueError(f"Unknown model {name!r}; choose from {sorted(models)}")
    if name == "cloud_mask_unet":
        return models[name](in_channels=in_channels, classes=out_channels, base=base)
    return models[name](in_channels=in_channels, out_channels=out_channels, base=base)
