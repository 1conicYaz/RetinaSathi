"""Compact U-Net for lesion and vessel segmentation."""

from __future__ import annotations

import torch
from torch import nn


class DoubleConv(nn.Sequential):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__(
            nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False), nn.BatchNorm2d(output_channels), nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False), nn.BatchNorm2d(output_channels), nn.ReLU(inplace=True),
        )


class CompactUNet(nn.Module):
    def __init__(self, output_channels: int, base_channels: int = 16) -> None:
        super().__init__(); channels = [base_channels, base_channels*2, base_channels*4, base_channels*8]
        self.encoders = nn.ModuleList([DoubleConv(3, channels[0]), DoubleConv(channels[0], channels[1]), DoubleConv(channels[1], channels[2])])
        self.pool = nn.MaxPool2d(2); self.bridge = DoubleConv(channels[2], channels[3])
        self.upconvs = nn.ModuleList([nn.ConvTranspose2d(channels[3], channels[2], 2, 2), nn.ConvTranspose2d(channels[2], channels[1], 2, 2), nn.ConvTranspose2d(channels[1], channels[0], 2, 2)])
        self.decoders = nn.ModuleList([DoubleConv(channels[3], channels[2]), DoubleConv(channels[2], channels[1]), DoubleConv(channels[1], channels[0])])
        self.output = nn.Conv2d(channels[0], output_channels, 1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        skips = []; value = image
        for encoder in self.encoders:
            value = encoder(value); skips.append(value); value = self.pool(value)
        value = self.bridge(value)
        for upconv, decoder, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            value = upconv(value)
            if value.shape[-2:] != skip.shape[-2:]: value = nn.functional.interpolate(value, skip.shape[-2:], mode="bilinear", align_corners=False)
            value = decoder(torch.cat([skip, value], dim=1))
        return self.output(value)


def dice_bce_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    probabilities = torch.sigmoid(logits); dimensions = (0, 2, 3)
    intersection = (probabilities * targets).sum(dim=dimensions)
    dice = (2*intersection + 1) / (probabilities.sum(dim=dimensions) + targets.sum(dim=dimensions) + 1)
    return nn.functional.binary_cross_entropy_with_logits(logits, targets) + (1-dice).mean()
