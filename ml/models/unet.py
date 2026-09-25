"""Compact U-Net for lesion and vessel segmentation."""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


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


class UpBlock(nn.Module):
    def __init__(self,input_channels:int,skip_channels:int,output_channels:int)->None:
        super().__init__();self.up=nn.ConvTranspose2d(input_channels,output_channels,2,2);self.conv=DoubleConv(output_channels+skip_channels,output_channels)
    def forward(self,value:torch.Tensor,skip:torch.Tensor)->torch.Tensor:
        value=self.up(value)
        if value.shape[-2:]!=skip.shape[-2:]:value=nn.functional.interpolate(value,skip.shape[-2:],mode="bilinear",align_corners=False)
        return self.conv(torch.cat([value,skip],dim=1))


class MobileUNet(nn.Module):
    """U-Net decoder over an ImageNet-pretrained MobileNetV3-small encoder."""
    def __init__(self,output_channels:int,pretrained:bool=True)->None:
        super().__init__();weights=MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        self.encoder=mobilenet_v3_small(weights=weights).features
        self.up4=UpBlock(576,48,96);self.up3=UpBlock(96,24,48);self.up2=UpBlock(48,16,24);self.up1=UpBlock(24,16,16)
        self.final_up=nn.ConvTranspose2d(16,16,2,2);self.output=nn.Conv2d(16,output_channels,1)
    def forward(self,image:torch.Tensor)->torch.Tensor:
        skips={};value=image
        for index,layer in enumerate(self.encoder):
            value=layer(value)
            if index in (0,1,3,8):skips[index]=value
        value=self.up4(value,skips[8]);value=self.up3(value,skips[3]);value=self.up2(value,skips[1]);value=self.up1(value,skips[0]);value=self.final_up(value)
        if value.shape[-2:]!=image.shape[-2:]:value=nn.functional.interpolate(value,image.shape[-2:],mode="bilinear",align_corners=False)
        return self.output(value)


def build_lesion_model(config:dict[str,object],output_channels:int=4,pretrained:bool=True)->nn.Module:
    architecture=str(config.get("experiment",{}).get("architecture","compact_unet"))
    if architecture=="mobile_unet":return MobileUNet(output_channels,pretrained=pretrained)
    if architecture=="compact_unet":return CompactUNet(output_channels)
    raise ValueError(f"Unknown lesion architecture: {architecture}")


def dice_bce_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    probabilities = torch.sigmoid(logits); dimensions = (0, 2, 3)
    intersection = (probabilities * targets).sum(dim=dimensions)
    dice = (2*intersection + 1) / (probabilities.sum(dim=dimensions) + targets.sum(dim=dimensions) + 1)
    return nn.functional.binary_cross_entropy_with_logits(logits, targets) + (1-dice).mean()


def focal_tversky_loss(logits: torch.Tensor, targets: torch.Tensor, alpha: float = .7, beta: float = .3, gamma: float = .75) -> torch.Tensor:
    """Imbalance-aware loss for tiny retinal lesions.

    ``alpha`` gives false negatives more weight, which is appropriate for a
    screening-support candidate. Focal BCE still penalizes false positives.
    """
    probabilities = torch.sigmoid(logits)
    dims = (0, 2, 3)
    true_positive = (probabilities * targets).sum(dim=dims)
    false_negative = ((1-probabilities) * targets).sum(dim=dims)
    false_positive = (probabilities * (1-targets)).sum(dim=dims)
    tversky = (true_positive + 1) / (true_positive + alpha*false_negative + beta*false_positive + 1)
    # IDRiD lesions occupy very different pixel fractions. These bounded
    # foreground weights stop MA/HE gradients disappearing into background.
    default_weights = (24., 8., 3., 6.) if logits.shape[1] == 4 else (1.,) * logits.shape[1]
    positive_weights = logits.new_tensor(default_weights).view(1,-1,1,1)
    bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none", pos_weight=positive_weights)
    focal = ((1-torch.exp(-bce))**2 * bce).mean()
    return focal + torch.pow(1-tversky, gamma).mean()
