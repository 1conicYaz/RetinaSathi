"""Optic-disc and fovea coordinate regression from verified IDRiD labels."""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


class DiscFoveaLocalizer(nn.Module):
    def __init__(self, pretrained: bool = True) -> None:
        super().__init__(); base=mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT if pretrained else None)
        self.features=base.features; self.pool=base.avgpool
        self.head=nn.Sequential(nn.Linear(576,256),nn.Hardswish(),nn.Dropout(0.2),nn.Linear(256,8))

    def forward(self,image:torch.Tensor)->tuple[torch.Tensor,torch.Tensor]:
        raw=self.head(self.pool(self.features(image)).flatten(1))
        return torch.sigmoid(raw[:,:4]),raw[:,4:].clamp(-8.0,2.0)


def localization_loss(coordinates:torch.Tensor,log_variance:torch.Tensor,target:torch.Tensor)->torch.Tensor:
    squared=(coordinates-target).square()
    gaussian=0.5*(torch.exp(-log_variance)*squared+log_variance)
    return torch.nn.functional.smooth_l1_loss(coordinates,target)+0.05*gaussian.mean()
