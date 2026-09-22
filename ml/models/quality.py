"""DeepDRiD multi-task quality model."""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


class QualityModel(nn.Module):
    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        base = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT if pretrained else None)
        self.features = base.features
        self.pool = base.avgpool
        self.shared = nn.Sequential(nn.Linear(576, 256), nn.Hardswish(), nn.Dropout(0.2))
        self.overall = nn.Linear(256, 2)
        self.attributes = nn.Linear(256, 3)

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.shared(self.pool(self.features(image)).flatten(1))
        return self.overall(embedding), torch.sigmoid(self.attributes(embedding))
