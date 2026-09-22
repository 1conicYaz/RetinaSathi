from __future__ import annotations

import torch
from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


class RetinaSathiNet(nn.Module):
    """Compact multi-task network for IDRiD DR grade and DME risk."""

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        backbone = mobilenet_v3_small(weights=weights)
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        self.embedding = nn.Sequential(
            nn.Linear(576, 256),
            nn.Hardswish(),
            nn.Dropout(p=0.25),
        )
        self.grade_head = nn.Linear(256, 5)
        self.dme_head = nn.Linear(256, 3)

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feature_map = self.features(image)
        pooled = self.avgpool(feature_map).flatten(1)
        embedding = self.embedding(pooled)
        return self.grade_head(embedding), self.dme_head(embedding)

    def freeze_backbone(self) -> None:
        for parameter in self.features.parameters():
            parameter.requires_grad = False

    def unfreeze_last_blocks(self, count: int = 3) -> None:
        for block in self.features[-count:]:
            for parameter in block.parameters():
                parameter.requires_grad = True
