"""RetinaSathi V3 dual-head diabetic-retinopathy classifier."""

from __future__ import annotations

import torch
from torch import nn

from .classifier_v2 import FeatureBackbone


class DualHeadDRClassifier(nn.Module):
    """Shared retinal encoder with referral and ordinal severity heads."""

    def __init__(
        self,
        architecture: str = "efficientnet_b3",
        pretrained: bool = True,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.architecture = architecture
        self.backbone = FeatureBackbone(architecture, pretrained)
        self.dropout = nn.Dropout(dropout)
        self.referable_head = nn.Linear(self.backbone.dimension, 1)
        self.ordinal_head = nn.Linear(self.backbone.dimension, 4)

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.dropout(self.backbone(image))
        referable_logit = self.referable_head(embedding).squeeze(1)
        ordinal_logits = self.ordinal_head(embedding)
        return referable_logit, ordinal_logits

    def gradcam_layer(self) -> nn.Module:
        return self.backbone.features[-1]
