"""Ordinal diabetic-retinopathy classifier architectures."""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import (
    ConvNeXt_Tiny_Weights,
    EfficientNet_B3_Weights,
    EfficientNet_V2_S_Weights,
    convnext_tiny,
    efficientnet_b3,
    efficientnet_v2_s,
)


ARCHITECTURES = ("efficientnet_b3", "efficientnet_v2_s", "convnext_tiny")


class FeatureBackbone(nn.Module):
    def __init__(self, architecture: str, pretrained: bool) -> None:
        super().__init__()
        if architecture == "efficientnet_b3":
            base = efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT if pretrained else None)
            self.features, self.pool = base.features, base.avgpool
            self.normalization = nn.Identity(); self.dimension = base.classifier[-1].in_features
        elif architecture == "efficientnet_v2_s":
            base = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT if pretrained else None)
            self.features, self.pool = base.features, base.avgpool
            self.normalization = nn.Identity(); self.dimension = base.classifier[-1].in_features
        elif architecture == "convnext_tiny":
            base = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None)
            self.features, self.pool = base.features, base.avgpool
            self.normalization = base.classifier[0]; self.dimension = base.classifier[-1].in_features
        else:
            raise ValueError(f"Unsupported architecture {architecture!r}; choose one of {ARCHITECTURES}")

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        feature_map = self.features(image)
        pooled = self.pool(feature_map)
        return self.normalization(pooled).flatten(1)


class OrdinalDRClassifier(nn.Module):
    """Four cumulative DR thresholds plus an optional three-class DME head."""

    def __init__(self, architecture: str = "efficientnet_b3", pretrained: bool = True, dme_head: bool = True) -> None:
        super().__init__()
        self.architecture = architecture
        self.backbone = FeatureBackbone(architecture, pretrained)
        self.dropout = nn.Dropout(0.25)
        self.ordinal_head = nn.Linear(self.backbone.dimension, 4)
        self.dme_head = nn.Linear(self.backbone.dimension, 3) if dme_head else None

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        embedding = self.dropout(self.backbone(image))
        return self.ordinal_head(embedding), self.dme_head(embedding) if self.dme_head else None

    def gradcam_layer(self) -> nn.Module:
        return self.backbone.features[-1]


class NominalDRClassifier(nn.Module):
    """Cross-entropy comparison model using the identical feature backbone."""

    def __init__(self, architecture: str = "efficientnet_b3", pretrained: bool = True, dme_head: bool = True) -> None:
        super().__init__()
        self.architecture = architecture
        self.backbone = FeatureBackbone(architecture, pretrained)
        self.dropout = nn.Dropout(0.25)
        self.grade_head = nn.Linear(self.backbone.dimension, 5)
        self.dme_head = nn.Linear(self.backbone.dimension, 3) if dme_head else None

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        embedding = self.dropout(self.backbone(image))
        return self.grade_head(embedding), self.dme_head(embedding) if self.dme_head else None

    def gradcam_layer(self) -> nn.Module:
        return self.backbone.features[-1]


def coral_targets(grades: torch.Tensor, levels: int = 4) -> torch.Tensor:
    thresholds = torch.arange(levels, device=grades.device).unsqueeze(0)
    return (grades.unsqueeze(1) > thresholds).to(torch.float32)


def ordinal_probabilities(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    cumulative = torch.sigmoid(logits / temperature)
    cumulative = torch.cummin(cumulative, dim=1).values
    endpoints = torch.cat(
        [torch.ones_like(cumulative[:, :1]), cumulative, torch.zeros_like(cumulative[:, :1])], dim=1
    )
    probabilities = endpoints[:, :-1] - endpoints[:, 1:]
    return probabilities.clamp_min(0) / probabilities.sum(dim=1, keepdim=True).clamp_min(1e-8)


def coral_loss(logits: torch.Tensor, grades: torch.Tensor, positive_weights: torch.Tensor | None = None) -> torch.Tensor:
    return nn.functional.binary_cross_entropy_with_logits(logits, coral_targets(grades), pos_weight=positive_weights)
