"""RetinaSathi V3.1 multi-head diabetic-retinopathy classifier."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from ml.artifact_integrity import sha256

from .classifier_v2 import FeatureBackbone


class MultiHeadDRClassifier(nn.Module):
    """Shared encoder with referable, ordinal, and direct five-grade heads."""

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
        self.nominal_head = nn.Linear(self.backbone.dimension, 5)

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embedding = self.dropout(self.backbone(image))
        return (
            self.referable_head(embedding).squeeze(1),
            self.ordinal_head(embedding),
            self.nominal_head(embedding),
        )

    def gradcam_layer(self) -> nn.Module:
        return self.backbone.features[-1]


def load_v3_0_weights(
    model: MultiHeadDRClassifier,
    checkpoint_path: Path,
    expected_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Load only a compatible V3.0 encoder and heads into V3.1."""

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    architecture = checkpoint.get("config", {}).get("experiment", {}).get("architecture")
    if architecture != model.architecture:
        raise ValueError(
            f"V3.0 architecture {architecture!r} does not match V3.1 architecture {model.architecture!r}"
        )
    checkpoint_manifest = checkpoint.get("manifest_sha256")
    if expected_manifest_sha256 is not None and checkpoint_manifest != expected_manifest_sha256:
        raise ValueError("V3.0 checkpoint manifest hash does not match the validated V3 manifest")

    incompatible = model.load_state_dict(checkpoint["model"], strict=False)
    expected_missing = {"nominal_head.weight", "nominal_head.bias"}
    if set(incompatible.missing_keys) != expected_missing or incompatible.unexpected_keys:
        raise ValueError(
            "V3.0 checkpoint is not structurally compatible: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )
    return {
        "source_checkpoint": str(checkpoint_path),
        "source_checkpoint_sha256": sha256(checkpoint_path),
        "source_epoch": checkpoint.get("epoch"),
        "source_manifest_sha256": checkpoint_manifest,
        "architecture": architecture,
        "new_parameters": sorted(expected_missing),
    }
