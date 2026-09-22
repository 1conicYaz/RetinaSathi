"""Frozen official DINOv2-S/14 encoder with RetinaSathi screening heads."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import torch
from torch import nn

from ml.artifact_integrity import sha256


class DINOv2MultiHeadDRClassifier(nn.Module):
    """Use a permanently frozen DINOv2 feature encoder and three trainable heads."""

    dimension = 384
    patch_size = 14

    def __init__(self, backbone: nn.Module, dropout: float = 0.25) -> None:
        super().__init__()
        self.backbone = backbone
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        self.backbone.eval()
        self.dropout = nn.Dropout(dropout)
        self.referable_head = nn.Linear(self.dimension, 1)
        self.ordinal_head = nn.Linear(self.dimension, 4)
        self.nominal_head = nn.Linear(self.dimension, 5)

    def train(self, mode: bool = True) -> "DINOv2MultiHeadDRClassifier":
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if image.ndim != 4 or image.shape[-2] % self.patch_size or image.shape[-1] % self.patch_size:
            raise ValueError("DINOv2 input height and width must be divisible by 14")
        with torch.no_grad():
            embedding = self.backbone(image)
        if embedding.ndim != 2 or embedding.shape[1] != self.dimension:
            raise ValueError(f"Expected DINOv2-S/14 features shaped [batch, 384], got {tuple(embedding.shape)}")
        embedding = self.dropout(embedding)
        return (
            self.referable_head(embedding).squeeze(1),
            self.ordinal_head(embedding),
            self.nominal_head(embedding),
        )

    def trainable_parameters(self):
        return (parameter for parameter in self.parameters() if parameter.requires_grad)


def load_official_dinov2_vits14(
    provenance_path: Path,
    dropout: float = 0.25,
) -> tuple[DINOv2MultiHeadDRClassifier, dict[str, Any]]:
    """Load only the locally pinned and checksum-verified official encoder."""

    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    source = Path(provenance["local_source_path"])
    weight = Path(provenance["weight_path"])
    if not source.is_dir():
        raise FileNotFoundError(f"Pinned DINOv2 source is missing: {source}")
    if not weight.is_file():
        raise FileNotFoundError(f"DINOv2 weights are missing: {weight}")
    actual_hash = sha256(weight)
    if actual_hash != provenance["weight_sha256"]:
        raise ValueError("DINOv2 weight checksum does not match the pinned provenance")
    resolved_commit = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if resolved_commit != provenance["resolved_commit"]:
        raise ValueError("DINOv2 source commit does not match the pinned provenance")
    backbone = torch.hub.load(str(source), provenance["model_name"], source="local", pretrained=True)
    return DINOv2MultiHeadDRClassifier(backbone, dropout=dropout), provenance
