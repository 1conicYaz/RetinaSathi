"""Partially fine-tuned DINOv2-S/14 with RetinaSathi screening heads."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn

from .classifier_v3_2 import load_official_dinov2_vits14


class PartialDINOv2MultiHeadDRClassifier(nn.Module):
    """Train the final DINOv2 blocks and RetinaSathi's three prediction heads."""

    dimension = 384
    patch_size = 14

    def __init__(self, backbone: nn.Module, dropout: float = 0.25, unfreeze_blocks: int = 2) -> None:
        super().__init__()
        if unfreeze_blocks < 1 or unfreeze_blocks > len(backbone.blocks):
            raise ValueError(f"unfreeze_blocks must be between 1 and {len(backbone.blocks)}")
        self.backbone = backbone
        self.unfreeze_blocks = unfreeze_blocks
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        for block in self.backbone.blocks[-unfreeze_blocks:]:
            for parameter in block.parameters():
                parameter.requires_grad_(True)
        for parameter in self.backbone.norm.parameters():
            parameter.requires_grad_(True)
        self.dropout = nn.Dropout(dropout)
        self.referable_head = nn.Linear(self.dimension, 1)
        self.ordinal_head = nn.Linear(self.dimension, 4)
        self.nominal_head = nn.Linear(self.dimension, 5)

    def train(self, mode: bool = True) -> "PartialDINOv2MultiHeadDRClassifier":
        super().train(mode)
        self.backbone.eval()
        for block in self.backbone.blocks[-self.unfreeze_blocks:]:
            block.train(mode)
        self.backbone.norm.train(mode)
        return self

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if image.ndim != 4 or image.shape[-2] % self.patch_size or image.shape[-1] % self.patch_size:
            raise ValueError("DINOv2 input height and width must be divisible by 14")
        embedding = self.backbone(image)
        if embedding.ndim != 2 or embedding.shape[1] != self.dimension:
            raise ValueError(f"Expected DINOv2-S/14 features shaped [batch, 384], got {tuple(embedding.shape)}")
        embedding = self.dropout(embedding)
        return (
            self.referable_head(embedding).squeeze(1),
            self.ordinal_head(embedding),
            self.nominal_head(embedding),
        )

    def backbone_trainable_parameters(self) -> Iterable[nn.Parameter]:
        return (parameter for parameter in self.backbone.parameters() if parameter.requires_grad)

    def head_parameters(self) -> Iterable[nn.Parameter]:
        modules = (self.referable_head, self.ordinal_head, self.nominal_head)
        return (parameter for module in modules for parameter in module.parameters())

    def trainable_parameters(self) -> Iterable[nn.Parameter]:
        return (parameter for parameter in self.parameters() if parameter.requires_grad)


def load_partial_dinov2_vits14(
    provenance_path: Path,
    dropout: float = 0.25,
    unfreeze_blocks: int = 2,
) -> tuple[PartialDINOv2MultiHeadDRClassifier, dict[str, Any]]:
    frozen, provenance = load_official_dinov2_vits14(provenance_path, dropout=dropout)
    return PartialDINOv2MultiHeadDRClassifier(
        frozen.backbone,
        dropout=dropout,
        unfreeze_blocks=unfreeze_blocks,
    ), provenance
