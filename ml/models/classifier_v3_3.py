"""Frozen official RETFound-MAE CFP encoder with RetinaSathi screening heads."""

from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path
from types import ModuleType
from typing import Any

import torch
from torch import nn

from ml.artifact_integrity import sha256


class RETFoundMultiHeadDRClassifier(nn.Module):
    """Use a permanently frozen RETFound ViT-L/16 encoder and three heads."""

    dimension = 1024
    patch_size = 16

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

    def train(self, mode: bool = True) -> "RETFoundMultiHeadDRClassifier":
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if image.ndim != 4 or image.shape[-2] % self.patch_size or image.shape[-1] % self.patch_size:
            raise ValueError("RETFound input height and width must be divisible by 16")
        with torch.no_grad():
            embedding = self.backbone.forward_features(image)
        if embedding.ndim == 3 and embedding.shape[1] == 1:
            embedding = embedding.squeeze(1)
        if embedding.ndim != 2 or embedding.shape[1] != self.dimension:
            raise ValueError(f"Expected RETFound features shaped [batch, 1024], got {tuple(embedding.shape)}")
        embedding = self.dropout(embedding)
        return (
            self.referable_head(embedding).squeeze(1),
            self.ordinal_head(embedding),
            self.nominal_head(embedding),
        )

    def trainable_parameters(self):
        return (parameter for parameter in self.parameters() if parameter.requires_grad)

    def head_state_dict(self) -> dict[str, torch.Tensor]:
        prefixes = ("referable_head.", "ordinal_head.", "nominal_head.")
        return {name: value for name, value in self.state_dict().items() if name.startswith(prefixes)}

    def load_head_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        expected = set(self.head_state_dict())
        if set(state) != expected:
            raise ValueError(f"RETFound head checkpoint keys differ: expected={sorted(expected)}, got={sorted(state)}")
        incompatible = self.load_state_dict(state, strict=False)
        if incompatible.unexpected_keys:
            raise ValueError(f"Unexpected RETFound head checkpoint keys: {incompatible.unexpected_keys}")


def _load_official_module(source: Path) -> ModuleType:
    module_path = source / "models_vit.py"
    spec = importlib.util.spec_from_file_location("retfound_official_models_vit", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load official RETFound model definitions from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_official_retfound_mae_cfp(
    provenance_path: Path,
    dropout: float = 0.25,
) -> tuple[RETFoundMultiHeadDRClassifier, dict[str, Any]]:
    """Load a checksum- and commit-bound official RETFound CFP encoder."""

    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    source = Path(provenance["local_source_path"])
    weight = Path(provenance["weight_path"])
    if not source.is_dir():
        raise FileNotFoundError(f"Pinned RETFound source is missing: {source}")
    if not weight.is_file():
        raise FileNotFoundError(f"RETFound weights are missing: {weight}")
    if sha256(weight) != provenance["weight_sha256"]:
        raise ValueError("RETFound weight checksum does not match the pinned provenance")
    commit = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != provenance["resolved_commit"]:
        raise ValueError("RETFound source commit does not match the pinned provenance")

    official = _load_official_module(source)
    backbone = official.RETFound_mae(
        img_size=int(provenance["input_size"]),
        num_classes=5,
        drop_path_rate=0.0,
        global_pool=False,
    )
    checkpoint = torch.load(weight, map_location="cpu", weights_only=False, mmap=True)
    checkpoint_state = checkpoint["model"]
    expected_state = backbone.state_dict()
    usable = {
        name: value
        for name, value in checkpoint_state.items()
        if name in expected_state and value.shape == expected_state[name].shape
    }
    incompatible = backbone.load_state_dict(usable, strict=False, assign=True)
    expected_missing = set(provenance["expected_missing_keys"])
    if set(incompatible.missing_keys) != expected_missing or incompatible.unexpected_keys:
        raise ValueError(
            "RETFound checkpoint is not structurally compatible: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )
    if len(usable) != int(provenance["loaded_encoder_keys"]):
        raise ValueError("RETFound loaded encoder key count differs from pinned provenance")
    backbone.head = nn.Identity()
    return RETFoundMultiHeadDRClassifier(backbone, dropout=dropout), provenance
