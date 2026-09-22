#!/usr/bin/env python3
"""Export the frozen V3.4 classifier and bind the ONNX file to its evidence.

Preprocessing and calibration intentionally stay outside the graph so MATLAB can
show each retinal-processing stage and audit the final decision policy.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from torch import nn

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from ml.artifact_integrity import sha256
from ml.models import load_partial_dinov2_vits14


class ExportWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        referable, ordinal, nominal = self.model(image)
        return referable.unsqueeze(1), ordinal, nominal


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    validation_path = args.validation.resolve()
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    checkpoint_digest = sha256(checkpoint)
    if validation.get("checkpoint_sha256") != checkpoint_digest:
        raise ValueError("Validation report is not bound to the requested checkpoint")
    if validation.get("manifest_sha256") != state.get("manifest_sha256"):
        raise ValueError("Checkpoint and validation report use different data manifests")
    if validation.get("official_test_used") is not False:
        raise ValueError("V3.4 export requires the pre-locked-test validation report")

    config = state["config"]
    repository_root = REPOSITORY_ROOT
    provenance = Path(str(config["experiment"]["foundation_provenance"]))
    if not provenance.is_absolute():
        provenance = repository_root / provenance
    model, foundation = load_partial_dinov2_vits14(
        provenance,
        dropout=float(config["training"]["dropout"]),
        unfreeze_blocks=int(config["training"]["unfreeze_blocks"]),
    )
    model.load_state_dict(state["model"], strict=True)
    wrapper = ExportWrapper(model.eval())
    image_size = int(config["preprocessing"]["input_size"])
    sample = torch.randn(1, 3, image_size, image_size)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch_outputs = [value.numpy() for value in wrapper(sample)]
    torch.onnx.export(
        wrapper,
        sample,
        args.output,
        input_names=["image"],
        output_names=["referable_logit", "ordinal_logits", "nominal_logits"],
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )

    session = ort.InferenceSession(str(args.output), providers=["CPUExecutionProvider"])
    onnx_outputs = session.run(None, {"image": sample.numpy()})
    differences = [float(np.max(np.abs(left - right))) for left, right in zip(torch_outputs, onnx_outputs)]
    if max(differences) > 1e-4:
        raise RuntimeError(f"ONNX numerical parity failed: {differences}")

    manifest_path = args.manifest or args.output.with_suffix(".manifest.json")
    def portable(path: Path) -> str:
        try:
            return str(path.resolve().relative_to(repository_root))
        except ValueError:
            return path.name

    public_foundation = {
        key: value
        for key, value in foundation.items()
        if key not in {"local_source_path", "weight_path", "license_path"}
    }
    payload = {
        "schema_version": 1,
        "model_version": f"classifier-v3.4-seed{config['experiment']['seed']}",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "research_status": "candidate_source_gate_passed",
        "official_test_used": False,
        "checkpoint": {"path": portable(checkpoint), "sha256": checkpoint_digest},
        "validation": {"path": portable(validation_path), "sha256": sha256(validation_path)},
        "onnx": {"path": portable(args.output), "sha256": sha256(args.output)},
        "input": {
            "name": "image",
            "shape": [1, 3, image_size, image_size],
            "layout": "NCHW",
            "dtype": "float32",
            "preprocessing": config["preprocessing"],
            "normalization_mean_rgb": [0.485, 0.456, 0.406],
            "normalization_std_rgb": [0.229, 0.224, 0.225],
        },
        "outputs": ["referable_logit", "ordinal_logits", "nominal_logits"],
        "calibration": {
            "ordinal_temperature": validation["ordinal_temperature"],
            "nominal_temperature": validation["nominal_temperature"],
            "binary_temperature": validation["binary_temperature"],
            "referable_threshold": validation["threshold_gate"]["selected"]["threshold"],
            "grade_fusion": "0.5 * ordinal_probabilities + 0.5 * nominal_softmax",
        },
        "foundation_model": public_foundation,
        "pytorch_onnx_max_abs_difference": {
            "referable_logit": differences[0],
            "ordinal_logits": differences[1],
            "nominal_logits": differences[2],
        },
        "limitations": [
            "Research screening support only; human review is required.",
            "DME is not assessed by this model.",
            "Source validation is not prospective clinical validation.",
        ],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"onnx": str(args.output), "manifest": str(manifest_path), "parity": differences}, indent=2))


if __name__ == "__main__":
    main()
