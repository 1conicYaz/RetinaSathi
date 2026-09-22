"""Export a locked/candidate V2 classifier checkpoint to a validated ONNX artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import onnx
import torch
from torch import nn

from ml.artifact_integrity import (
    require_internal_validation_binding,
    require_locked_evaluation_binding,
    sha256,
)
from ml.training.train_classifier_v2 import build_model


class ExportableClassifierV2(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feature_map = self.model.backbone.features(image)
        pooled = self.model.backbone.pool(feature_map)
        embedding = self.model.backbone.normalization(pooled).flatten(1)
        embedding = self.model.dropout(embedding)
        grade_head = self.model.ordinal_head if hasattr(self.model, "ordinal_head") else self.model.grade_head
        grade_logits = grade_head(embedding)
        dme_logits = self.model.dme_head(embedding) if self.model.dme_head is not None else torch.zeros((image.shape[0], 3), device=image.device)
        return grade_logits, dme_logits, feature_map


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--status", choices=("candidate_not_tested", "locked_tested"), default="candidate_not_tested")
    parser.add_argument("--evaluation", type=Path)
    args = parser.parse_args()
    manifest_path = args.output.with_suffix(".manifest.json")
    if args.output.exists() or manifest_path.exists():
        raise SystemExit(f"Refusing to overwrite export: {args.output}")
    validation = json.loads(args.validation.read_text())
    try:
        checkpoint_digest = require_internal_validation_binding(
            args.checkpoint, validation
        )
    except ValueError as error:
        raise SystemExit(f"Refusing export: {error}") from error
    evaluation_digest = None
    if args.status == "locked_tested":
        if args.evaluation is None:
            raise SystemExit("A locked-tested export requires --evaluation")
        evaluation = json.loads(args.evaluation.read_text())
        try:
            require_locked_evaluation_binding(
                args.checkpoint, evaluation, args.validation
            )
        except ValueError as error:
            raise SystemExit(f"Refusing export: {error}") from error
        evaluation_digest = sha256(args.evaluation)
    elif args.evaluation is not None:
        raise SystemExit("--evaluation is only valid with --status locked_tested")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = state["config"]
    model = build_model(config, pretrained=False)
    model.load_state_dict(state["model"])
    wrapper = ExportableClassifierV2(model.eval()).eval()
    size = int(config["preprocessing"]["input_size"])
    sample = torch.zeros(1, 3, size, size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        torch.onnx.export(wrapper, sample, args.output, input_names=["image"], output_names=["grade_logits", "dme_logits", "feature_map"], opset_version=17, do_constant_folding=True, dynamo=False)
    onnx.checker.check_model(onnx.load(args.output))
    grade_head = model.ordinal_head if hasattr(model, "ordinal_head") else model.grade_head
    exact_linear_gradcam = isinstance(model.backbone.normalization, nn.Identity)
    manifest = {
        "schema_version": "1.0",
        "model_version": args.version,
        "status": args.status,
        "architecture": config["experiment"]["architecture"],
        "datasets": ["APTOS_2019", "IDRiD"] if config["data"]["aptos_pretrain"] else ["IDRiD"],
        "objective": config["training"]["objective"],
        "dme_head": model.dme_head is not None,
        "input_size": size,
        "preprocessing": config["preprocessing"],
        "temperature": validation["temperature"],
        "referable_threshold": validation["referable_threshold"],
        "validation": validation["validation"],
        "checkpoint_sha256": checkpoint_digest,
        "validation_sha256": sha256(args.validation),
        "locked_evaluation_sha256": evaluation_digest,
        "official_test_used": args.status == "locked_tested",
        "onnx_sha256": sha256(args.output),
        "checkpoint_epoch": state["epoch"],
        "explainability": {
            "method": "gradcam_exact_linear_gap" if exact_linear_gradcam else "not_available_for_backbone",
            "target": "expected_ordinal_severity" if config["training"]["objective"] == "coral" else "predicted_grade_logit",
            "grade_head_weights": grade_head.weight.detach().cpu().tolist() if exact_linear_gradcam else None,
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "sha256": manifest["onnx_sha256"], "status": args.status}))


if __name__ == "__main__":
    main()
