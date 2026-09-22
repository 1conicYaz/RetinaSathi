from __future__ import annotations

import argparse
from pathlib import Path

import onnx
import torch
from torch import nn

try:
    from ml.model import RetinaSathiNet
except ModuleNotFoundError:  # Preserve direct `python ml/export_onnx.py` usage.
    from model import RetinaSathiNet


class ExportableRetinaSathi(nn.Module):
    def __init__(self, model: RetinaSathiNet) -> None:
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feature_map = self.model.features(image)
        pooled = self.model.avgpool(feature_map).flatten(1)
        embedding = self.model.embedding(pooled)
        return self.model.grade_head(embedding), self.model.dme_head(embedding), feature_map


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the RetinaSathi checkpoint to ONNX.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = RetinaSathiNet(pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    wrapper = ExportableRetinaSathi(model).eval()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            torch.zeros(1, 3, int(checkpoint.get("image_size", 224)), int(checkpoint.get("image_size", 224))),
            args.output,
            input_names=["image"],
            output_names=["grade_logits", "dme_logits", "feature_map"],
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
        )
    onnx.checker.check_model(onnx.load(args.output))
    print(f"Exported and validated {args.output}")


if __name__ == "__main__":
    main()
