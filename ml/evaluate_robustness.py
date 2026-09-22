"""Evaluate a fixed classifier on internal-validation perturbations only."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter

from compute.model_runtime import RetinaSathiPredictor
from ml.evaluation import classifier_metrics
from ml.artifact_integrity import sha256
from ml.training.train_classifier_v2 import RetinalTransform, build_model, device_for_training, grade_probabilities


def perturbations() -> dict[str, Callable[[Image.Image], Image.Image]]:
    return {
        "normal": lambda image: image,
        "dark": lambda image: ImageEnhance.Brightness(image).enhance(0.42),
        "overexposed": lambda image: ImageEnhance.Brightness(image).enhance(1.75),
        "blurred": lambda image: image.filter(ImageFilter.GaussianBlur(7.0)),
        "low_contrast": lambda image: ImageEnhance.Contrast(image).enhance(0.38),
        "resolution_256": lambda image: image.resize((256, 256), Image.Resampling.LANCZOS),
        "resolution_1024": lambda image: image.resize((1024, 1024), Image.Resampling.LANCZOS),
        "partially_cropped": partial_crop,
    }


def partial_crop(image: Image.Image) -> Image.Image:
    output = image.copy()
    width, height = output.size
    output.paste((0, 0, 0), (0, 0, round(width * 0.28), height))
    return output


def validation_rows(split_path: Path) -> list[dict[str, str]]:
    with split_path.open(encoding="utf-8", newline="") as handle:
        return [row for row in csv.DictReader(handle) if row["split"] == "validation"]


def add_prediction_consistency(groups: dict[str, dict[str, object]]) -> None:
    """Compare every perturbation with normal before removing private predictions."""
    normal_predictions = np.asarray(groups["normal"]["predicted_grades"])
    for group in groups.values():
        predictions = np.asarray(group.pop("predicted_grades"))
        group["prediction_consistency_vs_normal"] = float(
            (predictions == normal_predictions).mean()
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", type=Path, default=Path("runs/splits/idrid_internal_train_validation.csv"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"Robustness artifact already exists: {args.output}")
    calibration = json.loads(args.validation.read_text())
    if calibration.get("official_test_used") is not False or calibration.get("checkpoint_sha256") != sha256(args.checkpoint):
        raise SystemExit("Validation/checkpoint provenance mismatch")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = state["config"]
    objective = str(config["training"]["objective"])
    temperature = float(calibration["temperature"])
    threshold = float(calibration["referable_threshold"])
    model = build_model(config, pretrained=False)
    model.load_state_dict(state["model"])
    device = device_for_training(); model.to(device).eval()
    transform = RetinalTransform(config, False)
    rows = validation_rows(args.split)
    image_root = args.data_root / "IDRiD/images/B. Disease Grading/1. Original Images/a. Training Set"
    groups: dict[str, dict[str, object]] = {}
    for name, operation in perturbations().items():
        truth: list[int] = []; probabilities: list[np.ndarray] = []; rejected = 0
        for row in rows:
            with Image.open(image_root / f"{row['image_id']}.jpg") as source:
                image = operation(source.convert("RGB"))
            if RetinaSathiPredictor.quality(image)["label"] == "poor":
                rejected += 1
            tensor = transform(image).unsqueeze(0).to(device)
            with torch.inference_mode():
                logits, _ = model(tensor)
            truth.append(int(row["grade"]))
            probabilities.append(grade_probabilities(logits.cpu(), objective, temperature)[0].numpy())
        truth_array = np.asarray(truth)
        probability_array = np.stack(probabilities)
        metrics = classifier_metrics(truth_array, probability_array, threshold)
        groups[name] = {
            "samples": len(rows),
            "quality_gate_rejected": rejected,
            "quality_gate_rejection_rate": rejected / len(rows),
            "prediction_consistency_vs_normal": None,
            "uncertain_count_confidence_below_0_60": int((probability_array.max(axis=1) < 0.60).sum()),
            "metrics": metrics,
            "predicted_grades": probability_array.argmax(axis=1).tolist(),
        }
    add_prediction_consistency(groups)
    result = {
        "status": "internal_validation_robustness_only",
        "checkpoint_sha256": sha256(args.checkpoint),
        "official_test_used": False,
        "groups": groups,
        "synthetic_api_safety_cases": {
            "black_frame": "covered_by_compute_test",
            "non_fundus": "covered_by_compute_test",
            "corrupt_image": "covered_by_api_test",
            "large_file": "covered_by_api_test",
        },
        "limitations": [
            "Perturbations are deterministic simulations, not prospective capture failures.",
            "Classifier metrics include all images; quality-gate rejection is reported separately.",
            "No official IDRiD test or Messidor image is accessed by this script.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "groups": len(groups), "official_test_used": False}))


if __name__ == "__main__":
    main()
