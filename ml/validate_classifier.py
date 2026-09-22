"""Calibrate and evaluate one classifier checkpoint on internal validation only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ml.calibration import fit_temperature
from ml.evaluation import best_referable_threshold, classifier_metrics
from ml.artifact_integrity import sha256
from ml.training.train_classifier_v2 import (
    RetinalTransform,
    build_model,
    device_for_training,
    grade_probabilities,
    make_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, default=Path("runs/splits"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise SystemExit(f"Validation artifact already exists: {args.output}")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = state["config"]
    objective = str(config["training"]["objective"])
    dataset = make_dataset(
        args.data_root,
        args.split_dir,
        "IDRiD",
        "validation",
        RetinalTransform(config, False),
    )
    # These evaluation sets are small; single-process loading avoids macOS
    # shared-memory manager failures and makes the pass deterministic.
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
    device = device_for_training()
    model = build_model(config, pretrained=False)
    model.load_state_dict(state["model"])
    model.to(device).eval()
    logits_parts: list[torch.Tensor] = []
    truth_parts: list[torch.Tensor] = []
    with torch.inference_mode():
        for images, grades, _, _ in loader:
            logits, _ = model(images.to(device))
            logits_parts.append(logits.cpu())
            truth_parts.append(grades)
    logits = torch.cat(logits_parts)
    truth = torch.cat(truth_parts)
    uncalibrated = grade_probabilities(logits, objective).numpy()
    temperature = fit_temperature(logits.to(device), truth.to(device), objective=objective)
    calibrated = grade_probabilities(logits, objective, temperature).numpy()
    threshold = best_referable_threshold(truth.numpy(), calibrated)
    result = {
        "model_status": "candidate_not_tested",
        "checkpoint_sha256": sha256(args.checkpoint),
        "checkpoint_epoch": int(state["epoch"]),
        "checkpoint_phase": str(state["phase"]),
        "temperature": temperature,
        "referable_threshold": threshold,
        "validation_samples": len(dataset),
        "validation_uncalibrated": classifier_metrics(truth.numpy(), uncalibrated),
        "validation": classifier_metrics(truth.numpy(), calibrated, threshold),
        "official_test_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "validation_qwk": result["validation"]["qwk"]}))


if __name__ == "__main__":
    main()
