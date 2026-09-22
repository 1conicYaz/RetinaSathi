#!/usr/bin/env python3
"""Evaluate one V3 checkpoint on validation, calibration, and source-validation splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from ml.calibration import fit_binary_temperature, fit_temperature
from ml.datasets import V3ManifestDataset
from ml.evaluation import select_safety_threshold
from ml.models import DualHeadDRClassifier
from ml.training.train_classifier_v2 import RetinalTransform, device_for_training
from ml.training.train_classifier_v3 import (
    calibrated_evaluation,
    make_loader,
    ordinal_positive_weights,
    referable_positive_weight,
    report_view,
    run_epoch,
    source_breakdown,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    config = yaml.safe_load((args.run_dir / "config.yaml").read_text(encoding="utf-8"))
    manifest = Path(config["data"]["manifest"])
    experiment = config["experiment"]
    selection = config["selection"]
    validation_fold = str(experiment["validation_fold"])
    calibration_fold = str(experiment["calibration_fold"])
    training_folds = {str(fold) for fold in range(5)} - {validation_fold, calibration_fold}
    transform = RetinalTransform(config, False)
    train_data = V3ManifestDataset(manifest, args.data_root, transform, {"development_pool"}, training_folds)
    validation_data = V3ManifestDataset(manifest, args.data_root, transform, {"development_pool"}, {validation_fold})
    calibration_data = V3ManifestDataset(manifest, args.data_root, transform, {"development_pool"}, {calibration_fold})
    source_data = V3ManifestDataset(manifest, args.data_root, transform, {"source_validation"})
    device = device_for_training()
    model = DualHeadDRClassifier(
        str(experiment["architecture"]),
        pretrained=False,
        dropout=float(config["training"]["dropout"]),
    ).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    common = {
        "device": device,
        "ordinal_weights": ordinal_positive_weights(train_data, device),
        "referable_weight": referable_positive_weight(train_data, device),
        "ordinal_loss_weight": float(config["training"]["ordinal_loss_weight"]),
        "referable_loss_weight": float(config["training"]["referable_loss_weight"]),
    }
    validation_result = run_epoch(model, make_loader(validation_data, config, False), **common)
    calibration_result = run_epoch(model, make_loader(calibration_data, config, False), **common)
    source_result = run_epoch(model, make_loader(source_data, config, False), **common)
    ordinal_temperature = fit_temperature(
        calibration_result["ordinal_logits"].to(device),
        calibration_result["grades"].to(device),
        objective="coral",
    )
    binary_temperature = fit_binary_temperature(
        calibration_result["binary_logits"].to(device),
        (calibration_result["grades"] >= 2).to(device),
    )
    calibration_at_half = calibrated_evaluation(
        calibration_result, ordinal_temperature, binary_temperature, 0.5
    )
    threshold_gate = select_safety_threshold(
        calibration_result["grades"].numpy() >= 2,
        calibration_at_half["referable_scores"],
        float(selection["minimum_referable_sensitivity"]),
        float(selection["minimum_referable_specificity"]),
    )
    threshold = float(threshold_gate["selected"]["threshold"])
    validation = calibrated_evaluation(validation_result, ordinal_temperature, binary_temperature, threshold)
    calibration = calibrated_evaluation(calibration_result, ordinal_temperature, binary_temperature, threshold)
    source = calibrated_evaluation(source_result, ordinal_temperature, binary_temperature, threshold)
    source_gate = (
        float(source["referable"]["sensitivity"]) >= float(selection["minimum_referable_sensitivity"])
        and float(source["referable"]["specificity"]) >= float(selection["minimum_referable_specificity"])
    )
    report = {
        "checkpoint": str(args.checkpoint),
        "epoch": int(checkpoint["epoch"]),
        "ordinal_temperature": ordinal_temperature,
        "binary_temperature": binary_temperature,
        "calibration_gate": threshold_gate,
        "source_gate_passed": source_gate,
        "uncalibrated": {
            "validation": report_view(validation_result),
            "calibration": report_view(calibration_result),
            "source_validation": report_view(source_result),
        },
        "validation": {
            "grade": validation["grade"],
            "referable": validation["referable"],
            "by_source": source_breakdown(validation_result, validation, threshold),
        },
        "calibration": {
            "grade": calibration["grade"],
            "referable": calibration["referable"],
            "by_source": source_breakdown(calibration_result, calibration, threshold),
        },
        "source_validation": {
            "grade": source["grade"],
            "referable": source["referable"],
            "by_source": source_breakdown(source_result, source, threshold),
        },
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "epoch": report["epoch"],
                "source_gate_passed": source_gate,
                "source_grade": source["grade"],
                "source_referable": source["referable"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
