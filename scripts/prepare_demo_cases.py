"""Prepare local-only, internal-validation SIH demo cases for a fixed candidate."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Callable

import torch
from PIL import Image, ImageEnhance, ImageFilter

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from compute.model_runtime import RetinaSathiPredictor
from ml.artifact_integrity import sha256
from ml.training.train_classifier_v2 import (
    RetinalTransform,
    build_model,
    device_for_training,
    grade_probabilities,
)


def select_highest_confidence(
    records: list[dict[str, Any]],
    predicate: Callable[[dict[str, Any]], bool],
    label: str,
) -> dict[str, Any]:
    """Select one case while failing clearly if the split lacks that category."""
    candidates = [record for record in records if predicate(record)]
    if not candidates:
        raise SystemExit(f"No {label} case is available in the internal validation split")
    return max(candidates, key=lambda record: record["confidence"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--split",
        type=Path,
        default=Path("runs/splits/idrid_internal_train_validation.csv"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("runs/demo_cases")
    )
    args = parser.parse_args()

    if args.output_dir.exists():
        raise SystemExit(f"Demo directory already exists: {args.output_dir}")

    validation = json.loads(args.validation.read_text())
    checkpoint_digest = sha256(args.checkpoint)
    if (
        validation.get("official_test_used") is not False
        or validation.get("checkpoint_sha256") != checkpoint_digest
    ):
        raise SystemExit("Validation/checkpoint provenance mismatch")

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = state["config"]
    objective = str(config["training"]["objective"])
    temperature = float(validation["temperature"])
    threshold = float(validation["referable_threshold"])
    model = build_model(config, pretrained=False)
    model.load_state_dict(state["model"])
    device = device_for_training()
    model.to(device).eval()
    transform = RetinalTransform(config, training=False)

    with args.split.open(encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["split"] == "validation"]
    if not rows:
        raise SystemExit(f"No internal validation records found in {args.split}")

    image_root = (
        args.data_root
        / "IDRiD/images/B. Disease Grading/1. Original Images/a. Training Set"
    )
    records: list[dict[str, Any]] = []
    for row in rows:
        path = image_root / f"{row['image_id']}.jpg"
        with Image.open(path) as source:
            image = source.convert("RGB")
        with torch.inference_mode():
            logits, _ = model(transform(image).unsqueeze(0).to(device))
        probabilities = grade_probabilities(logits.cpu(), objective, temperature)[0]
        predicted = int(probabilities.argmax())
        confidence = float(probabilities.max())
        referable = float(probabilities[2:].sum()) >= threshold
        quality = RetinaSathiPredictor.quality(image)
        records.append(
            {
                "path": path,
                "truth_grade": int(row["grade"]),
                "predicted_grade": predicted,
                "confidence": confidence,
                "referable": referable,
                "quality": quality["label"],
            }
        )

    gradeable = [record for record in records if record["quality"] != "poor"] or records
    choices = {
        "good_nonreferable": select_highest_confidence(
            gradeable, lambda record: record["truth_grade"] < 2, "non-referable"
        ),
        "referable": select_highest_confidence(
            gradeable, lambda record: record["truth_grade"] >= 2, "referable"
        ),
        "severe": select_highest_confidence(
            gradeable, lambda record: record["truth_grade"] >= 3, "severe"
        ),
        "uncertain": min(gradeable, key=lambda record: record["confidence"]),
    }

    args.output_dir.mkdir(parents=True)
    manifest: dict[str, Any] = {
        "status": "local_demo_cases_not_evaluation",
        "checkpoint_sha256": checkpoint_digest,
        "official_test_used": False,
        "source_split": "internal IDRiD validation",
        "cases": {},
    }
    for name, record in choices.items():
        destination = args.output_dir / f"{name}.jpg"
        shutil.copy2(record["path"], destination)
        public_record = {key: value for key, value in record.items() if key != "path"}
        manifest["cases"][name] = {**public_record, "file": destination.name}

    with Image.open(choices["good_nonreferable"]["path"]) as source:
        poor = ImageEnhance.Brightness(source.convert("RGB")).enhance(0.18)
        poor = poor.filter(ImageFilter.GaussianBlur(12))
    poor.save(args.output_dir / "poor_quality.jpg", quality=90)
    manifest["cases"]["poor_quality"] = {
        "file": "poor_quality.jpg",
        "synthetic_transform": "brightness 0.18 plus Gaussian blur radius 12",
        "expected_action": "retake; no DR/DME output",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "cases": len(manifest["cases"]),
                "official_test_used": False,
            }
        )
    )


if __name__ == "__main__":
    main()
