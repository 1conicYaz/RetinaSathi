"""Compare true Grad-CAM with real IDRiD masks on the segmentation validation split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ml.datasets.segmentation import LESION_PATHS, manifest_ids
from ml.explainability import GradCAM, attention_mask_metrics, ordinal_severity_score
from ml.artifact_integrity import sha256
from ml.training.train_classifier_v2 import RetinalTransform, build_model, device_for_training, grade_probabilities


def aligned_masks(image: Image.Image, masks: list[Image.Image], size: int) -> tuple[np.ndarray, np.ndarray]:
    rgb = np.asarray(image.convert("RGB")); visible = rgb.max(axis=2) > 8
    rows = np.flatnonzero(visible.any(axis=1)); columns = np.flatnonzero(visible.any(axis=0))
    if rows.size and columns.size:
        padding = round(min(rgb.shape[:2]) * 0.01)
        box = (max(0, int(columns[0]) - padding), max(0, int(rows[0]) - padding), min(rgb.shape[1], int(columns[-1]) + padding + 1), min(rgb.shape[0], int(rows[-1]) + padding + 1))
        image = image.crop(box); masks = [mask.crop(box) for mask in masks]
    square_size = max(image.size)
    image_square = Image.new("RGB", (square_size, square_size)); image_square.paste(image, ((square_size-image.width)//2, (square_size-image.height)//2))
    retina = np.asarray(image_square.resize((size, size), Image.Resampling.NEAREST)).max(axis=2) > 8
    union = Image.new("L", (square_size, square_size))
    offset = ((square_size-image.width)//2, (square_size-image.height)//2)
    for mask in masks:
        canvas = Image.new("L", (square_size, square_size)); canvas.paste(mask, offset)
        union = Image.fromarray(np.maximum(np.asarray(union), np.asarray(canvas)).astype(np.uint8))
    lesion = np.asarray(union.resize((size, size), Image.Resampling.NEAREST)) > 0
    return lesion, retina


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", type=Path, default=Path("runs/splits/idrid_segmentation_train_validation.csv"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"Explainability artifact already exists: {args.output}")
    calibration = json.loads(args.validation.read_text())
    digest = sha256(args.checkpoint)
    if calibration.get("official_test_used") is not False or calibration.get("checkpoint_sha256") != digest:
        raise SystemExit("Validation/checkpoint provenance mismatch")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False); config = state["config"]
    model = build_model(config, pretrained=False); model.load_state_dict(state["model"])
    device = device_for_training(); model.to(device).eval()
    objective = str(config["training"]["objective"]); temperature = float(calibration["temperature"])
    transform = RetinalTransform(config, False); size = int(config["preprocessing"]["input_size"])
    root = args.data_root / "IDRiD/original_extracted/A. Segmentation"
    records = []
    for image_id in manifest_ids(args.split, "validation"):
        image = Image.open(root / f"1. Original Images/a. Training Set/{image_id}.jpg").convert("RGB")
        masks = []
        for folder, suffix in LESION_PATHS.values():
            path = root / f"2. All Segmentation Groundtruths/a. Training Set/{folder}/{image_id}_{suffix}.tif"
            masks.append(Image.open(path).convert("L") if path.exists() else Image.new("L", image.size))
        lesion, retina = aligned_masks(image, masks, size)
        tensor = transform(image).unsqueeze(0).to(device)
        with torch.inference_mode():
            logits, _ = model(tensor)
            probabilities = grade_probabilities(logits.cpu(), objective, temperature)[0].numpy()
        if objective == "coral":
            score_fn = lambda output: ordinal_severity_score(output, temperature)
        else:
            grade = int(probabilities.argmax()); score_fn = lambda output, selected=grade: output[0][:, selected]
        with GradCAM(model, model.gradcam_layer()) as gradcam:
            heatmap = gradcam(tensor, score_fn)[0, 0].cpu().numpy()
        records.append(attention_mask_metrics(heatmap, lesion, retina))
    lesion_records = [record for record in records if record["lesion_present"]]
    result = {
        "status": "experimental_explainability_analysis_not_clinical_proof",
        "checkpoint_sha256": digest,
        "official_test_used": False,
        "split": "IDRiD segmentation training subset held-out validation",
        "samples": len(records),
        "samples_with_any_lesion": len(lesion_records),
        "mean_top_20pct_attention_lesion_iou": float(np.mean([record["top_20pct_attention_lesion_iou"] for record in lesion_records])) if lesion_records else None,
        "pointing_game_accuracy": float(np.mean([record["pointing_game_hit"] for record in lesion_records])) if lesion_records else None,
        "mean_attention_inside_retina_fraction": float(np.mean([record["attention_inside_retina_fraction"] for record in records])),
        "mean_attention_outside_retina_fraction": float(np.mean([record["attention_outside_retina_fraction"] for record in records])),
        "limitations": [
            "Attention is compared with the union of four lesion masks and is not a lesion prediction.",
            "The classifier may have seen some segmentation-subset images during classifier training.",
            "These are spatial sanity checks, not localization sensitivity or clinical validity.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "samples": len(records), "official_test_used": False}))


if __name__ == "__main__":
    main()
