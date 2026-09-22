#!/usr/bin/env python3
"""Create deterministic leakage-free local manifests and a shareable summary."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


SEED = 26038


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return [{str(key).strip(): str(value).strip() for key, value in row.items() if key} for row in csv.DictReader(handle)]


def stratified_split(rows: list[dict[str, object]], label_key: str, validation_fraction: float, seed: int) -> list[dict[str, object]]:
    by_class: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_class[int(row[label_key])].append(row)
    rng = random.Random(seed)
    result: list[dict[str, object]] = []
    for label, members in sorted(by_class.items()):
        ordered = sorted(members, key=lambda item: str(item["image_id"]))
        rng.shuffle(ordered)
        validation_count = max(1, round(len(ordered) * validation_fraction))
        validation_ids = {str(item["image_id"]) for item in ordered[:validation_count]}
        for item in sorted(ordered, key=lambda value: str(value["image_id"])):
            result.append({**item, "split": "validation" if str(item["image_id"]) in validation_ids else "train"})
    return sorted(result, key=lambda item: (str(item["split"]), str(item["image_id"])))


def distribution(rows: list[dict[str, object]], key: str) -> dict[str, dict[str, int]]:
    output: dict[str, dict[str, int]] = {}
    for split in ("train", "validation"):
        output[split] = dict(sorted(Counter(str(row[key]) for row in rows if row["split"] == split).items()))
    return output


def manifest_sha(rows: list[dict[str, object]]) -> str:
    normalized = "\n".join(json.dumps(row, sort_keys=True, separators=(",", ":")) for row in rows)
    return hashlib.sha256(normalized.encode()).hexdigest()


def write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader(); writer.writerows(rows)


def validate_images(rows: list[dict[str, object]], directory: Path, suffix: str) -> list[str]:
    return [str(row["image_id"]) for row in rows if not (directory / f"{row['image_id']}{suffix}").is_file()]


def create(data_root: Path, output_dir: Path) -> dict[str, object]:
    aptos_labels = data_root / "APTOS_2019/extracted/train.csv"
    aptos_source = read_rows(aptos_labels)
    aptos_rows = [{"dataset": "APTOS_2019", "image_id": row["id_code"], "grade": int(row["diagnosis"])} for row in aptos_source]
    aptos_split = stratified_split(aptos_rows, "grade", 0.20, SEED)

    idrid_labels = data_root / "IDRiD/grading/B. Disease Grading/2. Groundtruths/a. IDRiD_Disease Grading_Training Labels.csv"
    idrid_test_labels = data_root / "IDRiD/grading/B. Disease Grading/2. Groundtruths/b. IDRiD_Disease Grading_Testing Labels.csv"
    idrid_source = read_rows(idrid_labels)
    idrid_rows = [
        {
            "dataset": "IDRiD",
            "image_id": row["Image name"],
            "grade": int(row["Retinopathy grade"]),
            "dme": int(row["Risk of macular edema"]),
        }
        for row in idrid_source
    ]
    idrid_split = stratified_split(idrid_rows, "grade", 0.20, SEED)

    regular_root = data_root / "DeepDRiD/original_extracted/DeepDRiD-1.1/regular_fundus_images"
    deep_train_path = regular_root / "regular-fundus-training/regular-fundus-training.csv"
    deep_validation_path = regular_root / "regular-fundus-validation/regular-fundus-validation.csv"
    deep_train = read_rows(deep_train_path)
    deep_validation = read_rows(deep_validation_path)
    train_patients = {row["patient_id"] for row in deep_train}
    validation_patients = {row["patient_id"] for row in deep_validation}

    idrid_segmentation_images = data_root / "IDRiD/images/A. Segmentation/1. Original Images/a. Training Set"
    segmentation_ids = sorted(path.stem for path in idrid_segmentation_images.glob("*.jpg"))
    shuffled_segmentation = segmentation_ids.copy(); random.Random(SEED).shuffle(shuffled_segmentation)
    segmentation_validation = set(shuffled_segmentation[: max(1, round(0.20 * len(shuffled_segmentation)))])
    segmentation_rows = [
        {"dataset": "IDRiD_SEGMENTATION", "image_id": image_id, "split": "validation" if image_id in segmentation_validation else "train"}
        for image_id in segmentation_ids
    ]
    drive_images = data_root / "DRIVE/training/images/training/images"
    drive_ids = sorted(path.name.split("_")[0] for path in drive_images.glob("*_training.tif"))
    shuffled_drive = drive_ids.copy(); random.Random(SEED).shuffle(shuffled_drive)
    drive_validation = set(shuffled_drive[:4])
    drive_rows = [
        {"dataset": "DRIVE", "image_id": image_id, "split": "validation" if image_id in drive_validation else "train"}
        for image_id in drive_ids
    ]

    write_manifest(output_dir / "aptos_train_validation.csv", aptos_split)
    write_manifest(output_dir / "idrid_internal_train_validation.csv", idrid_split)
    write_manifest(output_dir / "idrid_segmentation_train_validation.csv", segmentation_rows)
    write_manifest(output_dir / "drive_internal_train_validation.csv", drive_rows)
    summary = {
        "schema_version": "1.0",
        "seed": SEED,
        "validation_fraction": 0.20,
        "local_manifest_directory": "runs/splits (gitignored; regenerate from licensed source labels)",
        "aptos": {
            "source_label_sha256": file_sha256(aptos_labels),
            "manifest_sha256": manifest_sha(aptos_split),
            "records": len(aptos_split),
            "grade_distribution": distribution(aptos_split, "grade"),
            "missing_images": validate_images(aptos_split, data_root / "APTOS_2019/extracted/train_images", ".png"),
            "policy": "Stratified split of labelled Kaggle training partition only",
        },
        "idrid": {
            "source_label_sha256": file_sha256(idrid_labels),
            "manifest_sha256": manifest_sha(idrid_split),
            "records": len(idrid_split),
            "grade_distribution": distribution(idrid_split, "grade"),
            "dme_distribution": distribution(idrid_split, "dme"),
            "missing_images": validate_images(idrid_split, data_root / "IDRiD/images/B. Disease Grading/1. Original Images/a. Training Set", ".jpg"),
            "official_test": {
                "status": "locked_not_used_for_selection",
                "records": len(read_rows(idrid_test_labels)),
                "source_label_sha256": file_sha256(idrid_test_labels),
            },
            "patient_split_status": "patient identifiers unavailable in grading labels; image-level stratification documented",
        },
        "deepdrid_quality": {
            "policy": "Use official regular-fundus training/validation partitions grouped by patient_id",
            "train_records": len(deep_train),
            "validation_records": len(deep_validation),
            "train_patients": len(train_patients),
            "validation_patients": len(validation_patients),
            "patient_overlap": sorted(train_patients & validation_patients),
            "source_hashes": {
                "train": file_sha256(deep_train_path),
                "validation": file_sha256(deep_validation_path),
            },
        },
        "idrid_segmentation": {
            "records": len(segmentation_rows),
            "train": sum(row["split"] == "train" for row in segmentation_rows),
            "validation": sum(row["split"] == "validation" for row in segmentation_rows),
            "manifest_sha256": manifest_sha(segmentation_rows),
            "official_test": {"status": "locked_not_used_for_selection", "records": 27},
        },
        "drive_vessels": {
            "records": len(drive_rows),
            "train": sum(row["split"] == "train" for row in drive_rows),
            "validation": sum(row["split"] == "validation" for row in drive_rows),
            "manifest_sha256": manifest_sha(drive_rows),
            "official_test": {"status": "locked_not_used_for_selection", "records": 20},
        },
    }
    if summary["aptos"]["missing_images"] or summary["idrid"]["missing_images"] or summary["deepdrid_quality"]["patient_overlap"]:  # type: ignore[index]
        raise ValueError("Dataset validation failed; inspect missing images or patient overlap")
    return summary


def render_markdown(summary: dict[str, object]) -> str:
    aptos = summary["aptos"]  # type: ignore[index]
    idrid = summary["idrid"]  # type: ignore[index]
    deep = summary["deepdrid_quality"]  # type: ignore[index]
    segmentation = summary["idrid_segmentation"]  # type: ignore[index]
    drive = summary["drive_vessels"]  # type: ignore[index]
    return f"""# Data Integrity and Split Validation

Generated by `scripts/create_splits.py`. Raw images and licensed row-level label manifests remain local and gitignored.

| Dataset | Strategy | Result |
|---|---|---|
| APTOS 2019 | Seeded, class-stratified 80/20 split of labelled training data | {aptos['records']} records; 0 missing images |
| IDRiD grading | Seeded, grade-stratified 80/20 split of official training only | {idrid['records']} records; 0 missing images |
| IDRiD official test | Locked until architecture, thresholds and calibration are frozen | {idrid['official_test']['records']} records; not used for V2 selection |
| DeepDRiD quality | Official regular-fundus train/validation, grouped by `patient_id` | {deep['train_patients']} train patients; {deep['validation_patients']} validation patients; 0 overlap |
| IDRiD segmentation | Seeded 80/20 split of official training images | {segmentation['train']} train; {segmentation['validation']} validation; official 27-image test locked |
| DRIVE vessels | Seeded 16/4 split of official training images | {drive['train']} train; {drive['validation']} validation; official 20-image test locked |

## Reproducibility

- Seed: `{summary['seed']}`
- Validation fraction: `{summary['validation_fraction']}`
- Exact local manifests: `runs/splits/` (regenerated, never committed)
- Shareable source/manifest hashes and class distributions: `artifacts/split_summary.json`
- IDRiD grading labels do not expose patient IDs; this limitation is explicit rather than guessed from filenames.

## Safety gates

- No official IDRiD test record enters a training or validation manifest.
- All labelled APTOS/IDRiD records resolve to an image.
- DeepDRiD patient sets are disjoint.
- Original files are read only; processed caches must be written outside raw/canonical roots.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/splits"))
    parser.add_argument("--summary", type=Path, default=Path("artifacts/split_summary.json"))
    parser.add_argument("--docs-output", type=Path, default=Path("docs/DATA_VALIDATION.md"))
    args = parser.parse_args()
    summary = create(args.data_root.resolve(), args.output_dir)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n")
    args.docs_output.write_text(render_markdown(summary))
    print(json.dumps({"aptos": summary["aptos"]["records"], "idrid": summary["idrid"]["records"], "deepdrid_overlap": len(summary["deepdrid_quality"]["patient_overlap"])}))


if __name__ == "__main__":
    main()
