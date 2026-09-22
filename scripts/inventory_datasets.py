#!/usr/bin/env python3
"""Create a deterministic, privacy-safe inventory of the RetinaSathi datasets.

The scanner never opens image pixels, mutates source data, or hashes large archives.
Paths written to the manifest are relative to the configured dataset root so the
artifact can be shared without exposing a contributor's home directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, UnidentifiedImageError


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".ppm"}
MASK_SUFFIXES = {".gif"}
ANNOTATION_SUFFIXES = {".csv", ".xls", ".xlsx"}
ARCHIVE_SUFFIXES = {".zip", ".7z", ".tar", ".gz"}


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    canonical_roots: tuple[str, ...]
    purpose: str
    access: str
    licence: str
    split_policy: str


SPECS = (
    DatasetSpec(
        "APTOS_2019",
        ("extracted", "raw"),
        "DR classification (grades 0-4)",
        "Kaggle authentication and accepted competition rules",
        "Kaggle competition terms; do not redistribute",
        "Create train/validation from labelled training data; Kaggle test labels are unavailable",
    ),
    DatasetSpec(
        "DeepDRiD",
        ("original_extracted/DeepDRiD-1.1", "raw"),
        "Image quality and DR grading",
        "Official DeepDRiD release",
        "Bundled CC BY-SA 4.0 licence",
        "Respect official training/validation/evaluation partitions and patient identifiers",
    ),
    DatasetSpec(
        "IDRiD",
        ("images", "masks", "grading"),
        "DR/DME grading, lesion segmentation, optic-disc and fovea localization",
        "Official IEEE DataPort / challenge release",
        "CC BY 4.0 stated by the official release",
        "Official test is locked; derive V2 train/validation only from official training",
    ),
    DatasetSpec(
        "DRIVE",
        ("training", "test"),
        "Retinal vessel segmentation",
        "DRIVE Grand Challenge membership",
        "Official site terms",
        "Use official training images for fitting; report official test only after model lock",
    ),
    DatasetSpec(
        "Messidor",
        ("images", "annotations"),
        "External DR validation (grades 0-3) and macular-edema risk (0-2)",
        "ADCIS form and email verification",
        "Research/education access; do not redistribute",
        "External validation only; never merge labels blindly with a 0-4 training scale",
    ),
    DatasetSpec(
        "EyePACS",
        ("raw",),
        "Optional large-scale DR classification (grades 0-4)",
        "Kaggle authentication and accepted competition rules",
        "Kaggle competition terms; do not redistribute",
        "Opt-in only; archive remains unextracted when disk headroom is insufficient",
    ),
)

GRADE_SCALES = {
    "APTOS_2019": "DR 0-4",
    "DeepDRiD": "DR 0-4; quality fields retain official scales",
    "IDRiD": "DR 0-4; DME risk 0-2",
    "DRIVE": "not applicable",
    "Messidor": "DR 0-3; macular-edema risk 0-2",
    "EyePACS": "DR 0-4",
}

PATIENT_IDS = {
    "APTOS_2019": "not present in verified label columns",
    "DeepDRiD": "yes: explicit patient_id in official CSV files",
    "IDRiD": "not present in verified grading label columns",
    "DRIVE": "not present in verified file layout",
    "Messidor": "not verified in legacy workbook labels",
    "EyePACS": "unknown while archive remains compressed",
}

DUPLICATE_RISKS = {
    "APTOS_2019": "Extracted images plus source archive; canonical scan reports both roles but does not double-count archive contents",
    "DeepDRiD": "Source archive and extracted release coexist; image counts use extracted files only",
    "IDRiD": "The same fundus may appear in grading, segmentation, and localization tasks; task files are not unique patients",
    "DRIVE": "One photograph has image, FOV, and one or two manual masks; file count is not photograph count",
    "Messidor": "Original extraction and normalized image views coexist; canonical image view only",
    "EyePACS": "Archive contents uninspected; left/right eye and patient grouping must be handled after extraction",
}


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def iter_files(paths: Iterable[Path]) -> Iterable[Path]:
    for root in paths:
        if root.is_file():
            yield root
        elif root.exists():
            yield from (path for path in root.rglob("*") if path.is_file())


def classify(path: Path) -> str:
    suffix = path.suffix.lower()
    parts = {part.lower() for part in path.parts}
    if suffix in ANNOTATION_SUFFIXES:
        return "annotation"
    if suffix in ARCHIVE_SUFFIXES or path.name.lower().endswith(".tar.gz"):
        return "archive"
    if suffix in MASK_SUFFIXES or "masks" in parts or any("groundtruth" in part for part in parts):
        return "mask"
    if suffix in IMAGE_SUFFIXES:
        return "image"
    return "other"


def infer_split(relative_path: Path) -> str:
    value = relative_path.as_posix().lower()
    if any(token in value for token in ("validation", "valid", "val_")):
        return "validation"
    if any(token in value for token in ("testing", "/test", "test_")):
        return "test"
    if any(token in value for token in ("training", "/train", "train_")):
        return "train"
    if "evaluation" in value or "challenge" in value:
        return "evaluation"
    return "unspecified"


def csv_rows(path: Path) -> int | None:
    if path.suffix.lower() != ".csv":
        return None
    try:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            return max(sum(1 for _ in csv.reader(handle)) - 1, 0)
    except (UnicodeDecodeError, OSError):
        return None


def image_dimensions(files: list[Path], dataset_root: Path) -> tuple[dict[str, int], list[str]]:
    dimensions: Counter[str] = Counter()
    invalid: list[str] = []
    for path in files:
        if classify(path) != "image":
            continue
        try:
            with Image.open(path) as image:
                dimensions[f"{image.width}x{image.height}"] += 1
        except (UnidentifiedImageError, OSError):
            invalid.append(path.relative_to(dataset_root).as_posix())
    return dict(dimensions.most_common()), invalid


def known_class_distribution(spec: DatasetSpec, dataset_dir: Path) -> dict[str, object]:
    if spec.name == "APTOS_2019":
        path = dataset_dir / "extracted/train.csv"
        rows = read_csv_dicts(path)
        return {"label_partition": "train", "dr_grade": dict(sorted(Counter(row["diagnosis"] for row in rows).items()))}
    if spec.name == "IDRiD":
        path = dataset_dir / "grading/B. Disease Grading/2. Groundtruths/a. IDRiD_Disease Grading_Training Labels.csv"
        rows = read_csv_dicts(path)
        return {
            "label_partition": "official training only",
            "dr_grade": dict(sorted(Counter(row["Retinopathy grade"] for row in rows).items())),
            "dme_risk": dict(sorted(Counter(row["Risk of macular edema"] for row in rows).items())),
        }
    if spec.name == "DeepDRiD":
        base = dataset_dir / "original_extracted/DeepDRiD-1.1/regular_fundus_images"
        rows = read_csv_dicts(base / "regular-fundus-training/regular-fundus-training.csv")
        return {
            "label_partition": "official regular-fundus training",
            "overall_quality": dict(sorted(Counter(row["Overall quality"] for row in rows).items())),
            "patient_dr_level": dict(sorted(Counter(row["patient_DR_Level"] for row in rows).items())),
            "patients": len({row["patient_id"] for row in rows}),
        }
    return {"status": "not available from a verified plain-text label file"}


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return [
            {str(key).strip(): str(value).strip() for key, value in row.items() if key}
            for row in csv.DictReader(handle)
        ]


def scan_dataset(dataset_root: Path, spec: DatasetSpec) -> dict[str, object]:
    dataset_dir = dataset_root / spec.name
    roots = [dataset_dir / relative for relative in spec.canonical_roots]
    files = sorted(iter_files(roots))
    kinds = Counter(classify(path) for path in files)
    splits: dict[str, Counter[str]] = {}
    for path in files:
        split = infer_split(path.relative_to(dataset_dir))
        splits.setdefault(split, Counter())[classify(path)] += 1

    annotations = []
    for path in files:
        if classify(path) != "annotation":
            continue
        annotations.append(
            {
                "path": path.relative_to(dataset_root).as_posix(),
                "bytes": path.stat().st_size,
                "rows": csv_rows(path),
                "sha256": sha256(path),
            }
        )

    archives = [
        {
            "path": path.relative_to(dataset_root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": None,
            "hash_policy": "skipped_for_large_binary",
        }
        for path in files
        if classify(path) == "archive"
    ]
    total_bytes = sum(path.stat().st_size for path in files)
    dimensions, invalid_images = image_dimensions(files, dataset_root)
    return {
        "name": spec.name,
        "present": dataset_dir.is_dir(),
        "purpose": spec.purpose,
        "access": spec.access,
        "licence": spec.licence,
        "split_policy": spec.split_policy,
        "grade_scale": GRADE_SCALES[spec.name],
        "patient_identifiers": PATIENT_IDS[spec.name],
        "annotation_types": sorted({path.suffix.lower().lstrip(".") for path in files if classify(path) in {"annotation", "mask"}}),
        "class_distribution": known_class_distribution(spec, dataset_dir),
        "duplicate_risk": DUPLICATE_RISKS[spec.name],
        "canonical_roots": list(spec.canonical_roots),
        "counts": {"files": len(files), **dict(sorted(kinds.items()))},
        "split_counts": {
            split: dict(sorted(counts.items())) for split, counts in sorted(splits.items())
        },
        "bytes": total_bytes,
        "gib": round(total_bytes / 1024**3, 3),
        "image_dimensions": dimensions,
        "invalid_images": invalid_images,
        "missing_files": [relative for relative, root in zip(spec.canonical_roots, roots) if not root.exists()],
        "annotations": annotations,
        "archives": archives,
    }


def build_inventory(dataset_root: Path) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "dataset_root": "${RETINASATHI_DATA_ROOT}",
        "scanner_policy": {
            "read_only": True,
            "paths_are_relative": True,
            "large_archive_hashes": "skipped",
            "counts_use_canonical_roots_only": True,
        },
        "datasets": [scan_dataset(dataset_root, spec) for spec in SPECS],
    }


def render_markdown(inventory: dict[str, object]) -> str:
    rows = []
    for dataset in inventory["datasets"]:  # type: ignore[index]
        counts = dataset["counts"]
        status = "Present" if dataset["present"] else "Missing"
        rows.append(
            f"| {dataset['name']} | {status} | {counts.get('image', 0)} | "
            f"{counts.get('mask', 0)} | {counts.get('annotation', 0)} | "
            f"{counts.get('archive', 0)} | {dataset['gib']:.3f} |"
        )
    details = []
    for dataset in inventory["datasets"]:  # type: ignore[index]
        split_text = ", ".join(
            f"{name}: " + "/".join(f"{kind}={count}" for kind, count in counts.items())
            for name, counts in dataset["split_counts"].items()
        )
        details.extend(
            [
                f"### {dataset['name']}",
                "",
                f"- Purpose: {dataset['purpose']}",
                f"- Access/licence: {dataset['access']}; {dataset['licence']}",
                f"- Split rule: **{dataset['split_policy']}**",
                f"- Grade scale: {dataset['grade_scale']}",
                f"- Patient identifiers: {dataset['patient_identifiers']}",
                f"- Duplicate risk: {dataset['duplicate_risk']}",
                f"- Canonical scan roots: {', '.join(dataset['canonical_roots'])}",
                f"- Image dimensions (count): {', '.join(f'{size} ({count})' for size, count in dataset['image_dimensions'].items()) or 'not inspected'}",
                f"- Invalid image headers: {len(dataset['invalid_images'])}",
                f"- Observed splits: {split_text or 'none'}",
                "",
            ]
        )
    return "\n".join(
        [
            "# Dataset Inventory",
            "",
            "Generated by `scripts/inventory_datasets.py` from local files. Counts are file-role counts, not patient counts. Canonical roots deliberately exclude duplicate extraction layouts. Raw datasets are read-only and are never committed or redistributed.",
            "",
            "| Dataset | Status | Images | Masks | Annotations | Archives | Canonical GiB |",
            "|---|---:|---:|---:|---:|---:|---:|",
            *rows,
            "",
            "## Leakage and interpretation rules",
            "",
            "- IDRiD official test data is locked until architecture, thresholds, and calibration are frozen.",
            "- Messidor is external validation; its 0-3 scale must be mapped explicitly before comparison with 0-4 models.",
            "- DeepDRiD patient/eye identifiers must drive grouping so related images cannot cross splits.",
            "- DRIVE's 40 photographs produce multiple image/mask files; file counts must not be reported as patient or photograph counts.",
            "- EyePACS stays opt-in and compressed until enough disk exists for extraction plus working headroom.",
            "",
            "## Dataset details",
            "",
            *details,
            "## Reproduce",
            "",
            "```bash",
            "python scripts/inventory_datasets.py --data-root /path/to/SIH26038_Datasets",
            "```",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, default=Path("artifacts/dataset_inventory.json"))
    parser.add_argument("--markdown-output", type=Path, default=Path("docs/DATASET_INVENTORY.md"))
    args = parser.parse_args()
    if not args.data_root.is_dir():
        raise SystemExit(f"Dataset root not found: {args.data_root}")
    inventory = build_inventory(args.data_root.resolve())
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")
    args.markdown_output.write_text(render_markdown(inventory), encoding="utf-8")
    print(f"Wrote {args.json_output} and {args.markdown_output}")


if __name__ == "__main__":
    main()
