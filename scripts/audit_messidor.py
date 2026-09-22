"""Validate the licensed-local Messidor image/annotation pairing and label semantics."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd


def referable_from_messidor_grade(grade: int) -> bool:
    """Use the documented external binary endpoint: Messidor grade >=2."""
    if grade not in (0, 1, 2, 3):
        raise ValueError(f"Unexpected Messidor retinopathy grade: {grade}")
    return grade >= 2


def audit(root: Path) -> dict[str, object]:
    annotation_files = sorted((root / "annotations").rglob("*.xls"))
    if len(annotation_files) != 12:
        raise ValueError(f"Expected 12 annotation workbooks, found {len(annotation_files)}")
    frames = [pd.read_excel(path) for path in annotation_files]
    rows = pd.concat(frames, ignore_index=True)
    rows.columns = [str(column).strip() for column in rows.columns]
    required = {"Image name", "Ophthalmologic department", "Retinopathy grade", "Risk of macular edema"}
    if set(rows.columns) != required:
        raise ValueError(f"Unexpected Messidor columns: {list(rows.columns)}")
    names = rows["Image name"].astype(str).tolist()
    grades = [int(value) for value in rows["Retinopathy grade"]]
    edema = [int(value) for value in rows["Risk of macular edema"]]
    available = {path.name for path in (root / "images").rglob("*.tif")}
    missing = sorted(name for name in names if name not in available)
    duplicates = len(names) - len(set(names))
    if missing or duplicates:
        raise ValueError(f"Messidor pairing failed: {len(missing)} missing, {duplicates} duplicates")
    if not set(grades).issubset({0, 1, 2, 3}) or not set(edema).issubset({0, 1, 2}):
        raise ValueError("Messidor contains an unexpected label value")
    departments = Counter(str(value) for value in rows["Ophthalmologic department"])
    return {
        "dataset": "Messidor",
        "annotation_workbooks": len(annotation_files),
        "labelled_images": len(names),
        "unique_image_names": len(set(names)),
        "image_files": len(available),
        "missing_images": len(missing),
        "duplicate_annotations": duplicates,
        "retinopathy_grade_distribution": dict(sorted(Counter(grades).items())),
        "macular_edema_risk_distribution": dict(sorted(Counter(edema).items())),
        "referable_definition": "retinopathy_grade >= 2",
        "referable_distribution": dict(sorted(Counter(referable_from_messidor_grade(grade) for grade in grades).items(), key=lambda item: str(item[0]))),
        "department_distribution": dict(sorted(departments.items())),
        "use_policy": "external_validation_only_after_model_lock",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/messidor_audit.json"))
    args = parser.parse_args()
    result = audit(args.data_root / "Messidor")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
