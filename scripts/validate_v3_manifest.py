#!/usr/bin/env python3
"""Validate the generated RetinaSathi V3 manifest before model training."""

from __future__ import annotations

import argparse
import csv
import json
import hashlib
from collections import Counter, defaultdict
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("runs/v3_data/v3_initial_manifest.csv"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/v3_manifest_validation.json"))
    args = parser.parse_args()
    rows = read_rows(args.manifest)
    eligible = [row for row in rows if row["role"] == "development_pool"]
    errors: list[str] = []

    if len({row["image_uid"] for row in rows}) != len(rows):
        errors.append("image_uid values are not unique")
    if any(row["validation_status"] != "valid" for row in eligible):
        errors.append("eligible development rows include invalid or review-status files")
    if any(row["source_name"] == "IDRiD" and row["original_partition"] == "official_testing" for row in eligible):
        errors.append("IDRiD official test leaked into the development pool")
    if any(not row["cv_fold"] for row in eligible):
        errors.append("eligible development rows are missing cv_fold")
    if any(row["cv_fold"] for row in rows if row["role"] != "development_pool"):
        errors.append("non-development rows received cv_fold assignments")

    sha_groups: dict[str, list[str]] = defaultdict(list)
    patient_folds: dict[str, set[str]] = defaultdict(set)
    for row in eligible:
        sha_groups[row["sha256"]].append(row["image_uid"])
        patient_folds[row["patient_group_id"]].add(row["cv_fold"])
    duplicate_sha = {sha: uids for sha, uids in sha_groups.items() if len(uids) > 1}
    split_groups = {group: sorted(folds) for group, folds in patient_folds.items() if len(folds) > 1}
    if duplicate_sha:
        errors.append("exact duplicate images remain more than once in the eligible pool")
    if split_groups:
        errors.append("patient/group identifiers cross development folds")

    deep_train_groups = {
        row["patient_group_id"]
        for row in rows
        if row["source_name"] == "DeepDRiD" and row["role"] == "development_pool"
    }
    deep_validation_groups = {
        row["patient_group_id"]
        for row in rows
        if row["source_name"] == "DeepDRiD" and row["role"] == "source_validation"
    }
    deep_overlap = sorted(deep_train_groups & deep_validation_groups)
    if deep_overlap:
        errors.append("DeepDRiD patient IDs overlap between development and source validation")

    report = {
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "manifest_sha256": sha256_file(args.manifest),
        "records": len(rows),
        "eligible_development_records": len(eligible),
        "eligible_sources": dict(sorted(Counter(row["source_name"] for row in eligible).items())),
        "eligible_grades": dict(sorted(Counter(row["mapped_icdr_grade"] for row in eligible).items())),
        "eligible_referable": dict(sorted(Counter(row["referable_label"] for row in eligible).items())),
        "folds": dict(sorted(Counter(row["cv_fold"] for row in eligible).items())),
        "eligible_exact_duplicate_groups": len(duplicate_sha),
        "groups_crossing_folds": len(split_groups),
        "deepdrid_train_validation_patient_overlap": deep_overlap,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
