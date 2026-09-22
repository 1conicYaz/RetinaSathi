#!/usr/bin/env python3
"""Build an auditable, source-aware RetinaSathi V3 grading manifest.

This script never edits raw images. It validates labels and image files, computes
exact and perceptual hashes, assigns deterministic group-aware development folds,
and writes only manifests and aggregate audit reports.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from PIL import Image, UnidentifiedImageError
from scipy.fft import dctn


SEED = 26038
FOLDS = 5
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


@dataclass
class Record:
    image_uid: str
    source_name: str
    source_version: str
    country: str
    original_partition: str
    role: str
    relative_path: str
    patient_group_id: str
    patient_grouping_status: str
    eye: str
    field: str
    original_grade: int
    original_grade_system: str
    mapped_icdr_grade: int
    referable_label: int
    dme_or_maculopathy_label: str
    quality_label: str
    label_source: str
    licence_record: str
    label_alignment_status: str = "not_applicable"
    cv_fold: str = ""
    file_bytes: int = 0
    width: int = 0
    height: int = 0
    image_format: str = ""
    image_mode: str = ""
    sha256: str = ""
    dhash64: str = ""
    phash64: str = ""
    validation_status: str = "pending"
    validation_note: str = ""


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = []
        for raw in csv.DictReader(handle):
            rows.append({str(key).strip(): str(value).strip() for key, value in raw.items() if key})
        return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def dhash64(image: Image.Image) -> str:
    gray = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    pixels = list(gray.get_flattened_data())
    value = 0
    for row in range(8):
        for column in range(8):
            value = (value << 1) | int(pixels[row * 9 + column] > pixels[row * 9 + column + 1])
    return f"{value:016x}"


def phash64(image: Image.Image) -> str:
    gray = image.convert("L").resize((32, 32), Image.Resampling.LANCZOS)
    coefficients = dctn(gray, type=2, norm="ortho")[:8, :8]
    flattened = coefficients.flatten()
    median = sorted(float(value) for value in flattened[1:])[len(flattened[1:]) // 2]
    value = 0
    for coefficient in flattened:
        value = (value << 1) | int(float(coefficient) > median)
    return f"{value:016x}"


def validate_record(record: Record, root: Path) -> None:
    path = root / record.relative_path
    if not path.is_file():
        record.validation_status = "missing"
        record.validation_note = "Label row does not resolve to an image file."
        return
    if path.suffix.lower() not in IMAGE_SUFFIXES:
        record.validation_status = "unsupported_extension"
        record.validation_note = path.suffix
        return
    try:
        record.file_bytes = path.stat().st_size
        record.sha256 = sha256_file(path)
        with Image.open(path) as image:
            image.load()
            record.width, record.height = image.size
            record.image_format = str(image.format or "")
            record.image_mode = image.mode
            record.dhash64 = dhash64(image)
            record.phash64 = phash64(image)
        if min(record.width, record.height) < 256:
            record.validation_status = "review"
            record.validation_note = "Image dimension below 256 pixels."
        else:
            record.validation_status = "valid"
    except (OSError, ValueError, UnidentifiedImageError) as error:
        record.validation_status = "invalid"
        record.validation_note = f"{type(error).__name__}: {error}"


def aptos_records(root: Path) -> list[Record]:
    labels = root / "APTOS_2019/extracted/train.csv"
    records = []
    for row in read_csv(labels):
        image_id = row["id_code"]
        grade = int(row["diagnosis"])
        records.append(
            Record(
                image_uid=f"aptos2019:train:{image_id}",
                source_name="APTOS_2019",
                source_version="Kaggle_2019",
                country="India",
                original_partition="labelled_train",
                role="development_pool",
                relative_path=f"APTOS_2019/extracted/train_images/{image_id}.png",
                patient_group_id=f"aptos2019:unknown:{image_id}",
                patient_grouping_status="patient_id_unavailable_image_level_group",
                eye="unknown",
                field="unknown",
                original_grade=grade,
                original_grade_system="APTOS_0_4",
                mapped_icdr_grade=grade,
                referable_label=int(grade >= 2),
                dme_or_maculopathy_label="unknown",
                quality_label="unknown",
                label_source="APTOS_2019/extracted/train.csv",
                licence_record="Kaggle competition terms; do not redistribute",
            )
        )
    return records


def idrid_records(root: Path, partition: str) -> list[Record]:
    if partition == "training":
        labels_rel = "IDRiD/grading/B. Disease Grading/2. Groundtruths/a. IDRiD_Disease Grading_Training Labels.csv"
        images_rel = "IDRiD/images/B. Disease Grading/1. Original Images/a. Training Set"
        role = "development_pool"
    else:
        labels_rel = "IDRiD/grading/B. Disease Grading/2. Groundtruths/b. IDRiD_Disease Grading_Testing Labels.csv"
        images_rel = "IDRiD/images/B. Disease Grading/1. Original Images/b. Testing Set"
        role = "historical_test_excluded_from_v3_selection"
    records = []
    for row in read_csv(root / labels_rel):
        image_id = row["Image name"]
        grade = int(row["Retinopathy grade"])
        dme = row.get("Risk of macular edema", "unknown")
        records.append(
            Record(
                image_uid=f"idrid:{partition}:{image_id}",
                source_name="IDRiD",
                source_version="official_challenge_release",
                country="India",
                original_partition=f"official_{partition}",
                role=role,
                relative_path=f"{images_rel}/{image_id}.jpg",
                patient_group_id=f"idrid:{partition}:unknown:{image_id}",
                patient_grouping_status="patient_id_unavailable_image_level_group",
                eye="unknown",
                field="unknown",
                original_grade=grade,
                original_grade_system="ICDR_0_4",
                mapped_icdr_grade=grade,
                referable_label=int(grade >= 2),
                dme_or_maculopathy_label=dme,
                quality_label="unknown",
                label_source=labels_rel,
                licence_record="Official IDRiD release; CC BY 4.0 stated by source",
            )
        )
    return records


def deepdrid_records(root: Path, partition: str) -> list[Record]:
    base = Path("DeepDRiD/original_extracted/DeepDRiD-1.1/regular_fundus_images")
    folder = f"regular-fundus-{partition}"
    labels_rel = base / folder / f"{folder}.csv"
    records = []
    for row in read_csv(root / labels_rel):
        image_id = row["image_id"]
        patient_id = row["patient_id"]
        parts = image_id.rsplit("_", 1)
        eye_field = parts[-1].lower() if len(parts) == 2 else ""
        eye = "left" if eye_field.startswith("l") else "right" if eye_field.startswith("r") else "unknown"
        field = eye_field[1:] if len(eye_field) > 1 else "unknown"
        labelled_eyes = [
            (label_eye, row[column])
            for label_eye, column in (("left", "left_eye_DR_Level"), ("right", "right_eye_DR_Level"))
            if row[column]
        ]
        if len(labelled_eyes) == 1:
            labelled_eye, grade_text = labelled_eyes[0]
            grade = int(grade_text)
            label_alignment_status = "aligned" if labelled_eye == eye else "filename_eye_label_mismatch"
        else:
            grade = int(row["patient_DR_Level"])
            label_alignment_status = "missing_or_ambiguous_eye_grade_patient_grade_for_audit_only"
        if label_alignment_status == "aligned":
            role = "development_pool" if partition == "training" else "source_validation"
        else:
            role = "quarantine_label_alignment"
        relative_path = base / folder / "Images" / patient_id / f"{image_id}.jpg"
        records.append(
            Record(
                image_uid=f"deepdrid1.1:{partition}:{image_id}",
                source_name="DeepDRiD",
                source_version="1.1",
                country="China",
                original_partition=f"official_{partition}",
                role=role,
                relative_path=str(relative_path),
                patient_group_id=f"deepdrid1.1:patient:{patient_id}",
                patient_grouping_status="official_patient_id",
                eye=eye,
                field=field,
                original_grade=grade,
                original_grade_system="DeepDRiD_eye_DR_0_4",
                mapped_icdr_grade=grade,
                referable_label=int(grade >= 2),
                dme_or_maculopathy_label="unknown",
                quality_label=row["Overall quality"],
                label_source=str(labels_rel),
                licence_record="DeepDRiD bundled CC BY-SA 4.0 licence",
                label_alignment_status=label_alignment_status,
            )
        )
    return records


def group_label(records: Iterable[Record]) -> int:
    return max(record.mapped_icdr_grade for record in records)


def assign_grouped_folds(records: list[Record], seed: int = SEED, folds: int = FOLDS) -> None:
    eligible = [record for record in records if record.role == "development_pool"]
    groups: dict[str, list[Record]] = defaultdict(list)
    for record in eligible:
        groups[record.patient_group_id].append(record)
    by_source_grade: dict[tuple[str, int], list[str]] = defaultdict(list)
    for group_id, members in groups.items():
        by_source_grade[(members[0].source_name, group_label(members))].append(group_id)
    rng = random.Random(seed)
    for key in sorted(by_source_grade):
        group_ids = sorted(by_source_grade[key])
        rng.shuffle(group_ids)
        for index, group_id in enumerate(group_ids):
            fold = str(index % folds)
            for record in groups[group_id]:
                record.cv_fold = fold


def hamming_hex(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


def apply_exact_duplicate_policy(records: list[Record]) -> None:
    groups: dict[str, list[Record]] = defaultdict(list)
    for record in records:
        if record.validation_status in {"valid", "review"}:
            groups[record.sha256].append(record)
    for sha256, members in groups.items():
        if len(members) < 2:
            continue
        grades = {member.mapped_icdr_grade for member in members}
        protected_roles = {member.role for member in members if member.role != "development_pool"}
        if len(grades) > 1:
            for member in members:
                member.role = "quarantine_exact_duplicate_label_conflict"
                member.patient_group_id = f"exact:{sha256}"
            continue
        if protected_roles:
            for member in members:
                member.role = "quarantine_cross_role_exact_duplicate"
                member.patient_group_id = f"exact:{sha256}"
            continue
        ordered = sorted(members, key=lambda member: member.image_uid)
        canonical = ordered[0]
        canonical.patient_group_id = f"exact:{sha256}"
        for duplicate in ordered[1:]:
            duplicate.role = "duplicate_excluded"
            duplicate.patient_group_id = f"exact:{sha256}"


def apply_reviewed_near_duplicate_policy(records: list[Record], decisions_path: Path) -> dict[str, int]:
    summary = {"confirmed_duplicate_pairs": 0, "rejected_pairs": 0, "pending_pairs": 0, "quarantined_records": 0}
    if not decisions_path.is_file():
        return summary
    by_uid = {record.image_uid: record for record in records}
    flagged_uids: set[str] = set()
    for row in read_csv(decisions_path):
        decision = row["decision"]
        if decision == "confirmed_duplicate":
            summary["confirmed_duplicate_pairs"] += 1
            flagged_uids.update((row["left_image_uid"], row["right_image_uid"]))
        elif decision == "not_duplicate":
            summary["rejected_pairs"] += 1
        else:
            summary["pending_pairs"] += 1
    # Pull every exact copy connected to a visually confirmed near-duplicate.
    flagged_hashes = {by_uid[uid].sha256 for uid in flagged_uids if uid in by_uid}
    for record in records:
        if record.sha256 in flagged_hashes:
            flagged_uids.add(record.image_uid)
    for uid in flagged_uids:
        record = by_uid.get(uid)
        if record is None:
            raise ValueError(f"Near-duplicate decision references unknown image_uid: {uid}")
        if not record.role.startswith("quarantine"):
            record.role = "quarantine_confirmed_near_duplicate_cluster"
        record.patient_group_id = f"near_duplicate_cluster:{min(flagged_uids)}"
    summary["quarantined_records"] = len(flagged_uids)
    return summary


def duplicate_report(records: list[Record]) -> dict[str, object]:
    valid = [record for record in records if record.validation_status in {"valid", "review"}]
    exact: dict[str, list[Record]] = defaultdict(list)
    for record in valid:
        exact[record.sha256].append(record)
    exact_groups = [members for members in exact.values() if len(members) > 1]

    # Compare across sources and roles. This keeps the initial audit tractable and
    # targets leakage-risk duplicates rather than same-source repeated fields.
    near_pairs: list[dict[str, object]] = []
    for index, left in enumerate(valid):
        for right in valid[index + 1 :]:
            if left.source_name == right.source_name and left.role == right.role:
                continue
            if left.sha256 == right.sha256:
                continue
            phash_distance = hamming_hex(left.phash64, right.phash64)
            dhash_distance = hamming_hex(left.dhash64, right.dhash64)
            if phash_distance <= 2 and dhash_distance <= 6:
                near_pairs.append(
                    {
                        "left": left.image_uid,
                        "right": right.image_uid,
                        "left_source": left.source_name,
                        "right_source": right.source_name,
                        "phash_distance": phash_distance,
                        "dhash_distance": dhash_distance,
                        "status": "manual_review_required",
                    }
                )
    return {
        "exact_duplicate_groups": [
            {
                "sha256": members[0].sha256,
                "records": [member.image_uid for member in members],
                "roles": sorted({member.role for member in members}),
                "sources": sorted({member.source_name for member in members}),
            }
            for members in exact_groups
        ],
        "near_duplicate_candidates": near_pairs,
        "near_duplicate_method": "64-bit pHash distance <= 2 and dHash distance <= 6 across different sources or roles; candidates require visual review",
    }


def distributions(records: list[Record]) -> dict[str, object]:
    output: dict[str, object] = {}
    for source in sorted({record.source_name for record in records}):
        source_rows = [record for record in records if record.source_name == source]
        output[source] = {
            "records": len(source_rows),
            "roles": dict(sorted(Counter(record.role for record in source_rows).items())),
            "grades": dict(sorted(Counter(str(record.mapped_icdr_grade) for record in source_rows).items())),
            "referable": dict(sorted(Counter(str(record.referable_label) for record in source_rows).items())),
            "validation": dict(sorted(Counter(record.validation_status for record in source_rows).items())),
            "patients_or_groups": len({record.patient_group_id for record in source_rows}),
        }
    return output


def eligible_distribution(records: list[Record]) -> dict[str, object]:
    eligible = [record for record in records if record.role == "development_pool"]
    by_source: dict[str, object] = {}
    for source in sorted({record.source_name for record in eligible}):
        rows = [record for record in eligible if record.source_name == source]
        by_source[source] = {
            "records": len(rows),
            "grades": dict(sorted(Counter(str(record.mapped_icdr_grade) for record in rows).items())),
            "referable": dict(sorted(Counter(str(record.referable_label) for record in rows).items())),
            "groups": len({record.patient_group_id for record in rows}),
        }
    return {
        "records": len(eligible),
        "grades": dict(sorted(Counter(str(record.mapped_icdr_grade) for record in eligible).items())),
        "referable": dict(sorted(Counter(str(record.referable_label) for record in eligible).items())),
        "sources": by_source,
    }


def write_csv(path: Path, records: list[Record]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(record) for record in records]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_data_card(path: Path, audit: dict[str, object]) -> None:
    dist = audit["distribution"]
    duplicates = audit["duplicates"]
    lines = [
        "# RetinaSathi V3 Initial Data Audit",
        "",
        f"Generated: {audit['generated_at']}",
        "",
        "Raw images were read only. No image was edited or removed by this audit.",
        "",
        "## Sources",
        "",
        "| Source | Records | Groups | Grades 0/1/2/3/4 | Validation |",
        "|---|---:|---:|---|---|",
    ]
    for source, summary in dist.items():
        grades = summary["grades"]
        grade_text = "/".join(str(grades.get(str(grade), 0)) for grade in range(5))
        validation_text = ", ".join(f"{key}: {value}" for key, value in summary["validation"].items())
        lines.append(f"| {source} | {summary['records']} | {summary['patients_or_groups']} | {grade_text} | {validation_text} |")
    lines.extend(
        [
            "",
            "## Split policy",
            "",
            "- APTOS 2019 labelled training images are development data; patient identifiers are unavailable.",
            "- IDRiD official training is development data; its consumed official test is historical evidence only.",
            "- DeepDRiD official training enters the development pool with patient grouping.",
            "- DeepDRiD official validation remains a source-validation partition.",
            "- Messidor is excluded from V3 selection because its V2 result has already been inspected.",
            "- A new untouched external source or prospective Indian cohort is still required before final model promotion.",
            "",
            "## Duplicate audit",
            "",
            f"- Exact duplicate groups: {len(duplicates['exact_duplicate_groups'])}",
            f"- Near-duplicate candidates requiring review: {len(duplicates['near_duplicate_candidates'])}",
            f"- Visually confirmed near-duplicate pairs: {audit['near_duplicate_decisions']['confirmed_duplicate_pairs']}",
            f"- Visually rejected lookalike pairs: {audit['near_duplicate_decisions']['rejected_pairs']}",
            "- Confirmed duplicate clusters are quarantined in the manifest; raw images are retained.",
            "",
            "## Training eligibility",
            "",
            *[f"- {role}: {count}" for role, count in audit["role_distribution"].items()],
            "",
            "Eligible development grade counts (0/1/2/3/4): "
            + "/".join(str(audit["eligible_distribution"]["grades"].get(str(grade), 0)) for grade in range(5)),
            "",
            "Eligible referable counts: "
            + ", ".join(
                f"{label}: {count}" for label, count in audit["eligible_distribution"]["referable"].items()
            ),
            "",
            "## Blocking conditions",
            "",
            f"- Missing or invalid labelled files: {audit['blocking_record_count']}",
            "- EyePACS remains deferred until extraction, patient/eye grouping, licence confirmation, and duplicate audit are complete.",
            "- Final Indian clinical validation remains unavailable.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--manifest", type=Path, default=Path("runs/v3_data/v3_initial_manifest.csv"))
    parser.add_argument("--audit", type=Path, default=Path("artifacts/v3_data_audit.json"))
    parser.add_argument("--data-card", type=Path, default=Path("docs/V3_DATA_CARD.md"))
    parser.add_argument(
        "--near-duplicate-decisions",
        type=Path,
        default=Path("configs/v3_near_duplicate_decisions.csv"),
    )
    args = parser.parse_args()
    root = args.data_root.resolve()

    records = [
        *aptos_records(root),
        *idrid_records(root, "training"),
        *idrid_records(root, "testing"),
        *deepdrid_records(root, "training"),
        *deepdrid_records(root, "validation"),
    ]
    if len({record.image_uid for record in records}) != len(records):
        raise ValueError("image_uid collision detected")
    for index, record in enumerate(records, start=1):
        validate_record(record, root)
        if index % 500 == 0:
            print(f"validated {index}/{len(records)}", flush=True)

    apply_exact_duplicate_policy(records)
    near_duplicate_decisions = apply_reviewed_near_duplicate_policy(records, args.near_duplicate_decisions)
    assign_grouped_folds(records)

    duplicates = duplicate_report(records)
    blockers = [record for record in records if record.validation_status in {"missing", "invalid", "unsupported_extension"}]
    audit = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_root": str(root),
        "seed": SEED,
        "folds": FOLDS,
        "records": len(records),
        "development_records": sum(record.role == "development_pool" for record in records),
        "source_validation_records": sum(record.role == "source_validation" for record in records),
        "historical_excluded_records": sum("historical" in record.role for record in records),
        "blocking_record_count": len(blockers),
        "blocking_records": [record.image_uid for record in blockers],
        "role_distribution": dict(sorted(Counter(record.role for record in records).items())),
        "fold_distribution": dict(
            sorted(Counter(record.cv_fold for record in records if record.role == "development_pool").items())
        ),
        "label_alignment": dict(sorted(Counter(record.label_alignment_status for record in records).items())),
        "near_duplicate_decisions": near_duplicate_decisions,
        "eligible_distribution": eligible_distribution(records),
        "distribution": distributions(records),
        "duplicates": duplicates,
    }
    write_csv(args.manifest, records)
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    write_data_card(args.data_card, audit)
    print(json.dumps({"records": len(records), "blockers": len(blockers), "manifest": str(args.manifest)}))
    if blockers:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
