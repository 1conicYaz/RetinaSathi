#!/usr/bin/env python3
"""Create a conservative post-training audit without touching the locked test set."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import yaml


def wilson_interval(successes: int, total: int, z: float = 1.96) -> list[float]:
    if total == 0:
        return [float("nan"), float("nan")]
    proportion = successes / total
    denominator = 1 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    spread = z * math.sqrt(
        proportion * (1 - proportion) / total + z * z / (4 * total * total)
    ) / denominator
    return [centre - spread, centre + spread]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    final_path = args.run_dir / "final_validation.json"
    final = json.loads(final_path.read_text(encoding="utf-8"))
    config = yaml.safe_load((args.run_dir / "config.yaml").read_text(encoding="utf-8"))
    selection = config["selection"]
    source = final["source_validation"]
    referral = source["referable"]
    matrix = source["grade"]["confusion_matrix"]
    minimum_sensitivity = float(
        selection.get("minimum_source_sensitivity", selection.get("minimum_referable_sensitivity", 0.90))
    )
    minimum_specificity = float(
        selection.get("minimum_source_specificity", selection.get("minimum_referable_specificity", 0.85))
    )
    sensitivity = float(referral["sensitivity"])
    specificity = float(referral["specificity"])
    architecture = str(config.get("experiment", {}).get("architecture", "unknown"))
    grade_rows = []
    for grade, row in enumerate(matrix):
        total = int(sum(row))
        correct = int(row[grade])
        grade_rows.append(
            {
                "grade": grade,
                "correct": correct,
                "total": total,
                "recall": correct / total if total else None,
                "recall_wilson_95": wilson_interval(correct, total),
            }
        )
    true_positive = int(referral["true_positive"])
    false_negative = int(referral["false_negative"])
    true_negative = int(referral["true_negative"])
    false_positive = int(referral["false_positive"])
    source_gate_passed = sensitivity >= minimum_sensitivity and specificity >= minimum_specificity
    next_actions = ["Do not alter the threshold using independent-source labels."]
    if architecture == "dinov2_vits14":
        next_actions.extend(
            [
                "Do not repeat seeds or unfreeze DINOv2 blocks because this frozen pilot did not beat V3.1.",
                "Compare gated RETFound CFP weights only after access and licence review.",
            ]
        )
    else:
        next_actions.extend(
            [
                "Compare a frozen DINOv2-S encoder under the identical manifest, heads, and gates.",
                "Compare gated RETFound CFP weights only after access and licence review.",
            ]
        )
    next_actions.extend(
        [
            "Review development-fold false positives and weak Grade 1/2/4 cases with an ophthalmologist.",
            "Require a new de-identified Indian-camera cohort before a final clinical generalization claim.",
        ]
    )
    audit = {
        "decision": "advance_to_locked_test" if source_gate_passed else "hold_locked_test",
        "official_test_used": bool(final["official_test_used"]),
        "selected_epoch": int(final["selected_epoch"]),
        "checkpoint_sha256": final["checkpoint_sha256"],
        "calibration_gate_passed": bool(final["threshold_gate"]["passed"]),
        "source_gate": {
            "passed": source_gate_passed,
            "minimum_sensitivity": minimum_sensitivity,
            "minimum_specificity": minimum_specificity,
            "sensitivity": sensitivity,
            "sensitivity_wilson_95": wilson_interval(true_positive, true_positive + false_negative),
            "specificity": specificity,
            "specificity_wilson_95": wilson_interval(true_negative, true_negative + false_positive),
        },
        "source_grade_recall": grade_rows,
        "primary_findings": [
            (
                "Independent-source sensitivity and specificity satisfy the predeclared gate."
                if source_gate_passed
                else "Independent-source gate failed: "
                + ", ".join(
                    item
                    for item, failed in (
                        (f"sensitivity {sensitivity:.1%} < {minimum_sensitivity:.1%}", sensitivity < minimum_sensitivity),
                        (f"specificity {specificity:.1%} < {minimum_specificity:.1%}", specificity < minimum_specificity),
                    )
                    if failed
                )
            ),
            "Grade 1, Grade 2, and Grade 4 recall require focused improvement.",
            "The official locked test remains unused.",
        ],
        "next_actions": next_actions,
    }
    output_path = args.run_dir / "candidate_audit.json"
    output_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
