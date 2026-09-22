"""Deterministic sampling, weighting, selection, and promotion policy for V3.1."""

from __future__ import annotations

from collections import Counter
from typing import Mapping, Sequence

import numpy as np
import torch


def joint_source_grade_sample_weights(
    rows: Sequence[Mapping[str, str]],
    max_ratio: float = 10.0,
) -> list[float]:
    if max_ratio < 1:
        raise ValueError("max_ratio must be at least 1")
    counts = Counter((row["source_name"], int(row["mapped_icdr_grade"])) for row in rows)
    raw = [1.0 / counts[(row["source_name"], int(row["mapped_icdr_grade"]))] for row in rows]
    floor = max(raw) / max_ratio
    bounded = [max(value, floor) for value in raw]
    mean = sum(bounded) / max(len(bounded), 1)
    return [value / mean for value in bounded]


def nominal_class_weights(
    dataset: object,
    device: torch.device,
    minimum: float = 0.5,
    maximum: float = 3.0,
) -> torch.Tensor:
    rows = getattr(dataset, "rows")
    counts = Counter(int(row["mapped_icdr_grade"]) for row in rows)
    if any(counts[grade] == 0 for grade in range(5)):
        raise ValueError("All five DR grades are required to compute nominal class weights")
    values = np.asarray([1.0 / np.sqrt(counts[grade]) for grade in range(5)], dtype=np.float32)
    values /= values.mean()
    values = np.clip(values, minimum, maximum)
    return torch.as_tensor(values, device=device, dtype=torch.float32)


def selection_key(
    result: Mapping[str, object],
    weights: Mapping[str, object] | None = None,
) -> tuple[float, float, float, float, float]:
    """Validation-only composite policy fixed before V3.1 training."""

    referable = result["referable_metrics"]  # type: ignore[assignment]
    grade = result["grade_metrics"]  # type: ignore[assignment]
    recalls = grade["per_class_recall"]  # type: ignore[index]
    minority_recall = min(float(recalls[index]) for index in (1, 2, 4))
    auprc = referable["auprc"]  # type: ignore[index]
    auprc_value = float(auprc if auprc is not None else -1.0)
    qwk = float(grade["qwk"])  # type: ignore[index]
    macro_f1 = float(grade["macro_f1"])  # type: ignore[index]
    policy = weights or {
        "referable_auprc": 0.45,
        "qwk": 0.25,
        "macro_f1": 0.20,
        "minimum_grade_1_2_4_recall": 0.10,
    }
    if abs(sum(float(value) for value in policy.values()) - 1.0) > 1e-6:
        raise ValueError("Checkpoint selection weights must sum to 1.0")
    composite = (
        float(policy["referable_auprc"]) * auprc_value
        + float(policy["qwk"]) * qwk
        + float(policy["macro_f1"]) * macro_f1
        + float(policy["minimum_grade_1_2_4_recall"]) * minority_recall
    )
    return (composite, auprc_value, qwk, macro_f1, minority_recall)


def candidate_status(
    calibration_gate_passed: bool,
    source_metrics: Mapping[str, object],
    minimum_source_sensitivity: float,
    minimum_source_specificity: float,
) -> str:
    if not calibration_gate_passed:
        return "failed_calibration_safety_gate"
    if (
        float(source_metrics["sensitivity"]) < minimum_source_sensitivity
        or float(source_metrics["specificity"]) < minimum_source_specificity
    ):
        return "failed_independent_source_gate"
    return "candidate_source_gate_passed"
