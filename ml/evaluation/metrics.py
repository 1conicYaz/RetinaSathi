"""Clinically relevant, empty-class-safe evaluation metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)


def expected_calibration_error(confidence: np.ndarray, correct: np.ndarray, bins: int = 15) -> float:
    boundaries = np.linspace(0, 1, bins + 1)
    value = 0.0
    for lower, upper in zip(boundaries[:-1], boundaries[1:]):
        selected = (confidence > lower) & (confidence <= upper)
        if selected.any():
            value += selected.mean() * abs(correct[selected].mean() - confidence[selected].mean())
    return float(value)


def classifier_metrics(truth: np.ndarray, probabilities: np.ndarray, referable_threshold: float = 0.5) -> dict[str, object]:
    predicted = probabilities.argmax(axis=1)
    referable_truth = truth >= 2
    referable_score = probabilities[:, 2:].sum(axis=1)
    referable_predicted = referable_score >= referable_threshold
    confidence = probabilities.max(axis=1)
    one_hot = np.eye(5)[truth]
    return {
        "qwk": float(cohen_kappa_score(truth, predicted, weights="quadratic")),
        "macro_f1": float(f1_score(truth, predicted, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
        "per_class_recall": recall_score(truth, predicted, labels=np.arange(5), average=None, zero_division=0).tolist(),
        "referable_sensitivity": float(((referable_predicted & referable_truth).sum()) / max(referable_truth.sum(), 1)),
        "referable_specificity": float(((~referable_predicted & ~referable_truth).sum()) / max((~referable_truth).sum(), 1)),
        "confusion_matrix": confusion_matrix(truth, predicted, labels=np.arange(5)).tolist(),
        "ece": expected_calibration_error(confidence, predicted == truth),
        "multiclass_brier": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
    }


def best_referable_threshold(truth: np.ndarray, probabilities: np.ndarray) -> float:
    referable_truth = truth >= 2
    scores = probabilities[:, 2:].sum(axis=1)
    candidates = np.unique(np.concatenate(([0.0, 0.5, 1.0], scores)))
    def youden(threshold: float) -> float:
        predicted = scores >= threshold
        sensitivity = (predicted & referable_truth).sum() / max(referable_truth.sum(), 1)
        specificity = (~predicted & ~referable_truth).sum() / max((~referable_truth).sum(), 1)
        return float(sensitivity + specificity - 1)
    return float(max(candidates, key=lambda threshold: (youden(float(threshold)), -abs(float(threshold) - 0.5))))


def referable_metrics(truth: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> dict[str, object]:
    truth = np.asarray(truth, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    predicted = scores >= threshold
    tp = int((predicted & truth).sum())
    fn = int((~predicted & truth).sum())
    tn = int((~predicted & ~truth).sum())
    fp = int((predicted & ~truth).sum())
    return {
        "threshold": float(threshold),
        "true_positive": tp,
        "false_negative": fn,
        "true_negative": tn,
        "false_positive": fp,
        "sensitivity": float(tp / max(tp + fn, 1)),
        "specificity": float(tn / max(tn + fp, 1)),
        "ppv": float(tp / max(tp + fp, 1)),
        "npv": float(tn / max(tn + fn, 1)),
        "auroc": float(roc_auc_score(truth, scores)) if len(np.unique(truth)) == 2 else None,
        "auprc": float(average_precision_score(truth, scores)) if truth.any() else None,
        "brier": float(np.mean((scores - truth.astype(float)) ** 2)),
    }


def select_safety_threshold(
    truth: np.ndarray,
    scores: np.ndarray,
    minimum_sensitivity: float = 0.90,
    minimum_specificity: float = 0.85,
) -> dict[str, object]:
    candidates = np.unique(np.concatenate(([0.0, 0.5, 1.0], np.asarray(scores, dtype=float))))
    evaluated = [referable_metrics(truth, scores, float(threshold)) for threshold in candidates]
    passing = [
        result
        for result in evaluated
        if result["sensitivity"] >= minimum_sensitivity and result["specificity"] >= minimum_specificity
    ]
    if passing:
        selected = max(passing, key=lambda result: (result["specificity"], result["ppv"], result["threshold"]))
        return {
            "passed": True,
            "minimum_sensitivity": minimum_sensitivity,
            "minimum_specificity": minimum_specificity,
            "selected": selected,
        }
    selected = max(
        evaluated,
        key=lambda result: (
            min(result["sensitivity"] / max(minimum_sensitivity, 1e-8), 1.0)
            + min(result["specificity"] / max(minimum_specificity, 1e-8), 1.0),
            result["sensitivity"],
            result["specificity"],
        ),
    )
    return {
        "passed": False,
        "minimum_sensitivity": minimum_sensitivity,
        "minimum_specificity": minimum_specificity,
        "selected": selected,
        "failure_reason": "No calibration-set threshold satisfied both predeclared gates.",
    }
