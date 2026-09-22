from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from ml.calibration import fit_binary_temperature, fit_temperature
from ml.evaluation import best_referable_threshold, classifier_metrics, referable_metrics, select_safety_threshold
from ml.evaluate_robustness import add_prediction_consistency
from ml.models import DualHeadDRClassifier, coral_loss, coral_targets, ordinal_probabilities
from ml.select_classifier import factor_coverage, require_master_comparisons
from ml.training.pilot_classifier import (
    balanced_indices,
)
from ml.training.train_classifier_v2 import class_weights, load_aptos_backbone


class OrdinalClassifierTests(unittest.TestCase):
    def test_coral_targets_respect_grade_order(self) -> None:
        targets = coral_targets(torch.tensor([0, 2, 4]))
        expected = torch.tensor([[0, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 1]], dtype=torch.float32)
        torch.testing.assert_close(targets, expected)

    def test_ordinal_probabilities_are_valid(self) -> None:
        probabilities = ordinal_probabilities(torch.tensor([[3.0, 1.0, -1.0, -3.0]]))
        self.assertTrue(bool((probabilities >= 0).all()))
        torch.testing.assert_close(probabilities.sum(1), torch.ones(1))

    def test_ordinal_loss_is_finite(self) -> None:
        loss = coral_loss(torch.zeros(3, 4), torch.tensor([0, 2, 4]))
        self.assertTrue(bool(torch.isfinite(loss)))

    def test_v3_dual_head_shapes_and_gradients(self) -> None:
        model = DualHeadDRClassifier("efficientnet_b3", pretrained=False, dropout=0.0)
        images = torch.randn(2, 3, 64, 64)
        binary_logits, ordinal_logits = model(images)
        self.assertEqual(tuple(binary_logits.shape), (2,))
        self.assertEqual(tuple(ordinal_logits.shape), (2, 4))
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            binary_logits, torch.tensor([0.0, 1.0])
        ) + coral_loss(ordinal_logits, torch.tensor([0, 4]))
        loss.backward()
        self.assertIsNotNone(model.referable_head.weight.grad)
        self.assertIsNotNone(model.ordinal_head.weight.grad)


class ClinicalMetricsTests(unittest.TestCase):
    def test_perfect_predictions_have_perfect_primary_metrics(self) -> None:
        truth = np.arange(5)
        probabilities = np.eye(5)
        metrics = classifier_metrics(truth, probabilities)
        self.assertEqual(metrics["qwk"], 1.0)
        self.assertEqual(metrics["macro_f1"], 1.0)
        self.assertEqual(metrics["referable_sensitivity"], 1.0)
        self.assertEqual(metrics["referable_specificity"], 1.0)
        self.assertEqual(best_referable_threshold(truth, probabilities), 0.5)

    def test_temperature_is_positive(self) -> None:
        logits = torch.tensor([[3.0, 2.0, 1.0, -1.0], [-2.0, -3.0, -4.0, -5.0]])
        temperature = fit_temperature(logits, torch.tensor([3, 0]), iterations=5)
        self.assertGreater(temperature, 0)

    def test_nominal_temperature_is_positive(self) -> None:
        logits = torch.tensor([[3.0, 2.0, 1.0, -1.0, -2.0], [-2.0, -3.0, -4.0, -5.0, 2.0]])
        temperature = fit_temperature(logits, torch.tensor([0, 4]), iterations=5, objective="cross_entropy")
        self.assertGreater(temperature, 0)

    def test_binary_temperature_is_positive(self) -> None:
        logits = torch.tensor([-2.0, -1.0, 1.0, 2.0])
        temperature = fit_binary_temperature(logits, torch.tensor([0, 0, 1, 1]), iterations=5)
        self.assertGreater(temperature, 0)

    def test_safety_threshold_requires_both_gates(self) -> None:
        truth = np.array([0, 0, 0, 1, 1, 1], dtype=bool)
        scores = np.array([0.05, 0.10, 0.20, 0.80, 0.90, 0.95])
        result = select_safety_threshold(truth, scores, 0.90, 0.85)
        self.assertTrue(result["passed"])
        selected = result["selected"]
        self.assertGreaterEqual(selected["sensitivity"], 0.90)
        self.assertGreaterEqual(selected["specificity"], 0.85)
        self.assertEqual(referable_metrics(truth, scores, selected["threshold"])["false_negative"], 0)


class PilotComparisonTests(unittest.TestCase):
    def test_comparison_can_reuse_exact_aptos_backbone(self) -> None:
        source = torch.nn.Linear(2, 2)
        target = SimpleNamespace(backbone=torch.nn.Linear(2, 2))
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder) / "aptos.pt"
            torch.save(
                {
                    "phase": "APTOS_2019",
                    "config": {"experiment": {"architecture": "tiny"}},
                    "model": {
                        "backbone.weight": source.weight.detach().clone(),
                        "backbone.bias": source.bias.detach().clone(),
                    },
                },
                checkpoint,
            )
            load_aptos_backbone(target, checkpoint, "tiny")
        torch.testing.assert_close(target.backbone.weight, source.weight)
        torch.testing.assert_close(target.backbone.bias, source.bias)

    def test_cross_entropy_class_weights_upweight_rare_grade(self) -> None:
        dataset = SimpleNamespace(
            rows=[{"grade": 0}, {"grade": 0}, {"grade": 0}, {"grade": 1}]
        )
        weights = class_weights(dataset, torch.device("cpu"))
        self.assertGreater(float(weights[1]), float(weights[0]))

    def test_balanced_subset_is_deterministic_and_class_limited(self) -> None:
        dataset = SimpleNamespace(
            rows=[
                {"grade": 2},
                {"grade": 0},
                {"grade": 2},
                {"grade": 1},
                {"grade": 0},
                {"grade": 1},
            ]
        )
        self.assertEqual(balanced_indices(dataset, per_class=1), [1, 3, 0])

    def test_selection_gate_requires_every_master_prompt_factor(self) -> None:
        complete = factor_coverage(
            [
                {
                    "results": [
                        {
                            "architecture": "efficientnet_b3",
                            "objective": "coral",
                            "input_size": 384,
                            "dme_weight": 0.0,
                            "imbalance_strategy": "sampler_only",
                        },
                        {
                            "architecture": "convnext_tiny",
                            "objective": "cross_entropy",
                            "input_size": 512,
                            "dme_weight": 0.25,
                            "imbalance_strategy": "weighted_loss",
                        },
                    ]
                }
            ]
        )
        require_master_comparisons(complete)

        incomplete = dict(complete, input_sizes=[384])
        with self.assertRaisesRegex(ValueError, "384 and 512"):
            require_master_comparisons(incomplete)


class RobustnessReportTests(unittest.TestCase):
    def test_consistency_keeps_normal_reference_until_all_groups_are_compared(self) -> None:
        groups = {
            "normal": {"predicted_grades": [0, 1, 2]},
            "dark": {"predicted_grades": [0, 2, 2]},
        }

        add_prediction_consistency(groups)

        self.assertEqual(groups["normal"]["prediction_consistency_vs_normal"], 1.0)
        self.assertAlmostEqual(
            groups["dark"]["prediction_consistency_vs_normal"], 2 / 3
        )
        self.assertNotIn("predicted_grades", groups["normal"])
        self.assertNotIn("predicted_grades", groups["dark"])

if __name__ == "__main__":
    unittest.main()
