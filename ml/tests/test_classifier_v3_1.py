from __future__ import annotations

import tempfile
import unittest
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from ml.models import DualHeadDRClassifier


class V31ModelTests(unittest.TestCase):
    def test_three_heads_have_expected_shapes_and_gradients(self) -> None:
        from ml.models.classifier_v3_1 import MultiHeadDRClassifier

        model = MultiHeadDRClassifier("efficientnet_b3", pretrained=False, dropout=0.0)
        images = torch.randn(2, 3, 64, 64)
        binary, ordinal, nominal = model(images)
        self.assertEqual(tuple(binary.shape), (2,))
        self.assertEqual(tuple(ordinal.shape), (2, 4))
        self.assertEqual(tuple(nominal.shape), (2, 5))
        (binary.sum() + ordinal.sum() + nominal.sum()).backward()
        self.assertIsNotNone(model.referable_head.weight.grad)
        self.assertIsNotNone(model.ordinal_head.weight.grad)
        self.assertIsNotNone(model.nominal_head.weight.grad)

    def test_v30_transfer_loads_all_compatible_parameters(self) -> None:
        from ml.models.classifier_v3_1 import MultiHeadDRClassifier, load_v3_0_weights

        source = DualHeadDRClassifier("efficientnet_b3", pretrained=False, dropout=0.0)
        target = MultiHeadDRClassifier("efficientnet_b3", pretrained=False, dropout=0.0)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "v3_0.pt"
            torch.save(
                {
                    "model": source.state_dict(),
                    "epoch": 9,
                    "manifest_sha256": "manifest-hash",
                    "config": {"experiment": {"architecture": "efficientnet_b3"}},
                },
                path,
            )
            provenance = load_v3_0_weights(target, path, expected_manifest_sha256="manifest-hash")

        torch.testing.assert_close(target.referable_head.weight, source.referable_head.weight)
        torch.testing.assert_close(target.ordinal_head.weight, source.ordinal_head.weight)
        self.assertEqual(provenance["source_epoch"], 9)
        self.assertEqual(provenance["new_parameters"], ["nominal_head.bias", "nominal_head.weight"])

    def test_v30_transfer_rejects_architecture_mismatch(self) -> None:
        from ml.models.classifier_v3_1 import MultiHeadDRClassifier, load_v3_0_weights

        target = MultiHeadDRClassifier("efficientnet_b3", pretrained=False)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "wrong.pt"
            torch.save(
                {
                    "model": target.state_dict(),
                    "config": {"experiment": {"architecture": "convnext_tiny"}},
                },
                path,
            )
            with self.assertRaisesRegex(ValueError, "architecture"):
                load_v3_0_weights(target, path)


class V31TrainingPolicyTests(unittest.TestCase):
    def test_progress_update_is_atomic_and_machine_readable(self) -> None:
        from ml.training.train_classifier_v3_1 import write_progress

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "progress.json"
            write_progress(path, {"phase": "training", "epoch": 1, "batch": 25, "batches": 382})
            self.assertEqual(json.loads(path.read_text())["batch"], 25)
            self.assertFalse((Path(folder) / "progress.json.tmp").exists())

    def test_joint_source_grade_weights_prioritize_rare_bucket_with_cap(self) -> None:
        from ml.training.v3_1_components import joint_source_grade_sample_weights

        rows = [
            *[{"source_name": "A", "mapped_icdr_grade": "0"} for _ in range(20)],
            *[{"source_name": "A", "mapped_icdr_grade": "4"} for _ in range(2)],
            *[{"source_name": "B", "mapped_icdr_grade": "0"} for _ in range(4)],
        ]
        weights = joint_source_grade_sample_weights(rows, max_ratio=5.0)
        self.assertGreater(weights[20], weights[0])
        self.assertLessEqual(max(weights) / min(weights), 5.0)

    def test_nominal_weights_are_finite_bounded_and_upweight_rare_grade(self) -> None:
        from ml.training.v3_1_components import nominal_class_weights

        dataset = SimpleNamespace(
            rows=[
                *[{"mapped_icdr_grade": "0"} for _ in range(16)],
                *[{"mapped_icdr_grade": "1"} for _ in range(4)],
                *[{"mapped_icdr_grade": "2"} for _ in range(8)],
                *[{"mapped_icdr_grade": "3"} for _ in range(2)],
                {"mapped_icdr_grade": "4"},
            ]
        )
        weights = nominal_class_weights(dataset, torch.device("cpu"), minimum=0.5, maximum=3.0)
        self.assertTrue(bool(torch.isfinite(weights).all()))
        self.assertGreater(float(weights[4]), float(weights[0]))
        self.assertGreaterEqual(float(weights.min()), 0.5)
        self.assertLessEqual(float(weights.max()), 3.0)

    def test_selection_key_rewards_referral_and_grade_quality(self) -> None:
        from ml.training.v3_1_components import selection_key

        base = {
            "referable_metrics": {"auprc": 0.90},
            "grade_metrics": {"qwk": 0.70, "macro_f1": 0.50, "per_class_recall": [0.9, 0.2, 0.4, 0.6, 0.3]},
        }
        improved = {
            "referable_metrics": {"auprc": 0.90},
            "grade_metrics": {"qwk": 0.71, "macro_f1": 0.52, "per_class_recall": [0.9, 0.3, 0.5, 0.6, 0.4]},
        }
        self.assertGreater(selection_key(improved), selection_key(base))

        better_grading_with_small_auprc_tradeoff = {
            "referable_metrics": {"auprc": 0.89},
            "grade_metrics": {"qwk": 0.82, "macro_f1": 0.66, "per_class_recall": [0.9, 0.5, 0.6, 0.7, 0.5]},
        }
        self.assertGreater(selection_key(better_grading_with_small_auprc_tradeoff), selection_key(base))
        with self.assertRaisesRegex(ValueError, "sum to 1.0"):
            selection_key(base, {"referable_auprc": 1.0, "qwk": 1.0, "macro_f1": 0.0, "minimum_grade_1_2_4_recall": 0.0})

    def test_candidate_status_requires_calibration_and_source_gates(self) -> None:
        from ml.training.v3_1_components import candidate_status

        passing = {"sensitivity": 0.91, "specificity": 0.86}
        self.assertEqual(candidate_status(True, passing, 0.90, 0.85), "candidate_source_gate_passed")
        self.assertEqual(candidate_status(False, passing, 0.90, 0.85), "failed_calibration_safety_gate")
        self.assertEqual(
            candidate_status(True, {"sensitivity": 0.89, "specificity": 0.90}, 0.90, 0.85),
            "failed_independent_source_gate",
        )

    def test_v31_config_predeclares_guardband_and_locked_test(self) -> None:
        import yaml

        config = yaml.safe_load(Path("configs/classifier_v3_1.yaml").read_text())
        self.assertEqual(config["selection"]["minimum_calibration_sensitivity"], 0.95)
        self.assertEqual(config["selection"]["minimum_calibration_specificity"], 0.85)
        self.assertTrue(config["selection"]["official_test_locked"])


if __name__ == "__main__":
    unittest.main()
