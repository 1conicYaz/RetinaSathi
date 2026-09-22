from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from ml.preprocessing import preprocess_fundus

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model_runtime import RetinaSathiPredictor


class QualityGateTests(unittest.TestCase):
    @staticmethod
    def fundus_like(size: int = 512) -> Image.Image:
        rng = np.random.default_rng(26038)
        texture = np.clip(rng.normal(95, 25, (size, size)), 0, 255).astype(np.uint8)
        return Image.fromarray(np.stack([np.clip(texture * 1.4, 0, 255), texture, texture // 2], axis=-1).astype(np.uint8))

    def test_black_frame_is_rejected(self) -> None:
        image = Image.new("RGB", (512, 512), "black")
        result = RetinaSathiPredictor.quality(image)
        self.assertEqual(result["label"], "poor")
        self.assertIn("Retina is not visible", result["issues"])

    def test_textured_fundus_like_frame_is_usable(self) -> None:
        rng = np.random.default_rng(26038)
        base = np.full((512, 512), 98, dtype=np.float32)
        textured = np.clip(base + rng.normal(0, 24, base.shape), 0, 255).astype(np.uint8)
        image = Image.fromarray(np.stack([np.clip(textured * 1.35, 0, 255), textured, np.clip(textured * 0.7, 0, 255)], axis=-1).astype(np.uint8))
        result = RetinaSathiPredictor.quality(image)
        self.assertIn(result["label"], {"good", "usable"})

    def test_non_fundus_color_frame_is_rejected(self) -> None:
        rng = np.random.default_rng(17)
        blue = np.clip(rng.normal(120, 25, (512, 512)), 0, 255).astype(np.uint8)
        image = Image.fromarray(np.stack([blue // 2, blue, blue], axis=-1))
        result = RetinaSathiPredictor.quality(image)
        self.assertEqual(result["label"], "poor")
        self.assertIn("Image does not resemble a color fundus photograph", result["issues"])

    def test_incomplete_field_of_view_is_rejected(self) -> None:
        image = np.zeros((512, 512, 3), dtype=np.uint8)
        rng = np.random.default_rng(42)
        texture = np.clip(rng.normal(100, 22, (512, 210)), 0, 255).astype(np.uint8)
        image[:, :210] = np.stack([np.clip(texture * 1.4, 0, 255), texture, texture // 2], axis=-1)
        result = RetinaSathiPredictor.quality(Image.fromarray(image))
        self.assertEqual(result["label"], "poor")
        self.assertIn("Retinal field of view is incomplete", result["issues"])

    def test_dark_overexposed_low_contrast_and_blur_fail_safely(self) -> None:
        base = self.fundus_like()
        cases = {
            "dark": (ImageEnhance.Brightness(base).enhance(0.12), "Image is too dark"),
            "overexposed": (ImageEnhance.Brightness(base).enhance(3.0), "Image is overexposed"),
            "low_contrast": (ImageEnhance.Contrast(base).enhance(0.03), "Retinal contrast is low"),
            "blurred": (base.filter(ImageFilter.GaussianBlur(14)), "Image may be blurred"),
        }
        for name, (image, issue) in cases.items():
            with self.subTest(name=name):
                result = RetinaSathiPredictor.quality(image)
                self.assertIn(issue, result["issues"])
                self.assertNotEqual(result["label"], "good")

    def test_quality_handles_different_resolutions(self) -> None:
        for size in (128, 640, 1600):
            with self.subTest(size=size):
                result = RetinaSathiPredictor.quality(self.fundus_like(size))
                self.assertIn(result["label"], {"good", "usable"})


class ModelContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        checkpoint = Path(__file__).resolve().parents[1] / "artifacts" / "idrid_multitask.onnx"
        cls.predictor = RetinaSathiPredictor(checkpoint)

    def test_prediction_contract(self) -> None:
        rng = np.random.default_rng(26038)
        texture = np.clip(rng.normal(95, 28, (300, 300)), 0, 255).astype(np.uint8)
        pixels = np.stack([np.clip(texture * 1.4, 0, 255), texture, np.clip(texture * 0.65, 0, 255)], axis=-1).astype(np.uint8)
        result = self.predictor.predict(Image.fromarray(pixels))
        self.assertIn(result["dr_grade"], range(5))
        self.assertIn(result["dme_risk"], range(3))
        self.assertEqual(len(result["grade_probabilities"]), 5)
        self.assertTrue(str(result["explanation_image"]).startswith("data:image/jpeg;base64,"))
        self.assertTrue(str(result["explainability"]["heatmap_image"]).startswith("data:image/png;base64,"))
        self.assertEqual(result["api_version"], "2.0")
        self.assertEqual(result["lesions"]["status"], "not_trained")
        self.assertEqual(result["lesions"]["items"], [])
        self.assertIn("not_gradcam", result["explainability"]["method"])
        self.assertIsNone(result["dr"]["confidence_calibrated"])
        self.assertNotEqual(result["recommendation"]["urgency"], "routine")
        if not result["referable_dr"]:
            self.assertEqual(result["recommendation"]["urgency"], "review")
            self.assertIn("not calibrated", result["recommendation"]["text"])

    def test_ungradeable_image_has_no_clinical_prediction(self) -> None:
        result = self.predictor.predict(Image.new("RGB", (512, 512), "black"))
        self.assertEqual(result["quality"]["label"], "poor")
        self.assertEqual(result["dr"]["status"], "unavailable")
        self.assertIsNone(result["dr_grade"])
        self.assertIsNone(result["dme_risk"])
        self.assertIsNone(result["confidence"])
        self.assertEqual(result["explainability"]["method"], "not_run_quality_gate")

    def test_ordinal_logits_convert_to_five_probabilities(self) -> None:
        probabilities = RetinaSathiPredictor._grade_probabilities(np.asarray([3.0, 1.0, -1.0, -3.0]))
        self.assertEqual(probabilities.shape, (5,))
        self.assertAlmostEqual(float(probabilities.sum()), 1.0)
        self.assertTrue(bool((probabilities >= 0).all()))

    def test_exact_linear_gap_gradcam_is_class_specific(self) -> None:
        predictor = object.__new__(RetinaSathiPredictor)
        predictor.temperature = 1.0
        predictor.manifest = {
            "objective": "coral",
            "explainability": {
                "method": "gradcam_exact_linear_gap",
                "grade_head_weights": [[2.0, 0.0], [0.0, 2.0], [1.0, 0.0], [0.0, 1.0]],
            },
        }
        features = np.asarray([[[[1.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 1.0]]]])
        heatmap, method, status = predictor._explanation_heatmap(features, np.zeros(4), 2, (2, 2))
        self.assertEqual(method, "gradcam_exact_linear_gap")
        self.assertEqual(status, "ready")
        self.assertEqual(heatmap.shape, (2, 2))
        self.assertGreater(float(heatmap.max()), 0.0)

    def test_v2_runtime_preprocessing_matches_training(self) -> None:
        predictor = object.__new__(RetinaSathiPredictor)
        predictor.image_size = 64
        predictor.mean = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)[:, None, None]
        predictor.std = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)[:, None, None]
        predictor.manifest = {"preprocessing": {"enhancement": "ben_graham", "color_normalization": False}}
        array = np.zeros((48, 80, 3), dtype=np.uint8)
        array[5:43, 12:70] = (130, 70, 35)
        image = Image.fromarray(array)
        actual = predictor._preprocess(image)[0]
        expected_image = preprocess_fundus(image, 64, "ben_graham", False)
        expected = (np.asarray(expected_image, dtype=np.float32).transpose(2, 0, 1) / 255.0 - predictor.mean) / predictor.std
        self.assertTrue(np.allclose(actual, expected))

    def test_segmentation_runtime_uses_manifest_bilinear_resize(self) -> None:
        predictor = object.__new__(RetinaSathiPredictor)
        predictor.mean = np.zeros((3, 1, 1), dtype=np.float32)
        predictor.std = np.ones((3, 1, 1), dtype=np.float32)
        pixels = np.arange(6 * 10 * 3, dtype=np.uint8).reshape(6, 10, 3) + 12
        image = Image.fromarray(pixels)
        actual = predictor._fundus_tensor(
            image,
            8,
            {
                "enhancement": "none",
                "color_normalization": False,
                "resampling": "bilinear",
            },
        )[0]
        square = Image.new("RGB", (10, 10))
        square.paste(image, (0, 2))
        expected_image = square.resize((8, 8), Image.Resampling.BILINEAR)
        expected = np.asarray(expected_image, dtype=np.float32).transpose(2, 0, 1) / 255.0
        self.assertTrue(np.array_equal(actual, expected))

    def test_learned_quality_can_conservatively_reject_a_heuristic_pass(self) -> None:
        class PoorQualitySession:
            def run(self, _outputs, _inputs):
                return np.asarray([[5.0, -5.0]], dtype=np.float32), np.asarray([[0.8, 0.2, 0.3]], dtype=np.float32)

        predictor = object.__new__(RetinaSathiPredictor)
        predictor.mean = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)[:, None, None]
        predictor.std = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)[:, None, None]
        predictor.quality_session = PoorQualitySession()
        predictor.quality_manifest = {"input_size": 64, "good_probability_threshold": 0.5}
        image = QualityGateTests.fundus_like(128)
        result = predictor._assess_quality(image)
        self.assertEqual(result["label"], "poor")
        self.assertIn("Artifact may obscure retinal detail", result["issues"])
        self.assertIn("Fine retinal detail is unclear", result["issues"])

    def test_dr_only_candidate_does_not_fabricate_dme(self) -> None:
        class Session:
            def run(self, _outputs, _inputs):
                return np.asarray([[0.85, -0.4, -1.4, -2.2]],dtype=np.float32),np.zeros((1,3),dtype=np.float32),np.ones((1,2,4,4),dtype=np.float32)
        class Auxiliary:
            def analyze(self,_image,_tensor):
                return {"status":"not_trained","vessels":{"status":"not_trained"},"optic_disc":{"status":"not_trained"},"fovea":{"status":"not_trained"}},{"status":"not_trained","experimental":True,"items":[]}
        predictor=object.__new__(RetinaSathiPredictor);predictor.session=Session();predictor.auxiliary=Auxiliary();predictor.quality_session=None;predictor.quality_manifest={};predictor.manifest={};predictor.image_size=32;predictor.mean=np.asarray((.485,.456,.406),dtype=np.float32)[:,None,None];predictor.std=np.asarray((.229,.224,.225),dtype=np.float32)[:,None,None];predictor.status="candidate";predictor.dme_available=False;predictor.runtime_device="cpu";predictor.version="test";predictor.dataset="IDRiD";predictor.temperature=1.0;predictor.referable_threshold=.5
        result=predictor.predict(QualityGateTests.fundus_like(128))
        self.assertIsNone(result["dme_risk"]);self.assertEqual(result["dme"]["status"],"unavailable");self.assertEqual(result["dme_probabilities"],[])
        self.assertFalse(result["referable_dr"]);self.assertEqual(result["uncertainty"]["label"],"high");self.assertEqual(result["recommendation"]["urgency"],"review")

    def test_separate_referable_head_can_override_exact_grade_argmax(self) -> None:
        class Auxiliary:
            def analyze(self, _image, _tensor):
                return {"status": "not_trained", "vessels": {"status": "not_trained"}, "optic_disc": {"status": "not_trained"}, "fovea": {"status": "not_trained"}}, {"status": "not_trained", "experimental": True, "items": []}

        class BinaryHeadPredictor(RetinaSathiPredictor):
            def _prediction_signals(self, _tensor):
                probabilities = np.asarray([0.72, 0.12, 0.08, 0.05, 0.03], dtype=np.float32)
                return {
                    "raw_grade_probabilities": probabilities,
                    "grade_probabilities": probabilities,
                    "dme_probabilities": np.asarray([], dtype=np.float32),
                    "referable_score": 0.91,
                    "feature_map": np.ones((1, 1, 4, 4), dtype=np.float32),
                    "explanation_logits": probabilities,
                }

        predictor = object.__new__(BinaryHeadPredictor)
        predictor.auxiliary = Auxiliary(); predictor.quality_session = None; predictor.quality_manifest = {}
        predictor.manifest = {}; predictor.image_size = 32
        predictor.mean = np.asarray((.485, .456, .406), dtype=np.float32)[:, None, None]
        predictor.std = np.asarray((.229, .224, .225), dtype=np.float32)[:, None, None]
        predictor.status = "candidate"; predictor.dme_available = False; predictor.runtime_device = "cpu"
        predictor.version = "v3.4-test"; predictor.dataset = "test"; predictor.temperature = 1.0
        predictor.referable_threshold = 0.5

        result = predictor.predict(QualityGateTests.fundus_like(128))
        self.assertEqual(result["dr_grade"], 0)
        self.assertTrue(result["referable_dr"])
        self.assertEqual(result["recommendation"]["urgency"], "refer")


if __name__ == "__main__":
    unittest.main()
