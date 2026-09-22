from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "compute"))
from model_runtime_v3_4_onnx import V34OnnxPredictor


class V34OnnxRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = ROOT / "models" / "retinasathi-v3-4.onnx"
        cls.manifest = ROOT / "models" / "retinasathi-v3-4.manifest.json"
        cls.validation = ROOT / "runs" / "v3_4_seed_26038_mps_progress_20260920_gpu" / "final_validation.json"
        if not cls.model.exists():
            raise unittest.SkipTest("V3.4 ONNX artifact is not present")
        cls.predictor = V34OnnxPredictor(cls.model, cls.manifest, cls.validation)

    def test_contract_uses_calibrated_three_head_output(self) -> None:
        rng = np.random.default_rng(26038)
        texture = np.clip(rng.normal(95, 25, (512, 512)), 0, 255).astype(np.uint8)
        image = Image.fromarray(
            np.stack([np.clip(texture * 1.4, 0, 255), texture, texture // 2], axis=-1).astype(np.uint8)
        )
        result = self.predictor.predict(image)
        self.assertEqual(result["model_version"], "classifier-v3.4-seed26038")
        self.assertEqual(len(result["dr"]["probabilities"]), 5)
        self.assertAlmostEqual(sum(result["dr"]["probabilities"]), 1.0, places=4)
        self.assertEqual(result["dr"]["calibration_status"], "calibrated")
        self.assertEqual(result["dme"]["status"], "unavailable")
        self.assertEqual(result["explainability"]["status"], "unavailable")
        self.assertNotIn("heatmap_image", result["explainability"])
        self.assertEqual(result["explainability"]["attention_regions"], [])
        self.assertAlmostEqual(self.predictor.metrics["referable"]["sensitivity"], 0.9316037735849056)


if __name__ == "__main__":
    unittest.main()
