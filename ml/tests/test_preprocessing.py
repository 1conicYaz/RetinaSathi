from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parents[2]))
from ml.preprocessing import crop_fundus, preprocess_fundus


class FundusPreprocessingTests(unittest.TestCase):
    def test_black_border_is_removed_without_cropping_visible_pixels(self) -> None:
        pixels = np.zeros((120, 160, 3), dtype=np.uint8)
        pixels[20:101, 30:131] = (120, 60, 40)
        cropped = np.asarray(crop_fundus(Image.fromarray(pixels), padding_fraction=0))
        self.assertEqual(cropped.shape[:2], (81, 101))
        self.assertTrue((cropped[[0, -1], :, :].max(axis=2) > 8).all())

    def test_preprocessing_preserves_full_frame_and_is_deterministic(self) -> None:
        rng = np.random.default_rng(26038)
        image = Image.fromarray(rng.integers(0, 255, (180, 240, 3), dtype=np.uint8))
        first = np.asarray(preprocess_fundus(image, 384, enhancement="ben_graham"))
        second = np.asarray(preprocess_fundus(image, 384, enhancement="ben_graham"))
        self.assertEqual(first.shape, (384, 384, 3))
        np.testing.assert_array_equal(first, second)

    def test_clahe_and_color_normalization_are_configurable(self) -> None:
        image = Image.new("RGB", (128, 96), (90, 50, 30))
        result = preprocess_fundus(image, 64, enhancement="clahe", color_normalization=True)
        self.assertEqual(result.size, (64, 64))

    def test_unknown_enhancement_fails_loudly(self) -> None:
        with self.assertRaises(ValueError):
            preprocess_fundus(Image.new("RGB", (32, 32)), 32, enhancement="invented")


if __name__ == "__main__":
    unittest.main()
