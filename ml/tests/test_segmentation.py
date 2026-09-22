from __future__ import annotations

import unittest

import torch
import numpy as np

from ml.datasets import tiled_prediction
from ml.models import CompactUNet, dice_bce_loss
from ml.training.train_segmentation import metrics_from_counts


class SegmentationTests(unittest.TestCase):
    def test_unet_shape_and_loss(self) -> None:
        model=CompactUNet(4,base_channels=4); image=torch.randn(1,3,64,64); target=torch.zeros(1,4,64,64)
        logits=model(image); self.assertEqual(tuple(logits.shape),tuple(target.shape)); self.assertTrue(bool(torch.isfinite(dice_bce_loss(logits,target))))

    def test_tiled_prediction_reconstructs_shape(self) -> None:
        model=CompactUNet(1,base_channels=4).eval(); image=torch.randn(1,3,96,112)
        with torch.no_grad(): output=tiled_prediction(model,image,tile_size=64,overlap=16)
        self.assertEqual(tuple(output.shape),(1,1,96,112))

    def test_empty_ground_truth_is_not_counted_as_perfect_dice(self) -> None:
        metrics=metrics_from_counts(np.asarray([0,4]),np.asarray([0,1]),np.asarray([0,2]))
        self.assertIsNone(metrics["dice_per_class"][0])
        self.assertAlmostEqual(metrics["dice_per_class"][1],8/11)
        self.assertEqual(metrics["ground_truth_positive_pixels"],[0,6])


if __name__=="__main__": unittest.main()
