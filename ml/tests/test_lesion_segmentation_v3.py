from __future__ import annotations

import unittest

import numpy as np
import torch
from PIL import Image

from ml.lesions import extract_lesion_regions, gaussian_importance_map, overlay_lesions, retinal_field_mask, thresholds_from_manifest
from ml.models import MobileUNet, focal_tversky_loss
from ml.training.train_lesion_segmentation_v3 import select_thresholds


class LesionSegmentationV3Tests(unittest.TestCase):
    def test_loss_is_finite_and_differentiable(self):
        logits=torch.randn(2,4,32,32,requires_grad=True); targets=torch.zeros_like(logits); targets[:,:,15:17,15:17]=1
        loss=focal_tversky_loss(logits,targets); loss.backward()
        self.assertTrue(torch.isfinite(loss)); self.assertIsNotNone(logits.grad)

    def test_mobile_unet_preserves_spatial_shape(self):
        model=MobileUNet(4,pretrained=False).eval()
        with torch.no_grad():output=model(torch.zeros(1,3,128,128))
        self.assertEqual(tuple(output.shape),(1,4,128,128))

    def test_tiny_foreground_receives_more_loss_than_all_background(self):
        logits=torch.full((1,4,16,16),-2.0); empty=torch.zeros_like(logits); tiny=empty.clone(); tiny[:,0,8,8]=1
        self.assertGreater(float(focal_tversky_loss(logits,tiny)),float(focal_tversky_loss(logits,empty)))

    def test_class_specific_threshold_selection(self):
        counts=np.zeros((2,2,3),dtype=np.int64); counts[0]=[[8,8,2],[2,0,8]]; counts[1]=[[6,1,4],[8,2,2]]
        thresholds,metrics=select_thresholds(counts,[.2,.6],["a","b"])
        self.assertEqual(thresholds,{"a":.6,"b":.6}); self.assertGreaterEqual(metrics["b"]["dice"],.8)

    def test_regions_return_coordinates_and_overlay(self):
        probabilities=np.zeros((2,20,20),dtype=np.float32); probabilities[0,2:5,3:8]=.9; probabilities[1,10:12,10:12]=.8
        masks,regions=extract_lesion_regions(probabilities,["ma","he"],np.asarray([.5,.7]),minimum_pixels=3)
        self.assertEqual(regions[0]["type"],"ma"); self.assertEqual(regions[0]["x"],3)
        self.assertEqual(overlay_lesions(Image.new("RGB",(20,20)),masks).size,(20,20))

    def test_manifest_supports_per_class_and_legacy_thresholds(self):
        classes=["a","b"]
        np.testing.assert_allclose(thresholds_from_manifest({"thresholds":{"a":.2,"b":.7}},classes),[.2,.7])
        np.testing.assert_allclose(thresholds_from_manifest({"threshold":.5},classes),[.5,.5])

    def test_gaussian_blending_downweights_tile_edges(self):
        weights=gaussian_importance_map(32)
        self.assertEqual(weights.shape,(32,32))
        self.assertGreater(weights[16,16],weights[0,0])
        self.assertGreater(float(weights.min()),0)

    def test_retinal_field_mask_removes_camera_border_margin(self):
        rgb=np.zeros((100,100,3),dtype=np.uint8);rgb[10:90,10:90]=120
        field=retinal_field_mask(rgb,.05)
        self.assertTrue(field[50,50]);self.assertFalse(field[11,50]);self.assertFalse(field[5,50])


if __name__=="__main__": unittest.main()
