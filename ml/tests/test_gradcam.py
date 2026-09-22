from __future__ import annotations

import unittest

import torch
from torch import nn

from ml.explainability import GradCAM, attention_mask_metrics, ordinal_severity_score


class TinyClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__(); self.features = nn.Conv2d(3, 4, 3, padding=1); self.head = nn.Linear(4, 2)

    def forward(self, image):
        features = torch.relu(self.features(image)); return self.head(features.mean(dim=(-2, -1)))


class GradCAMTests(unittest.TestCase):
    def test_true_gradient_map_matches_input_size_and_range(self) -> None:
        torch.manual_seed(26038); model = TinyClassifier(); image = torch.randn(1, 3, 24, 24)
        with GradCAM(model, model.features) as gradcam:
            heatmap = gradcam(image, lambda logits: logits[:, 1])
        self.assertEqual(tuple(heatmap.shape), (1, 1, 24, 24))
        self.assertGreaterEqual(float(heatmap.min()), 0)
        self.assertLessEqual(float(heatmap.max()), 1)
        self.assertGreater(float(heatmap.max()), 0)

    def test_ordinal_severity_target_produces_true_gradient_map(self) -> None:
        class TinyOrdinal(nn.Module):
            def __init__(self) -> None:
                super().__init__(); self.features = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.ReLU()); self.head = nn.Linear(4, 4)

            def forward(self, image):
                features = self.features(image); return self.head(features.mean(dim=(-2, -1))), None

        torch.manual_seed(26038); model = TinyOrdinal(); image = torch.randn(1, 3, 16, 16)
        with torch.no_grad():
            model.head.weight.fill_(0.25)
            model.head.bias.zero_()
        with GradCAM(model, model.features) as gradcam:
            heatmap = gradcam(image, ordinal_severity_score)
        self.assertEqual(tuple(heatmap.shape), (1, 1, 16, 16))
        self.assertGreater(float(heatmap.max()), 0)

        # For global-average-pooling plus a linear head, the deployed analytic
        # form is exactly Grad-CAM up to the spatial constant removed by
        # min/max normalization.
        with torch.no_grad():
            activation = model.features(image)
            logits = model.head(activation.mean(dim=(-2, -1)))
            cumulative = torch.sigmoid(logits)
            coefficients = cumulative * (1 - cumulative)
            channel_weights = coefficients @ model.head.weight
            analytic = torch.relu((activation * channel_weights[:, :, None, None]).sum(dim=1, keepdim=True))
            analytic = (analytic - analytic.amin()) / (analytic.amax() - analytic.amin()).clamp_min(1e-8)
        self.assertTrue(torch.allclose(heatmap, analytic, atol=1e-5))

    def test_attention_validation_reports_overlap_and_border_leakage(self) -> None:
        heatmap = torch.zeros(8, 8).numpy(); heatmap[3:5, 3:5] = 1
        lesion = torch.zeros(8, 8, dtype=torch.bool).numpy(); lesion[3:5, 3:5] = True
        retina = torch.zeros(8, 8, dtype=torch.bool).numpy(); retina[1:7, 1:7] = True
        metrics = attention_mask_metrics(heatmap, lesion, retina)
        self.assertTrue(metrics["pointing_game_hit"])
        self.assertEqual(metrics["attention_outside_retina_fraction"], 0.0)
        self.assertGreater(metrics["top_20pct_attention_lesion_iou"], 0.0)


if __name__ == "__main__": unittest.main()
