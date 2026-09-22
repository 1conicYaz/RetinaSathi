from __future__ import annotations

import unittest

import torch
from torch import nn


class FakeBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(384, 384)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.projection(value)


class FakeDINO(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.patch = nn.Linear(3, 384)
        self.blocks = nn.ModuleList([FakeBlock() for _ in range(4)])
        self.norm = nn.LayerNorm(384)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        value = self.patch(image.mean(dim=(2, 3)))
        for block in self.blocks:
            value = block(value)
        return self.norm(value)


class V34ModelTests(unittest.TestCase):
    def make_model(self):
        from ml.models.classifier_v3_4 import PartialDINOv2MultiHeadDRClassifier

        return PartialDINOv2MultiHeadDRClassifier(FakeDINO(), dropout=0.0, unfreeze_blocks=2)

    def test_only_final_blocks_norm_and_heads_are_trainable(self) -> None:
        model = self.make_model()
        names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
        self.assertFalse(any(name.startswith("backbone.patch") for name in names))
        self.assertFalse(any(name.startswith("backbone.blocks.0") for name in names))
        self.assertFalse(any(name.startswith("backbone.blocks.1") for name in names))
        self.assertTrue(any(name.startswith("backbone.blocks.2") for name in names))
        self.assertTrue(any(name.startswith("backbone.blocks.3") for name in names))
        self.assertTrue(any(name.startswith("backbone.norm") for name in names))
        self.assertTrue(any(name.startswith("referable_head") for name in names))

    def test_gradients_reach_only_unfrozen_backbone_layers(self) -> None:
        model = self.make_model()
        outputs = model(torch.randn(2, 3, 28, 28))
        sum(output.sum() for output in outputs).backward()
        self.assertIsNone(model.backbone.blocks[1].projection.weight.grad)
        self.assertIsNotNone(model.backbone.blocks[2].projection.weight.grad)
        self.assertIsNotNone(model.backbone.blocks[3].projection.weight.grad)
        self.assertIsNotNone(model.backbone.norm.weight.grad)

    def test_train_mode_is_limited_to_adapted_layers(self) -> None:
        model = self.make_model()
        model.train()
        self.assertFalse(model.backbone.training)
        self.assertFalse(model.backbone.blocks[1].training)
        self.assertTrue(model.backbone.blocks[2].training)
        self.assertTrue(model.backbone.norm.training)

    def test_three_heads_have_expected_shapes(self) -> None:
        model = self.make_model()
        binary, ordinal, nominal = model(torch.randn(2, 3, 28, 28))
        self.assertEqual(tuple(binary.shape), (2,))
        self.assertEqual(tuple(ordinal.shape), (2, 4))
        self.assertEqual(tuple(nominal.shape), (2, 5))

    def test_rejects_invalid_unfreeze_count(self) -> None:
        from ml.models.classifier_v3_4 import PartialDINOv2MultiHeadDRClassifier

        with self.assertRaisesRegex(ValueError, "unfreeze_blocks"):
            PartialDINOv2MultiHeadDRClassifier(FakeDINO(), unfreeze_blocks=0)


if __name__ == "__main__":
    unittest.main()
