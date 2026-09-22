from __future__ import annotations

import unittest

import torch
from torch import nn


class FakeRETFound(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(3, 1024)

    def forward_features(self, image: torch.Tensor) -> torch.Tensor:
        return self.projection(image.mean(dim=(2, 3))).unsqueeze(1)


class V33ModelTests(unittest.TestCase):
    def make_model(self):
        from ml.models.classifier_v3_3 import RETFoundMultiHeadDRClassifier

        return RETFoundMultiHeadDRClassifier(FakeRETFound(), dropout=0.0)

    def test_three_heads_have_expected_shapes(self) -> None:
        model = self.make_model()
        binary, ordinal, nominal = model(torch.randn(2, 3, 32, 32))
        self.assertEqual(tuple(binary.shape), (2,))
        self.assertEqual(tuple(ordinal.shape), (2, 4))
        self.assertEqual(tuple(nominal.shape), (2, 5))

    def test_encoder_is_frozen_and_heads_receive_gradients(self) -> None:
        model = self.make_model()
        self.assertTrue(all(not parameter.requires_grad for parameter in model.backbone.parameters()))
        outputs = model(torch.randn(2, 3, 32, 32))
        sum(output.sum() for output in outputs).backward()
        self.assertTrue(all(parameter.grad is None for parameter in model.backbone.parameters()))
        self.assertIsNotNone(model.referable_head.weight.grad)
        self.assertIsNotNone(model.ordinal_head.weight.grad)
        self.assertIsNotNone(model.nominal_head.weight.grad)

    def test_encoder_remains_in_evaluation_mode(self) -> None:
        model = self.make_model()
        model.train()
        self.assertTrue(model.training)
        self.assertFalse(model.backbone.training)

    def test_rejects_input_not_divisible_by_patch_size(self) -> None:
        model = self.make_model()
        with self.assertRaisesRegex(ValueError, "divisible by 16"):
            model(torch.randn(1, 3, 30, 32))

    def test_head_state_excludes_frozen_encoder(self) -> None:
        model = self.make_model()
        state = model.head_state_dict()
        self.assertEqual(
            set(state),
            {
                "referable_head.weight", "referable_head.bias",
                "ordinal_head.weight", "ordinal_head.bias",
                "nominal_head.weight", "nominal_head.bias",
            },
        )
        self.assertFalse(any(name.startswith("backbone.") for name in state))


if __name__ == "__main__":
    unittest.main()
