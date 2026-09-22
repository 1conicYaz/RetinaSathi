"""True gradient-weighted class activation maps for local PyTorch models."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn

from ml.models import ordinal_probabilities


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self.forward_hook = target_layer.register_forward_hook(self._capture_activation)
        self.backward_hook = target_layer.register_full_backward_hook(self._capture_gradient)

    def _capture_activation(self, _module, _inputs, output) -> None:
        self.activations = output

    def _capture_gradient(self, _module, _gradient_input, gradient_output) -> None:
        self.gradients = gradient_output[0]

    def close(self) -> None:
        self.forward_hook.remove(); self.backward_hook.remove()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()

    def __call__(self, image: torch.Tensor, score_fn: Callable[[object], torch.Tensor]) -> torch.Tensor:
        self.model.zero_grad(set_to_none=True)
        if not image.requires_grad:
            image = image.detach().requires_grad_(True)
        output = self.model(image)
        score = score_fn(output)
        score.sum().backward()
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Target layer did not expose activations and gradients")
        weights = self.gradients.mean(dim=(-2, -1), keepdim=True)
        heatmap = torch.relu((weights * self.activations).sum(dim=1, keepdim=True))
        heatmap = nn.functional.interpolate(heatmap, image.shape[-2:], mode="bilinear", align_corners=False)
        minimum = heatmap.amin(dim=(-2, -1), keepdim=True)
        maximum = heatmap.amax(dim=(-2, -1), keepdim=True)
        return ((heatmap - minimum) / (maximum - minimum).clamp_min(1e-8)).detach()


def ordinal_grade_score(output: object, grade: int, temperature: float = 1.0) -> torch.Tensor:
    logits = output[0] if isinstance(output, tuple) else output
    return ordinal_probabilities(logits, temperature)[:, grade]


def ordinal_severity_score(output: object, temperature: float = 1.0) -> torch.Tensor:
    """Differentiable expected severity for CORAL-style cumulative logits."""
    logits = output[0] if isinstance(output, tuple) else output
    return torch.sigmoid(logits / temperature).sum(dim=1)
