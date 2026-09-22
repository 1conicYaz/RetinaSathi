"""Validation-only temperature scaling for ordinal logits."""

from __future__ import annotations

import torch

from ml.models.classifier_v2 import coral_targets


def fit_temperature(logits: torch.Tensor, grades: torch.Tensor, iterations: int = 50, objective: str = "coral") -> float:
    log_temperature = torch.zeros(1, requires_grad=True, device=logits.device)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=iterations)
    if objective not in {"coral", "cross_entropy"}:
        raise ValueError(f"Unsupported calibration objective: {objective}")
    targets = coral_targets(grades) if objective == "coral" else grades

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        temperature = log_temperature.exp().clamp(0.05, 10.0)
        scaled = logits / temperature
        loss = torch.nn.functional.binary_cross_entropy_with_logits(scaled, targets) if objective == "coral" else torch.nn.functional.cross_entropy(scaled, targets)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.detach().exp().clamp(0.05, 10.0).cpu())


def fit_binary_temperature(logits: torch.Tensor, targets: torch.Tensor, iterations: int = 50) -> float:
    log_temperature = torch.zeros(1, requires_grad=True, device=logits.device)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=iterations)

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        temperature = log_temperature.exp().clamp(0.05, 10.0)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits / temperature, targets.float())
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.detach().exp().clamp(0.05, 10.0).cpu())
