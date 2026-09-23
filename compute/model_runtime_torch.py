"""Local full-explainability predictor using a candidate or locked PyTorch model."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from compute.model_runtime import RetinaSathiPredictor
from compute.auxiliary_runtime import AuxiliaryRuntime
from ml.explainability import GradCAM, ordinal_severity_score
from ml.artifact_integrity import (
    require_internal_validation_binding,
    require_locked_evaluation_binding,
    sha256,
)
from ml.training.train_classifier_v2 import build_model, device_for_training, grade_probabilities


class _TorchSession:
    def __init__(self, predictor: "LocalPyTorchPredictor") -> None:
        self.predictor = predictor

    def run(self, _outputs, inputs: dict[str, np.ndarray]):
        predictor = self.predictor
        image = torch.from_numpy(inputs["image"]).to(predictor.device)
        with torch.inference_mode():
            grade_logits, dme_logits = predictor.model(image)
            probabilities = grade_probabilities(grade_logits.cpu(), predictor.objective, predictor.temperature or 1.0)
            grade = int(probabilities[0].argmax())
        if predictor.objective == "coral":
            score_fn = lambda output: ordinal_severity_score(output, predictor.temperature or 1.0)
        else:
            score_fn = lambda output: output[0][:, grade]
        with GradCAM(predictor.model, predictor.model.gradcam_layer()) as gradcam:
            predictor.latest_gradcam = gradcam(image, score_fn)[0, 0].cpu().numpy()
        return (
            grade_logits.detach().cpu().numpy(),
            dme_logits.detach().cpu().numpy() if dme_logits is not None else np.zeros((1, 3), dtype=np.float32),
            np.zeros((1, 1, 1, 1), dtype=np.float32),
        )


class LocalPyTorchPredictor(RetinaSathiPredictor):
    """Uses autograd Grad-CAM locally; cloud ONNX remains a lightweight backup."""

    def __init__(self, checkpoint_path: Path, validation_path: Path, evaluation_path: Path | None = None) -> None:
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        validation = json.loads(validation_path.read_text())
        checkpoint_digest = require_internal_validation_binding(
            checkpoint_path, validation
        )
        self.device = device_for_training()
        self.model = build_model(state["config"], pretrained=False)
        self.model.load_state_dict(state["model"])
        self.model.to(self.device).eval()
        self.objective = str(state["config"]["training"]["objective"])
        self.dme_available = self.model.dme_head is not None
        self.version = f"classifier-v2-{checkpoint_digest[:12]}"
        self.dataset = "APTOS_2019+IDRiD"
        self.image_size = int(state["config"]["preprocessing"]["input_size"])
        self.status = "candidate"
        if evaluation_path is not None:
            evaluation = json.loads(evaluation_path.read_text())
            require_locked_evaluation_binding(
                checkpoint_path, evaluation, validation_path
            )
            self.status = "ready"
        self.runtime_device = str(self.device)
        self.explainability_mode = "autograd_gradcam"
        self.temperature = float(validation["temperature"])
        self.referable_threshold = float(validation["referable_threshold"])
        self.model_sha256 = checkpoint_digest
        self.config_sha256 = sha256(validation_path)
        self.architecture = str(state["config"].get("model", {}).get("architecture", "EfficientNet-B3"))
        self.calibration_version = f"validation-sha256:{self.config_sha256[:12]}"
        self.manifest = {
            "model_version": self.version,
            "input_size": self.image_size,
            "objective": self.objective,
            "dme_head": self.dme_available,
            "preprocessing": state["config"]["preprocessing"],
            "validation": validation["validation"],
        }
        self.metrics = validation["validation"]
        self.latest_gradcam: np.ndarray | None = None
        self.mean = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)[:, None, None]
        self.std = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)[:, None, None]
        self._load_quality_model()
        self.auxiliary = AuxiliaryRuntime()
        self.session = _TorchSession(self)

    def _explanation_heatmap(
        self,
        _feature_map: np.ndarray,
        _grade_logits: np.ndarray,
        _grade: int,
        image_size: tuple[int, int],
    ) -> tuple[np.ndarray, str, str]:
        if self.latest_gradcam is None:
            raise RuntimeError("Grad-CAM was not generated")
        raw = Image.fromarray((self.latest_gradcam * 255).astype(np.uint8))
        resized = np.asarray(raw.resize(image_size, Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
        return resized, "gradcam_autograd", "ready"
