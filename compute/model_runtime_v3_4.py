"""Local runtime for the frozen RetinaSathi V3.4 DINOv2 research candidate."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from compute.auxiliary_runtime import AuxiliaryRuntime
from compute.model_runtime import RetinaSathiPredictor
from ml.artifact_integrity import sha256
from ml.models import load_partial_dinov2_vits14, ordinal_probabilities
from ml.training.train_classifier_v2 import device_for_training


class V34LocalPredictor(RetinaSathiPredictor):
    """Run one frozen V3.4 checkpoint with its saved calibration policy."""

    def __init__(self, checkpoint_path: Path, validation_path: Path) -> None:
        checkpoint_path = checkpoint_path.resolve()
        validation_path = validation_path.resolve()
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        checkpoint_digest = sha256(checkpoint_path)
        if validation.get("checkpoint_sha256") != checkpoint_digest:
            raise ValueError("V3.4 validation report is not bound to this checkpoint")
        if validation.get("manifest_sha256") != state.get("manifest_sha256"):
            raise ValueError("V3.4 checkpoint and validation report use different data manifests")
        if validation.get("official_test_used") is not False:
            raise ValueError("V3.4 local candidate expects a pre-locked-test validation report")

        config = state["config"]
        repository_root = checkpoint_path.parents[2]
        provenance_path = Path(str(config["experiment"]["foundation_provenance"]))
        if not provenance_path.is_absolute():
            provenance_path = repository_root / provenance_path
        self.model, foundation = load_partial_dinov2_vits14(
            provenance_path,
            dropout=float(config["training"]["dropout"]),
            unfreeze_blocks=int(config["training"]["unfreeze_blocks"]),
        )
        self.model.load_state_dict(state["model"], strict=True)
        self.device = device_for_training()
        self.model.to(self.device).eval()

        self.version = f"classifier-v3.4-seed26038-{checkpoint_digest[:12]}"
        self.dataset = "APTOS_2019+IDRiD+DeepDRiD"
        self.image_size = int(config["preprocessing"]["input_size"])
        self.status = "candidate"
        self.dme_available = False
        self.runtime_device = str(self.device)
        # The previous patch-token map was constant on audit probes. Keep V3.4
        # explanations disabled until a target- and input-sensitive method passes validation.
        self.explainability_mode = "disabled_not_technically_verified"
        self.temperature = float(validation["nominal_temperature"])
        self.ordinal_temperature = float(validation["ordinal_temperature"])
        self.nominal_temperature = float(validation["nominal_temperature"])
        self.binary_temperature = float(validation["binary_temperature"])
        self.referable_threshold = float(validation["threshold_gate"]["selected"]["threshold"])
        self.model_sha256 = checkpoint_digest
        self.config_sha256 = sha256(validation_path)
        self.architecture = "dinov2_vits14"
        self.calibration_version = f"validation-sha256:{self.config_sha256[:12]}"
        self.manifest = {
            "model_version": self.version,
            "input_size": self.image_size,
            "objective": "ordinal_nominal_binary_multi_head",
            "dme_head": False,
            "preprocessing": config["preprocessing"],
            "validation": validation["source_validation"],
            "foundation_model": foundation,
            "official_test_used": False,
        }
        self.metrics = validation["source_validation"]
        self.mean = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)[:, None, None]
        self.std = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)[:, None, None]
        self._load_quality_model()
        self.auxiliary = AuxiliaryRuntime()

    def _prediction_signals(self, tensor: np.ndarray) -> dict[str, object]:
        image = torch.from_numpy(tensor).to(self.device)
        with torch.inference_mode():
            binary_logits, ordinal_logits, nominal_logits = self.model(image)
            raw_ordinal = ordinal_probabilities(ordinal_logits, 1.0)
            raw_nominal = torch.softmax(nominal_logits, dim=1)
            raw_grade = 0.5 * raw_ordinal + 0.5 * raw_nominal
            calibrated_ordinal = ordinal_probabilities(ordinal_logits, self.ordinal_temperature)
            calibrated_nominal = torch.softmax(nominal_logits / self.nominal_temperature, dim=1)
            calibrated_grade = 0.5 * calibrated_ordinal + 0.5 * calibrated_nominal
            referable_score = torch.sigmoid(binary_logits / self.binary_temperature)

        return {
            "raw_grade_probabilities": raw_grade[0].detach().cpu().numpy(),
            "grade_probabilities": calibrated_grade[0].detach().cpu().numpy(),
            "dme_probabilities": np.asarray([], dtype=np.float32),
            "referable_score": float(referable_score[0].detach().cpu()),
            "feature_map": None,
            "explanation_logits": calibrated_grade[0].detach().cpu().numpy(),
        }
