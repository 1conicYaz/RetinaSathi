"""Deployment runtime for the frozen RetinaSathi V3.4 ONNX candidate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

try:
    from .auxiliary_runtime import AuxiliaryRuntime
    from .model_runtime import RetinaSathiPredictor
except ImportError:  # Supports the flat Docker application directory.
    from auxiliary_runtime import AuxiliaryRuntime
    from model_runtime import RetinaSathiPredictor


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class V34OnnxPredictor(RetinaSathiPredictor):
    """Run V3.4 with its exported preprocessing and calibration contract."""

    def __init__(self, model_path: Path, manifest_path: Path, validation_path: Path | None = None) -> None:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_digest = str(manifest["onnx"]["sha256"])
        actual_digest = _sha256(model_path)
        if actual_digest != expected_digest:
            raise ValueError("V3.4 ONNX hash does not match its deployment manifest")

        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.enable_cpu_mem_arena = False
        options.enable_mem_pattern = False
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        options.add_session_config_entry("session.disable_prepacking", "1")
        self.session = ort.InferenceSession(
            str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        output_names = [output.name for output in self.session.get_outputs()]
        expected_outputs = list(manifest["outputs"])
        if output_names != expected_outputs:
            raise ValueError(f"Unexpected V3.4 ONNX outputs: {output_names}")

        calibration = manifest["calibration"]
        preprocessing = manifest["input"]["preprocessing"]
        self.version = str(manifest["model_version"])
        self.dataset = "APTOS_2019+IDRiD+DeepDRiD"
        self.image_size = int(manifest["input"]["shape"][-1])
        self.status = "candidate"
        self.dme_available = False
        self.runtime_device = "cpu_onnxruntime"
        self.explainability_mode = "unavailable_in_onnx_cloud_runtime"
        self.temperature = float(calibration["nominal_temperature"])
        self.ordinal_temperature = float(calibration["ordinal_temperature"])
        self.nominal_temperature = float(calibration["nominal_temperature"])
        self.binary_temperature = float(calibration["binary_temperature"])
        self.referable_threshold = float(calibration["referable_threshold"])
        self.metrics: dict[str, object] = {}
        if validation_path is not None:
            validation = json.loads(validation_path.read_text(encoding="utf-8"))
            if validation.get("official_test_used") is not False:
                raise ValueError("V3.4 cloud candidate expects a pre-locked-test validation report")
            if abs(float(validation["ordinal_temperature"]) - self.ordinal_temperature) > 1e-7:
                raise ValueError("V3.4 ordinal calibration does not match its validation report")
            if abs(float(validation["nominal_temperature"]) - self.nominal_temperature) > 1e-7:
                raise ValueError("V3.4 nominal calibration does not match its validation report")
            if abs(float(validation["binary_temperature"]) - self.binary_temperature) > 1e-7:
                raise ValueError("V3.4 binary calibration does not match its validation report")
            self.metrics = dict(validation["validation"])
        self.manifest = {
            **manifest,
            "preprocessing": preprocessing,
            "dme_head": False,
            "official_test_used": False,
            "validation": self.metrics,
        }
        self.mean = np.asarray(manifest["input"]["normalization_mean_rgb"], dtype=np.float32)[:, None, None]
        self.std = np.asarray(manifest["input"]["normalization_std_rgb"], dtype=np.float32)[:, None, None]
        self._load_quality_model()
        self.auxiliary = AuxiliaryRuntime()

    def _prediction_signals(self, tensor: np.ndarray) -> dict[str, object]:
        referable_logit, ordinal_logits, nominal_logits = self.session.run(None, {"image": tensor})
        raw_ordinal = self._grade_probabilities(ordinal_logits[0])
        raw_nominal = self._softmax(nominal_logits[0])
        raw_grade = 0.5 * raw_ordinal + 0.5 * raw_nominal
        calibrated_ordinal = self._grade_probabilities(ordinal_logits[0], self.ordinal_temperature)
        calibrated_nominal = self._softmax(nominal_logits[0] / self.nominal_temperature)
        calibrated_grade = 0.5 * calibrated_ordinal + 0.5 * calibrated_nominal
        scaled_binary = np.clip(referable_logit[0, 0] / self.binary_temperature, -60.0, 60.0)
        referable_score = float(1.0 / (1.0 + np.exp(-scaled_binary)))
        return {
            "raw_grade_probabilities": raw_grade,
            "grade_probabilities": calibrated_grade,
            "dme_probabilities": np.asarray([], dtype=np.float32),
            "referable_score": referable_score,
            "feature_map": None,
            "explanation_logits": nominal_logits[0],
        }
