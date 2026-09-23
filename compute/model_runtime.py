from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageFilter

try:
    from .auxiliary_runtime import AuxiliaryRuntime
except ImportError:  # Supports the flat Docker application directory.
    from auxiliary_runtime import AuxiliaryRuntime


GRADE_LABELS = (
    "No diabetic retinopathy",
    "Mild non-proliferative DR",
    "Moderate non-proliferative DR",
    "Severe non-proliferative DR",
    "Proliferative diabetic retinopathy",
)
DME_LABELS = ("No apparent DME risk", "Possible DME risk", "High DME risk")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class RetinaSathiPredictor:
    def __init__(self, model_path: Path) -> None:
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.enable_cpu_mem_arena = False
        self.session = ort.InferenceSession(str(model_path), sess_options=options, providers=["CPUExecutionProvider"])
        self.version = "idrid-mobilenetv3-multitask-v0.1"
        self.dataset = "IDRiD"
        self.image_size = 224
        self.status = "baseline_v1"
        self.dme_available = True
        self.runtime_device = "cpu"
        self.explainability_mode = "baseline_feature_activation"
        self.temperature: float | None = None
        self.referable_threshold = 0.5
        manifest_path = model_path.with_suffix(".manifest.json")
        self.manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        if self.manifest:
            self.version = str(self.manifest["model_version"])
            self.dataset = "+".join(self.manifest.get("datasets", ["IDRiD"]))
            self.image_size = int(self.manifest["input_size"])
            self.dme_available = bool(self.manifest.get("dme_head", False))
            self.temperature = float(self.manifest["temperature"])
            self.referable_threshold = float(self.manifest["referable_threshold"])
            self.status = "ready" if self.manifest.get("status") == "locked_tested" else "candidate"
            self.explainability_mode = str(self.manifest.get("explainability", {}).get("method", "unavailable"))
        self.model_sha256 = _sha256_file(model_path)
        self.config_sha256 = _sha256_bytes(
            json.dumps(self.manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        self.architecture = str(self.manifest.get("architecture", "MobileNetV3"))
        self.calibration_version = f"manifest-sha256:{self.config_sha256[:12]}"
        self.mean = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)[:, None, None]
        self.std = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)[:, None, None]
        metrics_path = model_path.with_suffix(".metrics.json")
        self.metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
        self._load_quality_model()
        self.auxiliary = AuxiliaryRuntime()

    def _load_quality_model(self) -> None:
        self.quality_session: ort.InferenceSession | None = None
        self.quality_manifest: dict[str, object] = {}
        quality_path_value = os.getenv("QUALITY_MODEL_PATH")
        if not quality_path_value:
            return
        quality_path = Path(quality_path_value)
        manifest_path = quality_path.with_suffix(".manifest.json")
        if not quality_path.exists() or not manifest_path.exists():
            raise FileNotFoundError("QUALITY_MODEL_PATH requires both ONNX and manifest files")
        self.quality_session = ort.InferenceSession(str(quality_path), providers=["CPUExecutionProvider"])
        self.quality_manifest = json.loads(manifest_path.read_text())

    @staticmethod
    def quality(image: Image.Image) -> dict[str, float | str | list[str]]:
        resized_rgb = np.asarray(image.convert("RGB").resize((256, 256)), dtype=np.float32) / 255.0
        sample = np.asarray(image.convert("L").resize((256, 256)), dtype=np.float32) / 255.0
        yy, xx = np.ogrid[:256, :256]
        radius_squared = (xx - 127.5) ** 2 + (yy - 127.5) ** 2
        circular_field = radius_squared < 121**2
        boundary_ring = (radius_squared > 105**2) & (radius_squared < 118**2)
        foreground = (sample > 0.045) & circular_field
        outer_ring = ~circular_field
        dark_outer_fraction = float((sample[outer_ring] < 0.08).mean())
        boundary_coverage = float((sample[boundary_ring] > 0.045).mean())
        retinal_pixels = sample[foreground]
        if retinal_pixels.size < 100:
            return {"score": 0.0, "label": "poor", "issues": ["Retina is not visible"], "brightness": 0.0, "contrast": 0.0, "sharpness": 0.0}
        brightness = float(retinal_pixels.mean())
        contrast = float(retinal_pixels.std())
        foreground_fraction = float(foreground.sum() / circular_field.sum())
        channel_means = resized_rgb[foreground].mean(axis=0)
        resembles_fundus = bool(channel_means[0] > channel_means[1] * 1.05 and channel_means[0] > channel_means[2] * 1.10)
        vertical = np.abs(np.diff(sample, axis=0))
        horizontal = np.abs(np.diff(sample, axis=1))
        gradient = float((vertical[(foreground[1:] & foreground[:-1])].mean() + horizontal[(foreground[:, 1:] & foreground[:, :-1])].mean()) / 2)
        # Sensor noise and checkerboards can satisfy simple brightness/contrast tests.
        # A very high local gradient is therefore treated as unsupported input.
        excessive_texture = gradient > 0.065
        field_boundary_supported = dark_outer_fraction >= 0.45 and boundary_coverage >= 0.60
        exposure = max(0.0, 1.0 - abs(brightness - 0.38) / 0.32)
        score = float(0.35 * exposure + 0.25 * min(1.0, contrast / 0.08) + 0.40 * min(1.0, gradient / 0.008))
        issues: list[str] = []
        if brightness < 0.22:
            issues.append("Image is too dark")
        elif brightness > 0.58:
            issues.append("Image is overexposed")
        if contrast < 0.045:
            issues.append("Retinal contrast is low")
        if gradient < 0.0045:
            issues.append("Image may be blurred")
        elif excessive_texture:
            issues.append("Image contains excessive high-frequency noise or a non-retinal pattern")
        if foreground_fraction < 0.55:
            issues.append("Retinal field of view is incomplete")
        if not field_boundary_supported:
            issues.append("Circular retinal field boundary is not visible")
        if not resembles_fundus:
            issues.append("Image does not resemble a color fundus photograph")
        label = "good" if score >= 0.68 else "usable" if score >= 0.45 else "poor"
        if foreground_fraction < 0.55 or not field_boundary_supported or not resembles_fundus or excessive_texture:
            label = "poor"
        return {"score": round(score, 4), "label": label, "issues": issues, "brightness": round(brightness, 4), "contrast": round(contrast, 4), "sharpness": round(gradient, 4)}

    def _preprocess(self, image: Image.Image) -> np.ndarray:
        if self.manifest.get("preprocessing"):
            settings = self.manifest["preprocessing"]
            return self._fundus_tensor(image, self.image_size, settings)
        width, height = image.size
        scale = 256 / min(width, height)
        resized = image.resize((round(width * scale), round(height * scale)), Image.Resampling.BILINEAR)
        left = (resized.width - self.image_size) // 2
        top = (resized.height - self.image_size) // 2
        crop = resized.crop((left, top, left + self.image_size, top + self.image_size))
        tensor = np.asarray(crop, dtype=np.float32).transpose(2, 0, 1) / 255.0
        return ((tensor - self.mean) / self.std)[None]

    def _fundus_tensor(self, image: Image.Image, image_size: int, settings: dict[str, object]) -> np.ndarray:
        image = image.convert("RGB")
        rgb = np.asarray(image)
        visible = rgb.max(axis=2) > 8
        rows = np.flatnonzero(visible.any(axis=1)); columns = np.flatnonzero(visible.any(axis=0))
        if rows.size and columns.size:
            padding = round(min(rgb.shape[:2]) * 0.01)
            top = max(0, int(rows[0]) - padding); bottom = min(rgb.shape[0], int(rows[-1]) + padding + 1)
            left = max(0, int(columns[0]) - padding); right = min(rgb.shape[1], int(columns[-1]) + padding + 1)
            image = Image.fromarray(rgb[top:bottom, left:right])
        size = max(image.size)
        square = Image.new("RGB", (size, size), (0, 0, 0))
        square.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
        resampling = (
            Image.Resampling.BILINEAR
            if settings.get("resampling") == "bilinear"
            else Image.Resampling.LANCZOS
        )
        image = square.resize((image_size, image_size), resampling)
        if settings.get("color_normalization"):
            normalized = np.asarray(image, dtype=np.float32)
            means = normalized.reshape(-1, 3).mean(axis=0)
            normalized *= (float(means.mean()) / np.maximum(means, 1e-6))[None, None, :]
            image = Image.fromarray(np.clip(normalized, 0, 255).astype(np.uint8))
        enhancement = settings.get("enhancement", "none")
        if enhancement == "ben_graham":
            raw = np.asarray(image, dtype=np.float32)
            sigma = max(1.0, min(image.size) * 0.035)
            blurred = np.asarray(image.filter(ImageFilter.GaussianBlur(sigma)), dtype=np.float32)
            image = Image.fromarray(np.clip(4.0 * raw - 4.0 * blurred + 128.0, 0, 255).astype(np.uint8))
        elif enhancement != "none":
            raise ValueError(f"Unsupported deployed preprocessing enhancement: {enhancement}")
        tensor = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
        return ((tensor - self.mean) / self.std)[None]

    def _assess_quality(self, image: Image.Image) -> dict[str, object]:
        heuristic = self.quality(image)
        if heuristic["label"] == "poor" or self.quality_session is None:
            return heuristic
        size = int(self.quality_manifest["input_size"])
        logits, attributes = self.quality_session.run(None, {"image": self._fundus_tensor(image, size, {"enhancement": "none", "color_normalization": False})})
        good_probability = float(self._softmax(logits[0])[1])
        artifact, clarity, field = (float(value) for value in attributes[0])
        threshold = float(self.quality_manifest["good_probability_threshold"])
        learned_label = "poor" if good_probability < threshold else "usable" if good_probability < max(0.75, threshold) else "good"
        rank = {"poor": 0, "usable": 1, "good": 2}
        label = min(str(heuristic["label"]), learned_label, key=lambda value: rank[value])
        issues = list(heuristic["issues"])
        if artifact > 0.50: issues.append("Artifact may obscure retinal detail")
        if clarity < 0.50: issues.append("Fine retinal detail is unclear")
        if field < 0.50: issues.append("Optic disc or macula may be poorly framed")
        return {**heuristic, "label": label, "issues": list(dict.fromkeys(issues)), "learned_good_probability": round(good_probability, 5), "learned_attributes": {"artifact": round(artifact, 5), "clarity": round(clarity, 5), "field_definition": round(field, 5)}}

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - logits.max()
        values = np.exp(shifted)
        return values / values.sum()

    @staticmethod
    def _grade_probabilities(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        if logits.shape[-1] == 5:
            return RetinaSathiPredictor._softmax(logits / temperature)
        if logits.shape[-1] != 4:
            raise ValueError(f"Expected four ordinal or five nominal logits, received {logits.shape[-1]}")
        scaled = np.clip(logits / temperature, -60.0, 60.0)
        cumulative = 1 / (1 + np.exp(-scaled))
        cumulative = np.minimum.accumulate(cumulative)
        endpoints = np.concatenate(([1.0], cumulative, [0.0]))
        probabilities = np.maximum(endpoints[:-1] - endpoints[1:], 0)
        return probabilities / max(probabilities.sum(), 1e-8)

    @staticmethod
    def _heatmap(feature_map: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
        heatmap = np.maximum(feature_map[0], 0).mean(axis=0)
        heatmap -= heatmap.min()
        maximum = float(heatmap.max())
        if maximum > 0:
            heatmap /= maximum
        raw = Image.fromarray((heatmap * 255).astype(np.uint8))
        return np.asarray(raw.resize(image_size, Image.Resampling.BILINEAR), dtype=np.float32) / 255.0

    def _explanation_heatmap(
        self,
        feature_map: np.ndarray,
        grade_logits: np.ndarray,
        grade: int,
        image_size: tuple[int, int],
    ) -> tuple[np.ndarray, str, str]:
        explanation = self.manifest.get("explainability", {})
        if explanation.get("method") != "gradcam_exact_linear_gap":
            return self._heatmap(feature_map, image_size), "channel_mean_feature_activation_not_gradcam", "baseline_feature_activation"
        head_weights = np.asarray(explanation["grade_head_weights"], dtype=np.float32)
        objective = self.manifest.get("objective")
        if objective == "coral":
            scaled = np.clip(grade_logits / (self.temperature or 1.0), -60.0, 60.0)
            cumulative = 1.0 / (1.0 + np.exp(-scaled))
            derivatives = cumulative * (1.0 - cumulative) / (self.temperature or 1.0)
            channel_weights = derivatives @ head_weights
        else:
            channel_weights = head_weights[grade]
        raw_heatmap = np.maximum((feature_map[0] * channel_weights[:, None, None]).sum(axis=0), 0.0)
        raw_heatmap -= raw_heatmap.min()
        maximum = float(raw_heatmap.max())
        if maximum > 0:
            raw_heatmap /= maximum
        raw = Image.fromarray((raw_heatmap * 255).astype(np.uint8))
        resized = np.asarray(raw.resize(image_size, Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
        return resized, "gradcam_exact_linear_gap", "ready"

    def _prediction_signals(self, tensor: np.ndarray) -> dict[str, object]:
        """Return calibrated classifier signals behind the stable API contract."""
        grade_logits, dme_logits, feature_map = self.session.run(None, {"image": tensor})
        raw_grade_probabilities = self._grade_probabilities(grade_logits[0])
        grade_probabilities = self._grade_probabilities(grade_logits[0], self.temperature or 1.0)
        dme_probabilities = self._softmax(dme_logits[0]) if self.dme_available else np.asarray([], dtype=np.float32)
        return {
            "raw_grade_probabilities": raw_grade_probabilities,
            "grade_probabilities": grade_probabilities,
            "dme_probabilities": dme_probabilities,
            "referable_score": float(grade_probabilities[2:].sum()),
            "feature_map": feature_map,
            "explanation_logits": grade_logits[0],
        }

    @staticmethod
    def _overlay(image: Image.Image, heatmap: np.ndarray) -> str:
        display = image.convert("RGB")
        display.thumbnail((900, 900), Image.Resampling.LANCZOS)
        raw = Image.fromarray((heatmap * 255).astype(np.uint8))
        resized_heatmap = np.asarray(raw.resize(display.size, Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
        rgb = np.asarray(display, dtype=np.float32)
        color = np.zeros_like(rgb)
        color[..., 0] = 255 * resized_heatmap
        color[..., 1] = 190 * np.clip((resized_heatmap - 0.35) / 0.65, 0, 1)
        alpha = (0.55 * resized_heatmap)[..., None]
        combined = np.clip(rgb * (1 - alpha) + color * alpha, 0, 255).astype(np.uint8)
        output = io.BytesIO()
        Image.fromarray(combined).save(output, format="JPEG", quality=88, optimize=True)
        return "data:image/jpeg;base64," + base64.b64encode(output.getvalue()).decode("ascii")

    @staticmethod
    def _heatmap_image(heatmap: np.ndarray) -> str:
        """Render model influence as a standalone, high-contrast colour image."""
        values = np.clip(heatmap, 0.0, 1.0)
        anchors = np.asarray(
            [
                [18, 20, 76],
                [20, 92, 170],
                [23, 190, 207],
                [245, 224, 77],
                [238, 82, 44],
                [126, 20, 38],
            ],
            dtype=np.float32,
        )
        positions = values * (len(anchors) - 1)
        lower = np.floor(positions).astype(np.int32)
        upper = np.minimum(lower + 1, len(anchors) - 1)
        blend = (positions - lower)[..., None]
        rgb = anchors[lower] * (1.0 - blend) + anchors[upper] * blend
        output = io.BytesIO()
        Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).save(output, format="PNG", optimize=True)
        return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode("ascii")

    @staticmethod
    def _attention_regions(heatmap: np.ndarray) -> list[dict[str, float | str]]:
        """Return salient feature-map regions without assigning clinical labels."""
        raw = Image.fromarray((heatmap * 255).astype(np.uint8))
        small_heat = np.asarray(raw.resize((32, 32)), dtype=np.float32) / 255
        candidates: list[dict[str, float | str]] = []
        suppressed = np.zeros_like(small_heat, dtype=bool)
        for _ in range(4):
            available = np.where(suppressed, -1, small_heat)
            index = int(np.argmax(available))
            y, x = np.unravel_index(index, available.shape)
            probability = float(available[y, x])
            if probability < 0.48:
                break
            candidates.append(
                {
                    "region_type": "model_attention",
                    "strength": round(probability, 4),
                    "center_x": round((x + 0.5) / 32, 4),
                    "center_y": round((y + 0.5) / 32, 4),
                    "radius": 0.045,
                    "evidence": "This region influenced the model's screening output.",
                }
            )
            suppressed[max(0, y - 4):min(32, y + 5), max(0, x - 4):min(32, x + 5)] = True
        return candidates

    @staticmethod
    def _image_data_uri(image: Image.Image) -> str:
        display = image.convert("RGB")
        display.thumbnail((900, 900), Image.Resampling.LANCZOS)
        output = io.BytesIO()
        display.save(output, format="JPEG", quality=88, optimize=True)
        return "data:image/jpeg;base64," + base64.b64encode(output.getvalue()).decode("ascii")

    def _ungradeable(self, image: Image.Image, quality: dict[str, object], started_at: float) -> dict[str, object]:
        recommendation = "Retake the image before clinical interpretation. No DR or DME prediction was produced."
        preview = self._image_data_uri(image)
        return {
            "api_version": "2.0",
            "assessment": {"state": "retake_required", "reason": "Image quality gate stopped inference"},
            "model_version": self.version,
            "model_identity": self.runtime_identity(),
            "dataset": self.dataset,
            "quality": {"status": "ready" if self.quality_session is not None else "heuristic", "model_version": str(self.quality_manifest.get("model_version", "deterministic-quality-v1")), **quality},
            "dr": {"status": "unavailable", "grade": None, "label": "Not graded — image ungradeable", "probabilities": [], "confidence_raw": None, "confidence_calibrated": None, "calibration_status": "not_calibrated", "referable": None, "referable_score": None, "referable_threshold": self.referable_threshold, "referable_decision": None},
            "dme": {"status": "unavailable", "risk": None, "label": "Not assessed — image ungradeable", "probabilities": []},
            "structures": {"status": "unavailable", "vessels": {"status": "unavailable"}, "optic_disc": {"status": "unavailable"}, "fovea": {"status": "unavailable"}},
            "lesions": {"status": "unavailable", "experimental": True, "items": []},
            "explainability": {"status": "unavailable", "method": "not_run_quality_gate", "image": preview, "attention_regions": [], "clinical_interpretation": "No explanation was generated because inference was stopped by the quality gate."},
            "recommendation": {"text": recommendation, "urgency": "retake", "requires_human_review": True, "reasons": quality["issues"]},
            "uncertainty": {"label": "unavailable", "requires_manual_review": True, "reason": "Inference stopped at the quality gate"},
            "runtime": {"processing_mode": os.getenv("PROCESSING_MODE", "cloud"), "device": "cpu", "latency_ms": round((time.perf_counter() - started_at) * 1000, 2)},
            "dr_grade": None, "dr_label": "Not graded — image ungradeable", "grade_probabilities": [],
            "dme_risk": None, "dme_label": "Not assessed — image ungradeable", "dme_probabilities": [],
            "confidence": None, "referable_dr": None, "recommendation_text": recommendation,
            "explanation_image": preview, "legacy_lesions": [],
            "disclaimer": "Research screening support only. Not a medical diagnosis.",
        }

    def runtime_identity(self) -> dict[str, object]:
        """Return the immutable identity used by health, model-card and predictions."""
        return {
            "model_name": "RetinaSathi DR screening candidate",
            "model_version": self.version,
            "architecture": getattr(self, "architecture", "unavailable"),
            "model_sha256": getattr(self, "model_sha256", "0" * 64),
            "config_sha256": getattr(self, "config_sha256", "0" * 64),
            "input_size": self.image_size,
            "referable_threshold": self.referable_threshold,
            "calibration_version": getattr(self, "calibration_version", "unavailable"),
            "explanation_capability": getattr(self, "explainability_mode", "unavailable"),
            "build_commit": os.getenv("BUILD_COMMIT", "unavailable"),
            "deployment_revision": os.getenv("CONTAINER_APP_REVISION")
            or os.getenv("DEPLOYMENT_RELEASE", "unavailable"),
        }

    def predict(self, image: Image.Image) -> dict[str, object]:
        started_at = time.perf_counter()
        image = image.convert("RGB")
        quality = self._assess_quality(image)
        if quality["label"] == "poor":
            return self._ungradeable(image, quality, started_at)
        signals = self._prediction_signals(self._preprocess(image))
        raw_grade_probabilities = np.asarray(signals["raw_grade_probabilities"])
        grade_probabilities = np.asarray(signals["grade_probabilities"])
        dme_probabilities = np.asarray(signals["dme_probabilities"])
        grade = int(grade_probabilities.argmax())
        dme_risk = int(dme_probabilities.argmax()) if self.dme_available else None
        feature_map = signals.get("feature_map")
        if feature_map is None:
            heatmap = None
            explanation_method = "unavailable_in_onnx_cloud_runtime"
            explanation_status = "unavailable"
        else:
            heatmap, explanation_method, explanation_status = self._explanation_heatmap(
                np.asarray(feature_map), np.asarray(signals["explanation_logits"]), grade, image.size
            )
        referable_score = float(signals["referable_score"])
        referable = referable_score >= self.referable_threshold
        confidence = round(float(grade_probabilities[grade]), 5)
        raw_confidence = round(float(raw_grade_probabilities[grade]), 5)
        uncertainty_label = "unavailable" if self.temperature is None else "low" if confidence >= 0.8 else "medium" if confidence >= 0.6 else "high"
        uncertain_action = self.temperature is None or uncertainty_label == "high"
        clinical_action = referable or (dme_risk is not None and dme_risk >= 1)
        if clinical_action:
            recommendation = "Refer to an ophthalmologist for confirmatory examination."
        elif self.temperature is None:
            recommendation = "V1 confidence is not calibrated. Obtain qualified human review before deciding routine follow-up."
        elif uncertain_action:
            recommendation = "Prediction confidence is low. Obtain clinician review or recapture before routine disposition."
        else:
            recommendation = "Routine follow-up screening; review sooner if symptoms develop."
        attention_regions = self._attention_regions(heatmap) if heatmap is not None else []
        explanation_image = self._overlay(image, heatmap) if heatmap is not None else self._image_data_uri(image)
        heatmap_image = self._heatmap_image(heatmap) if heatmap is not None else None
        structures, lesions = self.auxiliary.analyze(image, self._fundus_tensor)
        runtime = {
            "processing_mode": os.getenv("PROCESSING_MODE", "cloud"),
            "device": self.runtime_device,
            "latency_ms": round((time.perf_counter() - started_at) * 1000, 2),
        }
        result = {
            "api_version": "2.0",
            "assessment": {
                "state": "uncertain" if uncertain_action and not clinical_action else "assessed_referable" if referable else "assessed_non_referable",
                "reason": "Low or uncalibrated confidence requires manual review" if uncertain_action and not clinical_action else "Dedicated referral head crossed its threshold" if referable else "Dedicated referral head remained below its threshold",
            },
            "model_version": self.version,
            "model_identity": self.runtime_identity(),
            "dataset": self.dataset,
            "quality": {"status": "ready" if self.quality_session is not None else "heuristic", "model_version": str(self.quality_manifest.get("model_version", "deterministic-quality-v1")), **quality},
            "dr": {
                "status": self.status,
                "grade": grade,
                "label": GRADE_LABELS[grade],
                "probabilities": [round(float(value), 5) for value in grade_probabilities],
                "confidence_raw": raw_confidence,
                "confidence_calibrated": confidence if self.temperature is not None else None,
                "calibration_status": "calibrated" if self.temperature is not None else "not_calibrated",
                "referable": referable,
                "referable_score": round(referable_score, 5),
                "referable_threshold": self.referable_threshold,
                "referable_decision": referable,
            },
            "dme": {
                "status": self.status if self.dme_available else "unavailable",
                "risk": dme_risk,
                "label": DME_LABELS[dme_risk] if dme_risk is not None else "Not assessed — classifier has no DME head",
                "probabilities": [round(float(value), 5) for value in dme_probabilities],
            },
            "structures": structures,
            "lesions": lesions,
            "explainability": {
                "status": explanation_status,
                "method": explanation_method,
                "image": explanation_image,
                "attention_regions": attention_regions,
                "clinical_interpretation": (
                    "Explanation unavailable for this model version because the prior method did not pass technical validation."
                    if heatmap is None
                    else "Model influence only; not a lesion map or anatomical confirmation."
                ),
            },
            "recommendation": {
                "text": recommendation,
                "urgency": "refer" if clinical_action else "review" if uncertain_action else "routine",
                "requires_human_review": True,
                "reasons": ["V1 confidence is uncalibrated"] if self.temperature is None and not clinical_action else ["Low calibrated confidence"] if uncertain_action and not clinical_action else ["Model output requires clinician review"],
            },
            "uncertainty": {"label": uncertainty_label, "requires_manual_review": True, "reason": "V1 confidence is not calibrated" if self.temperature is None else "Confidence is below 0.60" if uncertain_action else "All research outputs require human review"},
            "runtime": runtime,
            # V1 compatibility fields remain until the web client and stored records migrate.
            "dr_grade": grade, "dr_label": GRADE_LABELS[grade],
            "grade_probabilities": [round(float(value), 5) for value in grade_probabilities],
            "dme_risk": dme_risk, "dme_label": DME_LABELS[dme_risk] if dme_risk is not None else "Not assessed — classifier has no DME head",
            "dme_probabilities": [round(float(value), 5) for value in dme_probabilities],
            "confidence": confidence, "referable_dr": referable,
            "recommendation_text": recommendation, "explanation_image": explanation_image,
            "legacy_lesions": [],
            "disclaimer": "Research screening support only. Not a medical diagnosis.",
        }
        if heatmap_image is not None:
            result["explainability"]["heatmap_image"] = heatmap_image
        return result
