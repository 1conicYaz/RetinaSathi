from __future__ import annotations

import io
import os
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

try:
    from .model_runtime import RetinaSathiPredictor
except ImportError:  # Supports `uvicorn app:app` from the compute directory.
    from model_runtime import RetinaSathiPredictor


MODEL_PATH = Path(os.getenv("MODEL_PATH", "artifacts/idrid_multitask.onnx"))
MAX_IMAGE_BYTES = 15 * 1024 * 1024
allowed_origins = [origin.strip() for origin in os.getenv("ALLOWED_ORIGINS", "*").split(",") if origin.strip()]

app = FastAPI(title="RetinaSathi Inference API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials="*" not in allowed_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

predictor: RetinaSathiPredictor | None = None


def api_error(status_code: int, code: str, message: str, next_action: str) -> HTTPException:
    return HTTPException(status_code=status_code, detail={"code": code, "message": message, "next_action": next_action})


@app.on_event("startup")
def load_model() -> None:
    global predictor
    v34_onnx_manifest = os.getenv("V3_4_ONNX_MANIFEST")
    if v34_onnx_manifest and MODEL_PATH.exists():
        try:
            from .model_runtime_v3_4_onnx import V34OnnxPredictor
        except ImportError:
            from model_runtime_v3_4_onnx import V34OnnxPredictor
        validation_path = os.getenv("V3_4_VALIDATION_REPORT")
        predictor = V34OnnxPredictor(
            MODEL_PATH,
            Path(v34_onnx_manifest),
            Path(validation_path) if validation_path else None,
        )
        return
    v34_checkpoint = os.getenv("V3_4_CHECKPOINT")
    v34_validation = os.getenv("V3_4_VALIDATION")
    if v34_checkpoint and v34_validation:
        try:
            from .model_runtime_v3_4 import V34LocalPredictor
        except ImportError:
            from model_runtime_v3_4 import V34LocalPredictor
        predictor = V34LocalPredictor(Path(v34_checkpoint), Path(v34_validation))
        return
    checkpoint = os.getenv("PYTORCH_CHECKPOINT")
    validation = os.getenv("PYTORCH_VALIDATION")
    if checkpoint and validation:
        try:
            from .model_runtime_torch import LocalPyTorchPredictor
        except ImportError:
            from model_runtime_torch import LocalPyTorchPredictor
        evaluation = os.getenv("PYTORCH_EVALUATION")
        predictor = LocalPyTorchPredictor(Path(checkpoint), Path(validation), Path(evaluation) if evaluation else None)
        return
    if MODEL_PATH.exists():
        predictor = RetinaSathiPredictor(MODEL_PATH)


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "healthy" if predictor else "model_missing",
        "model_loaded": predictor is not None,
        "model_version": predictor.version if predictor else None,
        "quality_model_loaded": predictor.quality_session is not None if predictor else False,
    }


@app.get("/model-card")
def model_card() -> dict[str, object]:
    if predictor is None:
        raise api_error(503, "MODEL_UNAVAILABLE", "Model checkpoint is not available.", "Start the local model service or use the cloud backup.")
    validation_limit = (
        "V3.4 has source validation but still needs a one-time locked test and prospective clinical evaluation"
        if predictor.version.startswith("classifier-v3.4")
        else "Candidate models are not final until locked-test and external evaluation are complete"
    )
    explanation_limit = (
        "Cloud ONNX V3.4 does not provide an attention map; the experimental gradient-based view requires the local PyTorch runtime"
        if predictor.explainability_mode == "unavailable_in_onnx_cloud_runtime"
        else "Explanations show model attention and are not confirmed lesions"
    )
    return {
        "name": "RetinaSathi diabetic-retinopathy screening candidate",
        "version": predictor.version,
        "dataset": predictor.dataset,
        "status": predictor.status,
        "metrics": predictor.manifest.get("validation", predictor.metrics),
        "calibration_status": "calibrated" if predictor.temperature is not None else "not_calibrated",
        "referable_threshold": predictor.referable_threshold,
        "explainability_mode": predictor.explainability_mode,
        "quality_model": {
            "status": "ready" if predictor.quality_session is not None else "heuristic",
            "version": predictor.quality_manifest.get("model_version", "deterministic-quality-v1"),
            "validation": predictor.quality_manifest.get("validation"),
        },
        "intended_use": "Hackathon research prototype for DR screening assistance",
        "limitations": [
            "Trained on a small research dataset",
            "Requires prospective clinical validation",
            explanation_limit,
            "Poor-quality images must be retaken",
            validation_limit,
        ],
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict[str, object]:
    if predictor is None:
        raise api_error(503, "MODEL_UNAVAILABLE", "Model checkpoint is not available.", "Start the local model service or use the cloud backup.")
    if file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise api_error(415, "INVALID_FILE", "Upload a JPEG, PNG, or WebP retinal image.", "Choose a supported fundus image file.")
    content = await file.read(MAX_IMAGE_BYTES + 1)
    if len(content) > MAX_IMAGE_BYTES:
        raise api_error(413, "IMAGE_TOO_LARGE", "Image exceeds the 15 MB limit.", "Export a smaller JPEG or PNG without removing retinal detail.")
    try:
        with Image.open(io.BytesIO(content)) as source:
            image = source.convert("RGB")
    except (UnidentifiedImageError, OSError) as error:
        raise api_error(400, "CORRUPT_IMAGE", "The uploaded file is not a readable image.", "Open the source image locally, then recapture or export it again.") from error
    try:
        return predictor.predict(image)
    except Exception as error:
        raise api_error(500, "INFERENCE_FAILED", "The model could not complete this screening.", "Retry once, then switch runtime or contact the technical operator.") from error
