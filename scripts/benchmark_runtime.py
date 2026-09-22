"""Measure the deployed-compatible ONNX runtime without persisting retinal images."""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import statistics
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "compute"))
from model_runtime import RetinaSathiPredictor
from model_runtime_v3_4_onnx import V34OnnxPredictor


def synthetic_fundus(size: int = 1024) -> Image.Image:
    rng = np.random.default_rng(26038)
    yy, xx = np.ogrid[:size, :size]
    radius = np.sqrt((xx - size / 2) ** 2 + (yy - size / 2) ** 2)
    field = np.clip(1 - radius / (size * 0.49), 0, 1)
    noise = rng.normal(0, 8, (size, size))
    red = np.clip(45 + 105 * field + noise, 0, 255)
    green = np.clip(20 + 60 * field + noise, 0, 255)
    blue = np.clip(15 + 35 * field + noise, 0, 255)
    return Image.fromarray(np.stack([red, green, blue], axis=-1).astype(np.uint8))


def percentile(values: list[float], fraction: float) -> float:
    return float(np.percentile(np.asarray(values), fraction * 100))


def peak_rss_mib() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = 1024**2 if platform.system() == "Darwin" else 1024
    return round(value / divisor, 2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("compute/artifacts/idrid_multitask.onnx"))
    parser.add_argument("--v3-4-manifest", type=Path, help="Use the V3.4 three-head ONNX deployment runtime")
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--quality-model", type=Path)
    parser.add_argument("--lesion-model", type=Path)
    parser.add_argument("--vessel-model", type=Path)
    parser.add_argument("--localization-model", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/performance.json"))
    args = parser.parse_args()
    configured={"QUALITY_MODEL_PATH":args.quality_model,"LESION_MODEL_PATH":args.lesion_model,"VESSEL_MODEL_PATH":args.vessel_model,"LOCALIZATION_MODEL_PATH":args.localization_model}
    for name,path in configured.items():
        if path is not None:os.environ[name]=str(path)
    predictor = (
        V34OnnxPredictor(args.model, args.v3_4_manifest)
        if args.v3_4_manifest is not None
        else RetinaSathiPredictor(args.model)
    )
    image = synthetic_fundus()
    predictor.predict(image)
    preprocessing: list[float] = []
    classifier: list[float] = []
    total: list[float] = []
    auxiliary_times:dict[str,list[float]]={}
    for _ in range(args.runs):
        started = time.perf_counter(); tensor = predictor._preprocess(image); preprocessing.append((time.perf_counter() - started) * 1000)
        started = time.perf_counter(); predictor.session.run(None, {"image": tensor}); classifier.append((time.perf_counter() - started) * 1000)
        started = time.perf_counter(); predictor.predict(image); total.append((time.perf_counter() - started) * 1000)
    for label,session,manifest in (("lesions",predictor.auxiliary.lesion_session,predictor.auxiliary.lesion_manifest),("vessels",predictor.auxiliary.vessel_session,predictor.auxiliary.vessel_manifest)):
        if session is None:continue
        tensor=predictor._fundus_tensor(image,int(manifest["input_size"]),{"enhancement":"none","color_normalization":False});values=[]
        for _ in range(args.runs):
            started=time.perf_counter();session.run(None,{"image":tensor});values.append((time.perf_counter()-started)*1000)
        auxiliary_times[label]=values
    model_sizes={"classifier":round(args.model.stat().st_size/1024**2,2)}
    for label,path in (("quality",args.quality_model),("lesions",args.lesion_model),("vessels",args.vessel_model),("localization",args.localization_model)):
        if path is not None:model_sizes[label]=round(path.stat().st_size/1024**2,2)
    result = {
        "runtime": "onnxruntime_cpu",
        "model_version": predictor.version,
        "runs_after_warmup": args.runs,
        "input": "deterministic_synthetic_fundus_1024px",
        "preprocessing_ms": {"median": round(statistics.median(preprocessing), 2), "p95": round(percentile(preprocessing, 0.95), 2)},
        "classifier_ms": {"median": round(statistics.median(classifier), 2), "p95": round(percentile(classifier, 0.95), 2)},
        "total_screening_ms": {"median": round(statistics.median(total), 2), "p95": round(percentile(total, 0.95), 2)},
        "process_peak_rss_mib": peak_rss_mib(),
        "model_size_mib": model_sizes,
        "segmentation_latency_ms": {label:{"median":round(statistics.median(values),2),"p95":round(percentile(values,0.95),2)} for label,values in auxiliary_times.items()} or None,
        "cloud_configured_memory_mib": 512,
        "limitations": ["Synthetic input benchmark", "Concurrent workloads can affect timings", "Only explicitly supplied auxiliary models are measured", "Cloud live RSS is not exposed by the service"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
