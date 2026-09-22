#!/usr/bin/env python3
"""Create local-only Python reference tensors and logits for MATLAB parity."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image
from scipy.io import savemat

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from ml.preprocessing import preprocess_fundus


MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--per-grade", type=int, default=0)
    parser.add_argument("images", nargs="*", type=Path)
    args = parser.parse_args()
    image_records: list[tuple[Path, dict[str, object]]] = [
        (path, {"selection": "explicit"}) for path in args.images
    ]
    if args.per_grade:
        if args.manifest is None or args.data_root is None:
            parser.error("--manifest and --data-root are required with --per-grade")
        rows = list(csv.DictReader(args.manifest.open(encoding="utf-8")))
        for grade in range(5):
            candidates = sorted(
                (
                    row
                    for row in rows
                    if row["role"] == "development_pool"
                    and int(row["mapped_icdr_grade"]) == grade
                    and row["validation_status"] == "valid"
                ),
                key=lambda row: (row["source_name"], row["image_uid"]),
            )[: args.per_grade]
            if len(candidates) != args.per_grade:
                raise ValueError(f"Grade {grade} has only {len(candidates)} eligible parity cases")
            for row in candidates:
                image_records.append(
                    (
                        args.data_root / row["relative_path"],
                        {
                            "selection": "deterministic_development_pool",
                            "image_uid": row["image_uid"],
                            "source_name": row["source_name"],
                            "mapped_icdr_grade": grade,
                        },
                    )
                )
    if not image_records:
        parser.error("Provide image paths or use --per-grade with a manifest and data root")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    session = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    records = []
    for index, (image_path, metadata) in enumerate(image_records, start=1):
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        image = Image.open(image_path).convert("RGB")
        processed = preprocess_fundus(image, 392, "ben_graham", False)
        pixels = np.asarray(processed, dtype=np.float32) / 255.0
        normalized_hwc = (pixels - MEAN) / STD
        tensor = normalized_hwc.transpose(2, 0, 1)[None].astype(np.float32)
        outputs = session.run(None, {"image": tensor})
        stem = f"case_{index:02d}"
        copied_image = args.output_dir / f"{stem}_input{image_path.suffix.lower()}"
        copied_image.write_bytes(image_path.read_bytes())
        processed.save(args.output_dir / f"{stem}_python_enhanced.png")
        savemat(
            args.output_dir / f"{stem}_python_reference.mat",
            {
                "normalizedHWC": normalized_hwc,
                "referableLogit": outputs[0],
                "ordinalLogits": outputs[1],
                "nominalLogits": outputs[2],
            },
        )
        records.append({"case": stem, "copied_input": copied_image.name, **metadata})
    (args.output_dir / "fixture_manifest.json").write_text(
        json.dumps({"schema_version": 1, "cases": records}, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(records)} local parity cases to {args.output_dir}")


if __name__ == "__main__":
    main()
