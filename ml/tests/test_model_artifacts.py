from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from ml.artifact_integrity import (
    require_internal_validation_binding,
    require_locked_evaluation_binding,
    sha256,
)
from ml.export_onnx import ExportableRetinaSathi
from ml.model import RetinaSathiNet


class ModelArtifactTests(unittest.TestCase):
    def test_validation_and_locked_evaluation_bind_exact_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder) / "candidate.pt"
            checkpoint.write_bytes(b"checkpoint-a")
            digest = sha256(checkpoint)
            validation = {
                "official_test_used": False,
                "checkpoint_sha256": digest,
            }
            validation_path = Path(folder) / "validation.json"
            validation_path.write_text(json.dumps(validation))
            evaluation = {
                "status": "locked_test_and_external_evaluation_complete",
                "official_test_used_after_lock": True,
                "checkpoint_sha256": digest,
                "validation_sha256": sha256(validation_path),
            }
            self.assertEqual(
                require_internal_validation_binding(checkpoint, validation), digest
            )
            self.assertEqual(
                require_locked_evaluation_binding(
                    checkpoint, evaluation, validation_path
                ),
                digest,
            )

            mismatched = dict(validation, checkpoint_sha256="0" * 64)
            with self.assertRaisesRegex(ValueError, "does not match"):
                require_internal_validation_binding(checkpoint, mismatched)
            premature = dict(evaluation, official_test_used_after_lock=False)
            with self.assertRaisesRegex(ValueError, "after lock"):
                require_locked_evaluation_binding(checkpoint, premature)
            wrong_validation = Path(folder) / "wrong-validation.json"
            wrong_validation.write_text("{}")
            with self.assertRaisesRegex(ValueError, "validation artifact"):
                require_locked_evaluation_binding(
                    checkpoint, evaluation, wrong_validation
                )

    def test_checkpoint_restores_model_and_optimizer_state(self) -> None:
        model = torch.nn.Linear(3, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        loss = model(torch.ones(1, 3)).sum()
        loss.backward()
        optimizer.step()
        payload = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": 1}
        buffer = io.BytesIO()
        torch.save(payload, buffer)
        buffer.seek(0)
        restored = torch.load(buffer, map_location="cpu", weights_only=False)
        next_model = torch.nn.Linear(3, 2)
        next_optimizer = torch.optim.AdamW(next_model.parameters(), lr=0.01)
        next_model.load_state_dict(restored["model"])
        next_optimizer.load_state_dict(restored["optimizer"])
        self.assertEqual(restored["epoch"], 1)
        torch.testing.assert_close(next_model.weight, model.weight)
        self.assertEqual(len(next_optimizer.state), len(optimizer.state))

    def test_onnx_matches_pytorch_numerically(self) -> None:
        torch.manual_seed(26038)
        wrapper = ExportableRetinaSathi(RetinaSathiNet(pretrained=False)).eval()
        sample = torch.randn(1, 3, 64, 64)
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "smoke.onnx"
            with torch.inference_mode():
                expected = [tensor.numpy() for tensor in wrapper(sample)]
                torch.onnx.export(
                    wrapper,
                    sample,
                    output,
                    input_names=["image"],
                    output_names=["grade_logits", "dme_logits", "feature_map"],
                    opset_version=17,
                    dynamo=False,
                )
            onnx.checker.check_model(onnx.load(output))
            session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
            actual = session.run(None, {"image": sample.numpy()})
        for torch_value, onnx_value in zip(expected, actual, strict=True):
            np.testing.assert_allclose(onnx_value, torch_value, rtol=1e-4, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
