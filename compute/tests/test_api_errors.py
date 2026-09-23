from __future__ import annotations

import asyncio
import io
import sys
import unittest
from pathlib import Path

from fastapi import HTTPException, UploadFile
from PIL import Image
from starlette.datastructures import Headers

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app as api


class StructuredApiErrorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.previous = api.predictor
        self.previous_api_key = api.INFERENCE_API_KEY
        api.predictor = object()
        api.INFERENCE_API_KEY = ""

    def tearDown(self) -> None:
        api.predictor = self.previous
        api.INFERENCE_API_KEY = self.previous_api_key

    def upload(self, content: bytes, media_type: str) -> UploadFile:
        return UploadFile(io.BytesIO(content), filename="sample", headers=Headers({"content-type": media_type}))

    def assert_code(self, upload: UploadFile, code: str) -> None:
        with self.assertRaises(HTTPException) as caught:
            asyncio.run(api.predict(upload))
        self.assertEqual(caught.exception.detail["code"], code)
        self.assertTrue(caught.exception.detail["next_action"])

    def test_invalid_media_type_has_machine_code(self) -> None:
        self.assert_code(self.upload(b"text", "text/plain"), "INVALID_FILE")

    def test_corrupt_image_has_machine_code(self) -> None:
        self.assert_code(self.upload(b"not-an-image", "image/png"), "CORRUPT_IMAGE")

    def test_oversized_image_has_machine_code(self) -> None:
        self.assert_code(self.upload(b"0" * (api.MAX_IMAGE_BYTES + 1), "image/jpeg"), "IMAGE_TOO_LARGE")

    def test_predict_requires_configured_inference_key(self) -> None:
        api.INFERENCE_API_KEY = "server-secret"
        with self.assertRaises(HTTPException) as caught:
            asyncio.run(api.predict(self.upload(b"not-an-image", "image/png"), "wrong-secret"))
        self.assertEqual(caught.exception.status_code, 401)
        self.assertEqual(caught.exception.detail["code"], "UNAUTHORIZED")

    def test_decoded_pixel_limit_is_enforced(self) -> None:
        previous_pixels = api.MAX_IMAGE_PIXELS
        api.MAX_IMAGE_PIXELS = 100
        output = io.BytesIO()
        Image.new("RGB", (11, 11), "red").save(output, format="PNG")
        try:
            self.assert_code(self.upload(output.getvalue(), "image/png"), "IMAGE_DIMENSIONS_TOO_LARGE")
        finally:
            api.MAX_IMAGE_PIXELS = previous_pixels


if __name__ == "__main__":
    unittest.main()
