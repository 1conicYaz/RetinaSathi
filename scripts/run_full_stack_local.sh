#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
cleanup() { kill "$api_pid" 2>/dev/null || true; }
trap cleanup EXIT INT TERM
PROCESSING_MODE=local MODEL_PATH=compute/artifacts/idrid_multitask.onnx .venv/bin/uvicorn compute.app:app --host 127.0.0.1 --port 8000 &
api_pid=$!
VITE_INFERENCE_URL=http://127.0.0.1:8000 VITE_INFERENCE_MODE=local npm run dev -- --host 127.0.0.1
