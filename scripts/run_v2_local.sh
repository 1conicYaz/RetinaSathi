#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 CHECKPOINT VALIDATION_JSON [LOCKED_EVALUATION_JSON]" >&2
  exit 2
fi
api_pid=""
cleanup() { [[ -z "$api_pid" ]] || kill "$api_pid" 2>/dev/null || true; }
trap cleanup EXIT INT TERM
if [[ $# -eq 3 ]]; then
  PYTORCH_CHECKPOINT="$1" PYTORCH_VALIDATION="$2" PYTORCH_EVALUATION="$3" PROCESSING_MODE=local \
    .venv/bin/uvicorn compute.app:app --host 127.0.0.1 --port 8000 &
else
  PYTORCH_CHECKPOINT="$1" PYTORCH_VALIDATION="$2" PROCESSING_MODE=local \
    .venv/bin/uvicorn compute.app:app --host 127.0.0.1 --port 8000 &
fi
api_pid=$!
VITE_INFERENCE_URL=http://127.0.0.1:8000 VITE_INFERENCE_MODE=local npm run dev -- --host 127.0.0.1
