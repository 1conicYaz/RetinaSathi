#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

checkpoint="${1:-runs/v3_4_seed_26038_mps_progress_20260920_gpu/best.pt}"
validation="${2:-runs/v3_4_seed_26038_mps_progress_20260920_gpu/final_validation.json}"
api_pid=""
cleanup() { [[ -z "$api_pid" ]] || kill "$api_pid" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

V3_4_CHECKPOINT="$checkpoint" V3_4_VALIDATION="$validation" PROCESSING_MODE=local \
  .venv/bin/uvicorn compute.app:app --host 127.0.0.1 --port 8000 &
api_pid=$!

VITE_LOCAL_INFERENCE_URL=http://127.0.0.1:8000 \
VITE_INFERENCE_MODE=local \
VITE_MODEL_GENERATION=v3.4 \
VITE_LOCAL_MODEL_LABEL="Local V3.4 candidate" \
  npm run dev -- --host 127.0.0.1
