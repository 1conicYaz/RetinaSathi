#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m venv .venv
.venv/bin/python -m pip install -r ml/requirements.txt -r compute/requirements.txt
npm install
.venv/bin/python scripts/create_splits.py --data-root "${RETINASATHI_DATA_ROOT:?Set RETINASATHI_DATA_ROOT}"
echo "RetinaSathi local environment is ready."
