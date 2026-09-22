#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
.venv/bin/python -m ml.training.train_localization --data-root "${RETINASATHI_DATA_ROOT:?Set RETINASATHI_DATA_ROOT}" "$@"
