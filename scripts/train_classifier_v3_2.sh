#!/usr/bin/env bash
set -euo pipefail

.venv/bin/python -m ml.training.train_classifier_v3_2 \
  --data-root "${RETINASATHI_DATA_ROOT:?Set RETINASATHI_DATA_ROOT}" \
  "$@"
