#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

aptos_checkpoint="${1:?Usage: $0 RUN/best_aptos_2019.pt}"

.venv/bin/python -m ml.training.pilot_classifier \
  --data-root "${RETINASATHI_DATA_ROOT:?Set RETINASATHI_DATA_ROOT}" \
  --backbone-checkpoint "$aptos_checkpoint" \
  --dataset IDRiD \
  --architectures efficientnet_b3 \
  --objectives coral cross_entropy \
  --input-sizes 384 512 \
  --dme-weights 0 0.25 \
  --imbalance-strategies weighted_loss \
  --epochs 2 \
  --per-class-train 30 \
  --per-class-validation 15 \
  --batch-size 4 \
  --output artifacts/classifier_comparison_summary.json
