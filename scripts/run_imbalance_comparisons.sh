#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

aptos_checkpoint="${1:?Usage: $0 RUN/best_aptos_2019.pt}"

.venv/bin/python -m ml.training.pilot_classifier \
  --data-root "${RETINASATHI_DATA_ROOT:?Set RETINASATHI_DATA_ROOT}" \
  --backbone-checkpoint "$aptos_checkpoint" \
  --dataset IDRiD \
  --architectures efficientnet_b3 \
  --objectives coral \
  --input-sizes 384 \
  --dme-weights 0.25 \
  --imbalance-strategies sampler_only weighted_loss weighted_sampler none \
  --subset-strategy full \
  --epochs 2 \
  --per-class-validation 15 \
  --batch-size 4 \
  --output artifacts/classifier_imbalance_summary.json
