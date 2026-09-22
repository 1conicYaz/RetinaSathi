#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
npm run lint
npm run typecheck
npm run build
.venv/bin/python -m unittest discover -s compute/tests -v
.venv/bin/python -m unittest discover -s ml/tests -v
