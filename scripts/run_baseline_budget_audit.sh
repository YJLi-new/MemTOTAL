#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

python -m analysis \
  --config configs/exp/m5_baseline_budget_audit.yaml \
  --seed "${1:-961}" \
  --output_dir "${2:-results/generated/m5-baseline-budget-audit}" \
  --input_root "${3:-runs/verify}"
