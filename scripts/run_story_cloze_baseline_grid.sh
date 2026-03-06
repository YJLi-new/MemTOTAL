#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

SEED="${1:-991}"
OUTPUT_DIR="${2:-results/generated/m5-story-cloze-baseline-grid-smoke}"
if [[ $# -gt 0 ]]; then
  shift
fi
if [[ $# -gt 0 ]]; then
  shift
fi

python -m memtotal.baselines.grid_runner \
  --config configs/exp/m5_story_cloze_baseline_grid_smoke.yaml \
  --seed "${SEED}" \
  --output_dir "${OUTPUT_DIR}" \
  "$@"
