#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

SEED="${1:-1001}"
OUTPUT_DIR="${2:-results/generated/m5-story-cloze-baseline-grid-protocol-with-memgen-dual-smoke}"
if [[ $# -gt 0 ]]; then
  shift
fi
if [[ $# -gt 0 ]]; then
  shift
fi

python -m memtotal.baselines.grid_runner \
  --config configs/exp/m5_story_cloze_baseline_grid_protocol_with_memgen_dual_smoke.yaml \
  --seed "${SEED}" \
  --output_dir "${OUTPUT_DIR}" \
  "$@"
