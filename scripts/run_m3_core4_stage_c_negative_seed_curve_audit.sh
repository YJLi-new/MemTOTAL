#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

SEED="${1:-40201}"
INPUT_ROOT="${2:-/root/autodl-tmp/memtotal-stage-c-curve-suite-v2}"
REPORT_DIR="${3:-results/generated/m3-core4-stage-c-negative-seed-curve-audit-v1}"
shift $(( $# > 0 ? 1 : 0 ))
shift $(( $# > 0 ? 1 : 0 ))
shift $(( $# > 0 ? 1 : 0 ))
EXTRA_ARGS=("$@")

mkdir -p "${REPORT_DIR}"

./scripts/run_analysis.sh \
  --config configs/exp/m3_stage_c_negative_seed_curve_audit.yaml \
  --seed "${SEED}" \
  --output_dir "${REPORT_DIR}" \
  --input_root "${INPUT_ROOT}" \
  "${EXTRA_ARGS[@]}"
