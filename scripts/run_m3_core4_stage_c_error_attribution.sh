#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

SEED_BASE="${1:-56201}"
RUNS_ROOT="${2:-/root/autodl-tmp/memtotal-stage-c-qonly-seed-sweep-v6-case-dump}"
REPORT_DIR="${3:-results/generated/m3-core4-stage-c-error-attribution-v1}"
shift $(( $# > 0 ? 1 : 0 ))
shift $(( $# > 0 ? 1 : 0 ))
shift $(( $# > 0 ? 1 : 0 ))
EXTRA_ARGS=("$@")

mkdir -p "${REPORT_DIR}"

./scripts/run_analysis.sh \
  --config configs/exp/m3_stage_c_error_attribution.yaml \
  --seed "${SEED_BASE}" \
  --output_dir "${REPORT_DIR}" \
  --input_root "${RUNS_ROOT}" \
  "${EXTRA_ARGS[@]}"
