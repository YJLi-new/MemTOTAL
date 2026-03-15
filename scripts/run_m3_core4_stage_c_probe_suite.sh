#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

SEED_BASE="${1:-3201}"
RUNS_ROOT="${2:-/root/autodl-tmp/memtotal-stage-c-probe-suite}"
REPORT_DIR="${3:-results/generated/m3-core4-stage-c-probe-suite}"
QWEN25_STAGEB="${4:-runs/verify/m3-core4-qwen25/stage-b}"
QWEN3_STAGEB="${5:-runs/verify/m3-core4-qwen3/stage-b}"
if [[ $# -gt 0 ]]; then
  shift
fi
if [[ $# -gt 0 ]]; then
  shift
fi
if [[ $# -gt 0 ]]; then
  shift
fi
if [[ $# -gt 0 ]]; then
  shift
fi
if [[ $# -gt 0 ]]; then
  shift
fi
EXTRA_ARGS=("$@")

mkdir -p "${RUNS_ROOT}" "${REPORT_DIR}"
QWEN25_SEED="$((SEED_BASE + 0))"
QWEN3_SEED="$((SEED_BASE + 10))"

run_stage_c() {
  local config_path="$1"
  local seed="$2"
  local output_dir="$3"
  local resume_dir="$4"
  if python - <<PY
from memtotal.analysis.m3_probe import probe_run_matches_config
import sys
matched = probe_run_matches_config(${output_dir@Q}, ${config_path@Q}, ${seed})
sys.exit(0 if matched else 1)
PY
  then
    echo "reuse ${output_dir}"
    return
  fi
  ./scripts/run_train.sh \
    --config "${config_path}" \
    --seed "${seed}" \
    --output_dir "${output_dir}" \
    --resume "${resume_dir}" \
    "${EXTRA_ARGS[@]}"
}

run_gradient_audit() {
  local config_path="$1"
  local seed="$2"
  local output_dir="$3"
  local resume_dir="$4"
  if python - <<PY
from memtotal.analysis.m3_probe import probe_run_matches_config
import sys
matched = probe_run_matches_config(${output_dir@Q}, ${config_path@Q}, ${seed})
sys.exit(0 if matched else 1)
PY
  then
    echo "reuse ${output_dir}"
    return
  fi
  ./scripts/run_analysis.sh \
    --config "${config_path}" \
    --seed "${seed}" \
    --output_dir "${output_dir}" \
    --resume "${resume_dir}" \
    "${EXTRA_ARGS[@]}"
}

run_stage_c "configs/exp/m3_stage_c_core4_qwen25_smoke.yaml" "${QWEN25_SEED}" "${RUNS_ROOT}/qwen25-q-only" "${QWEN25_STAGEB}"
run_stage_c "configs/exp/m3_stage_c_core4_qwen25_w_only_smoke.yaml" "${QWEN25_SEED}" "${RUNS_ROOT}/qwen25-w-only" "${QWEN25_STAGEB}"
run_stage_c "configs/exp/m3_stage_c_core4_qwen25_w_plus_q_smoke.yaml" "${QWEN25_SEED}" "${RUNS_ROOT}/qwen25-w-plus-q" "${QWEN25_STAGEB}"
run_stage_c "configs/exp/m3_stage_c_core4_qwen3_smoke.yaml" "${QWEN3_SEED}" "${RUNS_ROOT}/qwen3-q-only" "${QWEN3_STAGEB}"
run_stage_c "configs/exp/m3_stage_c_core4_qwen3_w_only_smoke.yaml" "${QWEN3_SEED}" "${RUNS_ROOT}/qwen3-w-only" "${QWEN3_STAGEB}"
run_stage_c "configs/exp/m3_stage_c_core4_qwen3_w_plus_q_smoke.yaml" "${QWEN3_SEED}" "${RUNS_ROOT}/qwen3-w-plus-q" "${QWEN3_STAGEB}"

run_gradient_audit "configs/exp/m3_stage_c_gradient_audit_qwen25.yaml" "${QWEN25_SEED}" "${RUNS_ROOT}/qwen25-q-only-gradient-audit" "${QWEN25_STAGEB}"
run_gradient_audit "configs/exp/m3_stage_c_gradient_audit_qwen3.yaml" "${QWEN3_SEED}" "${RUNS_ROOT}/qwen3-q-only-gradient-audit" "${QWEN3_STAGEB}"

./scripts/run_analysis.sh \
  --config configs/exp/m3_stage_c_probe_summary.yaml \
  --seed "${SEED_BASE}" \
  --output_dir "${REPORT_DIR}" \
  --input_root "${RUNS_ROOT}" \
  "${EXTRA_ARGS[@]}"
