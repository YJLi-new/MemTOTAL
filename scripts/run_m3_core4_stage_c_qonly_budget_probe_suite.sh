#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

SEED_BASE="${1:-3301}"
RUNS_ROOT="${2:-/root/autodl-tmp/memtotal-stage-c-qonly-budget-probe-suite}"
REPORT_DIR="${3:-results/generated/m3-core4-stage-c-qonly-budget-probe-suite}"
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

TMP_CONFIG_DIR="${RUNS_ROOT}/tmp-configs"
mkdir -p "${TMP_CONFIG_DIR}" "${RUNS_ROOT}" "${REPORT_DIR}"

QWEN25_SEED="$((SEED_BASE + 0))"
QWEN3_SEED="$((SEED_BASE + 10))"

write_override() {
  local path="$1"
  local include_path="$2"
  local exp_name="$3"
  local variant_name="$4"
  shift 4
  {
    printf 'includes:\n'
    printf '  - %s\n' "${include_path}"
    printf '\nexperiment:\n'
    printf '  name: %s\n' "${exp_name}"
    printf '  method_variant: %s\n' "${variant_name}"
    if [[ $# -gt 0 ]]; then
      printf '\nruntime:\n'
      while [[ $# -gt 0 ]]; do
        printf '  %s: %s\n' "$1" "$2"
        shift 2
      done
    fi
  } > "${path}"
}

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

write_override \
  "${TMP_CONFIG_DIR}/qwen25_canonical.yaml" \
  "${ROOT_DIR}/configs/exp/m3_stage_c_core4_qwen25_smoke.yaml" \
  "m3_stage_c_core4_qwen25_qonly_budget_probe_canonical" \
  "m3-core4-stage-c-qonly-probe-qwen25-canonical"

write_override \
  "${TMP_CONFIG_DIR}/qwen25_lr_1.yaml" \
  "${ROOT_DIR}/configs/exp/m3_stage_c_core4_qwen25_smoke.yaml" \
  "m3_stage_c_core4_qwen25_qonly_budget_probe_lr_1" \
  "m3-core4-stage-c-qonly-probe-qwen25-lr-1" \
  adapt_learning_rate 1.0

write_override \
  "${TMP_CONFIG_DIR}/qwen25_lr_5.yaml" \
  "${ROOT_DIR}/configs/exp/m3_stage_c_core4_qwen25_smoke.yaml" \
  "m3_stage_c_core4_qwen25_qonly_budget_probe_lr_5" \
  "m3-core4-stage-c-qonly-probe-qwen25-lr-5" \
  adapt_learning_rate 5.0

write_override \
  "${TMP_CONFIG_DIR}/qwen25_steps_10.yaml" \
  "${ROOT_DIR}/configs/exp/m3_stage_c_core4_qwen25_smoke.yaml" \
  "m3_stage_c_core4_qwen25_qonly_budget_probe_steps_10" \
  "m3-core4-stage-c-qonly-probe-qwen25-steps-10" \
  adapt_steps 10

write_override \
  "${TMP_CONFIG_DIR}/qwen25_lr_5_steps_10.yaml" \
  "${ROOT_DIR}/configs/exp/m3_stage_c_core4_qwen25_smoke.yaml" \
  "m3_stage_c_core4_qwen25_qonly_budget_probe_lr_5_steps_10" \
  "m3-core4-stage-c-qonly-probe-qwen25-lr-5-steps-10" \
  adapt_learning_rate 5.0 \
  adapt_steps 10

write_override \
  "${TMP_CONFIG_DIR}/qwen3_canonical.yaml" \
  "${ROOT_DIR}/configs/exp/m3_stage_c_core4_qwen3_smoke.yaml" \
  "m3_stage_c_core4_qwen3_qonly_budget_probe_canonical" \
  "m3-core4-stage-c-qonly-probe-qwen3-canonical"

write_override \
  "${TMP_CONFIG_DIR}/qwen3_lr_1.yaml" \
  "${ROOT_DIR}/configs/exp/m3_stage_c_core4_qwen3_smoke.yaml" \
  "m3_stage_c_core4_qwen3_qonly_budget_probe_lr_1" \
  "m3-core4-stage-c-qonly-probe-qwen3-lr-1" \
  adapt_learning_rate 1.0

write_override \
  "${TMP_CONFIG_DIR}/qwen3_lr_5.yaml" \
  "${ROOT_DIR}/configs/exp/m3_stage_c_core4_qwen3_smoke.yaml" \
  "m3_stage_c_core4_qwen3_qonly_budget_probe_lr_5" \
  "m3-core4-stage-c-qonly-probe-qwen3-lr-5" \
  adapt_learning_rate 5.0

write_override \
  "${TMP_CONFIG_DIR}/qwen3_steps_10.yaml" \
  "${ROOT_DIR}/configs/exp/m3_stage_c_core4_qwen3_smoke.yaml" \
  "m3_stage_c_core4_qwen3_qonly_budget_probe_steps_10" \
  "m3-core4-stage-c-qonly-probe-qwen3-steps-10" \
  adapt_steps 10

write_override \
  "${TMP_CONFIG_DIR}/qwen3_lr_5_steps_10.yaml" \
  "${ROOT_DIR}/configs/exp/m3_stage_c_core4_qwen3_smoke.yaml" \
  "m3_stage_c_core4_qwen3_qonly_budget_probe_lr_5_steps_10" \
  "m3-core4-stage-c-qonly-probe-qwen3-lr-5-steps-10" \
  adapt_learning_rate 5.0 \
  adapt_steps 10

run_stage_c "${TMP_CONFIG_DIR}/qwen25_canonical.yaml" "${QWEN25_SEED}" "${RUNS_ROOT}/qwen25-canonical" "${QWEN25_STAGEB}"
run_stage_c "${TMP_CONFIG_DIR}/qwen25_lr_1.yaml" "${QWEN25_SEED}" "${RUNS_ROOT}/qwen25-lr-1" "${QWEN25_STAGEB}"
run_stage_c "${TMP_CONFIG_DIR}/qwen25_lr_5.yaml" "${QWEN25_SEED}" "${RUNS_ROOT}/qwen25-lr-5" "${QWEN25_STAGEB}"
run_stage_c "${TMP_CONFIG_DIR}/qwen25_steps_10.yaml" "${QWEN25_SEED}" "${RUNS_ROOT}/qwen25-steps-10" "${QWEN25_STAGEB}"
run_stage_c "${TMP_CONFIG_DIR}/qwen25_lr_5_steps_10.yaml" "${QWEN25_SEED}" "${RUNS_ROOT}/qwen25-lr-5-steps-10" "${QWEN25_STAGEB}"

run_stage_c "${TMP_CONFIG_DIR}/qwen3_canonical.yaml" "${QWEN3_SEED}" "${RUNS_ROOT}/qwen3-canonical" "${QWEN3_STAGEB}"
run_stage_c "${TMP_CONFIG_DIR}/qwen3_lr_1.yaml" "${QWEN3_SEED}" "${RUNS_ROOT}/qwen3-lr-1" "${QWEN3_STAGEB}"
run_stage_c "${TMP_CONFIG_DIR}/qwen3_lr_5.yaml" "${QWEN3_SEED}" "${RUNS_ROOT}/qwen3-lr-5" "${QWEN3_STAGEB}"
run_stage_c "${TMP_CONFIG_DIR}/qwen3_steps_10.yaml" "${QWEN3_SEED}" "${RUNS_ROOT}/qwen3-steps-10" "${QWEN3_STAGEB}"
run_stage_c "${TMP_CONFIG_DIR}/qwen3_lr_5_steps_10.yaml" "${QWEN3_SEED}" "${RUNS_ROOT}/qwen3-lr-5-steps-10" "${QWEN3_STAGEB}"

./scripts/run_analysis.sh \
  --config configs/exp/m3_stage_c_probe_summary.yaml \
  --seed "${SEED_BASE}" \
  --output_dir "${REPORT_DIR}" \
  --input_root "${RUNS_ROOT}" \
  "${EXTRA_ARGS[@]}"
