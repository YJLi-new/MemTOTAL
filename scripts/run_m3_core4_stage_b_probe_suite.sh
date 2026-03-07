#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

SEED_BASE="${1:-2701}"
RUNS_ROOT="${2:-/root/autodl-tmp/memtotal-stage-b-probe-suite}"
REPORT_DIR="${3:-results/generated/m3-core4-stage-b-probe-suite}"
QWEN25_STAGEA="${4:-runs/verify/m3-core4-qwen25/stage-a}"
QWEN3_STAGEA="${5:-runs/verify/m3-core4-qwen3/stage-a}"
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

run_probe() {
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
  "${ROOT_DIR}/configs/exp/m3_stage_b_core4_qwen25_smoke.yaml" \
  "m3_stage_b_core4_qwen25_probe_canonical" \
  "m3-core4-stage-b-probe-qwen25-canonical"

write_override \
  "${TMP_CONFIG_DIR}/qwen25_meta_episodes_20.yaml" \
  "${ROOT_DIR}/configs/exp/m3_stage_b_core4_qwen25_smoke.yaml" \
  "m3_stage_b_core4_qwen25_probe_meta_episodes_20" \
  "m3-core4-stage-b-probe-qwen25-meta-episodes-20" \
  meta_episodes 20

write_override \
  "${TMP_CONFIG_DIR}/qwen25_meta_lr_01.yaml" \
  "${ROOT_DIR}/configs/exp/m3_stage_b_core4_qwen25_smoke.yaml" \
  "m3_stage_b_core4_qwen25_probe_meta_lr_01" \
  "m3-core4-stage-b-probe-qwen25-meta-lr-01" \
  meta_learning_rate 0.1

write_override \
  "${TMP_CONFIG_DIR}/qwen3_canonical.yaml" \
  "${ROOT_DIR}/configs/exp/m3_stage_b_core4_qwen3_smoke.yaml" \
  "m3_stage_b_core4_qwen3_probe_canonical" \
  "m3-core4-stage-b-probe-qwen3-canonical"

write_override \
  "${TMP_CONFIG_DIR}/qwen3_meta_episodes_12.yaml" \
  "${ROOT_DIR}/configs/exp/m3_stage_b_core4_qwen3_smoke.yaml" \
  "m3_stage_b_core4_qwen3_probe_meta_episodes_12" \
  "m3-core4-stage-b-probe-qwen3-meta-episodes-12" \
  meta_episodes 12

write_override \
  "${TMP_CONFIG_DIR}/qwen3_meta_lr_01.yaml" \
  "${ROOT_DIR}/configs/exp/m3_stage_b_core4_qwen3_smoke.yaml" \
  "m3_stage_b_core4_qwen3_probe_meta_lr_01" \
  "m3-core4-stage-b-probe-qwen3-meta-lr-01" \
  meta_learning_rate 0.1

run_probe "${TMP_CONFIG_DIR}/qwen25_canonical.yaml" "$((SEED_BASE + 0))" "${RUNS_ROOT}/qwen25-canonical" "${QWEN25_STAGEA}"
run_probe "${TMP_CONFIG_DIR}/qwen25_meta_episodes_20.yaml" "$((SEED_BASE + 2))" "${RUNS_ROOT}/qwen25-meta-episodes-20" "${QWEN25_STAGEA}"
run_probe "${TMP_CONFIG_DIR}/qwen25_meta_lr_01.yaml" "$((SEED_BASE + 4))" "${RUNS_ROOT}/qwen25-meta-lr-01" "${QWEN25_STAGEA}"
run_probe "${TMP_CONFIG_DIR}/qwen3_canonical.yaml" "$((SEED_BASE + 10))" "${RUNS_ROOT}/qwen3-canonical" "${QWEN3_STAGEA}"
run_probe "${TMP_CONFIG_DIR}/qwen3_meta_episodes_12.yaml" "$((SEED_BASE + 12))" "${RUNS_ROOT}/qwen3-meta-episodes-12" "${QWEN3_STAGEA}"
run_probe "${TMP_CONFIG_DIR}/qwen3_meta_lr_01.yaml" "$((SEED_BASE + 14))" "${RUNS_ROOT}/qwen3-meta-lr-01" "${QWEN3_STAGEA}"

./scripts/run_analysis.sh \
  --config configs/exp/m3_stage_b_probe_summary.yaml \
  --seed "${SEED_BASE}" \
  --output_dir "${REPORT_DIR}" \
  --input_root "${RUNS_ROOT}" \
  "${EXTRA_ARGS[@]}"
