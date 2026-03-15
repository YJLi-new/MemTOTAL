#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

SEED_BASE="${1:-50201}"
RUNS_ROOT="${2:-/root/autodl-tmp/memtotal-stage-c-qonly-negative-count-sweep}"
REPORT_DIR="${3:-results/generated/m3-core4-stage-c-qonly-negative-count-sweep}"
QWEN25_STAGEB="${4:-runs/verify/m3-core4-qwen25/stage-b}"
QWEN3_STAGEB="${5:-runs/verify/m3-core4-qwen3/stage-b}"
SEED_COUNT="${6:-5}"
shift $(( $# > 0 ? 1 : 0 ))
shift $(( $# > 0 ? 1 : 0 ))
shift $(( $# > 0 ? 1 : 0 ))
shift $(( $# > 0 ? 1 : 0 ))
shift $(( $# > 0 ? 1 : 0 ))
shift $(( $# > 0 ? 1 : 0 ))
EXTRA_ARGS=("$@")

TMP_CONFIG_DIR="${RUNS_ROOT}/tmp-configs"
mkdir -p "${TMP_CONFIG_DIR}" "${RUNS_ROOT}" "${REPORT_DIR}"

write_override() {
  local path="$1"
  local include_path="$2"
  local exp_name="$3"
  local variant_name="$4"
  local negative_count="$5"
  {
    printf 'includes:\n'
    printf '  - %s\n' "${include_path}"
    printf '\nexperiment:\n'
    printf '  name: %s\n' "${exp_name}"
    printf '  method_variant: %s\n' "${variant_name}"
    printf '\nruntime:\n'
    printf '  retrieval_negative_count: %s\n' "${negative_count}"
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

for negative_count in 3 7 15; do
  write_override \
    "${TMP_CONFIG_DIR}/qwen25_neg${negative_count}.yaml" \
    "${ROOT_DIR}/configs/exp/m3_stage_c_core4_qwen25_smoke.yaml" \
    "m3_stage_c_core4_qwen25_qonly_neg${negative_count}" \
    "m3-core4-stage-c-qonly-neg${negative_count}-qwen25" \
    "${negative_count}"
  write_override \
    "${TMP_CONFIG_DIR}/qwen3_neg${negative_count}.yaml" \
    "${ROOT_DIR}/configs/exp/m3_stage_c_core4_qwen3_smoke.yaml" \
    "m3_stage_c_core4_qwen3_qonly_neg${negative_count}" \
    "m3-core4-stage-c-qonly-neg${negative_count}-qwen3" \
    "${negative_count}"
done

for ((index=0; index<SEED_COUNT; index++)); do
  qwen25_seed="$((SEED_BASE + index))"
  qwen3_seed="$((SEED_BASE + 100 + index))"
  for negative_count in 3 7 15; do
    run_stage_c \
      "${TMP_CONFIG_DIR}/qwen25_neg${negative_count}.yaml" \
      "${qwen25_seed}" \
      "${RUNS_ROOT}/qwen25-neg${negative_count}-seed-${qwen25_seed}" \
      "${QWEN25_STAGEB}"
    run_stage_c \
      "${TMP_CONFIG_DIR}/qwen3_neg${negative_count}.yaml" \
      "${qwen3_seed}" \
      "${RUNS_ROOT}/qwen3-neg${negative_count}-seed-${qwen3_seed}" \
      "${QWEN3_STAGEB}"
  done
done

./scripts/run_analysis.sh \
  --config configs/exp/m3_stage_c_seed_sweep_summary.yaml \
  --seed "${SEED_BASE}" \
  --output_dir "${REPORT_DIR}" \
  --input_root "${RUNS_ROOT}" \
  "${EXTRA_ARGS[@]}"
