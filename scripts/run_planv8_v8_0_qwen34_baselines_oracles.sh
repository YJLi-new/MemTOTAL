#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

export PLANV8_PRIMARY_BACKBONE_NAME="${PLANV8_PRIMARY_BACKBONE_NAME:-Qwen3-4B}"
export PLANV8_PRIMARY_BACKBONE_KEY="${PLANV8_PRIMARY_BACKBONE_KEY:-qwen34}"
export PLANV8_PRIMARY_ARM_PREFIX="${PLANV8_PRIMARY_ARM_PREFIX:-q34}"
export PLANV8_PRIMARY_MODEL_ID="${PLANV8_PRIMARY_MODEL_ID:-Qwen/Qwen3-4B}"
export PLANV8_PRIMARY_PREP_SCRIPT="${PLANV8_PRIMARY_PREP_SCRIPT:-scripts/prepare_local_qwen34_model.sh}"
export PLANV8_PRIMARY_MODEL_DIR="${PLANV8_PRIMARY_MODEL_DIR:-/root/autodl-tmp/models/Qwen3-4B}"

bash scripts/run_planv8_v8_0_primary_baselines_oracles.sh \
  "${1:-61109}" \
  "${2:-/root/autodl-tmp/runs/verify/planv8-v8-0-qwen34-baselines-oracles}" \
  "${3:-/root/autodl-tmp/results/generated/planv8-v8-0-qwen34-baselines-oracles}" \
  "${4:-/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct}" \
  "${5:-/root/autodl-tmp/models/Qwen3-4B}" \
  "${6:-results/generated/review/planv7-lr75e5-v7-0-metrics-oracle-qwen25/v7-0-summary.json}"
