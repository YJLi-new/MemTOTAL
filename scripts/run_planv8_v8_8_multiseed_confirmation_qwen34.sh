#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

export PLANV8_PRIMARY_BACKBONE_NAME="${PLANV8_PRIMARY_BACKBONE_NAME:-Qwen3-4B}"
export PLANV8_PRIMARY_BACKBONE_KEY="${PLANV8_PRIMARY_BACKBONE_KEY:-qwen34}"
export PLANV8_PRIMARY_MODEL_ID="${PLANV8_PRIMARY_MODEL_ID:-Qwen/Qwen3-4B}"
export PLANV8_PRIMARY_PREP_SCRIPT="${PLANV8_PRIMARY_PREP_SCRIPT:-scripts/prepare_local_qwen34_model.sh}"
export PLANV8_PRIMARY_MODEL_DIR="${PLANV8_PRIMARY_MODEL_DIR:-/root/autodl-tmp/models/Qwen3-4B}"

bash scripts/run_planv8_v8_8_multiseed_confirmation.sh \
  "${1:-61109}" \
  "${2:-/root/autodl-tmp/runs/verify/planv8-v8-8-multiseed-confirmation-qwen34}" \
  "${3:-/root/autodl-tmp/results/generated/planv8-v8-8-multiseed-confirmation-qwen34}" \
  "${4:-/root/autodl-tmp/models/Qwen3-4B}" \
  "${5:-results/generated/review/planv8-v8-0-qwen34-baselines-oracles/v8-0-summary.json}" \
  "${6:-results/generated/review/planv8-v8-0-qwen34-baselines-oracles/selected-prompt-modes.json}" \
  "${7:-/root/autodl-tmp/runs/verify/planv8-v8-3-reader-opd-qwen34}" \
  "${8:-results/generated/review/planv8-v8-3-reader-opd-qwen34/v8-3-summary.json}" \
  "${9:-/root/autodl-tmp/runs/verify/planv8-v8-5-bridge-revisit-qwen34}" \
  "${10:-results/generated/review/planv8-v8-5-bridge-revisit-qwen34/v8-5-summary.json}" \
  "${11:-/root/autodl-tmp/runs/verify/planv8-v8-6-writer-aux-qwen34}" \
  "${12:-results/generated/review/planv8-v8-6-writer-aux-qwen34/v8-6-summary.json}" \
  "${13:-results/generated/review/planv8-v8-7-comparators-qwen34/v8-7-summary.json}"
