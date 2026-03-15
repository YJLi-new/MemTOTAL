#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

export PLANV9_PRIMARY_BACKBONE_NAME="${PLANV9_PRIMARY_BACKBONE_NAME:-Qwen3-4B}"
export PLANV9_PRIMARY_BACKBONE_KEY="${PLANV9_PRIMARY_BACKBONE_KEY:-qwen34}"
export PLANV9_PRIMARY_MODEL_ID="${PLANV9_PRIMARY_MODEL_ID:-Qwen/Qwen3-4B}"
export PLANV9_PRIMARY_PREP_SCRIPT="${PLANV9_PRIMARY_PREP_SCRIPT:-scripts/prepare_local_qwen34_model.sh}"
export PLANV9_PRIMARY_MODEL_DIR="${PLANV9_PRIMARY_MODEL_DIR:-/root/autodl-tmp/models/Qwen3-4B}"

bash scripts/run_planv9_v9_0_flashmem_discrimination.sh \
  "${1:-61109}" \
  "${2:-/root/autodl-tmp/runs/verify/planv9-v9-0-flashmem-discrimination-qwen34}" \
  "${3:-/root/autodl-tmp/results/generated/planv9-v9-0-flashmem-discrimination-qwen34}" \
  "${4:-/root/autodl-tmp/models/Qwen3-4B}" \
  "${5:-runs/review/planv8-v8-0-qwen34-baselines-oracles}" \
  "${6:-results/generated/review/planv8-v8-0-qwen34-baselines-oracles/v8-0-summary.json}" \
  "${7:-results/generated/review/planv8-v8-0-qwen34-baselines-oracles/selected-prompt-modes.json}"
