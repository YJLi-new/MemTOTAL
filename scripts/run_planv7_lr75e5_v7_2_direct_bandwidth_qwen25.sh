#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

BASE_SEED="${1:-61109}"
RUN_ROOT="${2:-/root/autodl-tmp/runs/verify/planv7-lr75e5-v7-2-direct-bandwidth-qwen25}"
RESULT_ROOT="${3:-/root/autodl-tmp/results/generated/planv7-lr75e5-v7-2-direct-bandwidth-qwen25}"
RESUME_STAGE_B_ROOT="${4:-runs/verify/m3-story-cloze-real-pilot-qwen25/stage-b}"
TRAIN_STEPS="${5:-200}"
MODEL_DIR="${6:-/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct}"

export PLANV7_PROJECTOR_LR="${PLANV7_PROJECTOR_LR:-7.5e-5}"
export PLANV7_OWNER_LOCKED_PROJECTOR_LR="${PLANV7_OWNER_LOCKED_PROJECTOR_LR:-7.5e-5}"
export PLANV7_REPO_CONFIRMED_V65_PROJECTOR_LR_REFERENCE="${PLANV7_REPO_CONFIRMED_V65_PROJECTOR_LR_REFERENCE:-7.5e-5}"
export PLANV7_OWNER_OVERRIDE_NOTE="${PLANV7_OWNER_OVERRIDE_NOTE:-false}"
export PLANV7_EXPERIMENT_PREFIX="${PLANV7_EXPERIMENT_PREFIX:-planv7_lr75e5}"

exec bash scripts/run_planv7_v7_2_direct_bandwidth_qwen25.sh \
  "${BASE_SEED}" \
  "${RUN_ROOT}" \
  "${RESULT_ROOT}" \
  "${RESUME_STAGE_B_ROOT}" \
  "${TRAIN_STEPS}" \
  "${MODEL_DIR}"
