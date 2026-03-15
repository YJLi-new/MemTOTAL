#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

WITH_SUPERVISOR=1
if [[ "${1:-}" == "--no-supervisor" ]]; then
  WITH_SUPERVISOR=0
fi

SEED="${PLANV9_V91_SEED:-61109}"
RUN_ROOT="${PLANV9_V91_RUN_ROOT:-/root/autodl-tmp/runs/verify/planv9-v9-1-longhorizon-baselines-qwen34}"
RESULT_ROOT="${PLANV9_V91_RESULT_ROOT:-/root/autodl-tmp/results/generated/planv9-v9-1-longhorizon-baselines-qwen34}"
MODEL_DIR="${PLANV9_V91_MODEL_DIR:-/root/autodl-tmp/models/Qwen3-4B}"
V90_SUMMARY_PATH="${PLANV9_V91_V90_SUMMARY_PATH:-results/generated/review/planv9-v9-0-flashmem-discrimination-qwen34/v9-0-summary.json}"

RUNNER_SESSION="planv9_v91_q34"
WATCH_SESSION="planv9_v91_q34_watch"
POST_SESSION="planv9_v91_q34_post"
SUPERVISOR_SESSION="planv9_v91_q34_supervisor"

mkdir -p "${RUN_ROOT}" "${RESULT_ROOT}"

if [[ -f "${RESULT_ROOT}/v9-1-summary.json" ]]; then
  exit 0
fi

if ! tmux has-session -t "${RUNNER_SESSION}" 2>/dev/null; then
  tmux new-session -d -s "${RUNNER_SESSION}" \
    "cd '${ROOT_DIR}' && env -u HTTPS_PROXY -u HTTP_PROXY -u ALL_PROXY -u https_proxy -u http_proxy -u all_proxy bash scripts/run_planv9_v9_1_longhorizon_baselines_qwen34.sh '${SEED}' '${RUN_ROOT}' '${RESULT_ROOT}' '${MODEL_DIR}' '${V90_SUMMARY_PATH}' 2>&1 | tee '${RUN_ROOT}/tmux-session.log'"
fi

if ! tmux has-session -t "${WATCH_SESSION}" 2>/dev/null; then
  tmux new-session -d -s "${WATCH_SESSION}" \
    "cd '${ROOT_DIR}' && bash scripts/watch_planv9_v9_1_status.sh '${RUN_ROOT}' '${RESULT_ROOT}' '${RUNNER_SESSION}' 12 2>&1 | tee -a '${RUN_ROOT}/watch.log'"
fi

if ! tmux has-session -t "${POST_SESSION}" 2>/dev/null; then
  tmux new-session -d -s "${POST_SESSION}" \
    "cd '${ROOT_DIR}' && bash scripts/postpublish_planv9_v9_1.sh '${ROOT_DIR}' '${RESULT_ROOT}' '${RUN_ROOT}' 2>&1 | tee '${RUN_ROOT}/postpublish.log'"
fi

if [[ "${WITH_SUPERVISOR}" == "1" ]] && ! tmux has-session -t "${SUPERVISOR_SESSION}" 2>/dev/null; then
  tmux new-session -d -s "${SUPERVISOR_SESSION}" \
    "cd '${ROOT_DIR}' && bash scripts/supervise_planv9_v9_1_qwen34.sh '${ROOT_DIR}' 2>&1 | tee '${RUN_ROOT}/supervisor.log'"
fi
