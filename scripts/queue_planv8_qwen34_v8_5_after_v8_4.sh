#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

BASE_SEED="${1:-61109}"
V84_RUN_ROOT="${2:-/root/autodl-tmp/runs/verify/planv8-v8-4-external-writer-qwen34}"
V84_RESULT_ROOT="${3:-/root/autodl-tmp/results/generated/planv8-v8-4-external-writer-qwen34}"
V85_RUN_ROOT="${4:-/root/autodl-tmp/runs/verify/planv8-v8-5-bridge-revisit-qwen34}"
V85_RESULT_ROOT="${5:-/root/autodl-tmp/results/generated/planv8-v8-5-bridge-revisit-qwen34}"
MODEL_DIR="${6:-/root/autodl-tmp/models/Qwen3-4B}"
V84_SUMMARY_PATH="${V84_RESULT_ROOT}/v8-4-summary.json"
SELECTED_PROMPTS_PATH="${V84_RESULT_ROOT}/selected-prompt-modes.json"
RUN_SESSION="${7:-planv8_v85_q34}"
WATCH_SESSION="${8:-planv8_v85_q34_watch}"
POST_SESSION="${9:-planv8_v85_q34_post}"

mkdir -p "${V85_RUN_ROOT}" "${V85_RESULT_ROOT}"

while [ ! -f "${V84_SUMMARY_PATH}" ]; do
  sleep 30
done

python - "${V84_SUMMARY_PATH}" <<'PY'
import json
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text())
next_step = str(summary.get("recommended_next_step", "")).strip()
if next_step != "open_v8_5_bridge":
    raise SystemExit(
        f"V8-4 did not authorize V8-5 on qwen34; recommended_next_step={next_step!r}"
    )
PY

if ! tmux has-session -t "${RUN_SESSION}" 2>/dev/null; then
  tmux new-session -d -s "${RUN_SESSION}" \
    "mkdir -p ${V85_RUN_ROOT} ${V85_RESULT_ROOT} && \
     cd ${ROOT_DIR} && \
     bash scripts/run_planv8_v8_5_bridge_revisit_qwen34.sh \
       ${BASE_SEED} \
       ${V85_RUN_ROOT} \
       ${V85_RESULT_ROOT} \
       ${MODEL_DIR} \
       ${V84_RUN_ROOT} \
       ${V84_SUMMARY_PATH} \
       ${SELECTED_PROMPTS_PATH} \
     2>&1 | tee ${V85_RUN_ROOT}/tmux-session.log"
fi

if ! tmux has-session -t "${WATCH_SESSION}" 2>/dev/null; then
  tmux new-session -d -s "${WATCH_SESSION}" \
    "bash -lc 'while tmux has-session -t ${RUN_SESSION} 2>/dev/null; do ts=\$(date -u +%Y-%m-%dT%H:%M:%SZ); progress=\$(python ${ROOT_DIR}/scripts/planv8_watch_progress.py --run_root ${V85_RUN_ROOT} 2>/dev/null || echo \"{}\"); gpu_line=\$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | head -n 1 | sed \"s/,/\\//\" || true); echo \"\${ts} progress=\${progress} gpu_mib=\${gpu_line:-unknown}\"; sleep 120; done' > ${V85_RUN_ROOT}/watch.log 2>&1"
fi

if ! tmux has-session -t "${POST_SESSION}" 2>/dev/null; then
  tmux new-session -d -s "${POST_SESSION}" \
    "bash -lc 'while [ ! -f ${V85_RESULT_ROOT}/v8-5-summary.json ]; do sleep 30; done; \
     cd ${ROOT_DIR}; \
     bash scripts/publish_review_artifacts.sh; \
     git add docs/exec-plans/active/20260315-planv8-v8-5-bridge-revisit-qwen34.md \
       results/generated/review/planv8-v8-5-bridge-revisit-qwen34 \
       runs/review/planv8-v8-5-bridge-revisit-qwen34; \
     if ! git diff --cached --quiet; then \
       git commit -m \"feat: complete planv8 qwen34 v8-5 bridge revisit\"; \
       env -u HTTPS_PROXY -u HTTP_PROXY -u ALL_PROXY -u https_proxy -u http_proxy -u all_proxy gh auth setup-git; \
       env -u HTTPS_PROXY -u HTTP_PROXY -u ALL_PROXY -u https_proxy -u http_proxy -u all_proxy git -c http.version=HTTP/1.1 push origin main; \
       env -u HTTPS_PROXY -u HTTP_PROXY -u ALL_PROXY -u https_proxy -u http_proxy -u all_proxy bash scripts/push_github_review_snapshot.sh; \
     fi' > ${V85_RUN_ROOT}/postpublish.log 2>&1"
fi
