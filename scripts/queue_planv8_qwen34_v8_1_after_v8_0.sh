#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

BASE_SEED="${1:-61109}"
V80_RESULT_ROOT="${2:-/root/autodl-tmp/results/generated/planv8-v8-0-qwen34-baselines-oracles}"
V81_RUN_ROOT="${3:-/root/autodl-tmp/runs/verify/planv8-v8-1-reader-interface-scout-qwen34}"
V81_RESULT_ROOT="${4:-/root/autodl-tmp/results/generated/planv8-v8-1-reader-interface-scout-qwen34}"
MODEL_DIR="${5:-/root/autodl-tmp/models/Qwen3-4B}"
V80_SUMMARY_PATH="${V80_RESULT_ROOT}/v8-0-summary.json"
SELECTED_PROMPTS_PATH="${V80_RESULT_ROOT}/selected-prompt-modes.json"
RUN_SESSION="${6:-planv8_v81_q34}"
WATCH_SESSION="${7:-planv8_v81_q34_watch}"
POST_SESSION="${8:-planv8_v81_q34_post}"

mkdir -p "${V81_RUN_ROOT}" "${V81_RESULT_ROOT}"

while [ ! -f "${V80_SUMMARY_PATH}" ]; do
  sleep 30
done

python - "${V80_SUMMARY_PATH}" <<'PY'
import json
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text())
next_step = str(summary.get("recommended_next_step", "")).strip()
if next_step != "open_v8_1_reader_interface_scout":
    raise SystemExit(
        f"V8-0 did not authorize V8-1 on qwen34; recommended_next_step={next_step!r}"
    )
PY

if ! tmux has-session -t "${RUN_SESSION}" 2>/dev/null; then
  tmux new-session -d -s "${RUN_SESSION}" \
    "mkdir -p ${V81_RUN_ROOT} ${V81_RESULT_ROOT} && \
     cd ${ROOT_DIR} && \
     bash scripts/run_planv8_v8_1_reader_interface_scout_qwen34.sh \
       ${BASE_SEED} \
       ${V81_RUN_ROOT} \
       ${V81_RESULT_ROOT} \
       ${MODEL_DIR} \
       ${SELECTED_PROMPTS_PATH} \
       ${V80_SUMMARY_PATH} \
     2>&1 | tee ${V81_RUN_ROOT}/tmux-session.log"
fi

if ! tmux has-session -t "${WATCH_SESSION}" 2>/dev/null; then
  tmux new-session -d -s "${WATCH_SESSION}" \
    "bash -lc 'while tmux has-session -t ${RUN_SESSION} 2>/dev/null; do ts=\$(date -u +%Y-%m-%dT%H:%M:%SZ); metrics_count=\$(find ${V81_RUN_ROOT} -path \"*/metrics.json\" -type f 2>/dev/null | wc -l); gpu_line=\$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | head -n 1 | sed \"s/,/\\//\" || true); echo \"\${ts} metrics=\${metrics_count} gpu_mib=\${gpu_line:-unknown}\"; sleep 300; done' > ${V81_RUN_ROOT}/watch.log 2>&1"
fi

if ! tmux has-session -t "${POST_SESSION}" 2>/dev/null; then
  tmux new-session -d -s "${POST_SESSION}" \
    "bash -lc 'while [ ! -f ${V81_RESULT_ROOT}/v8-1-summary.json ]; do sleep 30; done; \
     cd ${ROOT_DIR}; \
     bash scripts/publish_review_artifacts.sh; \
     git add docs/exec-plans/active/20260314-planv8-v8-1-reader-interface-scout-qwen34.md \
       results/generated/review/planv8-v8-1-reader-interface-scout-qwen34 \
       runs/review/planv8-v8-1-reader-interface-scout-qwen34; \
     if ! git diff --cached --quiet; then \
       git commit -m \"feat: complete planv8 qwen34 v8-1 reader interface scout\"; \
       env -u HTTPS_PROXY -u HTTP_PROXY -u ALL_PROXY -u https_proxy -u http_proxy -u all_proxy gh auth setup-git; \
       env -u HTTPS_PROXY -u HTTP_PROXY -u ALL_PROXY -u https_proxy -u http_proxy -u all_proxy git -c http.version=HTTP/1.1 push origin main; \
       env -u HTTPS_PROXY -u HTTP_PROXY -u ALL_PROXY -u https_proxy -u http_proxy -u all_proxy bash scripts/push_github_review_snapshot.sh; \
     fi' > ${V81_RUN_ROOT}/postpublish.log 2>&1"
fi
