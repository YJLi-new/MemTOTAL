#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

BASE_SEED="${1:-61109}"
V87_RESULT_ROOT="${2:-/root/autodl-tmp/results/generated/planv8-v8-7-comparators-qwen34}"
V88_RUN_ROOT="${3:-/root/autodl-tmp/runs/verify/planv8-v8-8-multiseed-confirmation-qwen34}"
V88_RESULT_ROOT="${4:-/root/autodl-tmp/results/generated/planv8-v8-8-multiseed-confirmation-qwen34}"
MODEL_DIR="${5:-/root/autodl-tmp/models/Qwen3-4B}"
V80_SUMMARY_PATH="${6:-results/generated/review/planv8-v8-0-qwen34-baselines-oracles/v8-0-summary.json}"
SELECTED_PROMPTS_PATH="${7:-results/generated/review/planv8-v8-0-qwen34-baselines-oracles/selected-prompt-modes.json}"
V83_RUN_ROOT="${8:-/root/autodl-tmp/runs/verify/planv8-v8-3-reader-opd-qwen34}"
V83_SUMMARY_PATH="${9:-results/generated/review/planv8-v8-3-reader-opd-qwen34/v8-3-summary.json}"
V85_RUN_ROOT="${10:-/root/autodl-tmp/runs/verify/planv8-v8-5-bridge-revisit-qwen34}"
V85_SUMMARY_PATH="${11:-results/generated/review/planv8-v8-5-bridge-revisit-qwen34/v8-5-summary.json}"
V86_RUN_ROOT="${12:-/root/autodl-tmp/runs/verify/planv8-v8-6-writer-aux-qwen34}"
V86_SUMMARY_PATH="${13:-results/generated/review/planv8-v8-6-writer-aux-qwen34/v8-6-summary.json}"
RUN_SESSION="${14:-planv8_v88_q34}"
WATCH_SESSION="${15:-planv8_v88_q34_watch}"
POST_SESSION="${16:-planv8_v88_q34_post}"

V87_SUMMARY_PATH="${V87_RESULT_ROOT}/v8-7-summary.json"

mkdir -p "${V88_RUN_ROOT}" "${V88_RESULT_ROOT}"

while [ ! -f "${V87_SUMMARY_PATH}" ]; do
  sleep 30
done

python - "${V87_SUMMARY_PATH}" <<'PY'
import json
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text())
next_step = str(summary.get("recommended_next_step", "")).strip()
if next_step != "open_v8_8_multiseed_confirmation":
    raise SystemExit(
        f"V8-7 did not authorize V8-8 on qwen34; recommended_next_step={next_step!r}"
    )
PY

if ! tmux has-session -t "${RUN_SESSION}" 2>/dev/null; then
  tmux new-session -d -s "${RUN_SESSION}" \
    "mkdir -p ${V88_RUN_ROOT} ${V88_RESULT_ROOT} && \
     cd ${ROOT_DIR} && \
     bash scripts/run_planv8_v8_8_multiseed_confirmation_qwen34.sh \
       ${BASE_SEED} \
       ${V88_RUN_ROOT} \
       ${V88_RESULT_ROOT} \
       ${MODEL_DIR} \
       ${V80_SUMMARY_PATH} \
       ${SELECTED_PROMPTS_PATH} \
       ${V83_RUN_ROOT} \
       ${V83_SUMMARY_PATH} \
       ${V85_RUN_ROOT} \
       ${V85_SUMMARY_PATH} \
       ${V86_RUN_ROOT} \
       ${V86_SUMMARY_PATH} \
       ${V87_SUMMARY_PATH} \
     2>&1 | tee ${V88_RUN_ROOT}/tmux-session.log"
fi

if ! tmux has-session -t "${WATCH_SESSION}" 2>/dev/null; then
  tmux new-session -d -s "${WATCH_SESSION}" \
    "bash -lc 'while tmux has-session -t ${RUN_SESSION} 2>/dev/null; do ts=\$(date -u +%Y-%m-%dT%H:%M:%SZ); metrics_count=\$(find ${V88_RUN_ROOT} -path \"*/metrics.json\" -type f 2>/dev/null | wc -l); gpu_line=\$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | head -n 1 | sed \"s/,/\\//\" || true); echo \"\${ts} metrics=\${metrics_count} gpu_mib=\${gpu_line:-unknown}\"; sleep 180; done' > ${V88_RUN_ROOT}/watch.log 2>&1"
fi

if ! tmux has-session -t "${POST_SESSION}" 2>/dev/null; then
  tmux new-session -d -s "${POST_SESSION}" \
    "bash -lc 'while [ ! -f ${V88_RESULT_ROOT}/v8-8-summary.json ]; do sleep 30; done; \
     cd ${ROOT_DIR}; \
     bash scripts/publish_review_artifacts.sh; \
     git add docs/exec-plans/active/20260315-planv8-v8-8-multiseed-confirmation-qwen34.md \
       results/generated/review/planv8-v8-8-multiseed-confirmation-qwen34 \
       runs/review/planv8-v8-8-multiseed-confirmation-qwen34; \
     if ! git diff --cached --quiet; then \
       git commit -m \"feat: complete planv8 qwen34 v8-8 multiseed confirmation\"; \
       env -u HTTPS_PROXY -u HTTP_PROXY -u ALL_PROXY -u https_proxy -u http_proxy -u all_proxy gh auth setup-git; \
       env -u HTTPS_PROXY -u HTTP_PROXY -u ALL_PROXY -u https_proxy -u http_proxy -u all_proxy git push origin main; \
       env -u HTTPS_PROXY -u HTTP_PROXY -u ALL_PROXY -u https_proxy -u http_proxy -u all_proxy bash scripts/push_github_review_snapshot.sh; \
     fi' > ${V88_RUN_ROOT}/postpublish.log 2>&1"
fi
