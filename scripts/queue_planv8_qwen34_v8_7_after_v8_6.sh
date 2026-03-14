#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

BASE_SEED="${1:-61109}"
V86_RUN_ROOT="${2:-/root/autodl-tmp/runs/verify/planv8-v8-6-writer-aux-qwen34}"
V86_RESULT_ROOT="${3:-/root/autodl-tmp/results/generated/planv8-v8-6-writer-aux-qwen34}"
V87_RUN_ROOT="${4:-/root/autodl-tmp/runs/verify/planv8-v8-7-comparators-qwen34}"
V87_RESULT_ROOT="${5:-/root/autodl-tmp/results/generated/planv8-v8-7-comparators-qwen34}"
MODEL_DIR="${6:-/root/autodl-tmp/models/Qwen3-4B}"
V80_SUMMARY_PATH="${7:-results/generated/review/planv8-v8-0-qwen34-baselines-oracles/v8-0-summary.json}"
SELECTED_PROMPTS_PATH="${8:-results/generated/review/planv8-v8-0-qwen34-baselines-oracles/selected-prompt-modes.json}"
V76_SUMMARY_PATH="${9:-results/generated/review/planv7-lr75e5-v7-6-multiseed-confirmation-qwen25/v7-6-summary.json}"
RUN_SESSION="${10:-planv8_v87_q34}"
WATCH_SESSION="${11:-planv8_v87_q34_watch}"
POST_SESSION="${12:-planv8_v87_q34_post}"

V86_SUMMARY_PATH="${V86_RESULT_ROOT}/v8-6-summary.json"

mkdir -p "${V87_RUN_ROOT}" "${V87_RESULT_ROOT}"

while [ ! -f "${V86_SUMMARY_PATH}" ]; do
  sleep 30
done

python - "${V86_SUMMARY_PATH}" <<'PY'
import json
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text())
next_step = str(summary.get("recommended_next_step", "")).strip()
if next_step != "open_v8_7_comparators":
    raise SystemExit(
        f"V8-6 did not authorize V8-7 on qwen34; recommended_next_step={next_step!r}"
    )
PY

if ! tmux has-session -t "${RUN_SESSION}" 2>/dev/null; then
  tmux new-session -d -s "${RUN_SESSION}" \
    "mkdir -p ${V87_RUN_ROOT} ${V87_RESULT_ROOT} && \
     cd ${ROOT_DIR} && \
     bash scripts/run_planv8_v8_7_comparators_qwen34.sh \
       ${BASE_SEED} \
       ${V87_RUN_ROOT} \
       ${V87_RESULT_ROOT} \
       ${MODEL_DIR} \
       ${V80_SUMMARY_PATH} \
       ${SELECTED_PROMPTS_PATH} \
       ${V76_SUMMARY_PATH} \
       ${V86_SUMMARY_PATH} \
     2>&1 | tee ${V87_RUN_ROOT}/tmux-session.log"
fi

if ! tmux has-session -t "${WATCH_SESSION}" 2>/dev/null; then
  tmux new-session -d -s "${WATCH_SESSION}" \
    "bash -lc 'while tmux has-session -t ${RUN_SESSION} 2>/dev/null; do ts=\$(date -u +%Y-%m-%dT%H:%M:%SZ); metrics_count=\$(find ${V87_RUN_ROOT} -path \"*/metrics.json\" -type f 2>/dev/null | wc -l); gpu_line=\$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | head -n 1 | sed \"s/,/\\//\" || true); echo \"\${ts} metrics=\${metrics_count} gpu_mib=\${gpu_line:-unknown}\"; sleep 180; done' > ${V87_RUN_ROOT}/watch.log 2>&1"
fi

if ! tmux has-session -t "${POST_SESSION}" 2>/dev/null; then
  tmux new-session -d -s "${POST_SESSION}" \
    "bash -lc 'while [ ! -f ${V87_RESULT_ROOT}/v8-7-summary.json ]; do sleep 30; done; \
     cd ${ROOT_DIR}; \
     bash scripts/publish_review_artifacts.sh; \
     git add docs/exec-plans/active/20260315-planv8-v8-7-comparators-qwen34.md \
       results/generated/review/planv8-v8-7-comparators-qwen34 \
       runs/review/planv8-v8-7-comparators-qwen34; \
     if ! git diff --cached --quiet; then \
       git commit -m \"feat: complete planv8 qwen34 v8-7 comparators\"; \
       gh auth setup-git; \
       git push origin main; \
       bash scripts/push_github_review_snapshot.sh; \
     fi' > ${V87_RUN_ROOT}/postpublish.log 2>&1"
fi
