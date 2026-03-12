#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PRE_V73_MAIN_HEAD="${1:-1a9c9ac18babeaff27402cc22eaf40daf714230e}"
BASE_SEED="${2:-61109}"
RUN_ROOT="${3:-/root/autodl-tmp/runs/verify/planv7-lr75e5-v7-4-forced-consumption-qwen25}"
RESULT_ROOT="${4:-/root/autodl-tmp/results/generated/planv7-lr75e5-v7-4-forced-consumption-qwen25}"
RESUME_STAGE_B_ROOT="${5:-runs/verify/m3-story-cloze-real-pilot-qwen25/stage-b}"
TRAIN_STEPS="${6:-300}"
MODEL_DIR="${7:-/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct}"
V73_SUMMARY_JSON="${8:-/root/autodl-tmp/results/generated/planv7-lr75e5-v7-3-bridge-qwen25/v7-3-summary.json}"

QUEUE_LOG_DIR="${RUN_ROOT}"
mkdir -p "${QUEUE_LOG_DIR}" "${RESULT_ROOT}"

while [ ! -f "${V73_SUMMARY_JSON}" ]; do
  sleep 30
done

while tmux has-session -t planv7_lr75e5_v73_post 2>/dev/null; do
  sleep 30
done

while true; do
  current_head="$(gh api repos/YJLi-new/MemTOTAL/commits/main --jq '.sha' 2>/dev/null || true)"
  if [ -n "${current_head}" ] && [ "${current_head}" != "${PRE_V73_MAIN_HEAD}" ]; then
    break
  fi
  sleep 30
done

python - "${V73_SUMMARY_JSON}" <<'PY'
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
payload = json.loads(summary_path.read_text())
next_step = str(payload.get("recommended_next_step", "")).strip()
if next_step != "open_v7_4_forced_consumption":
    raise SystemExit(
        f"V7-3 restart line did not authorize V7-4; recommended_next_step={next_step!r}"
    )
PY

if tmux has-session -t planv7_lr75e5_v74 2>/dev/null; then
  exit 0
fi

tmux new-session -d -s planv7_lr75e5_v74 \
  "mkdir -p ${RUN_ROOT} ${RESULT_ROOT} && \
   cd ${ROOT_DIR} && \
   bash scripts/run_planv7_lr75e5_v7_4_forced_consumption_qwen25.sh \
     ${BASE_SEED} \
     ${RUN_ROOT} \
     ${RESULT_ROOT} \
     ${RESUME_STAGE_B_ROOT} \
     ${TRAIN_STEPS} \
     ${MODEL_DIR} \
     ${V73_SUMMARY_JSON} \
   2>&1 | tee ${RUN_ROOT}/tmux-session.log"

tmux new-session -d -s planv7_lr75e5_v74_post \
  "bash -lc 'while [ ! -f ${RESULT_ROOT}/v7-4-summary.json ]; do sleep 30; done; \
   cd ${ROOT_DIR}; \
   bash scripts/publish_review_artifacts.sh; \
   git add scripts/run_planv7_v7_4_forced_consumption_qwen25.sh \
     scripts/queue_planv7_lr75e5_v7_5_after_v7_4.sh \
     docs/exec-plans/active/20260312-planv7-lr75e5-restart.md \
     docs/exec-plans/active/20260312-planv7-lr75e5-v7-4-forced-consumption.md \
     docs/exec-plans/active/20260312-planv7-lr75e5-v7-5-targeted-aux-revisit.md \
     results/generated/review/planv7-lr75e5-v7-4-forced-consumption-qwen25 \
     runs/review/planv7-lr75e5-v7-4-forced-consumption-qwen25; \
   git commit -m \"feat: complete planv7 lr-updated v7-4 forced consumption\"; \
   git push origin main; \
   bash scripts/push_github_review_snapshot.sh' > ${RUN_ROOT}/postpublish.log 2>&1"
