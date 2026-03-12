#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PRE_V75_MAIN_HEAD="${1:-0a05e04a48a619e6e397d6e9bc39ba99c53504cc}"
BASE_SEED="${2:-61109}"
RUN_ROOT="${3:-/root/autodl-tmp/runs/verify/planv7-lr75e5-v7-6-multiseed-confirmation-qwen25}"
RESULT_ROOT="${4:-/root/autodl-tmp/results/generated/planv7-lr75e5-v7-6-multiseed-confirmation-qwen25}"
RESUME_STAGE_B_ROOT="${5:-runs/verify/m3-story-cloze-real-pilot-qwen25/stage-b}"
TRAIN_STEPS="${6:-300}"
MODEL_DIR="${7:-/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct}"
V75_SUMMARY_JSON="${8:-/root/autodl-tmp/results/generated/planv7-lr75e5-v7-5-targeted-aux-revisit-qwen25/v7-5-summary.json}"
V70_SUMMARY_JSON="${9:-/root/autodl-tmp/results/generated/planv7-lr75e5-v7-0-metrics-oracle-qwen25/v7-0-summary.json}"

mkdir -p "${RUN_ROOT}" "${RESULT_ROOT}"

while [ ! -f "${V75_SUMMARY_JSON}" ]; do
  sleep 30
done

while tmux has-session -t planv7_lr75e5_v75_post 2>/dev/null; do
  sleep 30
done

while true; do
  current_head="$(gh api repos/YJLi-new/MemTOTAL/commits/main --jq '.sha' 2>/dev/null || true)"
  if [ -n "${current_head}" ] && [ "${current_head}" != "${PRE_V75_MAIN_HEAD}" ]; then
    break
  fi
  sleep 30
done

python - "${V75_SUMMARY_JSON}" <<'PY'
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
payload = json.loads(summary_path.read_text())
next_step = str(payload.get("recommended_next_step", "")).strip()
base_arm = str(payload.get("base_for_v7_6_arm_id", "")).strip()
if next_step != "prepare_v7_6_decision_point":
    raise SystemExit(
        f"V7-5 restart line did not authorize V7-6 decision point; recommended_next_step={next_step!r}"
    )
if not base_arm:
    raise SystemExit("V7-5 summary missing base_for_v7_6_arm_id")
PY

if tmux has-session -t planv7_lr75e5_v76 2>/dev/null; then
  exit 0
fi

tmux new-session -d -s planv7_lr75e5_v76 \
  "mkdir -p ${RUN_ROOT} ${RESULT_ROOT} && \
   cd ${ROOT_DIR} && \
   bash scripts/run_planv7_lr75e5_v7_6_multiseed_confirmation_qwen25.sh \
     ${BASE_SEED} \
     ${RUN_ROOT} \
     ${RESULT_ROOT} \
     ${RESUME_STAGE_B_ROOT} \
     ${TRAIN_STEPS} \
     ${MODEL_DIR} \
     ${V75_SUMMARY_JSON} \
     ${V70_SUMMARY_JSON} \
   2>&1 | tee ${RUN_ROOT}/tmux-session.log"

tmux new-session -d -s planv7_lr75e5_v76_post \
  "bash -lc 'while [ ! -f ${RESULT_ROOT}/v7-6-summary.json ]; do sleep 30; done; \
   cd ${ROOT_DIR}; \
   bash scripts/publish_review_artifacts.sh; \
   git add scripts/run_planv7_v7_6_multiseed_confirmation_qwen25.sh \
     docs/exec-plans/active/20260312-planv7-lr75e5-restart.md \
     docs/exec-plans/active/20260312-planv7-lr75e5-v7-6-multiseed-confirmation.md \
     results/generated/review/planv7-lr75e5-v7-6-multiseed-confirmation-qwen25 \
     runs/review/planv7-lr75e5-v7-6-multiseed-confirmation-qwen25; \
   git commit -m \"feat: complete planv7 lr-updated v7-6 multiseed confirmation\"; \
   git push origin main; \
   bash scripts/push_github_review_snapshot.sh' > ${RUN_ROOT}/postpublish.log 2>&1"
