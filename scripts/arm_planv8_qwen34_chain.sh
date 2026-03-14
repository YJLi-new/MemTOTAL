#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

BASE_SEED="${1:-61109}"
V80_RUN_ROOT="${2:-/root/autodl-tmp/runs/verify/planv8-v8-0-qwen34-baselines-oracles}"
V80_RESULT_ROOT="${3:-/root/autodl-tmp/results/generated/planv8-v8-0-qwen34-baselines-oracles}"
QWEN25_MODEL_DIR="${4:-/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct}"
QWEN34_MODEL_DIR="${5:-/root/autodl-tmp/models/Qwen3-4B}"
QWEN25_REFERENCE_SUMMARY="${6:-results/generated/review/planv7-lr75e5-v7-0-metrics-oracle-qwen25/v7-0-summary.json}"
V80_RUN_SESSION="${7:-planv8_v80_q34}"
V80_WATCH_SESSION="${8:-planv8_v80_q34_watch}"
V80_POST_SESSION="${9:-planv8_v80_q34_post}"
V80_DLWATCH_SESSION="${10:-planv8_v80_q34_dlwatch}"
V81_QUEUE_SESSION="${11:-planv8_q34_queue_v81}"
SUPERWATCH_SESSION="${12:-planv8_q34_superwatch}"
V81_RUN_ROOT="${13:-/root/autodl-tmp/runs/verify/planv8-v8-1-reader-interface-scout-qwen34}"
V81_RESULT_ROOT="${14:-/root/autodl-tmp/results/generated/planv8-v8-1-reader-interface-scout-qwen34}"

mkdir -p "${V80_RUN_ROOT}" "${V80_RESULT_ROOT}" "${V81_RUN_ROOT}" "${V81_RESULT_ROOT}"

if ! tmux has-session -t "${V80_RUN_SESSION}" 2>/dev/null; then
  tmux new-session -d -s "${V80_RUN_SESSION}" \
    "mkdir -p ${V80_RUN_ROOT} ${V80_RESULT_ROOT} && \
     cd ${ROOT_DIR} && \
     bash scripts/run_planv8_v8_0_qwen34_baselines_oracles.sh \
       ${BASE_SEED} \
       ${V80_RUN_ROOT} \
       ${V80_RESULT_ROOT} \
       ${QWEN25_MODEL_DIR} \
       ${QWEN34_MODEL_DIR} \
       ${QWEN25_REFERENCE_SUMMARY} \
     2>&1 | tee -a ${V80_RUN_ROOT}/tmux-session.log"
fi

if ! tmux has-session -t "${V80_WATCH_SESSION}" 2>/dev/null; then
  tmux new-session -d -s "${V80_WATCH_SESSION}" \
    "bash -lc 'while tmux has-session -t ${V80_RUN_SESSION} 2>/dev/null; do ts=\$(date -u +%Y-%m-%dT%H:%M:%SZ); metrics_count=\$(find ${V80_RUN_ROOT} -path \"*/metrics.json\" -type f 2>/dev/null | wc -l); gpu_line=\$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | head -n 1 | sed \"s/,/\\//\" || true); echo \"\${ts} metrics=\${metrics_count} gpu_mib=\${gpu_line:-unknown}\"; sleep 300; done' >> ${V80_RUN_ROOT}/watch.log 2>&1"
fi

if ! tmux has-session -t "${V80_DLWATCH_SESSION}" 2>/dev/null; then
  tmux new-session -d -s "${V80_DLWATCH_SESSION}" \
    "bash -lc 'while tmux has-session -t ${V80_RUN_SESSION} 2>/dev/null; do ts=\$(date -u +%Y-%m-%dT%H:%M:%SZ); staged=\$(find ${QWEN34_MODEL_DIR} -maxdepth 1 -type f -name \"model-*.safetensors\" -printf \"%f=%s \" 2>/dev/null | sort | tr -d \"\\n\"); active=\$(ps -eo pid,etimes,cmd | awk '\''/[w]get -c -O ${QWEN34_MODEL_DIR//\//\\/}\\/model-/{print \$1\":\"\$2\":\"\$3; exit}'\'' ); echo \"\${ts} staged=\${staged:-none} active_wget=\${active:-none}\"; sleep 60; done' >> ${V80_RUN_ROOT}/download-watch-live.log 2>&1"
fi

if ! tmux has-session -t "${V80_POST_SESSION}" 2>/dev/null; then
  tmux new-session -d -s "${V80_POST_SESSION}" \
    "bash -lc 'while [ ! -f ${V80_RESULT_ROOT}/v8-0-summary.json ]; do sleep 30; done; \
     cd ${ROOT_DIR}; \
     bash scripts/publish_review_artifacts.sh; \
     git add docs/exec-plans/active/20260314-planv8-qwen34-restart.md \
       results/generated/review/planv8-v8-0-qwen34-baselines-oracles \
       runs/review/planv8-v8-0-qwen34-baselines-oracles; \
     if ! git diff --cached --quiet; then \
       git commit -m \"feat: complete planv8 qwen34 v8-0 baselines oracles\"; \
       gh auth setup-git; \
       git push origin main; \
       bash scripts/push_github_review_snapshot.sh; \
     fi' > ${V80_RUN_ROOT}/postpublish.log 2>&1"
fi

if ! tmux has-session -t "${V81_QUEUE_SESSION}" 2>/dev/null; then
  tmux new-session -d -s "${V81_QUEUE_SESSION}" \
    "cd ${ROOT_DIR} && \
     bash scripts/queue_planv8_qwen34_v8_1_after_v8_0.sh \
       ${BASE_SEED} \
       ${V80_RESULT_ROOT} \
       ${V81_RUN_ROOT} \
       ${V81_RESULT_ROOT} \
       ${QWEN34_MODEL_DIR}"
fi

if ! tmux has-session -t "${SUPERWATCH_SESSION}" 2>/dev/null; then
  tmux new-session -d -s "${SUPERWATCH_SESSION}" \
    "bash -lc 'while true; do ts=\$(date -u +%Y-%m-%dT%H:%M:%SZ); v80_state=down; v81q_state=down; v81_state=down; tmux has-session -t ${V80_RUN_SESSION} 2>/dev/null && v80_state=up; tmux has-session -t ${V81_QUEUE_SESSION} 2>/dev/null && v81q_state=up; tmux has-session -t planv8_v81_q34 2>/dev/null && v81_state=up; v80_metrics=\$(find ${V80_RUN_ROOT} -path \"*/metrics.json\" -type f 2>/dev/null | wc -l); v81_metrics=\$(find ${V81_RUN_ROOT} -path \"*/metrics.json\" -type f 2>/dev/null | wc -l); staged=\$(find ${QWEN34_MODEL_DIR} -maxdepth 1 -type f -name \"model-*.safetensors\" -printf \"%f=%s \" 2>/dev/null | sort | tr -d \"\\n\"); gpu=\$(nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -n 1 | sed \"s/,/\\//g\" || true); active=\$(ps -eo pid,etimes,cmd | awk '\''/[w]get -c -O ${QWEN34_MODEL_DIR//\//\\/}\\/model-/{print \$1\":\"\$2\":\"\$3; exit}'\'' ); v80_summary=no; v81_summary=no; [ -f ${V80_RESULT_ROOT}/v8-0-summary.json ] && v80_summary=yes; [ -f ${V81_RESULT_ROOT}/v8-1-summary.json ] && v81_summary=yes; echo \"\${ts} v80=\${v80_state} v80_metrics=\${v80_metrics} v80_summary=\${v80_summary} v81_queue=\${v81q_state} v81=\${v81_state} v81_metrics=\${v81_metrics} v81_summary=\${v81_summary} gpu_mib_util=\${gpu:-unknown} staged=\${staged:-none} active_wget=\${active:-none}\"; sleep 120; done' >> /root/autodl-tmp/runs/verify/planv8-qwen34-superwatch-live.log 2>&1"
fi
