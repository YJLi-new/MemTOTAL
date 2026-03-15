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
V80_REVIEW_NAMESPACE="${15:-planv8-v8-0-qwen34-baselines-oracles}"
V80_DOC_PATH="${16:-docs/exec-plans/active/20260314-planv8-qwen34-restart.md}"
V80_POST_COMMIT_MESSAGE="${17:-feat: complete planv8 qwen34 v8-0 baselines oracles}"
V82_QUEUE_SESSION="${18:-planv8_q34_queue_v82}"
V82_RUN_ROOT="${19:-/root/autodl-tmp/runs/verify/planv8-v8-2-reader-sweep-qwen34}"
V82_RESULT_ROOT="${20:-/root/autodl-tmp/results/generated/planv8-v8-2-reader-sweep-qwen34}"
V82_RUN_SESSION="${21:-planv8_v82_q34}"
V83_QUEUE_SESSION="${22:-planv8_q34_queue_v83}"
V83_RUN_ROOT="${23:-/root/autodl-tmp/runs/verify/planv8-v8-3-reader-opd-qwen34}"
V83_RESULT_ROOT="${24:-/root/autodl-tmp/results/generated/planv8-v8-3-reader-opd-qwen34}"
V83_RUN_SESSION="${25:-planv8_v83_q34}"
V84_QUEUE_SESSION="${26:-planv8_q34_queue_v84}"
V84_RUN_ROOT="${27:-/root/autodl-tmp/runs/verify/planv8-v8-4-external-writer-qwen34}"
V84_RESULT_ROOT="${28:-/root/autodl-tmp/results/generated/planv8-v8-4-external-writer-qwen34}"
V84_RUN_SESSION="${29:-planv8_v84_q34}"
V85_QUEUE_SESSION="${30:-planv8_q34_queue_v85}"
V85_RUN_ROOT="${31:-/root/autodl-tmp/runs/verify/planv8-v8-5-bridge-revisit-qwen34}"
V85_RESULT_ROOT="${32:-/root/autodl-tmp/results/generated/planv8-v8-5-bridge-revisit-qwen34}"
V85_RUN_SESSION="${33:-planv8_v85_q34}"
V86_QUEUE_SESSION="${34:-planv8_q34_queue_v86}"
V86_RUN_ROOT="${35:-/root/autodl-tmp/runs/verify/planv8-v8-6-writer-aux-qwen34}"
V86_RESULT_ROOT="${36:-/root/autodl-tmp/results/generated/planv8-v8-6-writer-aux-qwen34}"
V86_RUN_SESSION="${37:-planv8_v86_q34}"
V87_QUEUE_SESSION="${38:-planv8_q34_queue_v87}"
V87_RUN_ROOT="${39:-/root/autodl-tmp/runs/verify/planv8-v8-7-comparators-qwen34}"
V87_RESULT_ROOT="${40:-/root/autodl-tmp/results/generated/planv8-v8-7-comparators-qwen34}"
V87_RUN_SESSION="${41:-planv8_v87_q34}"
V88_QUEUE_SESSION="${42:-planv8_q34_queue_v88}"
V88_RUN_ROOT="${43:-/root/autodl-tmp/runs/verify/planv8-v8-8-multiseed-confirmation-qwen34}"
V88_RESULT_ROOT="${44:-/root/autodl-tmp/results/generated/planv8-v8-8-multiseed-confirmation-qwen34}"
V88_RUN_SESSION="${45:-planv8_v88_q34}"
V89_QUEUE_SESSION="${46:-planv8_q34_queue_v89}"
V89_RUN_ROOT="${47:-/root/autodl-tmp/runs/verify/planv8-v8-9-cdmi-qwen34}"
V89_RESULT_ROOT="${48:-/root/autodl-tmp/results/generated/planv8-v8-9-cdmi-qwen34}"
V89_RUN_SESSION="${49:-planv8_v89_q34}"

mkdir -p \
  "${V80_RUN_ROOT}" \
  "${V80_RESULT_ROOT}" \
  "${V81_RUN_ROOT}" \
  "${V81_RESULT_ROOT}" \
  "${V82_RUN_ROOT}" \
  "${V82_RESULT_ROOT}" \
  "${V83_RUN_ROOT}" \
  "${V83_RESULT_ROOT}" \
  "${V84_RUN_ROOT}" \
  "${V84_RESULT_ROOT}" \
  "${V85_RUN_ROOT}" \
  "${V85_RESULT_ROOT}" \
  "${V86_RUN_ROOT}" \
  "${V86_RESULT_ROOT}" \
  "${V87_RUN_ROOT}" \
  "${V87_RESULT_ROOT}" \
  "${V88_RUN_ROOT}" \
  "${V88_RESULT_ROOT}" \
  "${V89_RUN_ROOT}" \
  "${V89_RESULT_ROOT}"

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
     git add ${V80_DOC_PATH} \
       results/generated/review/${V80_REVIEW_NAMESPACE} \
       runs/review/${V80_REVIEW_NAMESPACE}; \
     if ! git diff --cached --quiet; then \
       git commit -m \"${V80_POST_COMMIT_MESSAGE}\"; \
       env -u HTTPS_PROXY -u HTTP_PROXY -u ALL_PROXY -u https_proxy -u http_proxy -u all_proxy gh auth setup-git; \
       env -u HTTPS_PROXY -u HTTP_PROXY -u ALL_PROXY -u https_proxy -u http_proxy -u all_proxy git -c http.version=HTTP/1.1 push origin main; \
       env -u HTTPS_PROXY -u HTTP_PROXY -u ALL_PROXY -u https_proxy -u http_proxy -u all_proxy bash scripts/push_github_review_snapshot.sh; \
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

if ! tmux has-session -t "${V82_QUEUE_SESSION}" 2>/dev/null; then
  tmux new-session -d -s "${V82_QUEUE_SESSION}" \
    "cd ${ROOT_DIR} && \
     bash scripts/queue_planv8_qwen34_v8_2_after_v8_1.sh \
       ${BASE_SEED} \
       ${V81_RESULT_ROOT} \
       ${V82_RUN_ROOT} \
       ${V82_RESULT_ROOT} \
       ${QWEN34_MODEL_DIR}"
fi

if ! tmux has-session -t "${V83_QUEUE_SESSION}" 2>/dev/null; then
  tmux new-session -d -s "${V83_QUEUE_SESSION}" \
    "cd ${ROOT_DIR} && \
     bash scripts/queue_planv8_qwen34_v8_3_after_v8_2.sh \
       ${BASE_SEED} \
       ${V82_RUN_ROOT} \
       ${V82_RESULT_ROOT} \
       ${V83_RUN_ROOT} \
       ${V83_RESULT_ROOT} \
       ${QWEN34_MODEL_DIR}"
fi

if ! tmux has-session -t "${V84_QUEUE_SESSION}" 2>/dev/null; then
  tmux new-session -d -s "${V84_QUEUE_SESSION}" \
    "cd ${ROOT_DIR} && \
     bash scripts/queue_planv8_qwen34_v8_4_after_v8_3.sh \
       ${BASE_SEED} \
       ${V83_RUN_ROOT} \
       ${V83_RESULT_ROOT} \
       ${V84_RUN_ROOT} \
       ${V84_RESULT_ROOT} \
       ${QWEN34_MODEL_DIR}"
fi

if ! tmux has-session -t "${V85_QUEUE_SESSION}" 2>/dev/null; then
  tmux new-session -d -s "${V85_QUEUE_SESSION}" \
    "cd ${ROOT_DIR} && \
     bash scripts/queue_planv8_qwen34_v8_5_after_v8_4.sh \
       ${BASE_SEED} \
       ${V84_RUN_ROOT} \
       ${V84_RESULT_ROOT} \
       ${V85_RUN_ROOT} \
       ${V85_RESULT_ROOT} \
       ${QWEN34_MODEL_DIR}"
fi

if ! tmux has-session -t "${V86_QUEUE_SESSION}" 2>/dev/null; then
  tmux new-session -d -s "${V86_QUEUE_SESSION}" \
    "cd ${ROOT_DIR} && \
     bash scripts/queue_planv8_qwen34_v8_6_after_v8_5.sh \
       ${BASE_SEED} \
       ${V85_RUN_ROOT} \
       ${V85_RESULT_ROOT} \
       ${V86_RUN_ROOT} \
       ${V86_RESULT_ROOT} \
       ${QWEN34_MODEL_DIR}"
fi

if ! tmux has-session -t "${V87_QUEUE_SESSION}" 2>/dev/null; then
  tmux new-session -d -s "${V87_QUEUE_SESSION}" \
    "cd ${ROOT_DIR} && \
     bash scripts/queue_planv8_qwen34_v8_7_after_v8_3.sh \
       ${BASE_SEED} \
       ${V83_RESULT_ROOT} \
       ${V87_RUN_ROOT} \
       ${V87_RESULT_ROOT} \
       ${QWEN34_MODEL_DIR}"
fi

if ! tmux has-session -t "${V88_QUEUE_SESSION}" 2>/dev/null; then
  tmux new-session -d -s "${V88_QUEUE_SESSION}" \
    "cd ${ROOT_DIR} && \
     bash scripts/queue_planv8_qwen34_v8_8_after_v8_7.sh \
       ${BASE_SEED} \
       ${V87_RESULT_ROOT} \
       ${V88_RUN_ROOT} \
       ${V88_RESULT_ROOT} \
       ${QWEN34_MODEL_DIR}"
fi

if ! tmux has-session -t "${V89_QUEUE_SESSION}" 2>/dev/null; then
  tmux new-session -d -s "${V89_QUEUE_SESSION}" \
    "cd ${ROOT_DIR} && \
     bash scripts/queue_planv8_qwen34_v8_9_after_v8_8.sh \
       ${BASE_SEED} \
       ${V88_RESULT_ROOT} \
       ${V89_RUN_ROOT} \
       ${V89_RESULT_ROOT} \
       ${QWEN34_MODEL_DIR}"
fi

if ! tmux has-session -t "${SUPERWATCH_SESSION}" 2>/dev/null; then
  tmux new-session -d -s "${SUPERWATCH_SESSION}" \
    "bash -lc 'while true; do ts=\$(date -u +%Y-%m-%dT%H:%M:%SZ); v80_state=down; v81q_state=down; v81_state=down; v82q_state=down; v82_state=down; v83q_state=down; v83_state=down; v84q_state=down; v84_state=down; v85q_state=down; v85_state=down; v86q_state=down; v86_state=down; v87q_state=down; v87_state=down; v88q_state=down; v88_state=down; v89q_state=down; v89_state=down; tmux has-session -t ${V80_RUN_SESSION} 2>/dev/null && v80_state=up; tmux has-session -t ${V81_QUEUE_SESSION} 2>/dev/null && v81q_state=up; tmux has-session -t planv8_v81_q34 2>/dev/null && v81_state=up; tmux has-session -t ${V82_QUEUE_SESSION} 2>/dev/null && v82q_state=up; tmux has-session -t ${V82_RUN_SESSION} 2>/dev/null && v82_state=up; tmux has-session -t ${V83_QUEUE_SESSION} 2>/dev/null && v83q_state=up; tmux has-session -t ${V83_RUN_SESSION} 2>/dev/null && v83_state=up; tmux has-session -t ${V84_QUEUE_SESSION} 2>/dev/null && v84q_state=up; tmux has-session -t ${V84_RUN_SESSION} 2>/dev/null && v84_state=up; tmux has-session -t ${V85_QUEUE_SESSION} 2>/dev/null && v85q_state=up; tmux has-session -t ${V85_RUN_SESSION} 2>/dev/null && v85_state=up; tmux has-session -t ${V86_QUEUE_SESSION} 2>/dev/null && v86q_state=up; tmux has-session -t ${V86_RUN_SESSION} 2>/dev/null && v86_state=up; tmux has-session -t ${V87_QUEUE_SESSION} 2>/dev/null && v87q_state=up; tmux has-session -t ${V87_RUN_SESSION} 2>/dev/null && v87_state=up; tmux has-session -t ${V88_QUEUE_SESSION} 2>/dev/null && v88q_state=up; tmux has-session -t ${V88_RUN_SESSION} 2>/dev/null && v88_state=up; tmux has-session -t ${V89_QUEUE_SESSION} 2>/dev/null && v89q_state=up; tmux has-session -t ${V89_RUN_SESSION} 2>/dev/null && v89_state=up; v80_metrics=\$(find ${V80_RUN_ROOT} -path \"*/metrics.json\" -type f 2>/dev/null | wc -l); v81_metrics=\$(find ${V81_RUN_ROOT} -path \"*/metrics.json\" -type f 2>/dev/null | wc -l); v82_progress=\$(python ${ROOT_DIR}/scripts/planv8_watch_progress.py --run_root ${V82_RUN_ROOT} 2>/dev/null || echo \"{}\"); v83_progress=\$(python ${ROOT_DIR}/scripts/planv8_watch_progress.py --run_root ${V83_RUN_ROOT} 2>/dev/null || echo \"{}\"); v84_progress=\$(python ${ROOT_DIR}/scripts/planv8_watch_progress.py --run_root ${V84_RUN_ROOT} 2>/dev/null || echo \"{}\"); v85_progress=\$(python ${ROOT_DIR}/scripts/planv8_watch_progress.py --run_root ${V85_RUN_ROOT} 2>/dev/null || echo \"{}\"); v86_progress=\$(python ${ROOT_DIR}/scripts/planv8_watch_progress.py --run_root ${V86_RUN_ROOT} 2>/dev/null || echo \"{}\"); v87_metrics=\$(find ${V87_RUN_ROOT} -path \"*/metrics.json\" -type f 2>/dev/null | wc -l); v88_metrics=\$(find ${V88_RUN_ROOT} -path \"*/metrics.json\" -type f 2>/dev/null | wc -l); v89_metrics=\$(find ${V89_RUN_ROOT} -path \"*/metrics.json\" -type f 2>/dev/null | wc -l); staged=\$(find ${QWEN34_MODEL_DIR} -maxdepth 1 -type f -name \"model-*.safetensors\" -printf \"%f=%s \" 2>/dev/null | sort | tr -d \"\\n\"); gpu=\$(nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -n 1 | sed \"s/,/\\//g\" || true); active=\$(ps -eo pid,etimes,cmd | awk '\''/[w]get -c -O ${QWEN34_MODEL_DIR//\//\\/}\\/model-/{print \$1\":\"\$2\":\"\$3; exit}'\'' ); v80_summary=no; v81_summary=no; v82_summary=no; v83_summary=no; v84_summary=no; v85_summary=no; v86_summary=no; v87_summary=no; v88_summary=no; v89_summary=no; [ -f ${V80_RESULT_ROOT}/v8-0-summary.json ] && v80_summary=yes; [ -f ${V81_RESULT_ROOT}/v8-1-summary.json ] && v81_summary=yes; [ -f ${V82_RESULT_ROOT}/v8-2-summary.json ] && v82_summary=yes; [ -f ${V83_RESULT_ROOT}/v8-3-summary.json ] && v83_summary=yes; [ -f ${V84_RESULT_ROOT}/v8-4-summary.json ] && v84_summary=yes; [ -f ${V85_RESULT_ROOT}/v8-5-summary.json ] && v85_summary=yes; [ -f ${V86_RESULT_ROOT}/v8-6-summary.json ] && v86_summary=yes; [ -f ${V87_RESULT_ROOT}/v8-7-summary.json ] && v87_summary=yes; [ -f ${V88_RESULT_ROOT}/v8-8-summary.json ] && v88_summary=yes; [ -f ${V89_RESULT_ROOT}/v8-9-summary.json ] && v89_summary=yes; echo \"\${ts} v80=\${v80_state} v80_metrics=\${v80_metrics} v80_summary=\${v80_summary} v81_queue=\${v81q_state} v81=\${v81_state} v81_metrics=\${v81_metrics} v81_summary=\${v81_summary} v82_queue=\${v82q_state} v82=\${v82_state} v82_summary=\${v82_summary} v82_progress=\${v82_progress} v83_queue=\${v83q_state} v83=\${v83_state} v83_summary=\${v83_summary} v83_progress=\${v83_progress} v84_queue=\${v84q_state} v84=\${v84_state} v84_summary=\${v84_summary} v84_progress=\${v84_progress} v85_queue=\${v85q_state} v85=\${v85_state} v85_summary=\${v85_summary} v85_progress=\${v85_progress} v86_queue=\${v86q_state} v86=\${v86_state} v86_summary=\${v86_summary} v86_progress=\${v86_progress} v87_queue=\${v87q_state} v87=\${v87_state} v87_summary=\${v87_summary} v87_metrics=\${v87_metrics} v88_queue=\${v88q_state} v88=\${v88_state} v88_summary=\${v88_summary} v88_metrics=\${v88_metrics} v89_queue=\${v89q_state} v89=\${v89_state} v89_summary=\${v89_summary} v89_metrics=\${v89_metrics} gpu_mib_util=\${gpu:-unknown} staged=\${staged:-none} active_wget=\${active:-none}\"; sleep 120; done' >> /root/autodl-tmp/runs/verify/planv8-qwen34-superwatch-live.log 2>&1"
fi
