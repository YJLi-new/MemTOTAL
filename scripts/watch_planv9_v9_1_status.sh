#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="${1:?run root required}"
RESULT_ROOT="${2:?result root required}"
RUNNER_SESSION="${3:?runner session required}"
EXPECTED_RUNS="${4:-12}"

while true; do
  timestamp="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  completed="$(find "${RUN_ROOT}" -path '*/metrics.json' | wc -l | tr -d ' ')"
  latest="$(find "${RUN_ROOT}" -path '*/metrics.json' -printf '%T@ %h\n' 2>/dev/null | sort -nr | head -n1 | awk '{print $2}' | xargs -r basename)"
  gpu="$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | head -n1 | tr -d ' ')"
  runner_state="down"
  if tmux has-session -t "${RUNNER_SESSION}" 2>/dev/null; then
    runner_state="up"
  fi
  summary_state="no"
  if [[ -f "${RESULT_ROOT}/v9-1-summary.json" ]]; then
    summary_state="yes"
  fi
  echo "${timestamp} completed=${completed}/${EXPECTED_RUNS} latest=${latest:-done} gpu=${gpu:-0,0} summary=${summary_state} runner=${runner_state}"
  if [[ "${summary_state}" == "yes" ]]; then
    exit 0
  fi
  sleep 60
done
