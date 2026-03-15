#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

WATCH_FILE="${WATCH_FILE:-runs/verify/memgen-story-cloze-qwen3-smoke-v2/metrics.json}"
OUTPUT_DIR="${OUTPUT_DIR:-results/generated/m5-story-cloze-baseline-grid-protocol-with-memgen-dual-smoke}"
SEED="${SEED:-1001}"
POLL_SECONDS="${POLL_SECONDS:-30}"
ONCE="${ONCE:-0}"
LOG_PATH="${LOG_PATH:-${OUTPUT_DIR}/watcher.log}"
STATE_PATH="${STATE_PATH:-${OUTPUT_DIR}/watcher_state.json}"

mkdir -p "$(dirname "${LOG_PATH}")"

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "${LOG_PATH}"
}

write_state() {
  local status="$1"
  local detail="$2"
  python - "$STATE_PATH" "$status" "$detail" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "status": sys.argv[2],
    "detail": sys.argv[3],
}
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
}

refresh_grid() {
  log "detected ${WATCH_FILE}; refreshing dual-import grid at ${OUTPUT_DIR}"
  ./scripts/run_story_cloze_baseline_grid_protocol_with_memgen_dual.sh "${SEED}" "${OUTPUT_DIR}" >>"${LOG_PATH}" 2>&1
  write_state "refreshed" "dual-import grid refreshed after qwen3 MemGen metrics appeared"
  log "refresh finished"
}

if [[ -f "${WATCH_FILE}" ]]; then
  refresh_grid
  exit 0
fi

if [[ "${ONCE}" == "1" ]]; then
  log "watch target not ready yet: ${WATCH_FILE}"
  write_state "waiting" "watch target not ready yet"
  exit 0
fi

log "waiting for ${WATCH_FILE}"
write_state "waiting" "polling for qwen3 MemGen metrics"
while [[ ! -f "${WATCH_FILE}" ]]; do
  sleep "${POLL_SECONDS}"
done

refresh_grid
