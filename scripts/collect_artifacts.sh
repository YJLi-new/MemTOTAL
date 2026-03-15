#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 <run_dir> [destination_dir]" >&2
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="$(cd "$1" && pwd)"
DEST_DIR="${2:-${ROOT_DIR}/results/reports/$(basename "${RUN_DIR}")}"

mkdir -p "${DEST_DIR}"

for file in config.snapshot.yaml run_info.json metrics.json predictions.jsonl checkpoint.pt profiling.json profiling.csv summary.csv summary.svg; do
  if [[ -f "${RUN_DIR}/${file}" ]]; then
    cp "${RUN_DIR}/${file}" "${DEST_DIR}/"
  fi
done

echo "artifacts-collected ${DEST_DIR}"
