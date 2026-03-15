#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_FILE="${ROOT_DIR}/data/toy/smoke_samples.jsonl"

if [[ ! -f "${DATA_FILE}" ]]; then
  echo "missing data file: ${DATA_FILE}" >&2
  exit 1
fi

echo "data-ok ${DATA_FILE}"

