#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${1:?root dir required}"

cd "${ROOT_DIR}"

while true; do
  bash scripts/arm_planv9_v9_1_qwen34.sh --no-supervisor
  sleep 600
done
