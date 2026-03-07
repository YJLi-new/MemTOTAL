#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

sync_dir() {
  local src="$1"
  local dst="$2"
  if [[ ! -d "${src}" ]]; then
    echo "missing source: ${src}" >&2
    return 1
  fi
  mkdir -p "$(dirname "${dst}")"
  rsync -a --delete "${src}/" "${dst}/"
}

sync_dir "runs/verify/m3-core4-qwen25/stage-b" "runs/review/m3-core4-qwen25-stage-b"
sync_dir "runs/verify/m3-core4-qwen3/stage-b" "runs/review/m3-core4-qwen3-stage-b"
sync_dir "/root/autodl-tmp/memtotal-stage-c-qonly-negative-count-sweep-v1" "runs/review/m3-core4-stage-c-qonly-negative-count-sweep-v1"
sync_dir "/root/autodl-tmp/memtotal-stage-c-qonly-retrieval-loss-sweep-v1" "runs/review/m3-core4-stage-c-qonly-retrieval-loss-sweep-v1"
sync_dir "/root/autodl-tmp/memtotal-stage-c-qonly-seed-sweep-v5-margin-canonical" "runs/review/m3-core4-stage-c-qonly-seed-sweep-v5-margin-canonical"
sync_dir "/root/autodl-tmp/memtotal-stage-c-qonly-seed-sweep-v6-case-dump" "runs/review/m3-core4-stage-c-qonly-seed-sweep-v6-case-dump"

sync_dir "results/generated/m3-core4-stage-c-qonly-negative-count-sweep-v1" "results/generated/review/m3-core4-stage-c-qonly-negative-count-sweep-v1"
sync_dir "results/generated/m3-core4-stage-c-qonly-retrieval-loss-sweep-v1" "results/generated/review/m3-core4-stage-c-qonly-retrieval-loss-sweep-v1"
sync_dir "results/generated/m3-core4-stage-c-qonly-seed-sweep-v5-margin-canonical" "results/generated/review/m3-core4-stage-c-qonly-seed-sweep-v5-margin-canonical"
sync_dir "results/generated/m3-core4-stage-c-qonly-seed-sweep-v6-case-dump" "results/generated/review/m3-core4-stage-c-qonly-seed-sweep-v6-case-dump"
sync_dir "results/generated/m3-core4-stage-c-error-attribution-v1" "results/generated/review/m3-core4-stage-c-error-attribution-v1"
sync_dir "results/generated/m3-core4-stage-c-margin-audit-v3-fixed-holdout" "results/generated/review/m3-core4-stage-c-margin-audit-v3-fixed-holdout"
sync_dir "results/generated/m3-core4-stage-c-negative-seed-curve-audit-v2-fixed-holdout" "results/generated/review/m3-core4-stage-c-negative-seed-curve-audit-v2-fixed-holdout"
sync_dir "results/generated/m3-core4-stage-c-curve-suite-v3-fixed-holdout" "results/generated/review/m3-core4-stage-c-curve-suite-v3-fixed-holdout"

echo "review artifacts refreshed"
