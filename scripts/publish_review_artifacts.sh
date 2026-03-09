#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

sync_dir() {
  local src="$1"
  local dst="$2"
  if [[ ! -d "${src}" ]]; then
    echo "missing source: ${src}" >&2
    return 0
  fi
  mkdir -p "$(dirname "${dst}")"
  rsync -a \
    --exclude='*.pt' \
    --exclude='*.ckpt' \
    --exclude='.analysis.lock' \
    --exclude='.suite.lock' \
    --exclude='snapshot_evals/' \
    --exclude='task_case_dump.jsonl' \
    "${src}/" "${dst}/"
}

sync_first_available() {
  local dst="$1"
  shift
  local src
  for src in "$@"; do
    if [[ -d "${src}" ]]; then
      sync_dir "${src}" "${dst}"
      return 0
    fi
  done
  echo "missing source: $*" >&2
  return 0
}

rebuild_docs_bundle() {
  local bundle_path="docs_review_bundle.zip"
  local temp_bundle
  temp_bundle="$(mktemp "${bundle_path}.tmp.XXXXXX")"
  rm -f "${temp_bundle}"
  mapfile -t doc_files < <(find docs -type f | sort)
  if [[ "${#doc_files[@]}" -eq 0 ]]; then
    echo "missing docs content for ${bundle_path}" >&2
    return 1
  fi
  zip -q -X "${temp_bundle}" "${doc_files[@]}"
  mv "${temp_bundle}" "${bundle_path}"
}

sync_dir "runs/verify/m3-core4-qwen25/stage-b" "runs/review/m3-core4-qwen25-stage-b"
sync_dir "runs/verify/m3-core4-qwen3/stage-b" "runs/review/m3-core4-qwen3-stage-b"
sync_dir "runs/verify/m3-story-cloze-real-pilot-qwen25" "runs/review/m3-story-cloze-real-pilot-qwen25"
sync_dir "runs/verify/m3-fever-real-pilot-qwen25" "runs/review/m3-fever-real-pilot-qwen25"
sync_dir "runs/verify/m4-fever-shared-injection-qwen25" "runs/review/m4-fever-shared-injection-qwen25"
sync_dir "runs/verify/m4-fever-dynamics-recovery-qwen25" "runs/review/m4-fever-dynamics-recovery-qwen25"
sync_dir "runs/verify/m4-fever-dynamics-recovery-stabilized-qwen25" "runs/review/m4-fever-dynamics-recovery-stabilized-qwen25"
sync_first_available "runs/review/m4-fever-deep-prompt-recovery-qwen25" \
  "runs/verify/m4-fever-deep-prompt-recovery-qwen25" \
  "/root/autodl-tmp/runs/verify/m4-fever-deep-prompt-recovery-qwen25"
sync_first_available "runs/review/m4-fever-anti-shortcut-recovery-qwen25" \
  "runs/verify/m4-fever-anti-shortcut-recovery-qwen25" \
  "/root/autodl-tmp/runs/verify/m4-fever-anti-shortcut-recovery-qwen25"
sync_first_available "runs/review/m4-fever-shared-injection-alignment-qwen25" \
  "runs/verify/m4-fever-shared-injection-alignment-qwen25" \
  "/root/autodl-tmp/runs/verify/m4-fever-shared-injection-alignment-qwen25"
sync_first_available "runs/review/m5-fever-writer-reasoner-alignment-qwen25" \
  "runs/verify/m5-fever-writer-reasoner-alignment-qwen25" \
  "/root/autodl-tmp/runs/verify/m5-fever-writer-reasoner-alignment-qwen25"
sync_first_available "runs/review/m5-fever-writer-objective-rewrite-qwen25" \
  "runs/verify/m5-fever-writer-objective-rewrite-qwen25" \
  "/root/autodl-tmp/runs/verify/m5-fever-writer-objective-rewrite-qwen25"
sync_first_available "runs/review/m5-fever-dense-teacher-qwen25" \
  "runs/verify/m5-fever-dense-teacher-qwen25" \
  "/root/autodl-tmp/runs/verify/m5-fever-dense-teacher-qwen25"
sync_first_available "runs/review/tl-poc-fever-qwen25" \
  "runs/verify/tl-poc-fever-qwen25" \
  "/root/autodl-tmp/runs/verify/tl-poc-fever-qwen25"
sync_first_available "runs/review/tl-bridge-rescue-fever-qwen25" \
  "runs/verify/tl-bridge-rescue-fever-qwen25" \
  "/root/autodl-tmp/runs/verify/tl-bridge-rescue-fever-qwen25"
sync_first_available "runs/review/tl-slot-basis-rescue-fever-qwen25" \
  "runs/verify/tl-slot-basis-rescue-fever-qwen25" \
  "/root/autodl-tmp/runs/verify/tl-slot-basis-rescue-fever-qwen25"
sync_first_available "runs/review/tl-reader-geometry-rescue-fever-qwen25" \
  "runs/verify/tl-reader-geometry-rescue-fever-qwen25" \
  "/root/autodl-tmp/runs/verify/tl-reader-geometry-rescue-fever-qwen25"
sync_first_available "runs/review/tl-reader-symmetry-break-fever-qwen25" \
  "runs/verify/tl-reader-symmetry-break-fever-qwen25" \
  "/root/autodl-tmp/runs/verify/tl-reader-symmetry-break-fever-qwen25"
sync_first_available "runs/review/tl-reader-local-bootstrap-fever-qwen25" \
  "runs/verify/tl-reader-local-bootstrap-fever-qwen25" \
  "/root/autodl-tmp/runs/verify/tl-reader-local-bootstrap-fever-qwen25"
sync_first_available "runs/review/tl-writer-value-fever-qwen25" \
  "runs/verify/tl-writer-value-fever-qwen25" \
  "/root/autodl-tmp/runs/verify/tl-writer-value-fever-qwen25"
sync_first_available "runs/review/tl-micro-lora-fever-qwen25" \
  "runs/verify/tl-micro-lora-fever-qwen25" \
  "/root/autodl-tmp/runs/verify/tl-micro-lora-fever-qwen25"
sync_first_available "runs/review/tl-bridge-multitask-qwen25" \
  "runs/verify/tl-bridge-multitask-qwen25" \
  "/root/autodl-tmp/runs/verify/tl-bridge-multitask-qwen25"
sync_first_available "runs/review/writer-weaver-qwen25-smoke" \
  "runs/verify/writer-weaver-qwen25-smoke" \
  "/root/autodl-tmp/runs/verify/writer-weaver-qwen25-smoke"
sync_first_available "runs/review/writer-weaver-qwen25-f1a" \
  "runs/verify/writer-weaver-qwen25-f1a" \
  "/root/autodl-tmp/runs/verify/writer-weaver-qwen25-f1a"
sync_dir "/root/autodl-tmp/memtotal-stage-c-qonly-negative-count-sweep-v1" "runs/review/m3-core4-stage-c-qonly-negative-count-sweep-v1"
sync_dir "/root/autodl-tmp/memtotal-stage-c-qonly-retrieval-loss-sweep-v1" "runs/review/m3-core4-stage-c-qonly-retrieval-loss-sweep-v1"
sync_dir "/root/autodl-tmp/memtotal-stage-c-qonly-seed-sweep-v5-margin-canonical" "runs/review/m3-core4-stage-c-qonly-seed-sweep-v5-margin-canonical"
sync_dir "/root/autodl-tmp/memtotal-stage-c-qonly-seed-sweep-v6-case-dump" "runs/review/m3-core4-stage-c-qonly-seed-sweep-v6-case-dump"

sync_dir "results/generated/m3-story-cloze-real-pilot-qwen25" "results/generated/review/m3-story-cloze-real-pilot-qwen25"
sync_dir "results/generated/m3-fever-real-pilot-qwen25" "results/generated/review/m3-fever-real-pilot-qwen25"
sync_dir "results/generated/m4-fever-shared-injection-qwen25" "results/generated/review/m4-fever-shared-injection-qwen25"
sync_dir "results/generated/m4-fever-dynamics-recovery-qwen25" "results/generated/review/m4-fever-dynamics-recovery-qwen25"
sync_dir "results/generated/m4-fever-dynamics-recovery-stabilized-qwen25" "results/generated/review/m4-fever-dynamics-recovery-stabilized-qwen25"
sync_first_available "results/generated/review/m4-fever-deep-prompt-recovery-qwen25" \
  "results/generated/m4-fever-deep-prompt-recovery-qwen25" \
  "/root/autodl-tmp/results/generated/m4-fever-deep-prompt-recovery-qwen25"
sync_first_available "results/generated/review/m4-fever-anti-shortcut-recovery-qwen25" \
  "results/generated/m4-fever-anti-shortcut-recovery-qwen25" \
  "/root/autodl-tmp/results/generated/m4-fever-anti-shortcut-recovery-qwen25"
sync_first_available "results/generated/review/m4-fever-shared-injection-alignment-qwen25" \
  "results/generated/m4-fever-shared-injection-alignment-qwen25" \
  "/root/autodl-tmp/results/generated/m4-fever-shared-injection-alignment-qwen25"
sync_first_available "results/generated/review/m5-fever-writer-reasoner-alignment-qwen25" \
  "results/generated/m5-fever-writer-reasoner-alignment-qwen25" \
  "/root/autodl-tmp/results/generated/m5-fever-writer-reasoner-alignment-qwen25"
sync_first_available "results/generated/review/m5-fever-writer-objective-rewrite-qwen25" \
  "results/generated/m5-fever-writer-objective-rewrite-qwen25" \
  "/root/autodl-tmp/results/generated/m5-fever-writer-objective-rewrite-qwen25"
sync_first_available "results/generated/review/m5-fever-dense-teacher-qwen25" \
  "results/generated/m5-fever-dense-teacher-qwen25" \
  "/root/autodl-tmp/results/generated/m5-fever-dense-teacher-qwen25"
sync_first_available "results/generated/review/tl-poc-fever-qwen25" \
  "results/generated/tl-poc-fever-qwen25" \
  "/root/autodl-tmp/results/generated/tl-poc-fever-qwen25"
sync_first_available "results/generated/review/tl-bridge-rescue-fever-qwen25" \
  "results/generated/tl-bridge-rescue-fever-qwen25" \
  "/root/autodl-tmp/results/generated/tl-bridge-rescue-fever-qwen25"
sync_first_available "results/generated/review/tl-slot-basis-rescue-fever-qwen25" \
  "results/generated/tl-slot-basis-rescue-fever-qwen25" \
  "/root/autodl-tmp/results/generated/tl-slot-basis-rescue-fever-qwen25"
sync_first_available "results/generated/review/tl-reader-geometry-rescue-fever-qwen25" \
  "results/generated/tl-reader-geometry-rescue-fever-qwen25" \
  "/root/autodl-tmp/results/generated/tl-reader-geometry-rescue-fever-qwen25"
sync_first_available "results/generated/review/tl-reader-symmetry-break-fever-qwen25" \
  "results/generated/tl-reader-symmetry-break-fever-qwen25" \
  "/root/autodl-tmp/results/generated/tl-reader-symmetry-break-fever-qwen25"
sync_first_available "results/generated/review/tl-reader-local-bootstrap-fever-qwen25" \
  "results/generated/tl-reader-local-bootstrap-fever-qwen25" \
  "/root/autodl-tmp/results/generated/tl-reader-local-bootstrap-fever-qwen25"
sync_first_available "results/generated/review/tl-writer-value-fever-qwen25" \
  "results/generated/tl-writer-value-fever-qwen25" \
  "/root/autodl-tmp/results/generated/tl-writer-value-fever-qwen25"
sync_first_available "results/generated/review/tl-micro-lora-fever-qwen25" \
  "results/generated/tl-micro-lora-fever-qwen25" \
  "/root/autodl-tmp/results/generated/tl-micro-lora-fever-qwen25"
sync_first_available "results/generated/review/tl-bridge-multitask-qwen25" \
  "results/generated/tl-bridge-multitask-qwen25" \
  "/root/autodl-tmp/results/generated/tl-bridge-multitask-qwen25"
sync_first_available "results/generated/review/writer-weaver-qwen25-smoke" \
  "results/generated/writer-weaver-qwen25-smoke" \
  "/root/autodl-tmp/results/generated/writer-weaver-qwen25-smoke"
sync_first_available "results/generated/review/writer-weaver-qwen25-f1a" \
  "results/generated/writer-weaver-qwen25-f1a" \
  "/root/autodl-tmp/results/generated/writer-weaver-qwen25-f1a"
sync_dir "results/generated/m3-core4-stage-c-qonly-negative-count-sweep-v1" "results/generated/review/m3-core4-stage-c-qonly-negative-count-sweep-v1"
sync_dir "results/generated/m3-core4-stage-c-qonly-retrieval-loss-sweep-v1" "results/generated/review/m3-core4-stage-c-qonly-retrieval-loss-sweep-v1"
sync_dir "results/generated/m3-core4-stage-c-qonly-seed-sweep-v5-margin-canonical" "results/generated/review/m3-core4-stage-c-qonly-seed-sweep-v5-margin-canonical"
sync_dir "results/generated/m3-core4-stage-c-qonly-seed-sweep-v6-case-dump" "results/generated/review/m3-core4-stage-c-qonly-seed-sweep-v6-case-dump"
sync_dir "results/generated/m3-core4-stage-c-error-attribution-v1" "results/generated/review/m3-core4-stage-c-error-attribution-v1"
sync_dir "results/generated/m3-core4-stage-c-margin-audit-v3-fixed-holdout" "results/generated/review/m3-core4-stage-c-margin-audit-v3-fixed-holdout"
sync_dir "results/generated/m3-core4-stage-c-negative-seed-curve-audit-v2-fixed-holdout" "results/generated/review/m3-core4-stage-c-negative-seed-curve-audit-v2-fixed-holdout"
sync_dir "results/generated/m3-core4-stage-c-curve-suite-v3-fixed-holdout" "results/generated/review/m3-core4-stage-c-curve-suite-v3-fixed-holdout"

rebuild_docs_bundle

echo "review artifacts refreshed"
