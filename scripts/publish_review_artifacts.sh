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

dir_has_files() {
  local src="$1"
  [[ -d "${src}" ]] || return 1
  find "${src}" -type f -print -quit | grep -q .
}

sync_first_available() {
  local dst="$1"
  shift
  local src
  for src in "$@"; do
    if dir_has_files "${src}"; then
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

clean_review_ephemera() {
  if [[ -d "runs/review" ]]; then
    find "runs/review" -type f \
      \( -name '*.pt' -o -name '*.ckpt' -o -name 'tmux-session.log' \) \
      -delete
  fi
}

clean_review_ephemera

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
sync_first_available "runs/review/writer-weaver-qwen25-f1b" \
  "runs/verify/writer-weaver-qwen25-f1b" \
  "/root/autodl-tmp/runs/verify/writer-weaver-qwen25-f1b"
sync_first_available "runs/review/writer-circuit-opening-qwen25" \
  "runs/verify/writer-circuit-opening-qwen25" \
  "/root/autodl-tmp/runs/verify/writer-circuit-opening-qwen25"
sync_first_available "runs/review/writer-deep-prefix-jointpeft-qwen25" \
  "runs/verify/writer-deep-prefix-jointpeft-qwen25" \
  "/root/autodl-tmp/runs/verify/writer-deep-prefix-jointpeft-qwen25"
sync_first_available "runs/review/planv6-v6-1-clean-baseline-qwen25" \
  "runs/verify/planv6-v6-1-clean-baseline-qwen25" \
  "/root/autodl-tmp/runs/verify/planv6-v6-1-clean-baseline-qwen25"
sync_first_available "runs/review/planv6-v6-2-support-screening-qwen25" \
  "runs/verify/planv6-v6-2-support-screening-qwen25" \
  "/root/autodl-tmp/runs/verify/planv6-v6-2-support-screening-qwen25" \
  "/tmp/memtotal-runs/planv6-v6-2-support-screening-qwen25"
sync_first_available "runs/review/planv6-v6-3-loss-screening-qwen25" \
  "runs/verify/planv6-v6-3-loss-screening-qwen25" \
  "/root/autodl-tmp/runs/verify/planv6-v6-3-loss-screening-qwen25" \
  "/tmp/memtotal-runs/planv6-v6-3-loss-screening-qwen25"
sync_first_available "runs/review/planv6-v6-4-mixed-matrix-qwen25" \
  "runs/verify/planv6-v6-4-mixed-matrix-qwen25" \
  "/root/autodl-tmp/runs/verify/planv6-v6-4-mixed-matrix-qwen25" \
  "/tmp/memtotal-runs/planv6-v6-4-mixed-matrix-qwen25"
sync_first_available "runs/review/planv6-v6-5-recipe-stabilization-qwen25" \
  "runs/verify/planv6-v6-5-recipe-stabilization-qwen25" \
  "/root/autodl-tmp/runs/verify/planv6-v6-5-recipe-stabilization-qwen25" \
  "/tmp/memtotal-runs/planv6-v6-5-recipe-stabilization-qwen25"
sync_first_available "runs/review/planv7-v7-0-metrics-oracle-qwen25" \
  "runs/verify/planv7-v7-0-metrics-oracle-qwen25" \
  "/root/autodl-tmp/runs/verify/planv7-v7-0-metrics-oracle-qwen25" \
  "/tmp/memtotal-runs/planv7-v7-0-metrics-oracle-qwen25"
sync_first_available "runs/review/planv7-lr75e5-v7-0-metrics-oracle-qwen25" \
  "runs/verify/planv7-lr75e5-v7-0-metrics-oracle-qwen25" \
  "/root/autodl-tmp/runs/verify/planv7-lr75e5-v7-0-metrics-oracle-qwen25" \
  "/tmp/memtotal-runs/planv7-lr75e5-v7-0-metrics-oracle-qwen25"
sync_first_available "runs/review/planv7-lr75e5-v7-1-width-depth-scout-qwen25" \
  "runs/verify/planv7-lr75e5-v7-1-width-depth-scout-qwen25" \
  "/root/autodl-tmp/runs/verify/planv7-lr75e5-v7-1-width-depth-scout-qwen25" \
  "/tmp/memtotal-runs/planv7-lr75e5-v7-1-width-depth-scout-qwen25"
sync_first_available "runs/review/planv7-lr75e5-v7-2-direct-bandwidth-qwen25" \
  "runs/verify/planv7-lr75e5-v7-2-direct-bandwidth-qwen25" \
  "/root/autodl-tmp/runs/verify/planv7-lr75e5-v7-2-direct-bandwidth-qwen25" \
  "/tmp/memtotal-runs/planv7-lr75e5-v7-2-direct-bandwidth-qwen25"
sync_first_available "runs/review/planv7-lr75e5-v7-3-bridge-qwen25" \
  "runs/verify/planv7-lr75e5-v7-3-bridge-qwen25" \
  "/root/autodl-tmp/runs/verify/planv7-lr75e5-v7-3-bridge-qwen25" \
  "/tmp/memtotal-runs/planv7-lr75e5-v7-3-bridge-qwen25"
sync_first_available "runs/review/planv7-lr75e5-v7-4-forced-consumption-qwen25" \
  "runs/verify/planv7-lr75e5-v7-4-forced-consumption-qwen25" \
  "/root/autodl-tmp/runs/verify/planv7-lr75e5-v7-4-forced-consumption-qwen25" \
  "/tmp/memtotal-runs/planv7-lr75e5-v7-4-forced-consumption-qwen25"
sync_first_available "runs/review/planv7-lr75e5-v7-5-targeted-aux-revisit-qwen25" \
  "runs/verify/planv7-lr75e5-v7-5-targeted-aux-revisit-qwen25" \
  "/root/autodl-tmp/runs/verify/planv7-lr75e5-v7-5-targeted-aux-revisit-qwen25" \
  "/tmp/memtotal-runs/planv7-lr75e5-v7-5-targeted-aux-revisit-qwen25"
sync_first_available "runs/review/planv7-lr75e5-v7-6-multiseed-confirmation-qwen25" \
  "runs/verify/planv7-lr75e5-v7-6-multiseed-confirmation-qwen25" \
  "/root/autodl-tmp/runs/verify/planv7-lr75e5-v7-6-multiseed-confirmation-qwen25" \
  "/tmp/memtotal-runs/planv7-lr75e5-v7-6-multiseed-confirmation-qwen25"
sync_first_available "runs/review/planv7-v7-1-width-depth-scout-qwen25" \
  "runs/verify/planv7-v7-1-width-depth-scout-qwen25" \
  "/root/autodl-tmp/runs/verify/planv7-v7-1-width-depth-scout-qwen25" \
  "/tmp/memtotal-runs/planv7-v7-1-width-depth-scout-qwen25"
sync_first_available "runs/review/planv7-v7-2-direct-bandwidth-qwen25" \
  "runs/verify/planv7-v7-2-direct-bandwidth-qwen25" \
  "/root/autodl-tmp/runs/verify/planv7-v7-2-direct-bandwidth-qwen25" \
  "/tmp/memtotal-runs/planv7-v7-2-direct-bandwidth-qwen25"
sync_first_available "runs/review/planv7-v7-3-bridge-qwen25" \
  "runs/verify/planv7-v7-3-bridge-qwen25" \
  "/root/autodl-tmp/runs/verify/planv7-v7-3-bridge-qwen25" \
  "/tmp/memtotal-runs/planv7-v7-3-bridge-qwen25"
sync_first_available "runs/review/planv7-v7-4-forced-consumption-qwen25" \
  "runs/verify/planv7-v7-4-forced-consumption-qwen25" \
  "/root/autodl-tmp/runs/verify/planv7-v7-4-forced-consumption-qwen25" \
  "/tmp/memtotal-runs/planv7-v7-4-forced-consumption-qwen25"
sync_first_available "runs/review/planv7-v7-5-targeted-aux-revisit-qwen25" \
  "runs/verify/planv7-v7-5-targeted-aux-revisit-qwen25" \
  "/root/autodl-tmp/runs/verify/planv7-v7-5-targeted-aux-revisit-qwen25" \
  "/tmp/memtotal-runs/planv7-v7-5-targeted-aux-revisit-qwen25"
sync_first_available "runs/review/planv7-v7-6-multiseed-confirmation-qwen25" \
  "runs/verify/planv7-v7-6-multiseed-confirmation-qwen25" \
  "/root/autodl-tmp/runs/verify/planv7-v7-6-multiseed-confirmation-qwen25" \
  "/tmp/memtotal-runs/planv7-v7-6-multiseed-confirmation-qwen25"
sync_first_available "runs/review/planv8-v8-0-qwen3-baselines-oracles" \
  "runs/verify/planv8-v8-0-qwen3-baselines-oracles-r1" \
  "/root/autodl-tmp/runs/verify/planv8-v8-0-qwen3-baselines-oracles-r1" \
  "/tmp/memtotal-runs/planv8-v8-0-qwen3-baselines-oracles-r1" \
  "runs/verify/planv8-v8-0-qwen3-baselines-oracles" \
  "/root/autodl-tmp/runs/verify/planv8-v8-0-qwen3-baselines-oracles" \
  "/tmp/memtotal-runs/planv8-v8-0-qwen3-baselines-oracles"
sync_first_available "runs/review/planv8-v8-0-qwen34-baselines-oracles" \
  "runs/verify/planv8-v8-0-qwen34-baselines-oracles-r1" \
  "/root/autodl-tmp/runs/verify/planv8-v8-0-qwen34-baselines-oracles-r1" \
  "/tmp/memtotal-runs/planv8-v8-0-qwen34-baselines-oracles-r1" \
  "runs/verify/planv8-v8-0-qwen34-baselines-oracles" \
  "/root/autodl-tmp/runs/verify/planv8-v8-0-qwen34-baselines-oracles" \
  "/tmp/memtotal-runs/planv8-v8-0-qwen34-baselines-oracles"
sync_first_available "runs/review/planv8-v8-1-reader-interface-scout" \
  "runs/verify/planv8-v8-1-reader-interface-scout-r1" \
  "/root/autodl-tmp/runs/verify/planv8-v8-1-reader-interface-scout-r1" \
  "/tmp/memtotal-runs/planv8-v8-1-reader-interface-scout-r1" \
  "runs/verify/planv8-v8-1-reader-interface-scout" \
  "/root/autodl-tmp/runs/verify/planv8-v8-1-reader-interface-scout" \
  "/tmp/memtotal-runs/planv8-v8-1-reader-interface-scout"
sync_first_available "runs/review/planv8-v8-1-reader-interface-scout-qwen34" \
  "runs/verify/planv8-v8-1-reader-interface-scout-qwen34" \
  "/root/autodl-tmp/runs/verify/planv8-v8-1-reader-interface-scout-qwen34" \
  "/tmp/memtotal-runs/planv8-v8-1-reader-interface-scout-qwen34"
sync_first_available "runs/review/planv8-v8-2-reader-sweep" \
  "runs/verify/planv8-v8-2-reader-sweep" \
  "/root/autodl-tmp/runs/verify/planv8-v8-2-reader-sweep" \
  "/tmp/memtotal-runs/planv8-v8-2-reader-sweep"
sync_first_available "runs/review/planv8-v8-2-reader-sweep-qwen34" \
  "runs/verify/planv8-v8-2-reader-sweep-qwen34" \
  "/root/autodl-tmp/runs/verify/planv8-v8-2-reader-sweep-qwen34" \
  "/tmp/memtotal-runs/planv8-v8-2-reader-sweep-qwen34"
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
sync_first_available "results/generated/review/writer-weaver-qwen25-f1b" \
  "results/generated/writer-weaver-qwen25-f1b" \
  "/root/autodl-tmp/results/generated/writer-weaver-qwen25-f1b"
sync_first_available "results/generated/review/writer-circuit-opening-qwen25" \
  "results/generated/writer-circuit-opening-qwen25" \
  "/root/autodl-tmp/results/generated/writer-circuit-opening-qwen25"
sync_first_available "results/generated/review/writer-deep-prefix-jointpeft-qwen25" \
  "results/generated/writer-deep-prefix-jointpeft-qwen25" \
  "/root/autodl-tmp/results/generated/writer-deep-prefix-jointpeft-qwen25"
sync_first_available "results/generated/review/planv6-v6-1-clean-baseline-qwen25" \
  "results/generated/planv6-v6-1-clean-baseline-qwen25" \
  "/root/autodl-tmp/results/generated/planv6-v6-1-clean-baseline-qwen25"
sync_first_available "results/generated/review/planv6-v6-2-support-screening-qwen25" \
  "results/generated/planv6-v6-2-support-screening-qwen25" \
  "/root/autodl-tmp/results/generated/planv6-v6-2-support-screening-qwen25" \
  "/tmp/memtotal-results/planv6-v6-2-support-screening-qwen25"
sync_first_available "results/generated/review/planv6-v6-3-loss-screening-qwen25" \
  "results/generated/planv6-v6-3-loss-screening-qwen25" \
  "/root/autodl-tmp/results/generated/planv6-v6-3-loss-screening-qwen25" \
  "/tmp/memtotal-results/planv6-v6-3-loss-screening-qwen25"
sync_first_available "results/generated/review/planv6-v6-4-mixed-matrix-qwen25" \
  "results/generated/planv6-v6-4-mixed-matrix-qwen25" \
  "/root/autodl-tmp/results/generated/planv6-v6-4-mixed-matrix-qwen25" \
  "/tmp/memtotal-results/planv6-v6-4-mixed-matrix-qwen25"
sync_first_available "results/generated/review/planv6-v6-5-recipe-stabilization-qwen25" \
  "results/generated/planv6-v6-5-recipe-stabilization-qwen25" \
  "/root/autodl-tmp/results/generated/planv6-v6-5-recipe-stabilization-qwen25" \
  "/tmp/memtotal-results/planv6-v6-5-recipe-stabilization-qwen25"
sync_first_available "results/generated/review/planv7-v7-0-metrics-oracle-qwen25" \
  "results/generated/planv7-v7-0-metrics-oracle-qwen25" \
  "/root/autodl-tmp/results/generated/planv7-v7-0-metrics-oracle-qwen25" \
  "/tmp/memtotal-results/planv7-v7-0-metrics-oracle-qwen25"
sync_first_available "results/generated/review/planv7-lr75e5-v7-0-metrics-oracle-qwen25" \
  "results/generated/planv7-lr75e5-v7-0-metrics-oracle-qwen25" \
  "/root/autodl-tmp/results/generated/planv7-lr75e5-v7-0-metrics-oracle-qwen25" \
  "/tmp/memtotal-results/planv7-lr75e5-v7-0-metrics-oracle-qwen25"
sync_first_available "results/generated/review/planv7-lr75e5-v7-1-width-depth-scout-qwen25" \
  "results/generated/planv7-lr75e5-v7-1-width-depth-scout-qwen25" \
  "/root/autodl-tmp/results/generated/planv7-lr75e5-v7-1-width-depth-scout-qwen25" \
  "/tmp/memtotal-results/planv7-lr75e5-v7-1-width-depth-scout-qwen25"
sync_first_available "results/generated/review/planv7-lr75e5-v7-2-direct-bandwidth-qwen25" \
  "results/generated/planv7-lr75e5-v7-2-direct-bandwidth-qwen25" \
  "/root/autodl-tmp/results/generated/planv7-lr75e5-v7-2-direct-bandwidth-qwen25" \
  "/tmp/memtotal-results/planv7-lr75e5-v7-2-direct-bandwidth-qwen25"
sync_first_available "results/generated/review/planv7-lr75e5-v7-3-bridge-qwen25" \
  "results/generated/planv7-lr75e5-v7-3-bridge-qwen25" \
  "/root/autodl-tmp/results/generated/planv7-lr75e5-v7-3-bridge-qwen25" \
  "/tmp/memtotal-results/planv7-lr75e5-v7-3-bridge-qwen25"
sync_first_available "results/generated/review/planv7-lr75e5-v7-4-forced-consumption-qwen25" \
  "results/generated/planv7-lr75e5-v7-4-forced-consumption-qwen25" \
  "/root/autodl-tmp/results/generated/planv7-lr75e5-v7-4-forced-consumption-qwen25" \
  "/tmp/memtotal-results/planv7-lr75e5-v7-4-forced-consumption-qwen25"
sync_first_available "results/generated/review/planv7-lr75e5-v7-5-targeted-aux-revisit-qwen25" \
  "results/generated/planv7-lr75e5-v7-5-targeted-aux-revisit-qwen25" \
  "/root/autodl-tmp/results/generated/planv7-lr75e5-v7-5-targeted-aux-revisit-qwen25" \
  "/tmp/memtotal-results/planv7-lr75e5-v7-5-targeted-aux-revisit-qwen25"
sync_first_available "results/generated/review/planv7-lr75e5-v7-6-multiseed-confirmation-qwen25" \
  "results/generated/planv7-lr75e5-v7-6-multiseed-confirmation-qwen25" \
  "/root/autodl-tmp/results/generated/planv7-lr75e5-v7-6-multiseed-confirmation-qwen25" \
  "/tmp/memtotal-results/planv7-lr75e5-v7-6-multiseed-confirmation-qwen25"
sync_first_available "results/generated/review/planv7-v7-1-width-depth-scout-qwen25" \
  "results/generated/planv7-v7-1-width-depth-scout-qwen25" \
  "/root/autodl-tmp/results/generated/planv7-v7-1-width-depth-scout-qwen25" \
  "/tmp/memtotal-results/planv7-v7-1-width-depth-scout-qwen25"
sync_first_available "results/generated/review/planv7-v7-2-direct-bandwidth-qwen25" \
  "results/generated/planv7-v7-2-direct-bandwidth-qwen25" \
  "/root/autodl-tmp/results/generated/planv7-v7-2-direct-bandwidth-qwen25" \
  "/tmp/memtotal-results/planv7-v7-2-direct-bandwidth-qwen25"
sync_first_available "results/generated/review/planv7-v7-3-bridge-qwen25" \
  "results/generated/planv7-v7-3-bridge-qwen25" \
  "/root/autodl-tmp/results/generated/planv7-v7-3-bridge-qwen25" \
  "/tmp/memtotal-results/planv7-v7-3-bridge-qwen25"
sync_first_available "results/generated/review/planv7-v7-4-forced-consumption-qwen25" \
  "results/generated/planv7-v7-4-forced-consumption-qwen25" \
  "/root/autodl-tmp/results/generated/planv7-v7-4-forced-consumption-qwen25" \
  "/tmp/memtotal-results/planv7-v7-4-forced-consumption-qwen25"
sync_first_available "results/generated/review/planv7-v7-5-targeted-aux-revisit-qwen25" \
  "results/generated/planv7-v7-5-targeted-aux-revisit-qwen25" \
  "/root/autodl-tmp/results/generated/planv7-v7-5-targeted-aux-revisit-qwen25" \
  "/tmp/memtotal-results/planv7-v7-5-targeted-aux-revisit-qwen25"
sync_first_available "results/generated/review/planv7-v7-6-multiseed-confirmation-qwen25" \
  "results/generated/planv7-v7-6-multiseed-confirmation-qwen25" \
  "/root/autodl-tmp/results/generated/planv7-v7-6-multiseed-confirmation-qwen25" \
  "/tmp/memtotal-results/planv7-v7-6-multiseed-confirmation-qwen25"
sync_first_available "results/generated/review/planv8-v8-0-qwen3-baselines-oracles" \
  "results/generated/planv8-v8-0-qwen3-baselines-oracles-r1" \
  "/root/autodl-tmp/results/generated/planv8-v8-0-qwen3-baselines-oracles-r1" \
  "/tmp/memtotal-results/planv8-v8-0-qwen3-baselines-oracles-r1" \
  "results/generated/planv8-v8-0-qwen3-baselines-oracles" \
  "/root/autodl-tmp/results/generated/planv8-v8-0-qwen3-baselines-oracles" \
  "/tmp/memtotal-results/planv8-v8-0-qwen3-baselines-oracles"
sync_first_available "results/generated/review/planv8-v8-0-qwen34-baselines-oracles" \
  "results/generated/planv8-v8-0-qwen34-baselines-oracles-r1" \
  "/root/autodl-tmp/results/generated/planv8-v8-0-qwen34-baselines-oracles-r1" \
  "/tmp/memtotal-results/planv8-v8-0-qwen34-baselines-oracles-r1" \
  "results/generated/planv8-v8-0-qwen34-baselines-oracles" \
  "/root/autodl-tmp/results/generated/planv8-v8-0-qwen34-baselines-oracles" \
  "/tmp/memtotal-results/planv8-v8-0-qwen34-baselines-oracles"
sync_first_available "results/generated/review/planv8-v8-1-reader-interface-scout" \
  "results/generated/planv8-v8-1-reader-interface-scout-r1" \
  "/root/autodl-tmp/results/generated/planv8-v8-1-reader-interface-scout-r1" \
  "/tmp/memtotal-results/planv8-v8-1-reader-interface-scout-r1" \
  "results/generated/planv8-v8-1-reader-interface-scout" \
  "/root/autodl-tmp/results/generated/planv8-v8-1-reader-interface-scout" \
  "/tmp/memtotal-results/planv8-v8-1-reader-interface-scout"
sync_first_available "results/generated/review/planv8-v8-1-reader-interface-scout-qwen34" \
  "results/generated/planv8-v8-1-reader-interface-scout-qwen34" \
  "/root/autodl-tmp/results/generated/planv8-v8-1-reader-interface-scout-qwen34" \
  "/tmp/memtotal-results/planv8-v8-1-reader-interface-scout-qwen34"
sync_first_available "results/generated/review/planv8-v8-2-reader-sweep" \
  "results/generated/planv8-v8-2-reader-sweep" \
  "/root/autodl-tmp/results/generated/planv8-v8-2-reader-sweep" \
  "/tmp/memtotal-results/planv8-v8-2-reader-sweep"
sync_first_available "results/generated/review/planv8-v8-2-reader-sweep-qwen34" \
  "results/generated/planv8-v8-2-reader-sweep-qwen34" \
  "/root/autodl-tmp/results/generated/planv8-v8-2-reader-sweep-qwen34" \
  "/tmp/memtotal-results/planv8-v8-2-reader-sweep-qwen34"
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
