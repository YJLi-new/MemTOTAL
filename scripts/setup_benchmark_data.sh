#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCHMARKS="${BENCHMARKS:-gsm8k,math,gpqa,triviaqa,story_cloze,kodcode,rocstories,fever,alfworld,memoryagentbench}"
MAX_EXAMPLES="${MAX_EXAMPLES:-4}"
SEED="${SEED:-701}"

cd "$ROOT_DIR"

python -m memtotal.tasks.setup_data \
  --benchmarks "$BENCHMARKS" \
  --max_examples "$MAX_EXAMPLES" \
  --seed "$SEED" \
  --output_root "$ROOT_DIR/data/benchmarks/materialized" \
  --manifest_root "$ROOT_DIR/data/benchmarks/manifests" \
  --summary_path "$ROOT_DIR/data/benchmarks/source_summary.json"

echo "benchmark-data-materialized $ROOT_DIR/data/benchmarks/materialized"
echo "benchmark-data-manifests $ROOT_DIR/data/benchmarks/manifests"
