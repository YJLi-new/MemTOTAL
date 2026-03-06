#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_ROOT="${1:-$ROOT_DIR/runs/verify/m4-benchmark-smoke/$STAMP}"
RESULT_ROOT="${2:-$ROOT_DIR/results/generated/m4-benchmark-smoke/$STAMP}"
SEED="${SEED:-601}"

CONFIGS=(
  "benchmark_gsm8k_qwen25_smoke.yaml"
  "benchmark_gpqa_qwen25_smoke.yaml"
  "benchmark_kodcode_qwen25_smoke.yaml"
  "benchmark_story_cloze_qwen25_smoke.yaml"
  "benchmark_fever_qwen25_smoke.yaml"
  "benchmark_alfworld_qwen25_smoke.yaml"
)

cd "$ROOT_DIR"

for config_name in "${CONFIGS[@]}"; do
  run_name="${config_name%.yaml}"
  python -m eval \
    --config "configs/exp/$config_name" \
    --seed "$SEED" \
    --output_dir "$RUN_ROOT/$run_name"
done

python -m analysis \
  --config "configs/exp/benchmark_gsm8k_qwen25_smoke.yaml" \
  --seed "$SEED" \
  --output_dir "$RESULT_ROOT" \
  --input_root "$RUN_ROOT"

echo "benchmark-smoke-runs $RUN_ROOT"
echo "benchmark-smoke-summary $RESULT_ROOT"
