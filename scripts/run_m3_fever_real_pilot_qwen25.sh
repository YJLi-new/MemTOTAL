#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

BASE_SEED="${1:-4501}"
RUN_ROOT="${2:-runs/verify/m3-fever-real-pilot-qwen25}"
RESULT_ROOT="${3:-results/generated/m3-fever-real-pilot-qwen25}"
RESUME_STAGE_B_ROOT="${4:-runs/verify/m3-story-cloze-real-pilot-qwen25/stage-b}"

export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf-cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

./scripts/prepare_local_qwen25_model.sh \
  "Qwen/Qwen2.5-1.5B-Instruct" \
  "/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct" \
  "${HF_HOME}"

python -m memtotal.tasks.setup_data \
  --benchmarks fever \
  --max_examples 256 \
  --seed 811 \
  --output_root data/benchmarks/materialized \
  --manifest_root data/benchmarks/manifests \
  --summary_path data/benchmarks/source_summary.json

if [[ ! -f "${RESUME_STAGE_B_ROOT}/queries_meta_init.pt" ]]; then
  ./scripts/10_pretrain_writer.sh \
    "${BASE_SEED}" \
    "${RUN_ROOT}/stage-a" \
    "configs/exp/m3_stage_a_core4_qwen25_real_pilot.yaml"

  ./scripts/20_meta_train_queries.sh \
    "$((BASE_SEED + 2))" \
    "${RUN_ROOT}/stage-b" \
    "${RUN_ROOT}/stage-a" \
    "configs/exp/m3_stage_b_core4_qwen25_real_pilot.yaml"

  RESUME_STAGE_B_ROOT="${RUN_ROOT}/stage-b"
fi

./scripts/run_analysis.sh \
  --config configs/exp/fever_real_pilot_split.yaml \
  --seed "$((BASE_SEED + 4))" \
  --output_dir "${RESULT_ROOT}/split"

./scripts/run_train.sh \
  --config configs/exp/m3_stage_c_real_qwen25_base_fever_screening.yaml \
  --seed "$((BASE_SEED + 6))" \
  --output_dir "${RUN_ROOT}/screen-base"

./scripts/run_train.sh \
  --config configs/exp/m3_stage_c_real_qwen25_shared_fever_screening.yaml \
  --seed "$((BASE_SEED + 8))" \
  --output_dir "${RUN_ROOT}/screen-shared" \
  --resume "${RESUME_STAGE_B_ROOT}"

./scripts/run_analysis.sh \
  --config configs/exp/fever_real_fixed_set_builder.yaml \
  --seed "$((BASE_SEED + 10))" \
  --output_dir "${RESULT_ROOT}/fixed-set" \
  --input_root "${RUN_ROOT}"

./scripts/run_train.sh \
  --config configs/exp/m3_stage_c_real_qwen25_base_fever_fixed64.yaml \
  --seed "$((BASE_SEED + 12))" \
  --output_dir "${RUN_ROOT}/pilot-A-base"

./scripts/run_train.sh \
  --config configs/exp/m3_stage_c_real_qwen25_shared_fever_fixed64.yaml \
  --seed "$((BASE_SEED + 14))" \
  --output_dir "${RUN_ROOT}/pilot-B-shared" \
  --resume "${RESUME_STAGE_B_ROOT}"

./scripts/run_train.sh \
  --config configs/exp/m3_stage_c_real_qwen25_shared_delta_fever_fixed64.yaml \
  --seed "$((BASE_SEED + 16))" \
  --output_dir "${RUN_ROOT}/pilot-F-shared-delta" \
  --resume "${RESUME_STAGE_B_ROOT}"

./scripts/run_train.sh \
  --config configs/exp/m3_stage_c_real_qwen25_shared_delta_shuffled_fever_fixed64.yaml \
  --seed "$((BASE_SEED + 18))" \
  --output_dir "${RUN_ROOT}/pilot-G-shared-delta-shuffled" \
  --resume "${RESUME_STAGE_B_ROOT}"

./scripts/run_analysis.sh \
  --config configs/exp/stage_c_real_pilot_compare_fever_delta.yaml \
  --seed "$((BASE_SEED + 20))" \
  --output_dir "${RESULT_ROOT}/compare" \
  --input_root "${RUN_ROOT}"

mkdir -p runs/review results/generated/review
rsync -a --exclude='*.pt' --exclude='*.ckpt' \
  "${RUN_ROOT}/" "runs/review/m3-fever-real-pilot-qwen25/"
rsync -a --exclude='*.pt' --exclude='*.ckpt' \
  "${RESULT_ROOT}/" "results/generated/review/m3-fever-real-pilot-qwen25/"

./scripts/publish_review_artifacts.sh

echo "fever real qwen25 pilot complete"
