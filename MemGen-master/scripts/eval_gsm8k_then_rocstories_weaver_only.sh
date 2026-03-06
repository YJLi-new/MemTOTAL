#!/bin/bash

set -euo pipefail

# Evaluate GSM8K with Weaver only (Trigger disabled) using the sequentially trained Weaver (GSM8K -> RocStories).
# Uses only GPU 4.

# --- tmux bootstrap for one-click runs ---
if [ -z "${RUN_UNDER_TMUX:-}" ]; then
  if ! command -v tmux >/dev/null 2>&1; then
    echo "tmux is required for this script. Please install tmux." >&2
    exit 1
  fi

  script_path="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
  session_name="memgen-$(basename "$0" .sh)-$(date +%Y%m%d%H%M%S)"
  args_str=""
  for arg in "$@"; do
    escaped_arg=$(printf '%s' "$arg" | sed 's/"/\\"/g')
    args_str="${args_str} \"${escaped_arg}\""
  done

  tmux new-session -d -s "${session_name}" "RUN_UNDER_TMUX=1 bash \"${script_path}\"${args_str}; exit_code=\$?; echo \"Process exited with code \$exit_code\"; echo \"Keeping tmux session alive for 15 minutes...\"; sleep 900; exit \$exit_code"
  tmux attach -t "${session_name}"
  exit 0
fi

# --- conda env activation ---
activate_conda() {
  if [ -f "${HOME}/.miniconda/etc/profile.d/conda.sh" ]; then
    # shellcheck source=/dev/null
    source "${HOME}/.miniconda/etc/profile.d/conda.sh"
  elif command -v conda >/dev/null 2>&1; then
    # shellcheck source=/dev/null
    source "$(conda info --base)/etc/profile.d/conda.sh"
  else
    echo "Conda not found. Please install Miniconda/Conda first." >&2
    exit 1
  fi

  conda activate memgen || { echo "Failed to activate conda env 'memgen'." >&2; exit 1; }
}

activate_conda

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
cmd="$(printf '%q ' "${SCRIPT_PATH}" "$@")"
cmd="${cmd% }"
export MEMGEN_LAUNCHER_SCRIPT="${SCRIPT_PATH}"
export MEMGEN_LAUNCHER_CMD="${cmd}"
export MEMGEN_LAUNCHER_PWD="$(pwd)"
cd "${PROJECT_ROOT}"

export DEBUG_MODE=true
export LOG_PATH="./debug_log_eval_gsm8k_then_rocstories_weaver_only.txt"
export HF_DATASETS_CACHE="$(pwd)/data_chache/hf_datasets"
export CUDA_VISIBLE_DEVICES=4
export MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29508}"
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_ASYNC_DISABLE=1
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$HOME/.triton}"
mkdir -p "${TRITON_CACHE_DIR}"
# Offline-friendly HF caches
export HF_HOME="${HF_HOME:-$(pwd)/data_chache/hf_home}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$(pwd)/data_chache/models}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Python executable not found: $PYTHON_BIN" >&2
    exit 1
fi

# Ensure DeepSpeed is available before launch.
if ! "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import deepspeed
PY
then
    echo "DeepSpeed is not installed in the current environment. Please run:" >&2
    echo "  conda activate memgen && pip install deepspeed" >&2
    exit 1
fi

# Ensure tensorboard is available (needed by torch.utils.tensorboard).
if ! "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import tensorboard
PY
then
    echo "tensorboard is not installed in the current environment. Please run:" >&2
    echo "  conda activate memgen && pip install tensorboard" >&2
    exit 1
fi

# Prefer local offline model if present.
LOCAL_QWEN="$(pwd)/data_chache/models/Qwen2.5-1.5B-Instruct"
DEFAULT_MODEL="${LOCAL_QWEN}"

REASONER_MODEL="${REASONER_MODEL:-${DEFAULT_MODEL}}"
WEAVER_MODEL="${WEAVER_MODEL:-${DEFAULT_MODEL}}"
TRIGGER_MODEL="${TRIGGER_MODEL:-${DEFAULT_MODEL}}"
TRIGGER_ACTIVE=False

# Evaluate on GSM8K only (for sequential training comparison)
DATASET_NAME="gsm8k"

# Augmentation configs (match weaver training defaults for GSM8K)
: "${MAX_PROMPT_AUG_NUM:=1}"
: "${MAX_INFERENCE_AUG_NUM:=5}"

PROMPT_LATENTS_LEN=${PROMPT_LATENTS_LEN:-16}
INFERENCE_LATENTS_LEN=${INFERENCE_LATENTS_LEN:-8}
TEMPERATURE=${TEMPERATURE:-1.0}

# Sequentially trained Weaver checkpoint (output of weaver_train_gsm8k_then_rocstories.sh)
DEFAULT_LOAD_MODEL_PATH="$(pwd)/results/train/rocstories/ssd/pn=1_pl=16_in=5_il=8_20251212-152134/weaver"
LOAD_MODEL_PATH="${LOAD_MODEL_PATH:-${DEFAULT_LOAD_MODEL_PATH}}"
if [ ! -e "${LOAD_MODEL_PATH}" ]; then
  echo "LOAD_MODEL_PATH not found: ${LOAD_MODEL_PATH}" >&2
  echo "Set LOAD_MODEL_PATH to the sequential weaver checkpoint directory on your server." >&2
  exit 1
fi

run_eval() {
    "$PYTHON_BIN" -m accelerate.commands.launch \
        --config_file=configs/zero2.yaml \
        main.py \
        --cfg-path configs/latent_memory/${DATASET_NAME}.yaml \
        --options \
        model.model_name ${REASONER_MODEL} \
        model.load_model_path ${LOAD_MODEL_PATH} \
        model.max_prompt_aug_num ${MAX_PROMPT_AUG_NUM} \
        model.max_inference_aug_num ${MAX_INFERENCE_AUG_NUM} \
        model.weaver.model_name ${WEAVER_MODEL} \
        model.weaver.prompt_latents_len ${PROMPT_LATENTS_LEN} \
        model.weaver.inference_latents_len ${INFERENCE_LATENTS_LEN} \
        model.trigger.model_name ${TRIGGER_MODEL} \
        model.trigger.active ${TRIGGER_ACTIVE} \
        run.mode evaluate \
        run.interaction.batch_size 4 \
        run.interaction.do_sample False \
        run.interaction.temperature ${TEMPERATURE} \
        run.interaction.max_response_length 1024
}

run_eval "$@"
