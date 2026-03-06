import argparse
from datetime import datetime
import getpass
import json
import os
import platform
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
import torch

from common.config import Config
from common.logger import setup_logger
from data import get_data_builder
from memgen.model import MemGenModel
from memgen.runner import MemGenRunner

def _safe_int(value: str | None, default: int = 0) -> int:
    try:
        return int(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def _should_write_launcher() -> bool:
    # Accelerate/torch distributed sets these; write only once per run.
    local_rank = _safe_int(os.environ.get("LOCAL_RANK"), 0)
    rank = _safe_int(os.environ.get("RANK"), 0)
    return local_rank == 0 and rank == 0


def _git_info() -> dict:
    try:
        root = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        status = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        return {
            "root": root,
            "commit": commit,
            "dirty": bool(status),
        }
    except Exception:
        return {}


def _collect_env() -> dict:
    allowlist = [
        # Launcher metadata
        "MEMGEN_LAUNCHER_SCRIPT",
        "MEMGEN_LAUNCHER_CMD",
        "MEMGEN_LAUNCHER_PWD",
        # Common run controls
        "PYTHON_BIN",
        "CUDA_VISIBLE_DEVICES",
        "MAIN_PROCESS_PORT",
        "MASTER_ADDR",
        "MASTER_PORT",
        "WORLD_SIZE",
        "RANK",
        "LOCAL_RANK",
        "NODE_RANK",
        # HF/Transformers caches
        "HF_HOME",
        "HUGGINGFACE_HUB_CACHE",
        "HF_DATASETS_CACHE",
        "TRANSFORMERS_OFFLINE",
        "HF_DATASETS_OFFLINE",
        # Triton cache
        "TRITON_CACHE_DIR",
        # Debug flags commonly used by scripts
        "DEBUG_MODE",
        "LOG_PATH",
        # Common script-level knobs (may be present when exported)
        "DATASET_NAME",
        "LOAD_MODEL_PATH",
        "REASONER_MODEL",
        "WEAVER_MODEL",
        "TRIGGER_MODEL",
        "MAX_PROMPT_AUG_NUM",
        "MAX_INFERENCE_AUG_NUM",
        "PROMPT_LATENTS_LEN",
        "INFERENCE_LATENTS_LEN",
        "TEMPERATURE",
    ]
    out = {k: os.environ.get(k) for k in allowlist if os.environ.get(k) is not None}
    # Also include any MEMGEN_* envs for extensibility.
    for k, v in os.environ.items():
        if k.startswith("MEMGEN_") and k not in out:
            out[k] = v
    return out


def write_launcher_json(working_dir: str, args: argparse.Namespace) -> None:
    if not _should_write_launcher():
        return

    os.makedirs(working_dir, exist_ok=True)
    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "cwd": os.getcwd(),
        "python": {
            "executable": sys.executable,
            "version": sys.version,
        },
        "system": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "user": getpass.getuser(),
            "pid": os.getpid(),
        },
        "git": _git_info(),
        "launcher": {
            "script": os.environ.get("MEMGEN_LAUNCHER_SCRIPT"),
            "cmd": os.environ.get("MEMGEN_LAUNCHER_CMD"),
            "pwd": os.environ.get("MEMGEN_LAUNCHER_PWD"),
        },
        "args": {
            "cfg_path": getattr(args, "cfg_path", None),
            "options": getattr(args, "options", None),
            "argv": sys.argv,
        },
        "env": _collect_env(),
    }

    tmp_path = os.path.join(working_dir, "launcher.json.tmp")
    final_path = os.path.join(working_dir, "launcher.json")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, sort_keys=True)
    os.replace(tmp_path, final_path)

def set_seed(random_seed: int, use_gpu: bool):

    random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    if use_gpu:
        torch.cuda.manual_seed_all(random_seed)

    torch.backends.cudnn.deterministic = True   
    torch.backends.cudnn.benchmark = False      

    print(f"set seed: {random_seed}")

def parse_args():
    parser = argparse.ArgumentParser(description="Memory Generator")

    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )

    args = parser.parse_args()

    return args

def build_working_dir(config: Config) -> str:
    
    # parent dir: <train/evaluate>/<dataset_name>/<reasoner_model_name>
    mode = config.run_cfg.mode
    dataset_name = config.dataset_cfg.name
    model_name = Path(config.model_cfg.model_name).name
    parent_dir = os.path.join("results", mode, dataset_name, model_name)

    # name: <prompt_aug_num>_<prompt_latents_len>_<inference_aug_num>_<inference_latents_len>_<timestamp>
    max_prompt_aug_num = config.model_cfg.max_prompt_aug_num
    prompt_latents_len = config.model_cfg.weaver.prompt_latents_len
    max_inference_aug_num = config.model_cfg.max_inference_aug_num
    inference_latents_len = config.model_cfg.weaver.inference_latents_len
    time = datetime.now().strftime("%Y%m%d-%H%M%S")
    working_dir = f"pn={max_prompt_aug_num}_pl={prompt_latents_len}_in={max_inference_aug_num}_il={inference_latents_len}_{time}" 

    return os.path.join(parent_dir, working_dir)

def main():

    args = parse_args()
    config = Config(args)

    set_seed(config.run_cfg.seed, use_gpu=True)
    
    # set up working directory
    working_dir = build_working_dir(config)
    write_launcher_json(working_dir, args)
    
    # set up logger
    config.run_cfg.log_dir = os.path.join(working_dir, "logs")
    setup_logger(output_dir=config.run_cfg.log_dir)

    config.pretty_print()

    # build components
    config_dict = config.to_dict()
    data_builder = get_data_builder(config_dict.get("dataset"))
    model = MemGenModel.from_config(config_dict.get("model"))
    
    runner = MemGenRunner(
        model=model,
        data_builder=data_builder,
        config=config_dict,
        working_dir=working_dir
    )

    # train or evaluate 
    if config.run_cfg.mode == "train":
        runner.train()
    
    elif config.run_cfg.mode == "evaluate":
        runner.evaluate()

if __name__ == "__main__":
    main()
