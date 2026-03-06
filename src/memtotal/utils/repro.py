from __future__ import annotations

import platform
import random
import subprocess
import sys
from datetime import datetime, timezone

import torch
import yaml


SUPPORTED_BACKBONES = (
    "Qwen2.5-1.5B-Instruct",
    "Qwen3-8B",
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def resolve_git_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return "nogit"


def collect_env_info() -> dict[str, str]:
    env_info = {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "pyyaml_version": yaml.__version__,
        "git_hash": resolve_git_hash(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    env_info.update(collect_gpu_info())
    return env_info


def collect_gpu_info() -> dict[str, str]:
    try:
        gpu_name = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        driver_version = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()[0].strip()
        memory_total = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()[0].strip()
        return {
            "gpu_name": gpu_name,
            "nvidia_driver_version": driver_version,
            "gpu_memory_total": memory_total,
        }
    except Exception:
        return {
            "gpu_name": "none",
            "nvidia_driver_version": "unknown",
            "gpu_memory_total": "unknown",
        }


def validate_backbone_name(name: str) -> None:
    if name not in SUPPORTED_BACKBONES:
        raise ValueError(
            f"Unsupported backbone '{name}'. Supported backbones: {SUPPORTED_BACKBONES}"
        )
