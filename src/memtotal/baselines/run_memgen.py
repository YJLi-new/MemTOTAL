from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from huggingface_hub import HfFolder

from memtotal.utils.config import load_config
from memtotal.utils.io import initialize_run_artifacts, write_json, write_jsonl
from memtotal.utils.repro import set_seed, validate_backbone_name


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MemTOTAL MemGen baseline adapter.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _resolve_load_model_path(config: dict) -> Path | None:
    baseline_cfg = config["baseline"]
    raw_path = baseline_cfg.get("load_model_path")
    if not raw_path:
        return None
    path = Path(raw_path)
    if not path.is_absolute():
        path = Path(baseline_cfg["repo_root"]).resolve() / path
    return path.resolve()


def _build_memgen_options(config: dict, seed: int) -> list[str]:
    baseline_cfg = config["baseline"]
    backbone_cfg = config["backbone"]
    options = [
        f"run.seed={seed}",
        f"run.mode={baseline_cfg['memgen_run_mode']}",
        f"dataset.name={baseline_cfg['task_name']}",
        f"model.model_name={backbone_cfg['model_id']}",
        f"model.weaver.model_name={backbone_cfg['model_id']}",
        f"model.trigger.model_name={backbone_cfg['model_id']}",
    ]
    load_model_path = _resolve_load_model_path(config)
    if load_model_path is not None:
        options.append(f"model.load_model_path={load_model_path}")
    trigger_active = baseline_cfg.get("trigger_active")
    if trigger_active is not None:
        options.append(f"model.trigger.active={str(trigger_active)}")
    if baseline_cfg.get("max_prompt_aug_num") is not None:
        options.append(f"model.max_prompt_aug_num={baseline_cfg['max_prompt_aug_num']}")
    if baseline_cfg.get("max_inference_aug_num") is not None:
        options.append(f"model.max_inference_aug_num={baseline_cfg['max_inference_aug_num']}")
    options.extend(baseline_cfg.get("extra_options", []))
    return options


def _build_command(config: dict, seed: int) -> list[str]:
    baseline_cfg = config["baseline"]
    repo_root = Path(baseline_cfg["repo_root"]).resolve()
    cfg_path = repo_root / baseline_cfg["memgen_config_path"]
    command = [
        sys.executable,
        "main.py",
        "--cfg-path",
        str(cfg_path),
        "--options",
        *_build_memgen_options(config, seed),
    ]
    return command


def _has_hf_auth_token() -> bool:
    if HfFolder.get_token():
        return True
    return any(
        bool(os.environ.get(name))
        for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_HOME_TOKEN")
    )


def _memgen_runtime_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    return env


def _baseline_metadata(config: dict) -> dict[str, object]:
    baseline_cfg = config["baseline"]
    load_model_path = _resolve_load_model_path(config)
    return {
        "trigger_active": baseline_cfg.get("trigger_active", False),
        "max_prompt_aug_num": baseline_cfg.get("max_prompt_aug_num"),
        "max_inference_aug_num": baseline_cfg.get("max_inference_aug_num"),
        "requires_trained_checkpoint": baseline_cfg.get("requires_trained_checkpoint", False),
        "insertion_profile": baseline_cfg.get("insertion_profile", "default"),
        "load_model_path": str(load_model_path) if load_model_path is not None else None,
    }


def _preflight_failure(config: dict) -> str | None:
    baseline_cfg = config["baseline"]
    load_model_path = _resolve_load_model_path(config)
    if baseline_cfg["task_name"] == "gpqa" and not _has_hf_auth_token():
        return (
            "GPQA requires Hugging Face authentication for gated dataset "
            "`Idavidrein/gpqa`. Run `huggingface-cli login` or export `HF_TOKEN` "
            "before launching this config."
        )
    if baseline_cfg.get("requires_trained_checkpoint", False) and load_model_path is None:
        return (
            "This MemGen config requires a trained checkpoint, but `load_model_path` is empty. "
            "Set `baseline.load_model_path` to a valid weaver/trigger checkpoint directory."
        )
    if load_model_path is not None and not load_model_path.exists():
        return f"Configured load_model_path does not exist: {load_model_path}"
    return None


def _write_launch_files(output_dir: Path, command: list[str], config: dict) -> None:
    shell_command = " ".join(subprocess.list2cmdline([part]) for part in command)
    (output_dir / "memgen_launch.sh").write_text(f"#!/usr/bin/env bash\nset -euo pipefail\n{shell_command}\n")
    write_json(
        output_dir / "memgen_launch.json",
        {
            "command": command,
            "repo_root": str(Path(config["baseline"]["repo_root"]).resolve()),
            "memgen_config_path": str(
                (Path(config["baseline"]["repo_root"]).resolve() / config["baseline"]["memgen_config_path"]).resolve()
            ),
            "task_name": config["baseline"]["task_name"],
            "backbone": config["backbone"]["name"],
            **_baseline_metadata(config),
        },
    )


def _expected_memgen_results_root(config: dict) -> Path:
    baseline_cfg = config["baseline"]
    model_slug = Path(config["backbone"]["model_id"]).name
    return (
        Path(baseline_cfg["repo_root"]).resolve()
        / "results"
        / baseline_cfg["memgen_run_mode"]
        / baseline_cfg["task_name"]
        / model_slug
    )


def _snapshot_working_dirs(results_root: Path) -> set[Path]:
    if not results_root.exists():
        return set()
    return {path for path in results_root.iterdir() if path.is_dir()}


def _resolve_new_working_dir(results_root: Path, before: set[Path]) -> Path | None:
    if not results_root.exists():
        return None
    after = {path for path in results_root.iterdir() if path.is_dir()}
    new_dirs = sorted(after - before, key=lambda item: item.stat().st_mtime)
    if new_dirs:
        return new_dirs[-1]
    existing_dirs = sorted(after, key=lambda item: item.stat().st_mtime)
    return existing_dirs[-1] if existing_dirs else None


def _translate_answer_json(answer_path: Path, output_dir: Path) -> dict[str, float | int]:
    records = []
    summary_metrics: dict[str, float | int] = {}
    for line in answer_path.read_text().splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if "summary_metrics" in payload:
            summary_metrics = payload["summary_metrics"]
            continue
        records.append(
            {
                "prompt": payload.get("prompt"),
                "solution": payload.get("solution"),
                "completion": payload.get("completion"),
                **{f"metric_{key}": value for key, value in payload.get("metrics", {}).items()},
            }
        )
    if records:
        write_jsonl(output_dir / "predictions.jsonl", records)
    summary_metrics["num_predictions"] = len(records)
    return summary_metrics


def _translate_conversations_txt(conversations_path: Path, output_dir: Path) -> dict[str, float | int]:
    text = conversations_path.read_text()
    pattern = re.compile(
        r"Conversation:\n(.*?)\nReward:\s*([0-9]+(?:\.[0-9]+)?)\n-+",
        flags=re.DOTALL,
    )
    records = []
    rewards = []
    for conversation, reward_text in pattern.findall(text):
        reward = float(reward_text)
        rewards.append(reward)
        records.append(
            {
                "conversation": conversation.strip(),
                "metric_compute_reward": reward,
            }
        )
    if records:
        write_jsonl(output_dir / "predictions.jsonl", records)
    compute_reward = sum(rewards) / len(rewards) if rewards else 0.0
    return {
        "compute_reward": compute_reward,
        "num_predictions": len(records),
    }


def _copy_raw_memgen_artifacts(working_dir: Path, output_dir: Path) -> dict[str, str]:
    raw_dir = output_dir / "memgen_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    copied: dict[str, str] = {}
    candidates = [
        working_dir / "launcher.json",
        working_dir / "logs" / "log.txt",
        working_dir / "evaluate" / "answer.json",
        working_dir / "evaluate" / "conversations.txt",
    ]
    for source in candidates:
        if not source.exists():
            continue
        destination = raw_dir / source.name
        shutil.copy2(source, destination)
        copied[source.name] = str(destination)
    return copied


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = load_config(args.config)
    validate_backbone_name(config["backbone"]["name"])
    set_seed(args.seed)
    output_dir = Path(args.output_dir).resolve()
    initialize_run_artifacts(
        output_dir=output_dir,
        config=config,
        seed=args.seed,
        argv=sys.argv if argv is None else ["memgen", *argv],
    )
    command = _build_command(config, args.seed)
    _write_launch_files(output_dir, command, config)
    results_root = _expected_memgen_results_root(config)
    before_dirs = _snapshot_working_dirs(results_root)
    metrics = {
        "mode": "memgen_adapter",
        "dry_run": args.dry_run,
        "task_name": config["baseline"]["task_name"],
        "backbone": config["backbone"]["name"],
        "memgen_results_root": str(results_root),
        **_baseline_metadata(config),
    }

    if args.dry_run:
        write_json(output_dir / "metrics.json", metrics)
        return 0

    preflight_error = _preflight_failure(config)
    if preflight_error is not None:
        metrics["returncode"] = 2
        metrics["preflight_error"] = preflight_error
        write_json(
            output_dir / "memgen_process.json",
            {
                "returncode": 2,
                "stderr_tail": preflight_error,
                "stdout_tail": "",
                "wall_time_sec": 0.0,
            },
        )
        write_json(output_dir / "metrics.json", metrics)
        return 2

    start_time = time.perf_counter()
    result = subprocess.run(
        command,
        cwd=Path(config["baseline"]["repo_root"]).resolve(),
        check=False,
        text=True,
        capture_output=True,
        env=_memgen_runtime_env(),
    )
    wall_time_sec = round(time.perf_counter() - start_time, 6)
    write_json(
        output_dir / "memgen_process.json",
        {
            "returncode": result.returncode,
            "wall_time_sec": wall_time_sec,
            "stdout_tail": result.stdout[-4000:],
            "stderr_tail": result.stderr[-4000:],
        },
    )
    metrics["returncode"] = result.returncode
    metrics["wall_time_sec"] = wall_time_sec
    working_dir = _resolve_new_working_dir(results_root, before_dirs)
    if working_dir is not None:
        metrics["memgen_working_dir"] = str(working_dir)
        copied = _copy_raw_memgen_artifacts(working_dir, output_dir)
        if copied:
            write_json(output_dir / "memgen_artifacts.json", copied)
        answer_path = working_dir / "evaluate" / "answer.json"
        conversations_path = working_dir / "evaluate" / "conversations.txt"
        if answer_path.exists():
            metrics.update(_translate_answer_json(answer_path, output_dir))
        elif conversations_path.exists():
            metrics.update(_translate_conversations_txt(conversations_path, output_dir))
    write_json(output_dir / "metrics.json", metrics)
    return 0 if result.returncode == 0 else result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
