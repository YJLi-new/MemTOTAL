from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from memtotal.utils.repro import collect_env_info


def ensure_dir(path: str | Path) -> Path:
    output = Path(path).resolve()
    output.mkdir(parents=True, exist_ok=True)
    return output


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True))


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    destination = Path(path)
    destination.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + ("\n" if rows else "")
    )


def snapshot_config(path: str | Path, config: dict[str, Any]) -> None:
    destination = Path(path)
    destination.write_text(yaml.safe_dump(config, sort_keys=False))


def initialize_run_artifacts(
    *,
    output_dir: str | Path,
    config: dict[str, Any],
    seed: int,
    argv: list[str],
) -> dict[str, Any]:
    run_dir = ensure_dir(output_dir)
    env_info = collect_env_info()
    task_cfg = config.get("task", {})
    run_info = {
        "seed": seed,
        "argv": argv,
        "experiment_name": config["experiment"]["name"],
        "stage": config["experiment"]["stage"],
        "method_variant": config["experiment"]["method_variant"],
        "task_name": task_cfg.get("name", ""),
        "benchmark_id": task_cfg.get("benchmark_id"),
        "task_domain": task_cfg.get("domain"),
        "task_split": task_cfg.get("split"),
        "smoke_subset": task_cfg.get("smoke_subset"),
        "backbone": config.get("backbone", {}).get("name", ""),
        **env_info,
    }
    snapshot_config(run_dir / "config.snapshot.yaml", config)
    write_json(run_dir / "run_info.json", run_info)
    return run_info
