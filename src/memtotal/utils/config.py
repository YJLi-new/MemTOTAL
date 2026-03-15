from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path).resolve()
    raw = yaml.safe_load(path.read_text()) or {}
    includes = raw.pop("includes", [])
    merged: dict[str, Any] = {}
    for include in includes:
        include_path = (path.parent / include).resolve()
        merged = _deep_merge(merged, load_config(include_path))
    merged = _deep_merge(merged, raw)
    merged["_meta"] = {
        "config_path": str(path),
        "includes": [str((path.parent / item).resolve()) for item in includes],
    }
    return merged

