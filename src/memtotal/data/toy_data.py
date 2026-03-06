from __future__ import annotations

import json
from pathlib import Path


def load_jsonl_dataset(dataset_path: str | Path) -> list[dict[str, object]]:
    path = Path(dataset_path).resolve()
    rows: list[dict[str, object]] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"Dataset is empty: {path}")
    return rows


def load_toy_dataset(dataset_path: str | Path) -> list[dict[str, object]]:
    return load_jsonl_dataset(dataset_path)
