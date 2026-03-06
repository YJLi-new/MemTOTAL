from __future__ import annotations

import json
from pathlib import Path


def load_toy_dataset(dataset_path: str | Path) -> list[dict[str, str]]:
    path = Path(dataset_path).resolve()
    rows: list[dict[str, str]] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"Dataset is empty: {path}")
    return rows

