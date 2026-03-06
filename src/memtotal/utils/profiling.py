from __future__ import annotations

import csv
import resource
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

from memtotal.utils.io import write_json


@dataclass
class ProfileTracker:
    output_dir: Path
    device: str = "cpu"
    event_name: str = "run"
    token_count: int = 0
    example_count: int = 0
    extra_metrics: dict[str, float | int | str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._start_time = time.perf_counter()
        self._cuda_enabled = self.device.startswith("cuda") and torch.cuda.is_available()
        if self._cuda_enabled:
            torch.cuda.reset_peak_memory_stats()

    def add_tokens(self, count: int) -> None:
        self.token_count += int(count)

    def add_example(self, count: int = 1) -> None:
        self.example_count += int(count)

    def set_metric(self, key: str, value: float | int | str) -> None:
        self.extra_metrics[key] = value

    def finalize(self) -> dict[str, float | int | str]:
        wall_time_sec = time.perf_counter() - self._start_time
        peak_device_memory_bytes = (
            int(torch.cuda.max_memory_allocated()) if self._cuda_enabled else 0
        )
        peak_rss_kb = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        payload: dict[str, float | int | str] = {
            "event_name": self.event_name,
            "device": self.device,
            "examples": self.example_count,
            "token_count": self.token_count,
            "wall_time_sec": round(wall_time_sec, 6),
            "tokens_per_sec": round(self.token_count / wall_time_sec, 6) if wall_time_sec else 0.0,
            "peak_device_memory_bytes": peak_device_memory_bytes,
            "peak_device_memory_mib": round(peak_device_memory_bytes / (1024 * 1024), 6),
            "peak_rss_kb": peak_rss_kb,
        }
        payload.update(self.extra_metrics)
        write_json(self.output_dir / "profiling.json", payload)
        self._write_csv(self.output_dir / "profiling.csv", payload)
        return payload

    def _write_csv(self, path: Path, payload: dict[str, float | int | str]) -> None:
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(payload.keys()))
            writer.writeheader()
            writer.writerow(payload)

