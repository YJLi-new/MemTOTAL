#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <run_root_or_stage_dir>" >&2
  exit 1
fi

TARGET="$1"

python - <<'PY' "$TARGET"
import json
import sys
from pathlib import Path

target = Path(sys.argv[1]).resolve()
profiling_files = []
if target.is_file() and target.name == "profiling.json":
    profiling_files = [target]
elif target.is_dir():
    profiling_files = sorted(target.rglob("profiling.json"))

if not profiling_files:
    raise SystemExit(f"no profiling.json files found under {target}")

for path in profiling_files:
    payload = json.loads(path.read_text())
    print(path)
    for key in [
        "event_name",
        "device",
        "examples",
        "token_count",
        "wall_time_sec",
        "tokens_per_sec",
        "peak_device_memory_mib",
        "peak_rss_kb",
    ]:
        print(f"  {key}: {payload.get(key)}")
PY

