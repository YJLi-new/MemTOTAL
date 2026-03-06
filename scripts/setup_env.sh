#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

python -m pip install -e .
python - <<'PY'
import importlib.util
import subprocess
import sys

auto_install = {
    "datasets": "datasets",
    "alfworld": "alfworld",
}
missing_auto = [package for module, package in auto_install.items() if importlib.util.find_spec(module) is None]
if missing_auto:
    subprocess.check_call([sys.executable, "-m", "pip", "install", *missing_auto])

required = ["torch", "yaml", "datasets", "alfworld", "textworld"]
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(f"Missing required modules: {missing}")
print("environment-ok")
PY
