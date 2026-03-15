#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Update M4 dual-gate summary from selection and compare outputs.")
    parser.add_argument("--selection_json", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--screen248_test_metrics")
    parser.add_argument("--fixed64_metrics")
    parser.add_argument("--overwrite-selection", action="store_true")
    return parser


def _load_gate_passed(path: str | None) -> bool:
    if not path:
        return False
    return bool(json.loads(Path(path).read_text()).get("gate_passed", False))


def main() -> int:
    args = build_arg_parser().parse_args()
    selection_path = Path(args.selection_json)
    selection = json.loads(selection_path.read_text())
    summary = {
        "selection_passed": bool(selection.get("selection_passed", False)),
        "screen248_test_gate_passed": _load_gate_passed(args.screen248_test_metrics),
        "fixed64_gate_passed": _load_gate_passed(args.fixed64_metrics),
    }
    summary["milestone_gate_passed"] = bool(
        summary["screen248_test_gate_passed"] and summary["fixed64_gate_passed"]
    )
    Path(args.output_json).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if args.overwrite_selection:
        selection["screen248_test_gate_passed"] = summary["screen248_test_gate_passed"]
        selection["fixed64_gate_passed"] = summary["fixed64_gate_passed"]
        selection["milestone_gate_passed"] = summary["milestone_gate_passed"]
        selection_path.write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
