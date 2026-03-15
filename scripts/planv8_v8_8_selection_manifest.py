#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text())


def build_selection_manifest(
    *,
    v83_summary_path: Path,
    v83_run_root: Path,
    v87_summary_path: Path,
    output_path: Path,
    v85_summary_path: Path | None = None,
    v85_run_root: Path | None = None,
    v86_summary_path: Path | None = None,
    v86_run_root: Path | None = None,
) -> dict[str, Any]:
    v83_summary = _load_json(v83_summary_path)
    v87_summary = _load_json(v87_summary_path)
    if v83_summary is None or v87_summary is None:
        raise ValueError("V8-3 and V8-7 summaries are required to build the V8-8 manifest.")
    if str(v87_summary.get("recommended_next_step", "")).strip() != "open_v8_8_multiseed_confirmation":
        raise ValueError("V8-7 summary does not authorize V8-8.")

    v85_summary = _load_json(v85_summary_path)
    v86_summary = _load_json(v86_summary_path)

    variants: list[dict[str, Any]] = []
    best_reader_arm = str(v83_summary.get("base_for_v8_4_arm_id") or v83_summary.get("best_arm_id") or "").strip()
    if best_reader_arm:
        variants.append(
            {
                "variant_id": "c1_reader_opd",
                "source_phase": "V8-3",
                "source_run_root": str(v83_run_root.resolve()),
                "arm_id": best_reader_arm,
                "interface_family": str(v83_summary.get("selected_interface_family_for_v8_4", "")).strip(),
                "bridge_family": "BR0",
                "auxiliary_family": str(v83_summary.get("best_alignment_aux_mode", "")).strip() or "reader_opd",
            }
        )

    best_current_arm = ""
    if v86_summary is not None and v86_run_root is not None:
        best_current_arm = str(v86_summary.get("base_for_v8_7_arm_id") or v86_summary.get("best_arm_id") or "").strip()
        if best_current_arm:
            variants.append(
                {
                    "variant_id": "c2_best_writer_route",
                    "source_phase": "V8-6",
                    "source_run_root": str(v86_run_root.resolve()),
                    "arm_id": best_current_arm,
                    "interface_family": str(v86_summary.get("selected_interface_family_for_v8_7", "")).strip(),
                    "bridge_family": str(v86_summary.get("selected_bridge_family_for_v8_7", "")).strip(),
                    "auxiliary_family": str(v86_summary.get("selected_aux_family_for_v8_7", "")).strip(),
                }
            )

    if v85_summary is not None and v85_run_root is not None and v86_summary is not None:
        bridge_best_arm = str(v85_summary.get("best_arm_id", "")).strip()
        bridge_is_distinct = str(v86_summary.get("best_arm_id", "")).strip() != "a0_none"
        if (
            bridge_best_arm
            and bridge_best_arm != "b0_no_bridge"
            and bool(v85_summary.get("best_arm_acceptance_qualified", False))
            and bridge_is_distinct
        ):
            variants.append(
                {
                    "variant_id": "c3_bridge_route",
                    "source_phase": "V8-5",
                    "source_run_root": str(v85_run_root.resolve()),
                    "arm_id": bridge_best_arm,
                    "interface_family": str(v85_summary.get("selected_interface_family_for_v8_6", "")).strip(),
                    "bridge_family": str(v85_summary.get("selected_bridge_family_for_v8_6", "")).strip(),
                    "auxiliary_family": "bridge_compressed",
                }
            )

    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for row in variants:
        key = (str(row.get("source_phase", "")).strip(), str(row.get("arm_id", "")).strip())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)

    if not deduped:
        raise ValueError("No V8-8 confirmation candidates were available.")

    payload = {
        "phase": "V8-8",
        "seeds": [61109, 61110, 61111],
        "promoted_variants": deduped,
        "v87_comparison_conclusion": str(v87_summary.get("comparison_conclusion", "")).strip(),
        "v87_recommended_next_step": str(v87_summary.get("recommended_next_step", "")).strip(),
        "base_for_v8_8_arm_id": str(v87_summary.get("base_for_v8_8_arm_id", "")).strip(),
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the PLANv8 V8-8 selection manifest.")
    parser.add_argument("--v83_summary", required=True)
    parser.add_argument("--v83_run_root", required=True)
    parser.add_argument("--v87_summary", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--v85_summary", default=None)
    parser.add_argument("--v85_run_root", default=None)
    parser.add_argument("--v86_summary", default=None)
    parser.add_argument("--v86_run_root", default=None)
    args = parser.parse_args()

    build_selection_manifest(
        v83_summary_path=Path(args.v83_summary),
        v83_run_root=Path(args.v83_run_root),
        v87_summary_path=Path(args.v87_summary),
        output_path=Path(args.output_json),
        v85_summary_path=Path(args.v85_summary) if args.v85_summary else None,
        v85_run_root=Path(args.v85_run_root) if args.v85_run_root else None,
        v86_summary_path=Path(args.v86_summary) if args.v86_summary else None,
        v86_run_root=Path(args.v86_run_root) if args.v86_run_root else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
