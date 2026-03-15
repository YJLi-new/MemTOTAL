from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.planv8_v8_8_selection_manifest import build_selection_manifest


class PlanV8V88SelectionManifestTest(unittest.TestCase):
    def _write_json(self, path: Path, payload: dict[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    def test_build_selection_manifest_supports_reader_only_route(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            v83_summary = root / "v8-3-summary.json"
            v87_summary = root / "v8-7-summary.json"
            output_json = root / "selection-manifest.json"
            v83_run_root = root / "v83-run"

            self._write_json(
                v83_summary,
                {
                    "base_for_v8_4_arm_id": "p5_opd_ansplusctx_centered",
                    "best_arm_id": "p5_opd_ansplusctx_centered",
                    "selected_interface_family_for_v8_4": "ri0_legacy_prefix",
                    "best_alignment_aux_mode": "opd_token_ce_centered",
                },
            )
            self._write_json(
                v87_summary,
                {
                    "comparison_conclusion": "comparators_support_open_v8_8_multiseed_confirmation",
                    "recommended_next_step": "open_v8_8_multiseed_confirmation",
                    "base_for_v8_8_arm_id": "p5_opd_ansplusctx_centered",
                },
            )

            payload = build_selection_manifest(
                v83_summary_path=v83_summary,
                v83_run_root=v83_run_root,
                v87_summary_path=v87_summary,
                output_path=output_json,
            )

            self.assertEqual(payload["base_for_v8_8_arm_id"], "p5_opd_ansplusctx_centered")
            self.assertEqual(len(payload["promoted_variants"]), 1)
            self.assertEqual(payload["promoted_variants"][0]["variant_id"], "c1_reader_opd")
            self.assertEqual(payload["promoted_variants"][0]["source_phase"], "V8-3")

    def test_build_selection_manifest_includes_writer_and_bridge_routes_when_available(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            v83_summary = root / "v8-3-summary.json"
            v85_summary = root / "v8-5-summary.json"
            v86_summary = root / "v8-6-summary.json"
            v87_summary = root / "v8-7-summary.json"
            output_json = root / "selection-manifest.json"

            self._write_json(
                v83_summary,
                {
                    "base_for_v8_4_arm_id": "p5_opd_ansplusctx_centered",
                    "selected_interface_family_for_v8_4": "ri0_legacy_prefix",
                    "best_alignment_aux_mode": "opd_token_ce_centered",
                },
            )
            self._write_json(
                v85_summary,
                {
                    "best_arm_id": "b2_bridge_compressed",
                    "best_arm_acceptance_qualified": True,
                    "selected_interface_family_for_v8_6": "ri2_cross_attn",
                    "selected_bridge_family_for_v8_6": "BR2",
                },
            )
            self._write_json(
                v86_summary,
                {
                    "best_arm_id": "a4_writer_opd_ansctx",
                    "base_for_v8_7_arm_id": "a4_writer_opd_ansctx",
                    "selected_interface_family_for_v8_7": "ri2_cross_attn",
                    "selected_bridge_family_for_v8_7": "BR2",
                    "selected_aux_family_for_v8_7": "writer_opd_answer_plus_context",
                },
            )
            self._write_json(
                v87_summary,
                {
                    "comparison_conclusion": "comparators_support_open_v8_8_multiseed_confirmation",
                    "recommended_next_step": "open_v8_8_multiseed_confirmation",
                    "base_for_v8_8_arm_id": "a4_writer_opd_ansctx",
                },
            )

            payload = build_selection_manifest(
                v83_summary_path=v83_summary,
                v83_run_root=root / "v83-run",
                v87_summary_path=v87_summary,
                output_path=output_json,
                v85_summary_path=v85_summary,
                v85_run_root=root / "v85-run",
                v86_summary_path=v86_summary,
                v86_run_root=root / "v86-run",
            )

            variant_ids = [row["variant_id"] for row in payload["promoted_variants"]]
            self.assertEqual(variant_ids, ["c1_reader_opd", "c2_best_writer_route", "c3_bridge_route"])


if __name__ == "__main__":
    unittest.main()
