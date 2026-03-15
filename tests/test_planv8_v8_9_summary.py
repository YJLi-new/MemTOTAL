from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.update_planv8_v8_9_summary import build_summary


class PlanV8V89SummaryTest(unittest.TestCase):
    def _write_json(self, path: Path, payload: dict[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    def _write_v80_summary(self, path: Path) -> None:
        self._write_json(
            path,
            {
                "selected_qwen34_baseline_scores": {
                    "gsm8k": 0.80,
                    "triviaqa": 0.03,
                }
            },
        )

    def _write_v88_summary(self, path: Path) -> None:
        self._write_json(
            path,
            {
                "comparison_conclusion": "multiseed_confirmation_success_open_v8_9_cdmi",
                "recommended_next_step": "open_v8_9_cdmi",
                "best_confirmed_variant_id": "c2_best_writer_route",
            },
        )

    def _write_manifest(self, path: Path, *, bridge_family: str = "BR2") -> None:
        self._write_json(
            path,
            {
                "best_confirmed_variant_id": "c2_best_writer_route",
                "source_phase": "V8-6",
                "source_run_root": "/tmp/source-run",
                "source_arm_id": "a4_writer_opd_ansctx",
                "source_interface_family": "ri2_cross_attn",
                "source_bridge_family": bridge_family,
                "source_auxiliary_family": "writer_opd_answer_plus_context",
                "source_prompt_variants": {
                    "gsm8k": "q3_gsm8k_nonthink",
                    "triviaqa": "q3_trivia_nonthink",
                },
            },
        )

    def _write_condition(
        self,
        result_root: Path,
        condition_id: str,
        *,
        benchmark_id: str,
        task_score: float,
        memory_path_variant: str = "two_level",
        short_slots: int = 16,
        reader_rank: float = 2.0,
        prefix_attention_mass_mean: float = 0.06,
        memory_token_attention_mass_mean: float = 0.04,
        cross_attn_gate_open_fraction: float = 0.10,
        train_events: list[dict[str, float]] | None = None,
    ) -> None:
        condition_root = result_root / condition_id
        condition_root.mkdir(parents=True, exist_ok=True)
        metrics = {
            "benchmark_id": benchmark_id,
            "best_adapt_task_score": task_score,
            "pilot_memory_path_variant": memory_path_variant,
            "pilot_memory_consumer_mode": "cross_attn_reader",
            "pilot_fuser_short_slots": short_slots,
            "reader_readout_effective_rank": reader_rank,
            "prefix_attention_mass_mean": prefix_attention_mass_mean,
            "memory_token_attention_mass_mean": memory_token_attention_mass_mean,
            "cross_attn_gate_open_fraction": cross_attn_gate_open_fraction,
            "prefix_attention_nontrivial_layer_count": 0 if memory_path_variant == "two_level" else 2,
            "prefix_artifact_stats": {
                "cross_attn_gate_open_fraction": cross_attn_gate_open_fraction,
                "memory_token_attention_mass_mean": memory_token_attention_mass_mean,
            },
        }
        self._write_json(condition_root / "metrics.json", metrics)
        if train_events is None:
            train_events = [{"loss": 1.0, "grad_norm_writer": 0.2, "grad_norm_reader": 0.1}]
        (condition_root / "train_events.json").write_text(json.dumps({"events": train_events}) + "\n")

    def test_build_summary_marks_closeout_ready_without_transfer_or_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            result_root = root / "results"
            manifest = root / "cdmi-manifest.json"
            v80_summary = root / "v8-0-summary.json"
            v88_summary = root / "v8-8-summary.json"
            self._write_manifest(manifest, bridge_family="BR0")
            self._write_v80_summary(v80_summary)
            self._write_v88_summary(v88_summary)

            self._write_condition(result_root, "c0_math_self", benchmark_id="gsm8k", task_score=0.83)
            self._write_condition(result_root, "c1_trivia_self", benchmark_id="triviaqa", task_score=0.04)
            self._write_condition(result_root, "c2_joint_math", benchmark_id="gsm8k", task_score=0.84)
            self._write_condition(
                result_root,
                "c3_joint_trivia",
                benchmark_id="triviaqa",
                task_score=0.05,
                cross_attn_gate_open_fraction=0.14,
            )
            self._write_condition(
                result_root,
                "c4_math_support_on_trivia",
                benchmark_id="triviaqa",
                task_score=0.05,
                cross_attn_gate_open_fraction=0.13,
            )
            self._write_condition(
                result_root,
                "c5_trivia_support_on_math",
                benchmark_id="gsm8k",
                task_score=0.84,
                cross_attn_gate_open_fraction=0.09,
            )

            summary = build_summary(
                result_root=result_root,
                manifest_path=manifest,
                v80_summary_path=v80_summary,
                v88_summary_path=v88_summary,
            )

            self.assertEqual(summary["comparison_conclusion"], "cdmi_profile_complete_paper_closeout_ready")
            self.assertEqual(summary["recommended_next_step"], "assemble_paper_closeout")
            self.assertEqual(summary["negative_transfer_rate"], 0.0)
            self.assertEqual(summary["cross_domain_leakage_rate"], 0.0)
            self.assertFalse(summary["compression_leakage_risk_flag"])

    def test_build_summary_flags_negative_transfer_and_compression_risk(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            result_root = root / "results"
            manifest = root / "cdmi-manifest.json"
            v80_summary = root / "v8-0-summary.json"
            v88_summary = root / "v8-8-summary.json"
            self._write_manifest(manifest, bridge_family="BR2")
            self._write_v80_summary(v80_summary)
            self._write_v88_summary(v88_summary)

            self._write_condition(result_root, "c0_math_self", benchmark_id="gsm8k", task_score=0.85)
            self._write_condition(result_root, "c1_trivia_self", benchmark_id="triviaqa", task_score=0.06)
            self._write_condition(result_root, "c2_joint_math", benchmark_id="gsm8k", task_score=0.82)
            self._write_condition(result_root, "c3_joint_trivia", benchmark_id="triviaqa", task_score=0.05)
            self._write_condition(result_root, "c4_math_support_on_trivia", benchmark_id="triviaqa", task_score=0.03)
            self._write_condition(result_root, "c5_trivia_support_on_math", benchmark_id="gsm8k", task_score=0.80)

            summary = build_summary(
                result_root=result_root,
                manifest_path=manifest,
                v80_summary_path=v80_summary,
                v88_summary_path=v88_summary,
            )

            self.assertEqual(summary["comparison_conclusion"], "cdmi_profile_complete_with_negative_transfer")
            self.assertEqual(
                summary["recommended_next_step"],
                "assemble_paper_closeout_with_cdmi_risk_memo",
            )
            self.assertEqual(summary["negative_transfer_rate"], 1.0)
            self.assertEqual(summary["cross_domain_leakage_rate"], 1.0)
            self.assertTrue(summary["compression_leakage_risk_flag"])
            self.assertTrue(summary["cross_domain_leakage_detected"])


if __name__ == "__main__":
    unittest.main()
