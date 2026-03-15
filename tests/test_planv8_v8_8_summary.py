from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.update_planv8_v8_8_summary import build_summary


class PlanV8V88SummaryTest(unittest.TestCase):
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
                    "fever": 0.79,
                }
            },
        )

    def _write_v87_summary(self, path: Path) -> None:
        self._write_json(
            path,
            {
                "comparison_conclusion": "comparators_support_open_v8_8_multiseed_confirmation",
                "recommended_next_step": "open_v8_8_multiseed_confirmation",
                "base_for_v8_8_arm_id": "a4_writer_opd_ansctx",
            },
        )

    def _write_selection_manifest(self, path: Path) -> None:
        self._write_json(
            path,
            {
                "seeds": [61109, 61110, 61111],
                "promoted_variants": [
                    {
                        "variant_id": "c1_reader_opd",
                        "source_phase": "V8-3",
                        "arm_id": "p4_opd_ansplusctx_w03",
                        "interface_family": "ri1_prepend_block",
                        "bridge_family": "BR0",
                        "auxiliary_family": "reader_opd",
                    },
                    {
                        "variant_id": "c2_best_writer_route",
                        "source_phase": "V8-6",
                        "arm_id": "a4_writer_opd_ansctx",
                        "interface_family": "ri2_cross_attn",
                        "bridge_family": "BR2",
                        "auxiliary_family": "writer_opd_answer_plus_context",
                    },
                ],
            },
        )

    def _write_task(
        self,
        result_root: Path,
        variant_id: str,
        seed: int,
        task_name: str,
        *,
        task_score: float,
        memory_path_variant: str = "single_level",
        short_slots: int = 0,
        prefix_attention_mass_mean: float = 0.06,
        reader_rank: float = 0.0,
    ) -> None:
        task_root = result_root / variant_id / f"seed_{seed}" / task_name
        task_root.mkdir(parents=True, exist_ok=True)
        metrics = {
            "benchmark_id": task_name,
            "best_adapt_task_score": task_score,
            "pilot_memory_path_variant": memory_path_variant,
            "pilot_fuser_short_slots": short_slots,
            "prefix_attention_nontrivial_layer_count": 2 if memory_path_variant == "single_level" else 0,
            "prefix_attention_mass_mean": prefix_attention_mass_mean,
            "reader_readout_effective_rank": reader_rank,
            "prefix_artifact_stats": {"cross_attn_gate_open_fraction": 0.10},
        }
        self._write_json(task_root / "metrics.json", metrics)
        (task_root / "train_events.json").write_text(
            json.dumps({"events": [{"loss": 1.0, "grad_norm_writer": 0.2, "grad_norm_reader": 0.1}]}) + "\n"
        )

    def test_build_summary_opens_v89_when_confirmation_succeeds(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            result_root = root / "results"
            v80_summary = root / "v8-0-summary.json"
            v87_summary = root / "v8-7-summary.json"
            selection_manifest = root / "selection-manifest.json"
            self._write_v80_summary(v80_summary)
            self._write_v87_summary(v87_summary)
            self._write_selection_manifest(selection_manifest)

            for seed in (61109, 61110, 61111):
                self._write_task(result_root, "c1_reader_opd", seed, "gsm8k", task_score=0.81)
                self._write_task(result_root, "c1_reader_opd", seed, "triviaqa", task_score=0.03)
                self._write_task(result_root, "c1_reader_opd", seed, "fever", task_score=0.79)
                self._write_task(
                    result_root,
                    "c2_best_writer_route",
                    seed,
                    "gsm8k",
                    task_score=0.84,
                    memory_path_variant="two_level",
                    short_slots=16,
                    reader_rank=2.0,
                )
                self._write_task(
                    result_root,
                    "c2_best_writer_route",
                    seed,
                    "triviaqa",
                    task_score=0.04,
                    memory_path_variant="two_level",
                    short_slots=16,
                    reader_rank=2.0,
                )
                self._write_task(
                    result_root,
                    "c2_best_writer_route",
                    seed,
                    "fever",
                    task_score=0.80,
                    memory_path_variant="two_level",
                    short_slots=16,
                    reader_rank=2.0,
                )

            summary = build_summary(
                result_root=result_root,
                selection_manifest_path=selection_manifest,
                v80_summary_path=v80_summary,
                v87_summary_path=v87_summary,
            )

            self.assertEqual(summary["comparison_conclusion"], "multiseed_confirmation_success_open_v8_9_cdmi")
            self.assertEqual(summary["recommended_next_step"], "open_v8_9_cdmi")
            self.assertEqual(summary["best_confirmed_variant_id"], "c2_best_writer_route")

    def test_build_summary_holds_when_primary_non_regression_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            result_root = root / "results"
            v80_summary = root / "v8-0-summary.json"
            v87_summary = root / "v8-7-summary.json"
            selection_manifest = root / "selection-manifest.json"
            self._write_v80_summary(v80_summary)
            self._write_v87_summary(v87_summary)
            self._write_selection_manifest(selection_manifest)

            for seed in (61109, 61110, 61111):
                self._write_task(result_root, "c1_reader_opd", seed, "gsm8k", task_score=0.80)
                self._write_task(result_root, "c1_reader_opd", seed, "triviaqa", task_score=0.02)
                self._write_task(result_root, "c1_reader_opd", seed, "fever", task_score=0.78)
                self._write_task(result_root, "c2_best_writer_route", seed, "gsm8k", task_score=0.83)
                self._write_task(result_root, "c2_best_writer_route", seed, "triviaqa", task_score=0.01)
                self._write_task(result_root, "c2_best_writer_route", seed, "fever", task_score=0.78)

            summary = build_summary(
                result_root=result_root,
                selection_manifest_path=selection_manifest,
                v80_summary_path=v80_summary,
                v87_summary_path=v87_summary,
            )

            self.assertEqual(summary["comparison_conclusion"], "multiseed_confirmation_failed_hold_v8_9")
            self.assertEqual(summary["recommended_next_step"], "hold_v8_9_confirmation_review")
            self.assertEqual(summary["best_confirmed_variant_id"], "")


if __name__ == "__main__":
    unittest.main()
