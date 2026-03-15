from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.update_planv8_v8_7_summary import build_summary


class PlanV8V87SummaryTest(unittest.TestCase):
    def _write_v80_summary(self, path: Path) -> None:
        path.write_text(
            json.dumps(
                {
                    "selected_qwen34_baseline_scores": {
                        "gsm8k": 0.80,
                        "triviaqa": 0.03,
                        "fever": 0.79,
                    },
                    "selected_prompt_modes_by_task": {
                        "gsm8k": "q3_gsm8k_nonthink",
                        "triviaqa": "q3_trivia_nonthink",
                        "fever": "answer_slot_labels",
                    },
                }
            )
            + "\n"
        )

    def _write_v76_summary(self, path: Path) -> None:
        path.write_text(
            json.dumps(
                {
                    "best_confirmed_variant_id": "p1",
                    "best_confirmed_promoted_arm_id": "a5_barlow",
                    "recommended_next_step": "open_stronger_integrated_writer_or_true_highdim_branch",
                    "branches": {
                        "p1": {
                            "tasks": {
                                "gsm8k": {"seed_rows": [{"task_score": 0.0}, {"task_score": 0.0}, {"task_score": 0.0}]},
                                "triviaqa": {"seed_rows": [{"task_score": 0.0}, {"task_score": 0.0}, {"task_score": 0.0}]},
                                "fever": {"seed_rows": [{"task_score": 0.296875}]},
                            }
                        }
                    },
                }
            )
            + "\n"
        )

    def _write_v83_summary(self, path: Path, *, gsm8k: float, triviaqa: float, fever: float) -> None:
        path.write_text(
            json.dumps(
                {
                    "phase": "V8-3",
                    "best_arm_id": "p5_opd_ansplusctx_centered",
                    "base_for_v8_4_arm_id": "p5_opd_ansplusctx_centered",
                    "selected_interface_family_for_v8_4": "ri0_legacy_prefix",
                    "best_alignment_aux_mode": "opd_token_ce_centered",
                    "arm_summaries": {
                        "p5_opd_ansplusctx_centered": {
                            "tasks": {
                                "gsm8k": {
                                    "task_score": gsm8k,
                                    "route_live": True,
                                    "prompt_variant": "q3_gsm8k_nonthink",
                                },
                                "triviaqa": {
                                    "task_score": triviaqa,
                                    "route_live": True,
                                    "prompt_variant": "q3_trivia_nonthink",
                                },
                                "fever": {
                                    "task_score": fever,
                                    "route_live": False,
                                    "prompt_variant": "answer_slot_labels",
                                },
                            }
                        }
                    },
                    "recommended_next_step": "open_v8_7_comparators",
                }
            )
            + "\n"
        )

    def _write_v86_summary(self, path: Path, *, gsm8k: float, triviaqa: float, fever: float) -> None:
        path.write_text(
            json.dumps(
                {
                    "phase": "V8-6",
                    "best_arm_id": "a4_writer_opd_ansctx",
                    "base_for_v8_7_arm_id": "a4_writer_opd_ansctx",
                    "selected_interface_family_for_v8_7": "ri2_cross_attn",
                    "selected_bridge_family_for_v8_7": "BR2",
                    "selected_aux_family_for_v8_7": "writer_opd_answer_plus_context",
                    "arm_summaries": {
                        "a4_writer_opd_ansctx": {
                            "tasks": {
                                "gsm8k": {
                                    "task_score": gsm8k,
                                    "route_live": True,
                                    "prompt_variant": "q3_gsm8k_nonthink",
                                },
                                "triviaqa": {
                                    "task_score": triviaqa,
                                    "route_live": True,
                                    "prompt_variant": "q3_trivia_nonthink",
                                },
                                "fever": {
                                    "task_score": fever,
                                    "route_live": True,
                                    "prompt_variant": "answer_slot_labels",
                                },
                            }
                        }
                    },
                }
            )
            + "\n"
        )

    def _write_metric(self, result_root: Path, arm: str, task: str, metric_name: str, score: float) -> None:
        task_root = result_root / arm / task
        task_root.mkdir(parents=True, exist_ok=True)
        metrics = {
            "benchmark_id": task,
            "metric_name": metric_name,
            "task_score": score,
        }
        if metric_name == "exact_match":
            metrics["exact_match"] = score
        elif metric_name == "accuracy":
            metrics["accuracy"] = score
            metrics["macro_f1"] = score
        else:
            metrics["compute_reward"] = score
        (task_root / "metrics.json").write_text(json.dumps(metrics) + "\n")

    def test_build_summary_opens_v88_when_best_v8_beats_floor_and_rag(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            result_root = root / "results"
            v80_summary = root / "v8-0-summary.json"
            v76_summary = root / "v7-6-summary.json"
            v86_summary = root / "v8-6-summary.json"
            self._write_v80_summary(v80_summary)
            self._write_v76_summary(v76_summary)
            self._write_v86_summary(v86_summary, gsm8k=0.86, triviaqa=0.05, fever=0.81)
            self._write_metric(result_root, "m1_text_rag", "gsm8k", "exact_match", 0.78)
            self._write_metric(result_root, "m1_text_rag", "triviaqa", "exact_match", 0.03)
            self._write_metric(result_root, "m1_text_rag", "fever", "accuracy", 0.76)
            self._write_metric(result_root, "m2_memgen", "gsm8k", "compute_reward", 0.50)

            summary = build_summary(
                result_root=result_root,
                v80_summary_path=v80_summary,
                v76_summary_path=v76_summary,
                best_v8_summary_path=v86_summary,
            )

            self.assertEqual(
                summary["comparison_conclusion"],
                "comparators_support_open_v8_8_multiseed_confirmation",
            )
            self.assertEqual(summary["recommended_next_step"], "open_v8_8_multiseed_confirmation")
            self.assertEqual(summary["base_for_v8_8_arm_id"], "a4_writer_opd_ansctx")

    def test_build_summary_holds_when_text_rag_is_stronger(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            result_root = root / "results"
            v80_summary = root / "v8-0-summary.json"
            v76_summary = root / "v7-6-summary.json"
            v86_summary = root / "v8-6-summary.json"
            self._write_v80_summary(v80_summary)
            self._write_v76_summary(v76_summary)
            self._write_v86_summary(v86_summary, gsm8k=0.83, triviaqa=0.03, fever=0.80)
            self._write_metric(result_root, "m1_text_rag", "gsm8k", "exact_match", 0.85)
            self._write_metric(result_root, "m1_text_rag", "triviaqa", "exact_match", 0.04)
            self._write_metric(result_root, "m1_text_rag", "fever", "accuracy", 0.78)

            summary = build_summary(
                result_root=result_root,
                v80_summary_path=v80_summary,
                v76_summary_path=v76_summary,
                best_v8_summary_path=v86_summary,
            )

            self.assertEqual(
                summary["comparison_conclusion"],
                "comparators_floor_cleared_but_text_rag_blocks_v8_8",
            )
            self.assertEqual(summary["recommended_next_step"], "hold_v8_8_rag_gap")

    def test_build_summary_accepts_v83_as_best_v8_reference(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            result_root = root / "results"
            v80_summary = root / "v8-0-summary.json"
            v76_summary = root / "v7-6-summary.json"
            v83_summary = root / "v8-3-summary.json"
            self._write_v80_summary(v80_summary)
            self._write_v76_summary(v76_summary)
            self._write_v83_summary(v83_summary, gsm8k=0.83, triviaqa=0.04, fever=0.79)
            self._write_metric(result_root, "m1_text_rag", "gsm8k", "exact_match", 0.81)
            self._write_metric(result_root, "m1_text_rag", "triviaqa", "exact_match", 0.03)
            self._write_metric(result_root, "m1_text_rag", "fever", "accuracy", 0.77)

            summary = build_summary(
                result_root=result_root,
                v80_summary_path=v80_summary,
                v76_summary_path=v76_summary,
                best_v8_summary_path=v83_summary,
            )

            self.assertEqual(summary["recommended_next_step"], "open_v8_8_multiseed_confirmation")
            self.assertEqual(summary["base_for_v8_8_arm_id"], "p5_opd_ansplusctx_centered")
            self.assertEqual(summary["base_for_v8_8_source_phase"], "V8-3")


if __name__ == "__main__":
    unittest.main()
