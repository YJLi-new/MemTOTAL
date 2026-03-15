from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.update_planv8_v8_4_summary import ALL_TASKS, ARM_ORDER, build_summary


class PlanV8V84SummaryTest(unittest.TestCase):
    def _write_v83_reference(
        self,
        path: Path,
        *,
        best_arm_id: str = "p4_opd_ansplusctx_w03",
        interface_family: str = "ri0_legacy_prefix",
        next_step: str = "open_v8_4_external_writer",
    ) -> None:
        path.write_text(
            json.dumps(
                {
                    "best_arm_id": best_arm_id,
                    "base_for_v8_4_arm_id": best_arm_id,
                    "selected_interface_family_for_v8_4": interface_family,
                    "recommended_next_step": next_step,
                }
            )
            + "\n"
        )

    def _write_task(
        self,
        result_root: Path,
        arm_id: str,
        task_name: str,
        *,
        task_score: float,
        control_logprob: float,
        branch_logprob: float,
        memory_tokens_count: int = 16,
        prefix_layers: int = 2,
        prefix_attention_mass_mean: float = 0.06,
        writer_grad_norm: float = 0.3,
        trainable_variant: str = "writer_then_joint",
    ) -> None:
        task_root = result_root / arm_id / task_name
        task_root.mkdir(parents=True, exist_ok=True)
        metrics = {
            "benchmark_id": task_name,
            "pilot_prompt_variant": {
                "gsm8k": "q3_gsm8k_nonthink",
                "triviaqa": "q3_trivia_nonthink",
                "fever": "answer_slot_labels",
            }[task_name],
            "best_adapt_task_score": float(task_score),
            "best_adapt_macro_f1": float(task_score),
            "pilot_train_steps": 300,
            "pilot_trainable_variant": trainable_variant,
            "pilot_writer_family": arm_id.split("_")[1] if arm_id != "w0_oracle64" else "EW0",
            "memory_tokens_count": int(memory_tokens_count),
            "prefix_attention_nontrivial_layer_count": int(prefix_layers),
            "prefix_attention_mass_mean": float(prefix_attention_mass_mean),
            "prefix_artifact_stats": {"memory_token_attention_mass_mean": 0.02},
            "train_final_phase": "stage_b_joint" if arm_id != "w0_oracle64" else "joint",
        }
        (task_root / "metrics.json").write_text(json.dumps(metrics) + "\n")
        (task_root / "train_events.json").write_text(
            json.dumps({"events": [{"loss": 1.0, "grad_norm_writer": writer_grad_norm}]}) + "\n"
        )
        case_rows = []
        for index in range(2):
            case_rows.append(
                {
                    "example_id": f"{task_name}-{index}",
                    "answer_logprob_with_memory": float(
                        branch_logprob if arm_id != "w0_oracle64" else control_logprob
                    ),
                    "prefix_attention_mass_mean": float(prefix_attention_mass_mean),
                }
            )
        (task_root / "task_case_dump.jsonl").write_text(
            "\n".join(json.dumps(row) for row in case_rows) + "\n"
        )

    def _write_flat_phase(self, result_root: Path) -> None:
        control_scores = {"gsm8k": 0.62, "triviaqa": 0.20, "fever": 0.58}
        control_logprobs = {"gsm8k": -12.0, "triviaqa": -5.0, "fever": -1.5}
        for task_name in ALL_TASKS:
            self._write_task(
                result_root,
                "w0_oracle64",
                task_name,
                task_score=control_scores[task_name],
                control_logprob=control_logprobs[task_name],
                branch_logprob=control_logprobs[task_name],
                trainable_variant="reader_only",
            )
        for arm_id in ARM_ORDER[1:]:
            for task_name in ALL_TASKS:
                self._write_task(
                    result_root,
                    arm_id,
                    task_name,
                    task_score=control_scores[task_name],
                    control_logprob=control_logprobs[task_name],
                    branch_logprob=control_logprobs[task_name],
                )

    def test_build_summary_promotes_trainable_writer_into_v85(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result_root = Path(temp_dir) / "results"
            result_root.mkdir(parents=True, exist_ok=True)
            self._write_flat_phase(result_root)
            v83_summary_path = Path(temp_dir) / "v8-3-summary.json"
            self._write_v83_reference(v83_summary_path)

            self._write_task(
                result_root,
                "w2_ext3layer64_lr2e5",
                "gsm8k",
                task_score=0.70,
                control_logprob=-12.0,
                branch_logprob=-8.0,
            )
            self._write_task(
                result_root,
                "w2_ext3layer64_lr2e5",
                "triviaqa",
                task_score=0.20,
                control_logprob=-5.0,
                branch_logprob=-4.0,
            )

            summary = build_summary(
                result_root=result_root,
                v83_summary_path=v83_summary_path,
            )

            self.assertEqual(summary["best_trainable_arm_id"], "w2_ext3layer64_lr2e5")
            self.assertEqual(summary["best_arm_id"], "w2_ext3layer64_lr2e5")
            self.assertTrue(summary["best_arm_acceptance_qualified"])
            self.assertEqual(summary["recommended_next_step"], "open_v8_5_bridge")
            self.assertEqual(
                summary["comparison_conclusion"],
                "external_writer_beats_oracle_open_v8_5_bridge",
            )

    def test_build_summary_routes_oracle_best_to_v87_comparators(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result_root = Path(temp_dir) / "results"
            result_root.mkdir(parents=True, exist_ok=True)
            self._write_flat_phase(result_root)
            v83_summary_path = Path(temp_dir) / "v8-3-summary.json"
            self._write_v83_reference(v83_summary_path)

            summary = build_summary(
                result_root=result_root,
                v83_summary_path=v83_summary_path,
            )

            self.assertEqual(summary["best_arm_id"], "w0_oracle64")
            self.assertFalse(summary["best_arm_acceptance_qualified"])
            self.assertEqual(summary["recommended_next_step"], "open_v8_7_comparators")
            self.assertEqual(
                summary["comparison_conclusion"],
                "external_writer_oracle_still_best_open_v8_7_comparators",
            )

    def test_build_summary_carries_v83_reference_into_v85_base(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result_root = Path(temp_dir) / "results"
            result_root.mkdir(parents=True, exist_ok=True)
            self._write_flat_phase(result_root)
            v83_summary_path = Path(temp_dir) / "v8-3-summary.json"
            self._write_v83_reference(
                v83_summary_path,
                best_arm_id="p5_opd_ansplusctx_centered",
                interface_family="ri1_prepend_block",
            )

            summary = build_summary(
                result_root=result_root,
                v83_summary_path=v83_summary_path,
            )

            self.assertEqual(summary["v83_best_arm_id"], "p5_opd_ansplusctx_centered")
            self.assertEqual(summary["v83_base_for_v8_4_arm_id"], "p5_opd_ansplusctx_centered")
            self.assertEqual(summary["v83_selected_interface_family_for_v8_4"], "ri1_prepend_block")
            self.assertEqual(summary["selected_interface_family_for_v8_5"], "ri1_prepend_block")


if __name__ == "__main__":
    unittest.main()
