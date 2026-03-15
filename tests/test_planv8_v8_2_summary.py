from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.update_planv8_v8_2_summary import ARM_ORDER, ALL_TASKS, build_summary


class PlanV8V82SummaryTest(unittest.TestCase):
    def _write_v81_reference(
        self,
        path: Path,
        *,
        best_arm_id: str = "i2_seq16_r64_mid8",
        interface_family: str = "ri1_prepend_block",
        memory_slots: int = 16,
        next_step: str = "open_v8_2_reader_sweep",
    ) -> None:
        path.write_text(
            json.dumps(
                {
                    "best_arm_id": best_arm_id,
                    "base_for_v8_2_arm_id": best_arm_id,
                    "selected_interface_family_for_v8_2": interface_family,
                    "recommended_next_step": next_step,
                    "arm_summaries": {
                        best_arm_id: {
                            "memory_slots": memory_slots,
                        }
                    },
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
        cross_attn_gate_open_fraction: float = 0.0,
        memory_token_attention_mass_mean: float = 0.02,
        train_events: list[dict[str, object]] | None = None,
        memory_consumer_mode: str = "reader_lora_sequence",
    ) -> None:
        task_root = result_root / arm_id / task_name
        task_root.mkdir(parents=True, exist_ok=True)
        grad_event = {"loss": 1.0}
        if arm_id != "control":
            grad_event["grad_norm_receiver_lora"] = 0.3
        payload_events = train_events if train_events is not None else ([grad_event] if arm_id != "control" else [])
        metrics = {
            "benchmark_id": task_name,
            "pilot_prompt_variant": {
                "gsm8k": "q3_gsm8k_nonthink",
                "triviaqa": "q3_trivia_think",
                "fever": "answer_slot_labels",
            }[task_name],
            "best_adapt_task_score": float(task_score),
            "best_adapt_macro_f1": float(task_score),
            "pilot_train_steps": 0 if arm_id == "control" else 300,
            "pilot_memory_consumer_mode": memory_consumer_mode,
            "memory_tokens_count": int(memory_tokens_count if arm_id != "control" else 0),
            "prefix_attention_nontrivial_layer_count": int(prefix_layers if arm_id != "control" else 0),
            "prefix_attention_mass_mean": float(prefix_attention_mass_mean if arm_id != "control" else 0.0),
            "cross_attn_gate_open_fraction": float(cross_attn_gate_open_fraction if arm_id != "control" else 0.0),
            "prefix_artifact_stats": {
                "memory_token_attention_mass_mean": float(
                    memory_token_attention_mass_mean if arm_id != "control" else 0.0
                )
            },
        }
        (task_root / "metrics.json").write_text(json.dumps(metrics) + "\n")
        (task_root / "train_events.json").write_text(json.dumps({"events": payload_events}) + "\n")
        case_rows = []
        for index in range(2):
            case_rows.append(
                {
                    "example_id": f"{task_name}-{index}",
                    "answer_logprob_with_memory": float(
                        branch_logprob if arm_id != "control" else control_logprob
                    ),
                    "prefix_attention_mass_mean": float(
                        prefix_attention_mass_mean if arm_id != "control" else 0.0
                    ),
                }
            )
        (task_root / "task_case_dump.jsonl").write_text(
            "\n".join(json.dumps(row) for row in case_rows) + "\n"
        )

    def _write_control_and_flat_arms(self, result_root: Path) -> None:
        control_scores = {"gsm8k": 0.60, "triviaqa": 0.20, "fever": 0.58}
        control_logprobs = {"gsm8k": -12.0, "triviaqa": -5.0, "fever": -1.5}
        for task_name in ALL_TASKS:
            self._write_task(
                result_root,
                "control",
                task_name,
                task_score=control_scores[task_name],
                control_logprob=control_logprobs[task_name],
                branch_logprob=control_logprobs[task_name],
                memory_tokens_count=0,
            )
        for arm_id in ARM_ORDER:
            for task_name in ALL_TASKS:
                self._write_task(
                    result_root,
                    arm_id,
                    task_name,
                    task_score=control_scores[task_name],
                    control_logprob=control_logprobs[task_name],
                    branch_logprob=control_logprobs[task_name],
                )

    def test_build_summary_promotes_signal_arm_into_v83(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result_root = Path(temp_dir) / "results"
            result_root.mkdir(parents=True, exist_ok=True)
            self._write_control_and_flat_arms(result_root)
            v81_summary_path = Path(temp_dir) / "v8-1-summary.json"
            self._write_v81_reference(v81_summary_path)

            self._write_task(
                result_root,
                "r2_mid12_r64_lr1e4",
                "gsm8k",
                task_score=0.68,
                control_logprob=-12.0,
                branch_logprob=-8.0,
                memory_tokens_count=16,
                prefix_layers=4,
                prefix_attention_mass_mean=0.09,
            )
            self._write_task(
                result_root,
                "r2_mid12_r64_lr1e4",
                "triviaqa",
                task_score=0.22,
                control_logprob=-5.0,
                branch_logprob=-4.0,
                memory_tokens_count=16,
                prefix_layers=4,
                prefix_attention_mass_mean=0.08,
            )

            summary = build_summary(
                result_root=result_root,
                v81_summary_path=v81_summary_path,
            )

            self.assertEqual(summary["best_arm_id"], "r2_mid12_r64_lr1e4")
            self.assertEqual(summary["best_interface_family"], "ri1_prepend_block")
            self.assertTrue(summary["best_arm_acceptance_qualified"])
            self.assertEqual(summary["recommended_next_step"], "open_v8_3_reader_opd")
            self.assertEqual(summary["comparison_conclusion"], "reader_sweep_score_signal_open_v8_3")

    def test_build_summary_marks_flat_sweep_as_last_consumer_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result_root = Path(temp_dir) / "results"
            result_root.mkdir(parents=True, exist_ok=True)
            self._write_control_and_flat_arms(result_root)
            v81_summary_path = Path(temp_dir) / "v8-1-summary.json"
            self._write_v81_reference(
                v81_summary_path,
                best_arm_id="i0_prefix_legacy_r2",
                interface_family="ri0_legacy_prefix",
            )

            summary = build_summary(
                result_root=result_root,
                v81_summary_path=v81_summary_path,
            )

            self.assertEqual(
                summary["comparison_conclusion"],
                "reader_sweep_flat_open_v8_3_last_consumer_attempt",
            )
            self.assertEqual(
                summary["recommended_next_step"],
                "open_v8_3_reader_opd_last_consumer_attempt",
            )
            self.assertEqual(summary["best_interface_family"], "ri0_legacy_prefix")
            self.assertFalse(summary["best_arm_acceptance_qualified"])

    def test_build_summary_carries_v81_reference_into_v83_base(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result_root = Path(temp_dir) / "results"
            result_root.mkdir(parents=True, exist_ok=True)
            self._write_control_and_flat_arms(result_root)
            v81_summary_path = Path(temp_dir) / "v8-1-summary.json"
            self._write_v81_reference(
                v81_summary_path,
                best_arm_id="i0_prefix_legacy_r2",
                interface_family="ri0_legacy_prefix",
                memory_slots=32,
                next_step="open_v8_2_reader_sweep_last_chance",
            )

            summary = build_summary(
                result_root=result_root,
                v81_summary_path=v81_summary_path,
            )

            self.assertEqual(summary["v81_base_for_v8_2_arm_id"], "i0_prefix_legacy_r2")
            self.assertEqual(summary["v81_selected_interface_family_for_v8_2"], "ri0_legacy_prefix")
            self.assertEqual(summary["arm_summaries"]["r0_mid8_r32_lr5e5"]["memory_slots"], 32)
            self.assertEqual(summary["selected_interface_family_for_v8_3"], "ri0_legacy_prefix")


if __name__ == "__main__":
    unittest.main()
