from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.update_planv8_v8_1_summary import ARM_ORDER, ALL_TASKS, build_summary


class PlanV8V81SummaryTest(unittest.TestCase):
    def _arm_defaults(self, arm_id: str) -> tuple[str, float]:
        if arm_id.startswith("i4_") or arm_id.startswith("i5_"):
            return "reader_cross_attn", 0.0
        if arm_id.startswith("i1_") or arm_id.startswith("i2_") or arm_id.startswith("i3_"):
            return "reader_lora_sequence", 0.0
        return "legacy_prefix", 0.0

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
        prefix_layers: int = 1,
        prefix_attention_mass_mean: float = 0.04,
        cross_attn_gate_open_fraction: float = 0.0,
        memory_token_attention_mass_mean: float = 0.0,
        train_events: list[dict[str, object]] | None = None,
    ) -> None:
        task_root = result_root / arm_id / task_name
        task_root.mkdir(parents=True, exist_ok=True)
        memory_consumer_mode, default_gate = self._arm_defaults(arm_id)
        gate_value = (
            cross_attn_gate_open_fraction
            if cross_attn_gate_open_fraction > 0.0
            else default_gate
        )
        grad_event = {"loss": 1.0}
        if arm_id.startswith("i4_") or arm_id.startswith("i5_"):
            grad_event["grad_norm_reader_cross_attn"] = 0.5
        elif arm_id != "control":
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
            "pilot_train_steps": 0 if arm_id == "control" else 200,
            "pilot_memory_consumer_mode": memory_consumer_mode,
            "memory_tokens_count": int(memory_tokens_count if arm_id != "control" else 0),
            "prefix_attention_nontrivial_layer_count": int(prefix_layers if arm_id != "control" else 0),
            "prefix_attention_mass_mean": float(prefix_attention_mass_mean if arm_id != "control" else 0.0),
            "cross_attn_gate_open_fraction": float(gate_value if arm_id != "control" else 0.0),
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

    def test_build_summary_promotes_signal_arm_into_v82(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result_root = Path(temp_dir) / "results"
            result_root.mkdir(parents=True, exist_ok=True)
            self._write_control_and_flat_arms(result_root)

            self._write_task(
                result_root,
                "i2_seq16_r64_mid8",
                "gsm8k",
                task_score=0.68,
                control_logprob=-12.0,
                branch_logprob=-8.0,
                memory_tokens_count=16,
                prefix_layers=2,
                prefix_attention_mass_mean=0.08,
            )
            self._write_task(
                result_root,
                "i2_seq16_r64_mid8",
                "triviaqa",
                task_score=0.20,
                control_logprob=-5.0,
                branch_logprob=-4.0,
                memory_tokens_count=16,
                prefix_layers=2,
                prefix_attention_mass_mean=0.07,
            )
            self._write_task(
                result_root,
                "i2_seq16_r64_mid8",
                "fever",
                task_score=0.58,
                control_logprob=-1.5,
                branch_logprob=-1.4,
                memory_tokens_count=16,
                prefix_layers=1,
                prefix_attention_mass_mean=0.03,
            )

            summary = build_summary(result_root=result_root)

            self.assertEqual(summary["best_arm_id"], "i2_seq16_r64_mid8")
            self.assertEqual(summary["best_interface_family"], "ri1_prepend_block")
            self.assertTrue(summary["best_arm_acceptance_qualified"])
            self.assertEqual(summary["recommended_next_step"], "open_v8_2_reader_sweep")
            self.assertEqual(summary["comparison_conclusion"], "reader_interface_score_signal_open_v8_2")

    def test_build_summary_marks_flat_matrix_as_last_chance(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result_root = Path(temp_dir) / "results"
            result_root.mkdir(parents=True, exist_ok=True)
            self._write_control_and_flat_arms(result_root)

            summary = build_summary(result_root=result_root)

            self.assertEqual(summary["comparison_conclusion"], "reader_interface_flat_open_v8_2_last_chance")
            self.assertEqual(summary["recommended_next_step"], "open_v8_2_reader_sweep_last_chance")
            self.assertFalse(summary["best_arm_acceptance_qualified"])


if __name__ == "__main__":
    unittest.main()
