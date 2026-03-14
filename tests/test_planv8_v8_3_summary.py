from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.update_planv8_v8_3_summary import ALL_TASKS, ARM_ORDER, build_summary


class PlanV8V83SummaryTest(unittest.TestCase):
    def _write_v82_reference(
        self,
        path: Path,
        *,
        best_arm_id: str = "r0_mid8_r32_lr5e5",
        interface_family: str = "ri0_legacy_prefix",
        next_step: str = "open_v8_3_reader_opd_last_consumer_attempt",
    ) -> None:
        path.write_text(
            json.dumps(
                {
                    "best_arm_id": best_arm_id,
                    "base_for_v8_3_arm_id": best_arm_id,
                    "selected_interface_family_for_v8_3": interface_family,
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
        cross_attn_gate_open_fraction: float = 0.0,
        memory_token_attention_mass_mean: float = 0.02,
        train_events: list[dict[str, object]] | None = None,
        alignment_aux_mode: str = "off",
        opd_positive_fraction: float = 0.0,
        opd_advantage: float = 0.0,
        effective_hint_mode: str = "answer_only",
    ) -> None:
        task_root = result_root / arm_id / task_name
        task_root.mkdir(parents=True, exist_ok=True)
        grad_event = {
            "loss": 1.0,
            "grad_norm_receiver_lora": 0.3,
            "opd_positive_token_fraction": opd_positive_fraction,
            "opd_mean_advantage": opd_advantage,
            "opd_hint_mode_effective": effective_hint_mode,
        }
        payload_events = train_events if train_events is not None else [grad_event]
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
            "pilot_memory_consumer_mode": "legacy_prefix",
            "pilot_alignment_aux_mode": alignment_aux_mode,
            "memory_tokens_count": int(memory_tokens_count),
            "prefix_attention_nontrivial_layer_count": int(prefix_layers),
            "prefix_attention_mass_mean": float(prefix_attention_mass_mean),
            "cross_attn_gate_open_fraction": float(cross_attn_gate_open_fraction),
            "prefix_artifact_stats": {
                "memory_token_attention_mass_mean": float(memory_token_attention_mass_mean)
            },
            "train_final_opd_positive_token_fraction": float(opd_positive_fraction),
            "train_final_opd_mean_advantage": float(opd_advantage),
            "train_final_opd_hint_mode_effective": effective_hint_mode,
            "train_final_opd_target_context_available": effective_hint_mode != "answer_only",
        }
        (task_root / "metrics.json").write_text(json.dumps(metrics) + "\n")
        (task_root / "train_events.json").write_text(json.dumps({"events": payload_events}) + "\n")
        case_rows = []
        for index in range(2):
            case_rows.append(
                {
                    "example_id": f"{task_name}-{index}",
                    "answer_logprob_with_memory": float(
                        branch_logprob if arm_id != "p0_ce_only" else control_logprob
                    ),
                    "prefix_attention_mass_mean": float(prefix_attention_mass_mean),
                }
            )
        (task_root / "task_case_dump.jsonl").write_text(
            "\n".join(json.dumps(row) for row in case_rows) + "\n"
        )

    def _write_flat_phase(self, result_root: Path) -> None:
        control_scores = {"gsm8k": 0.60, "triviaqa": 0.20, "fever": 0.58}
        control_logprobs = {"gsm8k": -12.0, "triviaqa": -5.0, "fever": -1.5}
        for task_name in ALL_TASKS:
            self._write_task(
                result_root,
                "p0_ce_only",
                task_name,
                task_score=control_scores[task_name],
                control_logprob=control_logprobs[task_name],
                branch_logprob=control_logprobs[task_name],
                alignment_aux_mode="off",
                opd_positive_fraction=0.0,
                opd_advantage=0.0,
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
                    alignment_aux_mode="opd_token_ce" if arm_id.startswith("p") and arm_id != "p1_teacher_choice_kl" else "teacher_choice_kl",
                    opd_positive_fraction=0.1,
                    opd_advantage=0.2,
                )

    def test_build_summary_promotes_score_signal_into_v84(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result_root = Path(temp_dir) / "results"
            result_root.mkdir(parents=True, exist_ok=True)
            self._write_flat_phase(result_root)
            v82_summary_path = Path(temp_dir) / "v8-2-summary.json"
            self._write_v82_reference(v82_summary_path)

            self._write_task(
                result_root,
                "p4_opd_ansplusctx_w03",
                "gsm8k",
                task_score=0.72,
                control_logprob=-12.0,
                branch_logprob=-7.0,
                alignment_aux_mode="opd_token_ce",
                opd_positive_fraction=0.7,
                opd_advantage=0.9,
                effective_hint_mode="answer_plus_rationale",
            )
            self._write_task(
                result_root,
                "p4_opd_ansplusctx_w03",
                "triviaqa",
                task_score=0.20,
                control_logprob=-5.0,
                branch_logprob=-3.5,
                alignment_aux_mode="opd_token_ce",
                opd_positive_fraction=0.6,
                opd_advantage=0.5,
                effective_hint_mode="answer_only",
            )

            summary = build_summary(
                result_root=result_root,
                v82_summary_path=v82_summary_path,
            )

            self.assertEqual(summary["best_arm_id"], "p4_opd_ansplusctx_w03")
            self.assertEqual(summary["best_interface_family"], "ri0_legacy_prefix")
            self.assertTrue(summary["best_arm_acceptance_qualified"])
            self.assertEqual(summary["recommended_next_step"], "open_v8_4_external_writer")
            self.assertEqual(
                summary["comparison_conclusion"],
                "reader_opd_score_signal_open_v8_4_external_writer",
            )

    def test_build_summary_routes_flat_phase_to_v87_comparators(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result_root = Path(temp_dir) / "results"
            result_root.mkdir(parents=True, exist_ok=True)
            self._write_flat_phase(result_root)
            v82_summary_path = Path(temp_dir) / "v8-2-summary.json"
            self._write_v82_reference(v82_summary_path)

            summary = build_summary(
                result_root=result_root,
                v82_summary_path=v82_summary_path,
            )

            self.assertEqual(
                summary["comparison_conclusion"],
                "reader_opd_flat_open_v8_7_comparators",
            )
            self.assertEqual(summary["recommended_next_step"], "open_v8_7_comparators")
            self.assertFalse(summary["best_arm_acceptance_qualified"])

    def test_build_summary_carries_v82_reference_into_v84_base(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result_root = Path(temp_dir) / "results"
            result_root.mkdir(parents=True, exist_ok=True)
            self._write_flat_phase(result_root)
            v82_summary_path = Path(temp_dir) / "v8-2-summary.json"
            self._write_v82_reference(
                v82_summary_path,
                best_arm_id="r2_mid12_r64_lr1e4",
                interface_family="ri1_prepend_block",
                next_step="open_v8_3_reader_opd",
            )

            summary = build_summary(
                result_root=result_root,
                v82_summary_path=v82_summary_path,
            )

            self.assertEqual(summary["v82_best_arm_id"], "r2_mid12_r64_lr1e4")
            self.assertEqual(summary["v82_base_for_v8_3_arm_id"], "r2_mid12_r64_lr1e4")
            self.assertEqual(summary["v82_selected_interface_family_for_v8_3"], "ri1_prepend_block")
            self.assertEqual(summary["selected_interface_family_for_v8_4"], "ri1_prepend_block")


if __name__ == "__main__":
    unittest.main()
