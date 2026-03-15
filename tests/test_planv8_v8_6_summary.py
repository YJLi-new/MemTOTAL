from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.update_planv8_v8_6_summary import ALL_TASKS, ARM_ORDER, build_summary


class PlanV8V86SummaryTest(unittest.TestCase):
    def _write_v85_reference(
        self,
        path: Path,
        *,
        best_arm_id: str = "b2_q32_s16",
        interface_family: str = "ri2_cross_attn",
        bridge_family: str = "BR2",
        next_step: str = "open_v8_6_writer_aux",
    ) -> None:
        path.write_text(
            json.dumps(
                {
                    "best_arm_id": best_arm_id,
                    "base_for_v8_6_arm_id": best_arm_id,
                    "selected_interface_family_for_v8_6": interface_family,
                    "selected_bridge_family_for_v8_6": bridge_family,
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
        memory_path_variant: str = "single_level",
        writer_slots: int = 64,
        reader_queries: int = 0,
        short_slots: int = 0,
        writer_rank: float = 8.0,
        common_mode_ratio: float = 0.35,
        pairwise_cosine: float = 0.20,
        prefix_attention_mass_mean: float = 0.06,
        writer_aux_grad: float = 0.0,
        writer_task_grad: float = 0.3,
        writer_total_grad: float = 0.4,
        writer_task_aux_cosine: float = 0.0,
        writer_aux_total_cosine: float = 0.0,
        reader_grad: float = 0.2,
        fuser_grad: float = 0.2,
        alignment_aux_mode: str = "off",
        reconstruction_aux_mode: str = "off",
        opd_advantage: float = 0.0,
        reconstruction_aux_loss: float = 0.0,
        barlow_aux_loss: float = 0.0,
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
            "pilot_memory_path_variant": memory_path_variant,
            "writer_memory_slots": int(writer_slots),
            "pilot_reader_num_queries": int(reader_queries),
            "pilot_fuser_short_slots": int(short_slots),
            "memory_tokens_count": int(writer_slots),
            "prefix_attention_nontrivial_layer_count": 2,
            "prefix_attention_mass_mean": float(prefix_attention_mass_mean),
            "prefix_artifact_stats": {"memory_token_attention_mass_mean": 0.02},
            "reader_readout_effective_rank": 2.0 if memory_path_variant == "two_level" else 0.0,
            "train_final_memory_slot_effective_rank": float(writer_rank),
            "memory_long_common_mode_energy_ratio": float(common_mode_ratio),
            "memory_long_pairwise_cosine_mean": float(pairwise_cosine),
            "train_grad_probe_writer_task_only_post_unfreeze_median": float(writer_task_grad),
            "train_grad_probe_writer_aux_only_post_unfreeze_median": float(writer_aux_grad),
            "train_grad_probe_writer_total_post_unfreeze_median": float(writer_total_grad),
            "train_grad_probe_writer_task_aux_cosine_post_unfreeze_median": float(writer_task_aux_cosine),
            "train_grad_probe_writer_aux_total_cosine_post_unfreeze_median": float(writer_aux_total_cosine),
            "pilot_alignment_aux_mode": alignment_aux_mode,
            "pilot_reconstruction_aux_mode": reconstruction_aux_mode,
            "train_final_opd_mean_advantage": float(opd_advantage),
            "train_reconstruction_aux_loss_post_unfreeze_median": float(reconstruction_aux_loss),
            "train_barlow_aux_loss_post_unfreeze_median": float(barlow_aux_loss),
            "pilot_active_aux_family": arm_id,
        }
        (task_root / "metrics.json").write_text(json.dumps(metrics) + "\n")
        (task_root / "train_events.json").write_text(
            json.dumps(
                {
                    "events": [
                        {
                            "loss": 1.0,
                            "grad_norm_writer": writer_total_grad,
                            "grad_norm_reader": reader_grad,
                            "grad_norm_fuser": fuser_grad,
                            "grad_probe_writer_task_only_norm": writer_task_grad,
                            "grad_probe_writer_aux_only_norm": writer_aux_grad,
                            "grad_probe_writer_total_norm": writer_total_grad,
                            "grad_probe_writer_task_aux_cosine": writer_task_aux_cosine,
                            "grad_probe_writer_aux_total_cosine": writer_aux_total_cosine,
                        }
                    ]
                }
            )
            + "\n"
        )
        case_rows = []
        for index in range(2):
            case_rows.append(
                {
                    "example_id": f"{task_name}-{index}",
                    "answer_logprob_with_memory": float(
                        branch_logprob if arm_id != "a0_none" else control_logprob
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
                "a0_none",
                task_name,
                task_score=control_scores[task_name],
                control_logprob=control_logprobs[task_name],
                branch_logprob=control_logprobs[task_name],
                writer_rank=8.0,
                common_mode_ratio=0.35,
                pairwise_cosine=0.20,
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
                    writer_rank=8.0,
                    common_mode_ratio=0.35,
                    pairwise_cosine=0.20,
                    writer_aux_grad=0.15,
                    writer_task_aux_cosine=0.05,
                    writer_aux_total_cosine=0.10,
                )

    def test_build_summary_promotes_primary_gain_aux_arm_into_v87(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result_root = Path(temp_dir) / "results"
            result_root.mkdir(parents=True, exist_ok=True)
            self._write_flat_phase(result_root)
            v85_summary_path = Path(temp_dir) / "v8-5-summary.json"
            self._write_v85_reference(v85_summary_path)

            self._write_task(
                result_root,
                "a4_writer_opd_ansctx",
                "gsm8k",
                task_score=0.71,
                control_logprob=-12.0,
                branch_logprob=-7.5,
                writer_rank=10.0,
                common_mode_ratio=0.28,
                pairwise_cosine=0.14,
                writer_aux_grad=0.20,
                writer_task_aux_cosine=0.08,
                writer_aux_total_cosine=0.12,
                alignment_aux_mode="opd_token_ce",
                opd_advantage=0.3,
            )
            self._write_task(
                result_root,
                "a4_writer_opd_ansctx",
                "triviaqa",
                task_score=0.20,
                control_logprob=-5.0,
                branch_logprob=-4.2,
                writer_rank=10.0,
                common_mode_ratio=0.28,
                pairwise_cosine=0.14,
                writer_aux_grad=0.20,
                writer_task_aux_cosine=0.08,
                writer_aux_total_cosine=0.12,
                alignment_aux_mode="opd_token_ce",
                opd_advantage=0.3,
            )

            summary = build_summary(
                result_root=result_root,
                v85_summary_path=v85_summary_path,
            )

            self.assertEqual(summary["best_aux_arm_id"], "a4_writer_opd_ansctx")
            self.assertEqual(summary["best_arm_id"], "a4_writer_opd_ansctx")
            self.assertEqual(summary["recommended_next_step"], "open_v8_7_comparators")
            self.assertEqual(
                summary["comparison_conclusion"],
                "writer_aux_primary_gain_open_v8_7_comparators",
            )
            self.assertEqual(summary["base_for_v8_7_arm_id"], "a4_writer_opd_ansctx")

    def test_build_summary_promotes_writer_health_gain_when_scores_are_flat(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result_root = Path(temp_dir) / "results"
            result_root.mkdir(parents=True, exist_ok=True)
            self._write_flat_phase(result_root)
            v85_summary_path = Path(temp_dir) / "v8-5-summary.json"
            self._write_v85_reference(v85_summary_path)

            for task_name in ALL_TASKS:
                self._write_task(
                    result_root,
                    "a1_barlow",
                    task_name,
                    task_score={"gsm8k": 0.62, "triviaqa": 0.20, "fever": 0.58}[task_name],
                    control_logprob={"gsm8k": -12.0, "triviaqa": -5.0, "fever": -1.5}[task_name],
                    branch_logprob={"gsm8k": -11.7, "triviaqa": -4.9, "fever": -1.4}[task_name],
                    writer_rank=10.5,
                    common_mode_ratio=0.23,
                    pairwise_cosine=0.10,
                    writer_aux_grad=0.18,
                    writer_task_aux_cosine=0.06,
                    writer_aux_total_cosine=0.11,
                    barlow_aux_loss=0.2,
                )

            summary = build_summary(
                result_root=result_root,
                v85_summary_path=v85_summary_path,
            )

            self.assertEqual(summary["best_aux_arm_id"], "a1_barlow")
            self.assertEqual(summary["best_arm_id"], "a1_barlow")
            self.assertEqual(
                summary["comparison_conclusion"],
                "writer_aux_writer_health_gain_open_v8_7_comparators",
            )

    def test_build_summary_keeps_base_route_when_aux_is_flat(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result_root = Path(temp_dir) / "results"
            result_root.mkdir(parents=True, exist_ok=True)
            self._write_flat_phase(result_root)
            v85_summary_path = Path(temp_dir) / "v8-5-summary.json"
            self._write_v85_reference(
                v85_summary_path,
                best_arm_id="b0_no_bridge",
                interface_family="ri0_legacy_prefix",
                bridge_family="BR0",
                next_step="open_v8_6_writer_aux_full_route",
            )

            summary = build_summary(
                result_root=result_root,
                v85_summary_path=v85_summary_path,
            )

            self.assertEqual(summary["best_arm_id"], "a0_none")
            self.assertEqual(
                summary["comparison_conclusion"],
                "writer_aux_flat_keep_base_route_open_v8_7_comparators",
            )
            self.assertEqual(summary["selected_interface_family_for_v8_7"], "ri0_legacy_prefix")
            self.assertEqual(summary["selected_bridge_family_for_v8_7"], "BR0")


if __name__ == "__main__":
    unittest.main()
