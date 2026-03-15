from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class PlanV7V72DirectBandwidthSummaryTest(unittest.TestCase):
    def test_summary_promotes_perlayer_and_runs_fever_guardrail_on_top_two(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script_path = repo_root / "scripts" / "update_planv7_v7_2_direct_bandwidth_summary.py"
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            result_root = tmp_path / "results"

            def write_json(path: Path, payload: dict[str, object]) -> None:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

            def write_events(path: Path, *, writer_grad: float, projector_grad: float, receiver_grad: float) -> None:
                path.parent.mkdir(parents=True, exist_ok=True)
                events = []
                for step in range(1, 201):
                    alpha = float(step - 1) / 199.0
                    loss = 8.0 + ((2.0 - 8.0) * alpha)
                    events.append(
                        {
                            "step": step,
                            "loss": loss,
                            "writer_frozen": False,
                            "grad_norm_writer": writer_grad,
                            "grad_norm_projector": projector_grad,
                            "grad_norm_receiver_lora": receiver_grad,
                            "grad_probe_writer_task_only_norm": writer_grad,
                            "grad_probe_writer_aux_only_norm": writer_grad / 2.0,
                            "grad_probe_writer_total_norm": writer_grad * 2.0,
                            "grad_probe_writer_task_aux_cosine": 0.1,
                            "grad_probe_writer_task_total_cosine": 0.6,
                            "grad_probe_writer_aux_total_cosine": 0.2,
                            "was_grad_clipped_writer": False,
                            "was_grad_clipped_projector": False,
                            "was_grad_clipped_receiver_lora": False,
                        }
                    )
                path.write_text(json.dumps(events, indent=2) + "\n")

            def write_generation_dump(path: Path, *, delta: float) -> None:
                path.parent.mkdir(parents=True, exist_ok=True)
                rows = []
                for idx in range(4):
                    rows.append(
                        {
                            "example_id": f"row-{idx}",
                            "prediction": "A",
                            "answer_logprob_with_memory": float(idx) + delta,
                        }
                    )
                path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

            def write_classification_dump(path: Path, *, correct_count: int, total: int = 4) -> None:
                path.parent.mkdir(parents=True, exist_ok=True)
                rows = []
                for idx in range(total):
                    is_correct = idx < correct_count
                    predicted = "SUPPORTS" if is_correct else "REFUTES"
                    rows.append(
                        {
                            "example_id": f"row-{idx}",
                            "gold_label": "SUPPORTS",
                            "predicted_label": predicted,
                            "predicted_correct": is_correct,
                            "final_margin": 0.6 if is_correct else -0.4,
                            "candidate_labels": ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"],
                            "candidate_texts": ["Supports", "Refutes", "Not enough info"],
                        }
                    )
                path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

            def generation_metrics(
                *,
                benchmark_id: str,
                task_score: float,
                task_case_dump_path: str,
                writer_slots: int,
                conditioning_layers: int,
                projector_rank: int,
                projector_mode: str,
                writer_rank: float,
                projected_rank: float,
                common_mode: float,
                pairwise_cosine: float,
            ) -> dict[str, object]:
                return {
                    "benchmark_id": benchmark_id,
                    "task_name": benchmark_id,
                    "task_metric_name": "exact_match",
                    "best_adapt_task_score": task_score,
                    "best_adapt_exact_match": task_score,
                    "delta_answer_logprob": 0.0,
                    "prefix_attention_mass_mean": 0.004,
                    "prefix_attention_mass_mean_by_layer": {"12": 0.003, "13": 0.002},
                    "projected_memory_effective_rank": projected_rank,
                    "memory_long_common_mode_energy_ratio": common_mode,
                    "train_final_support_state_effective_rank": 2.0,
                    "train_final_memory_long_effective_rank": writer_rank,
                    "train_final_writer_slot_basis_pairwise_cosine_mean": pairwise_cosine,
                    "writer_memory_slots": writer_slots,
                    "memory_long_slot_norm_std": 0.2,
                    "memory_long_slot_norm_mean": 1.0,
                    "pilot_train_steps": 200,
                    "train_loss_steps_1_50_median": 8.0,
                    "train_loss_steps_451_500_median": 2.0,
                    "snapshot_metrics": [{"step": 0, "prefix_l2": 1.0}, {"step": 200, "prefix_l2": 2.0}],
                    "pilot_bridge_mode": "writer_direct",
                    "pilot_memory_path_variant": "single_level",
                    "pilot_deep_prefix_layers": [12, 13, 14, 15],
                    "pilot_receiver_lora_target_layers": [12, 13, 14, 15],
                    "pilot_deep_prefix_rank": projector_rank,
                    "pilot_deep_prefix_projector_mode": projector_mode,
                    "pilot_writer_conditioning_layers": conditioning_layers,
                    "task_case_dump_path": task_case_dump_path,
                    "owner_locked_projector_lr": 7.5e-6,
                    "repo_confirmed_v65_projector_lr_reference": 7.5e-5,
                    "owner_override_note": True,
                }

            def fever_metrics(
                *,
                task_score: float,
                task_case_dump_path: str,
                writer_slots: int,
                conditioning_layers: int,
                projector_rank: int,
                projector_mode: str,
                writer_rank: float,
                projected_rank: float,
                common_mode: float,
                pairwise_cosine: float,
            ) -> dict[str, object]:
                return {
                    "benchmark_id": "fever",
                    "task_name": "fever",
                    "task_metric_name": "accuracy",
                    "best_adapt_task_score": task_score,
                    "best_adapt_exact_match": task_score,
                    "delta_answer_logprob": 0.0,
                    "prefix_attention_mass_mean": 0.004,
                    "prefix_attention_mass_mean_by_layer": {"12": 0.003, "13": 0.002},
                    "projected_memory_effective_rank": projected_rank,
                    "memory_long_common_mode_energy_ratio": common_mode,
                    "train_final_support_state_effective_rank": 2.0,
                    "train_final_memory_long_effective_rank": writer_rank,
                    "train_final_writer_slot_basis_pairwise_cosine_mean": pairwise_cosine,
                    "writer_memory_slots": writer_slots,
                    "memory_long_slot_norm_std": 0.2,
                    "memory_long_slot_norm_mean": 1.0,
                    "pilot_train_steps": 200,
                    "train_loss_steps_1_50_median": 8.0,
                    "train_loss_steps_451_500_median": 2.0,
                    "snapshot_metrics": [{"step": 0, "prefix_l2": 1.0}, {"step": 200, "prefix_l2": 2.0}],
                    "pilot_bridge_mode": "writer_direct",
                    "pilot_memory_path_variant": "single_level",
                    "pilot_deep_prefix_layers": [12, 13, 14, 15],
                    "pilot_receiver_lora_target_layers": [12, 13, 14, 15],
                    "pilot_deep_prefix_rank": projector_rank,
                    "pilot_deep_prefix_projector_mode": projector_mode,
                    "pilot_writer_conditioning_layers": conditioning_layers,
                    "task_case_dump_path": task_case_dump_path,
                    "owner_locked_projector_lr": 7.5e-6,
                    "repo_confirmed_v65_projector_lr_reference": 7.5e-5,
                    "owner_override_note": True,
                }

            for task_name, task_score in (("gsm8k", 0.10), ("triviaqa", 0.20)):
                control_dir = result_root / "control" / task_name
                control_case_dump = control_dir / "task_case_dump.jsonl"
                write_generation_dump(control_case_dump, delta=0.0)
                write_json(
                    control_dir / "metrics.json",
                    generation_metrics(
                        benchmark_id=task_name,
                        task_score=task_score,
                        task_case_dump_path=str(control_case_dump),
                        writer_slots=16,
                        conditioning_layers=2,
                        projector_rank=64,
                        projector_mode="shared_low_rank",
                        writer_rank=2.2,
                        projected_rank=3.0,
                        common_mode=0.992,
                        pairwise_cosine=0.80,
                    ),
                )
                write_events(control_dir / "train_events.json", writer_grad=0.0, projector_grad=0.0, receiver_grad=0.0)

            control_fever_dir = result_root / "control" / "fever"
            control_fever_dump = control_fever_dir / "task_case_dump.jsonl"
            write_classification_dump(control_fever_dump, correct_count=2)
            write_json(
                control_fever_dir / "metrics.json",
                fever_metrics(
                    task_score=0.50,
                    task_case_dump_path=str(control_fever_dump),
                    writer_slots=16,
                    conditioning_layers=2,
                    projector_rank=64,
                    projector_mode="shared_low_rank",
                    writer_rank=2.2,
                    projected_rank=3.0,
                    common_mode=0.992,
                    pairwise_cosine=0.80,
                ),
            )
            write_events(control_fever_dir / "train_events.json", writer_grad=0.0, projector_grad=0.0, receiver_grad=0.0)

            arm_specs = {
                "d_w1_shared": {
                    "writer_slots": 16,
                    "conditioning_layers": 2,
                    "projector_rank": 64,
                    "projector_mode": "shared_low_rank",
                    "writer_rank": 2.5,
                    "projected_rank": 4.0,
                    "common_mode": 0.991,
                    "pairwise_cosine": 0.82,
                    "gsm_delta": 0.0,
                    "trivia_delta": 0.0,
                    "case_delta": 0.05,
                },
                "d_w2_shared": {
                    "writer_slots": 32,
                    "conditioning_layers": 2,
                    "projector_rank": 64,
                    "projector_mode": "shared_low_rank",
                    "writer_rank": 5.2,
                    "projected_rank": 7.0,
                    "common_mode": 0.975,
                    "pairwise_cosine": 0.74,
                    "gsm_delta": 0.0,
                    "trivia_delta": 0.0,
                    "case_delta": 0.10,
                    "fever_score": 0.55,
                    "fever_correct": 2,
                },
                "d_w2_perlayer": {
                    "writer_slots": 32,
                    "conditioning_layers": 2,
                    "projector_rank": 128,
                    "projector_mode": "per_layer_low_rank",
                    "writer_rank": 6.5,
                    "projected_rank": 10.0,
                    "common_mode": 0.960,
                    "pairwise_cosine": 0.70,
                    "gsm_delta": 0.05,
                    "trivia_delta": 0.0,
                    "case_delta": 0.30,
                    "fever_score": 0.75,
                    "fever_correct": 3,
                },
            }

            for arm_id, spec in arm_specs.items():
                for task_name, base_score in (("gsm8k", 0.10), ("triviaqa", 0.20)):
                    task_dir = result_root / arm_id / task_name
                    case_dump = task_dir / "task_case_dump.jsonl"
                    write_generation_dump(case_dump, delta=spec["case_delta"])
                    task_delta = spec["gsm_delta"] if task_name == "gsm8k" else spec["trivia_delta"]
                    write_json(
                        task_dir / "metrics.json",
                        generation_metrics(
                            benchmark_id=task_name,
                            task_score=base_score + task_delta,
                            task_case_dump_path=str(case_dump),
                            writer_slots=spec["writer_slots"],
                            conditioning_layers=spec["conditioning_layers"],
                            projector_rank=spec["projector_rank"],
                            projector_mode=spec["projector_mode"],
                            writer_rank=spec["writer_rank"],
                            projected_rank=spec["projected_rank"],
                            common_mode=spec["common_mode"],
                            pairwise_cosine=spec["pairwise_cosine"],
                        ),
                    )
                    write_events(
                        task_dir / "train_events.json",
                        writer_grad=0.02,
                        projector_grad=0.05,
                        receiver_grad=0.02,
                    )

            for arm_id in ("d_w2_shared", "d_w2_perlayer"):
                spec = arm_specs[arm_id]
                task_dir = result_root / arm_id / "fever"
                case_dump = task_dir / "task_case_dump.jsonl"
                write_classification_dump(case_dump, correct_count=spec["fever_correct"])
                write_json(
                    task_dir / "metrics.json",
                    fever_metrics(
                        task_score=spec["fever_score"],
                        task_case_dump_path=str(case_dump),
                        writer_slots=spec["writer_slots"],
                        conditioning_layers=spec["conditioning_layers"],
                        projector_rank=spec["projector_rank"],
                        projector_mode=spec["projector_mode"],
                        writer_rank=spec["writer_rank"],
                        projected_rank=spec["projected_rank"],
                        common_mode=spec["common_mode"],
                        pairwise_cosine=spec["pairwise_cosine"],
                    ),
                )
                write_events(
                    task_dir / "train_events.json",
                    writer_grad=0.02,
                    projector_grad=0.05,
                    receiver_grad=0.02,
                )

            output_json = tmp_path / "summary.json"
            output_report = tmp_path / "summary.md"
            subprocess.run(
                [
                    sys.executable,
                    str(script_path),
                    "--result_root",
                    str(result_root),
                    "--output_json",
                    str(output_json),
                    "--output_report",
                    str(output_report),
                ],
                check=True,
            )

            summary = json.loads(output_json.read_text())
            self.assertEqual(
                summary["comparison_conclusion"],
                "promote_w2_perlayer_direct_and_bridge_control",
            )
            self.assertEqual(summary["promoted_arms"][0], "d_w2_perlayer")
            self.assertEqual(summary["promoted_arms"][1], "d_w2_shared")
            self.assertTrue(summary["acceptance"]["fever_guardrail_complete"])
            self.assertTrue(summary["evidence"]["bandwidth_strict_improvement_observed"])
            self.assertTrue(summary["evidence"]["projector_scaling_usefulness_change_observed"])
            self.assertEqual(
                summary["fever_guardrail"]["evaluated_arms"],
                ["d_w2_perlayer", "d_w2_shared"],
            )
            self.assertIn("d_w2_perlayer", summary["fever_guardrail"]["branches"])
            self.assertIn("d_w2_shared", summary["fever_guardrail"]["branches"])
            self.assertNotIn("d_w1_shared", summary["fever_guardrail"]["branches"])


if __name__ == "__main__":
    unittest.main()
