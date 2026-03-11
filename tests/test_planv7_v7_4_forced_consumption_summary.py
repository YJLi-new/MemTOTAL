from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class PlanV7V74ForcedConsumptionSummaryTest(unittest.TestCase):
    def test_summary_promotes_actual_score_branch_and_falls_back_to_v73_control_source(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script_path = repo_root / "scripts" / "update_planv7_v7_4_forced_consumption_summary.py"
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            result_root = tmp_path / "results"
            v73_summary_path = tmp_path / "v7-3-summary.json"

            def write_json(path: Path, payload: dict[str, object]) -> None:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

            def write_events(
                path: Path,
                *,
                writer_grad: float,
                projector_grad: float,
                receiver_grad: float,
            ) -> None:
                path.parent.mkdir(parents=True, exist_ok=True)
                events = []
                for step in range(1, 301):
                    alpha = float(step - 1) / 299.0
                    loss = 9.0 + ((3.0 - 9.0) * alpha)
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

            def metrics_payload(
                *,
                benchmark_id: str,
                task_score: float,
                task_case_dump_path: str,
                delta_answer_logprob: float,
                writer_slots: int,
                reader_queries: int,
                short_slots: int,
                active_writer_family: str,
                active_bridge_family: str,
                active_projector_family: str,
                memory_path_variant: str = "two_level",
                projector_token_source: str = "short_slots",
            ) -> dict[str, object]:
                return {
                    "benchmark_id": benchmark_id,
                    "task_name": benchmark_id,
                    "task_metric_name": "exact_match",
                    "best_adapt_task_score": task_score,
                    "best_adapt_exact_match": task_score,
                    "delta_answer_logprob": delta_answer_logprob,
                    "prefix_attention_mass_mean": 0.012,
                    "prefix_attention_mass_mean_by_layer": {"12": 0.004, "13": 0.003, "14": 0.003, "15": 0.002},
                    "projected_memory_effective_rank": 18.0,
                    "memory_long_common_mode_energy_ratio": 0.96,
                    "train_final_support_state_effective_rank": 2.0,
                    "train_final_memory_long_effective_rank": 8.0,
                    "train_final_writer_slot_basis_pairwise_cosine_mean": 0.55,
                    "writer_memory_slots": writer_slots,
                    "memory_long_slot_norm_std": 0.25,
                    "memory_long_slot_norm_mean": 1.0,
                    "pilot_train_steps": 300,
                    "train_loss_steps_1_50_median": 9.0,
                    "train_loss_tail_50_steps_median": 3.0,
                    "train_loss_steps_451_500_median": 0.0,
                    "snapshot_metrics": [{"step": 0, "prefix_l2": 1.0}, {"step": 300, "prefix_l2": 2.0}],
                    "pilot_bridge_mode": "writer_direct",
                    "pilot_memory_path_variant": memory_path_variant,
                    "pilot_projector_token_source": projector_token_source,
                    "pilot_reader_context_mode": "prompt_summary",
                    "pilot_reader_num_queries": reader_queries,
                    "pilot_fuser_short_slots": short_slots,
                    "pilot_deep_prefix_layers": [12, 13, 14, 15],
                    "pilot_receiver_lora_target_layers": [12, 13, 14, 15],
                    "pilot_deep_prefix_rank": 128,
                    "pilot_deep_prefix_projector_mode": "per_layer_low_rank",
                    "pilot_writer_conditioning_layers": 3,
                    "task_case_dump_path": task_case_dump_path,
                    "pilot_active_writer_family": active_writer_family,
                    "pilot_active_bridge_family": active_bridge_family,
                    "pilot_active_projector_family": active_projector_family,
                }

            v73_summary_path.write_text(
                json.dumps(
                    {
                        "bridge_arm_ranking": [{"arm_id": "b_w3_q16"}],
                        "direct_control_arm_id": "d_w1_shared",
                        "winning_depth": "D1",
                        "winning_depth_label": "mid4",
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )

            control_specs = {
                "gsm8k": 0.10,
                "triviaqa": 0.20,
            }
            for task_name, task_score in control_specs.items():
                control_dir = result_root / "control" / task_name
                control_case_dump = control_dir / "task_case_dump.jsonl"
                write_generation_dump(control_case_dump, delta=0.0)
                write_json(
                    control_dir / "metrics.json",
                    metrics_payload(
                        benchmark_id=task_name,
                        task_score=task_score,
                        task_case_dump_path=str(control_case_dump),
                        delta_answer_logprob=0.0,
                        writer_slots=64,
                        reader_queries=16,
                        short_slots=16,
                        active_writer_family="W3",
                        active_bridge_family="B2",
                        active_projector_family="P2",
                    ),
                )
                write_events(control_dir / "train_events.json", writer_grad=0.25, projector_grad=0.12, receiver_grad=0.08)

            arm_payloads = {
                ("f1_num_mask", "gsm8k"): {"score": 0.18, "delta": 0.25},
                ("f2_rx_only", "gsm8k"): {"score": 0.10, "delta": 0.30},
                ("f2_rx_only", "triviaqa"): {"score": 0.20, "delta": 0.10},
                ("f3_anneal", "gsm8k"): {"score": 0.10, "delta": 0.05},
                ("f4_dyn_budget", "gsm8k"): {"score": 0.10, "delta": 0.02},
                ("f4_dyn_budget", "triviaqa"): {"score": 0.19, "delta": 0.01},
            }
            for (arm_id, task_name), payload in arm_payloads.items():
                task_dir = result_root / arm_id / task_name
                case_dump = task_dir / "task_case_dump.jsonl"
                write_generation_dump(case_dump, delta=payload["delta"])
                writer_slots = 32 if (arm_id, task_name) == ("f4_dyn_budget", "triviaqa") else 64
                reader_queries = 8 if (arm_id, task_name) == ("f4_dyn_budget", "triviaqa") else 16
                short_slots = 8 if (arm_id, task_name) == ("f4_dyn_budget", "triviaqa") else 16
                active_writer_family = "W2" if (arm_id, task_name) == ("f4_dyn_budget", "triviaqa") else "W3"
                write_json(
                    task_dir / "metrics.json",
                    metrics_payload(
                        benchmark_id=task_name,
                        task_score=payload["score"],
                        task_case_dump_path=str(case_dump),
                        delta_answer_logprob=payload["delta"],
                        writer_slots=writer_slots,
                        reader_queries=reader_queries,
                        short_slots=short_slots,
                        active_writer_family=active_writer_family,
                        active_bridge_family="B2",
                        active_projector_family="P2",
                    ),
                )
                write_events(task_dir / "train_events.json", writer_grad=0.30, projector_grad=0.15, receiver_grad=0.09)

            output_json = tmp_path / "summary.json"
            output_report = tmp_path / "summary.md"
            subprocess.run(
                [
                    sys.executable,
                    str(script_path),
                    "--result_root",
                    str(result_root),
                    "--v73_summary",
                    str(v73_summary_path),
                    "--output_json",
                    str(output_json),
                    "--output_report",
                    str(output_report),
                ],
                check=True,
                cwd=repo_root,
            )

            summary = json.loads(output_json.read_text())
            self.assertEqual(summary["control_source_arm_id"], "b_w3_q16")
            self.assertEqual(summary["forced_consumption_arm_ranking"][0]["arm_id"], "f1_num_mask")
            self.assertEqual(
                summary["comparison_conclusion"],
                "forced_consumption_changes_primary_scores_move_to_v7_5",
            )
            self.assertEqual(summary["recommended_next_step"], "open_v7_5_targeted_aux_revisit")
            self.assertEqual(summary["base_for_v7_5_arm_id"], "f1_num_mask")
            self.assertEqual(summary["base_for_v7_5_source_phase"], "v7_4")
            self.assertTrue(summary["arms"]["f1_num_mask"]["acceptance_qualified"])
            self.assertTrue(summary["arms"]["f2_rx_only"]["diagnostic_only"])
            self.assertFalse(summary["arms"]["f4_dyn_budget"]["acceptance_qualified"])


if __name__ == "__main__":
    unittest.main()
