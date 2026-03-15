from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class PlanV7V73BridgeSummaryTest(unittest.TestCase):
    def test_summary_ranks_bridge_arms_against_v72_direct_control(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script_path = repo_root / "scripts" / "update_planv7_v7_3_bridge_summary.py"
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            result_root = tmp_path / "results"
            v72_summary_path = tmp_path / "v7-2-summary.json"

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
                            "grad_probe_writer_total_norm": writer_grad * 2.5,
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
                writer_slots: int,
                conditioning_layers: int,
                projector_rank: int,
                projector_mode: str,
                writer_rank: float,
                projected_rank: float,
                common_mode: float,
                pairwise_cosine: float,
                memory_path_variant: str,
                projector_token_source: str,
                reader_queries: int,
                short_slots: int,
            ) -> dict[str, object]:
                return {
                    "benchmark_id": benchmark_id,
                    "task_name": benchmark_id,
                    "task_metric_name": "exact_match",
                    "best_adapt_task_score": task_score,
                    "best_adapt_exact_match": task_score,
                    "delta_answer_logprob": 0.0,
                    "prefix_attention_mass_mean": 0.008,
                    "prefix_attention_mass_mean_by_layer": {"12": 0.004, "13": 0.003, "14": 0.002},
                    "projected_memory_effective_rank": projected_rank,
                    "memory_long_common_mode_energy_ratio": common_mode,
                    "train_final_support_state_effective_rank": 2.0,
                    "train_final_memory_long_effective_rank": writer_rank,
                    "train_final_writer_slot_basis_pairwise_cosine_mean": pairwise_cosine,
                    "writer_memory_slots": writer_slots,
                    "memory_long_slot_norm_std": 0.2,
                    "memory_long_slot_norm_mean": 1.0,
                    "pilot_train_steps": 300,
                    "train_loss_steps_1_50_median": 8.0,
                    "train_loss_tail_50_steps_median": 2.0,
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
                    "pilot_deep_prefix_rank": projector_rank,
                    "pilot_deep_prefix_projector_mode": projector_mode,
                    "pilot_writer_conditioning_layers": conditioning_layers,
                    "task_case_dump_path": task_case_dump_path,
                    "owner_locked_projector_lr": 7.5e-6,
                    "repo_confirmed_v65_projector_lr_reference": 7.5e-5,
                    "owner_override_note": True,
                }

            v72_summary_path.write_text(
                json.dumps(
                    {
                        "primary_arm_ranking": [{"arm_id": "d_w1_shared"}],
                        "winning_depth": "D1",
                        "winning_depth_label": "mid4",
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )

            for task_name, task_score in (("gsm8k", 0.10), ("triviaqa", 0.20)):
                control_dir = result_root / "control" / task_name
                control_case_dump = control_dir / "task_case_dump.jsonl"
                write_generation_dump(control_case_dump, delta=0.0)
                write_json(
                    control_dir / "metrics.json",
                    metrics_payload(
                        benchmark_id=task_name,
                        task_score=task_score,
                        task_case_dump_path=str(control_case_dump),
                        writer_slots=16,
                        conditioning_layers=2,
                        projector_rank=64,
                        projector_mode="shared_low_rank",
                        writer_rank=2.5,
                        projected_rank=4.0,
                        common_mode=0.992,
                        pairwise_cosine=0.80,
                        memory_path_variant="single_level",
                        projector_token_source="writer_slots",
                        reader_queries=0,
                        short_slots=0,
                    ),
                )
                write_events(control_dir / "train_events.json", writer_grad=0.20, projector_grad=0.10, receiver_grad=0.05)

            arm_specs = {
                "b_w3_q8": {
                    "writer_slots": 64,
                    "conditioning_layers": 3,
                    "projector_rank": 128,
                    "projector_mode": "per_layer_low_rank",
                    "writer_rank": 10.0,
                    "projected_rank": 14.0,
                    "common_mode": 0.970,
                    "pairwise_cosine": 0.72,
                    "gsm_delta": 0.0,
                    "trivia_delta": 0.0,
                    "case_delta": 0.15,
                    "reader_queries": 8,
                    "short_slots": 8,
                },
                "b_w3_q16": {
                    "writer_slots": 64,
                    "conditioning_layers": 3,
                    "projector_rank": 128,
                    "projector_mode": "per_layer_low_rank",
                    "writer_rank": 12.0,
                    "projected_rank": 18.0,
                    "common_mode": 0.960,
                    "pairwise_cosine": 0.68,
                    "gsm_delta": 0.05,
                    "trivia_delta": 0.02,
                    "case_delta": 0.30,
                    "reader_queries": 16,
                    "short_slots": 16,
                },
                "b_w3_q16_s8": {
                    "writer_slots": 64,
                    "conditioning_layers": 3,
                    "projector_rank": 128,
                    "projector_mode": "per_layer_low_rank",
                    "writer_rank": 9.0,
                    "projected_rank": 12.0,
                    "common_mode": 0.975,
                    "pairwise_cosine": 0.74,
                    "gsm_delta": 0.0,
                    "trivia_delta": 0.0,
                    "case_delta": 0.10,
                    "reader_queries": 16,
                    "short_slots": 8,
                },
                "b_w4_q16": {
                    "writer_slots": 96,
                    "conditioning_layers": 3,
                    "projector_rank": 256,
                    "projector_mode": "per_layer_low_rank",
                    "writer_rank": 13.0,
                    "projected_rank": 20.0,
                    "common_mode": 0.965,
                    "pairwise_cosine": 0.70,
                    "gsm_delta": -0.01,
                    "trivia_delta": 0.0,
                    "case_delta": 0.05,
                    "reader_queries": 16,
                    "short_slots": 16,
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
                        metrics_payload(
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
                            memory_path_variant="two_level",
                            projector_token_source="short_slots",
                            reader_queries=spec["reader_queries"],
                            short_slots=spec["short_slots"],
                        ),
                    )
                    write_events(
                        task_dir / "train_events.json",
                        writer_grad=0.25,
                        projector_grad=0.15,
                        receiver_grad=0.08,
                    )

            output_json = tmp_path / "summary.json"
            output_report = tmp_path / "summary.md"
            subprocess.run(
                [
                    sys.executable,
                    str(script_path),
                    "--result_root",
                    str(result_root),
                    "--v72_summary",
                    str(v72_summary_path),
                    "--output_json",
                    str(output_json),
                    "--output_report",
                    str(output_report),
                ],
                check=True,
                cwd=repo_root,
            )

            summary = json.loads(output_json.read_text())
            self.assertEqual(summary["direct_control_arm_id"], "d_w1_shared")
            self.assertEqual(summary["bridge_arm_ranking"][0]["arm_id"], "b_w3_q16")
            self.assertEqual(
                summary["comparison_conclusion"],
                "bridge_beats_direct_control_promote_bridge_winner",
            )
            self.assertEqual(
                summary["recommended_next_step"],
                "open_v7_4_forced_consumption_from_bridge_winner",
            )
            self.assertEqual(summary["direct_control"]["memory_path_variant"], "single_level")
            self.assertEqual(summary["arms"]["b_w3_q16"]["memory_path_variant"], "two_level")
            self.assertEqual(summary["arms"]["b_w3_q16"]["projector_token_source"], "short_slots")
            self.assertEqual(
                summary["arms"]["b_w3_q16"]["tasks"]["gsm8k"]["tail_window_source"],
                "train_loss_tail_50_steps_median",
            )


if __name__ == "__main__":
    unittest.main()
