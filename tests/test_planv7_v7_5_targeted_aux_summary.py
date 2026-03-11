from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class PlanV7V75TargetedAuxSummaryTest(unittest.TestCase):
    def test_summary_prefers_non_regressive_primary_gain_and_tracks_strict_metric_gains(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script_path = repo_root / "scripts" / "update_planv7_v7_5_targeted_aux_summary.py"
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            result_root = tmp_path / "results"
            v74_summary_path = tmp_path / "v7-4-summary.json"

            def write_json(path: Path, payload: dict[str, object]) -> None:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

            def write_events(path: Path, *, writer_grad: float, projector_grad: float) -> None:
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
                            "grad_norm_receiver_lora": 0.08,
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
                writer_rank_fraction: float,
                writer_rank: float,
                common_mode_ratio: float,
                active_aux_family: str,
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
                    "memory_long_common_mode_energy_ratio": common_mode_ratio,
                    "train_final_support_state_effective_rank": 2.0,
                    "train_final_memory_long_effective_rank": 8.0,
                    "train_final_writer_slot_basis_pairwise_cosine_mean": 0.25,
                    "writer_memory_slots": 64,
                    "memory_long_slot_norm_std": 0.25,
                    "memory_long_slot_norm_mean": 1.0,
                    "pilot_train_steps": 300,
                    "train_loss_steps_1_50_median": 9.0,
                    "train_loss_tail_50_steps_median": 3.0,
                    "train_loss_steps_451_500_median": 0.0,
                    "snapshot_metrics": [{"step": 0, "prefix_l2": 1.0}, {"step": 300, "prefix_l2": 2.0}],
                    "pilot_bridge_mode": "writer_direct",
                    "pilot_memory_path_variant": "two_level",
                    "pilot_projector_token_source": "short_slots",
                    "pilot_reader_context_mode": "prompt_summary",
                    "pilot_reader_num_queries": 16,
                    "pilot_fuser_short_slots": 16,
                    "pilot_deep_prefix_layers": [12, 13, 14, 15],
                    "pilot_receiver_lora_target_layers": [12, 13, 14, 15],
                    "pilot_deep_prefix_rank": 128,
                    "pilot_deep_prefix_projector_mode": "per_layer_low_rank",
                    "pilot_writer_conditioning_layers": 3,
                    "task_case_dump_path": task_case_dump_path,
                    "pilot_active_writer_family": "W3",
                    "pilot_active_bridge_family": "B2",
                    "pilot_active_projector_family": "P2",
                    "pilot_active_aux_family": active_aux_family,
                    "writer_memory_not_collapsed_strict": writer_rank_fraction >= 0.125 and common_mode_ratio <= 0.985,
                    "writer_rank_fraction": writer_rank_fraction,
                    "writer_memory_slot_effective_rank": writer_rank,
                    "slot_pairwise_cosine": 0.20,
                    "slot_pairwise_cosine_present": True,
                }

            v74_summary_path.write_text(
                json.dumps(
                    {
                        "base_for_v7_5_arm_id": "b_w3_q16",
                        "base_for_v7_5_source_phase": "v7_3",
                        "control_source_arm_id": "b_w3_q16",
                        "direct_control_arm_id": "d_w1_shared",
                        "winning_depth": "D1",
                        "winning_depth_label": "mid4",
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )

            baseline_scores = {"gsm8k": 0.10, "triviaqa": 0.20}
            for task_name, task_score in baseline_scores.items():
                task_dir = result_root / "a0_baseline" / task_name
                case_dump = task_dir / "task_case_dump.jsonl"
                write_generation_dump(case_dump, delta=0.0)
                write_json(
                    task_dir / "metrics.json",
                    metrics_payload(
                        benchmark_id=task_name,
                        task_score=task_score,
                        task_case_dump_path=str(case_dump),
                        delta_answer_logprob=0.0,
                        writer_rank_fraction=0.03,
                        writer_rank=2.0,
                        common_mode_ratio=0.995,
                        active_aux_family="A0_L5_baseline",
                    ),
                )
                write_events(task_dir / "train_events.json", writer_grad=0.20, projector_grad=0.10)

            arm_specs = {
                ("a1_reconstruction", "gsm8k"): {"score": 0.16, "delta": 0.40, "rank_fraction": 0.08, "rank": 4.0, "common": 0.990},
                ("a1_reconstruction", "triviaqa"): {"score": 0.20, "delta": 0.10, "rank_fraction": 0.04, "rank": 2.5, "common": 0.994},
                ("a2_vicreg", "gsm8k"): {"score": 0.10, "delta": 0.20, "rank_fraction": 0.09, "rank": 4.5, "common": 0.980},
                ("a2_vicreg", "triviaqa"): {"score": 0.20, "delta": 0.15, "rank_fraction": 0.07, "rank": 3.5, "common": 0.982},
                ("a3_contrastive", "gsm8k"): {"score": 0.13, "delta": 0.30, "rank_fraction": 0.05, "rank": 2.5, "common": 0.993},
                ("a3_contrastive", "triviaqa"): {"score": 0.17, "delta": -0.05, "rank_fraction": 0.05, "rank": 2.5, "common": 0.993},
                ("a4_reconstruction_vicreg", "gsm8k"): {"score": 0.10, "delta": 0.12, "rank_fraction": 0.10, "rank": 5.0, "common": 0.979},
                ("a4_reconstruction_vicreg", "triviaqa"): {"score": 0.20, "delta": 0.10, "rank_fraction": 0.08, "rank": 4.0, "common": 0.981},
                ("a5_barlow", "gsm8k"): {"score": 0.10, "delta": 0.05, "rank_fraction": 0.04, "rank": 2.5, "common": 0.994},
                ("a5_barlow", "triviaqa"): {"score": 0.20, "delta": 0.02, "rank_fraction": 0.04, "rank": 2.5, "common": 0.994},
            }
            family_labels = {
                "a1_reconstruction": "A1_L5_plus_reconstruction",
                "a2_vicreg": "A2_L5_plus_vicreg",
                "a3_contrastive": "A3_L5_plus_contrastive",
                "a4_reconstruction_vicreg": "A4_L5_plus_reconstruction_plus_vicreg",
                "a5_barlow": "A5_L5_plus_barlow",
            }
            for (arm_id, task_name), payload in arm_specs.items():
                task_dir = result_root / arm_id / task_name
                case_dump = task_dir / "task_case_dump.jsonl"
                write_generation_dump(case_dump, delta=payload["delta"])
                write_json(
                    task_dir / "metrics.json",
                    metrics_payload(
                        benchmark_id=task_name,
                        task_score=payload["score"],
                        task_case_dump_path=str(case_dump),
                        delta_answer_logprob=payload["delta"],
                        writer_rank_fraction=payload["rank_fraction"],
                        writer_rank=payload["rank"],
                        common_mode_ratio=payload["common"],
                        active_aux_family=family_labels[arm_id],
                    ),
                )
                write_events(task_dir / "train_events.json", writer_grad=0.25, projector_grad=0.12)

            output_json = tmp_path / "summary.json"
            output_report = tmp_path / "summary.md"
            subprocess.run(
                [
                    sys.executable,
                    str(script_path),
                    "--result_root",
                    str(result_root),
                    "--v74_summary",
                    str(v74_summary_path),
                    "--output_json",
                    str(output_json),
                    "--output_report",
                    str(output_report),
                ],
                check=True,
                cwd=repo_root,
            )

            summary = json.loads(output_json.read_text())
            self.assertEqual(summary["comparison_conclusion"], "aux_revisit_finds_primary_gain_open_v7_6")
            self.assertEqual(summary["recommended_next_step"], "open_v7_6_multiseed_confirmation")
            self.assertEqual(summary["base_for_v7_6_arm_id"], "a1_reconstruction")
            self.assertEqual(summary["aux_arm_ranking"][0]["arm_id"], "a1_reconstruction")
            self.assertTrue(summary["arms"]["a1_reconstruction"]["acceptance_qualified"])
            self.assertTrue(summary["arms"]["a2_vicreg"]["acceptance_qualified"])
            self.assertFalse(summary["arms"]["a3_contrastive"]["acceptance_qualified"])
            self.assertTrue(summary["evidence"]["reconstruction_branch_acceptance_qualified"])


if __name__ == "__main__":
    unittest.main()
