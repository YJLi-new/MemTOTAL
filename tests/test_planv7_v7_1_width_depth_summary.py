from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class PlanV7V71WidthDepthSummaryTest(unittest.TestCase):
    def test_summary_selects_mid4_when_primary_scores_tie_and_writer_metrics_are_better(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script_path = repo_root / "scripts" / "update_planv7_v7_1_width_depth_summary.py"
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

            def write_task_case_dump(path: Path, *, delta: float) -> None:
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

            def metrics(
                *,
                benchmark_id: str,
                task_score: float,
                task_case_dump_path: str,
                writer_slots: int,
                conditioning_layers: int,
                depth_layers: list[int],
                projector_rank: int,
                writer_rank: float,
                projected_rank: float,
                common_mode: float,
                pairwise_cosine: float,
                metric_name: str = "exact_match",
            ) -> dict[str, object]:
                return {
                    "benchmark_id": benchmark_id,
                    "task_name": benchmark_id,
                    "task_metric_name": metric_name,
                    "best_adapt_task_score": task_score,
                    "best_adapt_exact_match": task_score,
                    "delta_answer_logprob": 0.0,
                    "prefix_attention_mass_mean": 0.004,
                    "prefix_attention_mass_mean_by_layer": {str(depth_layers[0]): 0.003, str(depth_layers[1]): 0.002},
                    "projected_memory_effective_rank": projected_rank,
                    "memory_long_common_mode_energy_ratio": common_mode,
                    "train_final_support_state_effective_rank": 1.8,
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
                    "pilot_deep_prefix_layers": depth_layers,
                    "pilot_receiver_lora_target_layers": depth_layers,
                    "pilot_deep_prefix_rank": projector_rank,
                    "pilot_writer_conditioning_layers": conditioning_layers,
                    "task_case_dump_path": task_case_dump_path,
                    "owner_locked_projector_lr": 7.5e-6,
                    "repo_confirmed_v65_projector_lr_reference": 7.5e-5,
                    "owner_override_note": True,
                }

            for task_name, task_score in (("gsm8k", 0.10), ("triviaqa", 0.20)):
                control_dir = result_root / "control" / task_name
                control_case_dump = control_dir / "task_case_dump.jsonl"
                write_task_case_dump(control_case_dump, delta=0.0)
                write_json(
                    control_dir / "metrics.json",
                    metrics(
                        benchmark_id=task_name,
                        task_score=task_score,
                        task_case_dump_path=str(control_case_dump),
                        writer_slots=8,
                        conditioning_layers=1,
                        depth_layers=[0, 1, 2, 3],
                        projector_rank=32,
                        writer_rank=2.0,
                        projected_rank=2.5,
                        common_mode=0.994,
                        pairwise_cosine=0.85,
                    ),
                )
                write_events(control_dir / "train_events.json", writer_grad=0.0, projector_grad=0.0, receiver_grad=0.0)

            arm_specs = {
                "s00": {
                    "writer_slots": 8,
                    "conditioning_layers": 1,
                    "depth_layers": [0, 1, 2, 3],
                    "projector_rank": 32,
                    "writer_rank": 1.0,
                    "projected_rank": 4.5,
                    "common_mode": 0.9995,
                    "pairwise_cosine": 0.95,
                    "delta": 0.05,
                },
                "s01": {
                    "writer_slots": 8,
                    "conditioning_layers": 1,
                    "depth_layers": [12, 13, 14, 15],
                    "projector_rank": 32,
                    "writer_rank": 2.6,
                    "projected_rank": 3.0,
                    "common_mode": 0.992,
                    "pairwise_cosine": 0.70,
                    "delta": 0.20,
                },
                "s10": {
                    "writer_slots": 16,
                    "conditioning_layers": 2,
                    "depth_layers": [0, 1, 2, 3],
                    "projector_rank": 64,
                    "writer_rank": 1.1,
                    "projected_rank": 5.0,
                    "common_mode": 0.9992,
                    "pairwise_cosine": 0.94,
                    "delta": 0.04,
                },
                "s11": {
                    "writer_slots": 16,
                    "conditioning_layers": 2,
                    "depth_layers": [12, 13, 14, 15],
                    "projector_rank": 64,
                    "writer_rank": 3.2,
                    "projected_rank": 4.0,
                    "common_mode": 0.991,
                    "pairwise_cosine": 0.68,
                    "delta": 0.22,
                },
            }

            for arm_id, spec in arm_specs.items():
                for task_name, task_score in (("gsm8k", 0.10), ("triviaqa", 0.20)):
                    task_dir = result_root / arm_id / task_name
                    case_dump = task_dir / "task_case_dump.jsonl"
                    write_task_case_dump(case_dump, delta=spec["delta"])
                    write_json(
                        task_dir / "metrics.json",
                        metrics(
                            benchmark_id=task_name,
                            task_score=task_score,
                            task_case_dump_path=str(case_dump),
                            writer_slots=spec["writer_slots"],
                            conditioning_layers=spec["conditioning_layers"],
                            depth_layers=spec["depth_layers"],
                            projector_rank=spec["projector_rank"],
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
            self.assertEqual(summary["comparison_conclusion"], "select_mid4_for_v7_2")
            self.assertEqual(summary["winning_depth"], "D1")
            self.assertTrue(summary["acceptance"]["single_winning_depth_selected"])
            self.assertTrue(summary["acceptance"]["fever_not_used_to_override_primary"])
            self.assertEqual(summary["arms"]["s11"]["writer_family"], "W1")
            self.assertEqual(summary["arms"]["s11"]["projector_rank"], 64)
            self.assertTrue(summary["arms"]["s11"]["tasks"]["gsm8k"]["writer_memory_not_collapsed_strict"])
            self.assertFalse(summary["arms"]["s10"]["tasks"]["gsm8k"]["writer_memory_not_collapsed_strict"])
            self.assertIn(summary["arm_ranking"][0]["arm_id"], {"s01", "s11"})


if __name__ == "__main__":
    unittest.main()
