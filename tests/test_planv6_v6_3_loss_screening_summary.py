from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class PlanV6LossScreeningSummaryTest(unittest.TestCase):
    def test_loss_screening_summary_ranks_best_family(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script_path = repo_root / "scripts" / "update_planv6_v6_3_loss_screening_summary.py"
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            result_root = tmp_path / "results"

            def write_case_dump(path: Path, rows: list[dict[str, object]]) -> Path:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("".join(json.dumps(row) + "\n" for row in rows))
                return path

            def write_control(task_name: str) -> None:
                control_dir = result_root / task_name / "control"
                control_dir.mkdir(parents=True, exist_ok=True)
                control_rows = [
                    {
                        "example_id": "1",
                        "gold_label": "SUPPORTS",
                        "predicted_label": "REFUTES",
                        "predicted_correct": False,
                        "task_score": 0.0,
                        "final_margin": -0.2,
                        "candidate_labels": ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"],
                        "candidate_texts": ["Supports", "Refutes", "Not enough info"],
                    }
                ]
                case_path = write_case_dump(control_dir / "task_case_dump.jsonl", control_rows)
                metrics = {
                    "task_name": task_name,
                    "benchmark_id": task_name,
                    "task_metric_name": "accuracy",
                    "best_adapt_task_score": 0.0,
                    "best_adapt_exact_match": 0.0,
                    "prefix_attention_mass_mean_by_layer": {"0": 0.0, "1": 0.0},
                    "memory_long_common_mode_energy_ratio": 0.999999,
                    "train_final_support_state_effective_rank": 1.0,
                    "train_final_memory_long_effective_rank": 1.0,
                    "task_case_dump_path": str(case_path),
                }
                (control_dir / "metrics.json").write_text(json.dumps(metrics) + "\n")
                (control_dir / "train_events.json").write_text(json.dumps([]) + "\n")

            def build_events(*, writer_grad: float, task_only_grad: float, aux_only_grad: float) -> list[dict[str, object]]:
                events: list[dict[str, object]] = []
                for step in range(1, 101):
                    events.append(
                        {
                            "step": step,
                            "loss": 4.0 - (0.02 * step),
                            "grad_norm_writer": writer_grad,
                            "grad_norm_projector": 0.01,
                            "grad_norm_receiver_lora": 0.01,
                            "writer_frozen": False,
                            "gradient_probe_step_active": True,
                            "grad_probe_writer_task_only_norm": task_only_grad,
                            "grad_probe_writer_aux_only_norm": aux_only_grad,
                            "grad_probe_writer_total_norm": max(task_only_grad + aux_only_grad, writer_grad),
                            "grad_probe_writer_task_aux_cosine": 0.2,
                            "grad_probe_writer_task_total_cosine": 0.8,
                            "grad_probe_writer_aux_total_cosine": 0.4,
                            "was_grad_clipped_writer": False,
                            "was_grad_clipped_projector": False,
                            "was_grad_clipped_receiver_lora": False,
                        }
                    )
                return events

            def write_combo(
                combo_id: str,
                *,
                task_score: float,
                support_rank: float,
                common_mode_ratio: float,
                positive_margin: bool,
                writer_grad: float,
                task_only_grad: float,
                aux_only_grad: float,
            ) -> None:
                for task_name in ("fever", "gsm8k", "narrativeqa"):
                    combo_dir = result_root / task_name / combo_id
                    combo_dir.mkdir(parents=True, exist_ok=True)
                    rows = [
                        {
                            "example_id": "1",
                            "gold_label": "SUPPORTS",
                            "predicted_label": "SUPPORTS" if positive_margin else "REFUTES",
                            "predicted_correct": bool(positive_margin),
                            "task_score": task_score,
                            "final_margin": 0.4 if positive_margin else -0.1,
                            "candidate_labels": ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"],
                            "candidate_texts": ["Supports", "Refutes", "Not enough info"],
                        }
                    ]
                    case_path = write_case_dump(combo_dir / "task_case_dump.jsonl", rows)
                    metrics = {
                        "task_name": task_name,
                        "benchmark_id": task_name,
                        "task_metric_name": "accuracy",
                        "best_adapt_task_score": task_score,
                        "best_adapt_exact_match": task_score,
                        "prefix_attention_mass_mean": 0.01,
                        "prefix_attention_mass_mean_by_layer": {"0": 0.01, "1": 0.01},
                        "projected_memory_effective_rank": 4.0,
                        "memory_long_common_mode_energy_ratio": common_mode_ratio,
                        "train_final_support_state_effective_rank": support_rank,
                        "train_final_memory_long_effective_rank": max(1.0, support_rank),
                        "task_case_dump_path": str(case_path),
                        "pilot_train_steps": 100,
                    }
                    (combo_dir / "metrics.json").write_text(json.dumps(metrics) + "\n")
                    (combo_dir / "train_events.json").write_text(
                        json.dumps(build_events(
                            writer_grad=writer_grad,
                            task_only_grad=task_only_grad,
                            aux_only_grad=aux_only_grad,
                        )) + "\n"
                    )

            for task_name in ("fever", "gsm8k", "narrativeqa"):
                write_control(task_name)

            write_combo(
                "s3_multi_item_cross_attn_raw__l0_task_only",
                task_score=0.0,
                support_rank=1.1,
                common_mode_ratio=0.999999,
                positive_margin=False,
                writer_grad=0.02,
                task_only_grad=0.003,
                aux_only_grad=0.0,
            )
            write_combo(
                "s3_multi_item_cross_attn_raw__l2_contrastive",
                task_score=1.0,
                support_rank=2.0,
                common_mode_ratio=0.95,
                positive_margin=True,
                writer_grad=0.03,
                task_only_grad=0.01,
                aux_only_grad=0.002,
            )
            write_combo(
                "s5_hybrid_pooled_plus_items__l1_legacy",
                task_score=0.0,
                support_rank=1.8,
                common_mode_ratio=0.97,
                positive_margin=False,
                writer_grad=0.03,
                task_only_grad=0.001,
                aux_only_grad=0.02,
            )

            output_json = tmp_path / "v6-3-summary.json"
            output_report = tmp_path / "v6-3-summary.md"
            subprocess.run(
                [
                    sys.executable,
                    str(script_path),
                    "--result-root",
                    str(result_root),
                    "--output-json",
                    str(output_json),
                    "--output-report",
                    str(output_report),
                ],
                check=True,
                cwd=repo_root,
            )
            summary = json.loads(output_json.read_text())
            self.assertEqual(summary["comparison_conclusion"], "select_top_auxiliary_families")
            self.assertEqual(summary["top_auxiliary_families"][0], "l2_contrastive")
            self.assertIn("L2 contrastive", output_report.read_text())


if __name__ == "__main__":
    unittest.main()
