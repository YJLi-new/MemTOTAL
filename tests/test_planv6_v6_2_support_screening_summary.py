from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class PlanV6SupportScreeningSummaryTest(unittest.TestCase):
    def test_support_screening_summary_selects_top_two_modes(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script_path = repo_root / "scripts" / "update_planv6_v6_2_support_screening_summary.py"
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            result_root = tmp_path / "results"
            control_dir = result_root / "fever" / "control"
            control_dir.mkdir(parents=True, exist_ok=True)

            def write_case_dump(path: Path, rows: list[dict[str, object]]) -> Path:
                path.write_text("".join(json.dumps(row) + "\n" for row in rows))
                return path

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
                },
                {
                    "example_id": "2",
                    "gold_label": "REFUTES",
                    "predicted_label": "REFUTES",
                    "predicted_correct": True,
                    "task_score": 1.0,
                    "final_margin": 0.1,
                    "candidate_labels": ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"],
                    "candidate_texts": ["Supports", "Refutes", "Not enough info"],
                },
            ]
            control_case_path = write_case_dump(control_dir / "task_case_dump.jsonl", control_rows)
            control_metrics = {
                "task_name": "fever",
                "benchmark_id": "fever",
                "task_metric_name": "accuracy",
                "best_adapt_task_score": 0.5,
                "best_adapt_exact_match": 0.5,
                "prefix_attention_mass_mean_by_layer": {"0": 0.0, "1": 0.0},
                "memory_long_common_mode_energy_ratio": 0.999999,
                "train_final_support_state_effective_rank": 1.0,
                "train_final_memory_long_effective_rank": 1.0,
                "task_case_dump_path": str(control_case_path),
            }
            (control_dir / "metrics.json").write_text(json.dumps(control_metrics) + "\n")
            (control_dir / "train_events.json").write_text(json.dumps([]) + "\n")

            def build_events(*, loss_start: float, loss_end: float, writer_grad: float) -> list[dict[str, object]]:
                events: list[dict[str, object]] = []
                for step in range(1, 101):
                    frac = (step - 1) / 99.0
                    loss = ((1.0 - frac) * loss_start) + (frac * loss_end)
                    events.append(
                        {
                            "step": step,
                            "loss": loss,
                            "grad_norm_writer": writer_grad,
                            "grad_norm_projector": 0.01,
                            "grad_norm_receiver_lora": 0.01,
                            "writer_frozen": False,
                            "gradient_probe_step_active": True,
                            "grad_probe_writer_task_only_norm": writer_grad,
                            "grad_probe_writer_aux_only_norm": 0.0,
                            "grad_probe_writer_total_norm": writer_grad,
                            "grad_probe_writer_task_aux_cosine": 0.0,
                            "grad_probe_writer_task_total_cosine": 1.0,
                            "grad_probe_writer_aux_total_cosine": 0.0,
                            "was_grad_clipped_writer": False,
                            "was_grad_clipped_projector": False,
                            "was_grad_clipped_receiver_lora": False,
                        }
                    )
                return events

            def write_mode(
                mode_id: str,
                *,
                task_score: float,
                support_rank: float,
                common_mode_ratio: float,
                positive_margin: bool,
            ) -> None:
                writer_dir = result_root / "fever" / mode_id
                writer_dir.mkdir(parents=True, exist_ok=True)
                writer_rows = [
                    {
                        "example_id": "1",
                        "gold_label": "SUPPORTS",
                        "predicted_label": "SUPPORTS" if positive_margin else "REFUTES",
                        "predicted_correct": bool(positive_margin),
                        "task_score": 1.0 if positive_margin else 0.0,
                        "final_margin": 0.4 if positive_margin else -0.1,
                        "candidate_labels": ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"],
                        "candidate_texts": ["Supports", "Refutes", "Not enough info"],
                    },
                    {
                        "example_id": "2",
                        "gold_label": "REFUTES",
                        "predicted_label": "REFUTES",
                        "predicted_correct": True,
                        "task_score": 1.0,
                        "final_margin": 0.3,
                        "candidate_labels": ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"],
                        "candidate_texts": ["Supports", "Refutes", "Not enough info"],
                    },
                ]
                case_path = write_case_dump(writer_dir / "task_case_dump.jsonl", writer_rows)
                metrics = {
                    "task_name": "fever",
                    "benchmark_id": "fever",
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
                (writer_dir / "metrics.json").write_text(json.dumps(metrics) + "\n")
                (writer_dir / "train_events.json").write_text(
                    json.dumps(build_events(loss_start=3.0, loss_end=1.0, writer_grad=0.02)) + "\n"
                )

            write_mode(
                "s1_pooled_block_gated",
                task_score=1.0,
                support_rank=2.2,
                common_mode_ratio=0.95,
                positive_margin=True,
            )
            write_mode(
                "s2_structured_support_set",
                task_score=0.5,
                support_rank=1.8,
                common_mode_ratio=0.97,
                positive_margin=False,
            )
            write_mode(
                "s0_pooled_block_legacy",
                task_score=0.5,
                support_rank=1.0,
                common_mode_ratio=0.999999,
                positive_margin=False,
            )

            output_json = tmp_path / "v6-2-summary.json"
            output_report = tmp_path / "v6-2-summary.md"
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
            self.assertEqual(summary["comparison_conclusion"], "select_top_two_support_modes")
            self.assertEqual(
                summary["top_two_support_modes"],
                ["s1_pooled_block_gated", "s2_structured_support_set"],
            )
            self.assertIn("S1 pooled_block_gated", output_report.read_text())


if __name__ == "__main__":
    unittest.main()
