from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class PlanV6MixedMatrixSummaryTest(unittest.TestCase):
    def test_mixed_matrix_summary_selects_finalists(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script_path = repo_root / "scripts" / "update_planv6_v6_4_mixed_matrix_summary.py"
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

            def build_events(
                *,
                writer_grad: float,
                task_only_grad: float,
                aux_only_grad: float,
                loss_head: float,
                loss_tail: float,
            ) -> list[dict[str, object]]:
                events: list[dict[str, object]] = []
                for step in range(1, 101):
                    head_fraction = min(1.0, step / 50.0)
                    loss = loss_head - ((loss_head - loss_tail) * head_fraction)
                    events.append(
                        {
                            "step": step,
                            "loss": loss,
                            "grad_norm_writer": writer_grad,
                            "grad_norm_projector": 0.02,
                            "grad_norm_receiver_lora": 0.02,
                            "writer_frozen": False,
                            "gradient_probe_step_active": True,
                            "grad_probe_writer_task_only_norm": task_only_grad,
                            "grad_probe_writer_aux_only_norm": aux_only_grad,
                            "grad_probe_writer_total_norm": max(task_only_grad + aux_only_grad, writer_grad),
                            "grad_probe_writer_task_aux_cosine": 0.1,
                            "grad_probe_writer_task_total_cosine": 0.9,
                            "grad_probe_writer_aux_total_cosine": 0.3,
                            "was_grad_clipped_writer": False,
                            "was_grad_clipped_projector": False,
                            "was_grad_clipped_receiver_lora": False,
                        }
                    )
                return events

            def write_combo(combo_id: str, task_specs: dict[str, dict[str, float | bool]]) -> None:
                for task_name in ("fever", "gsm8k", "narrativeqa"):
                    spec = task_specs[task_name]
                    combo_dir = result_root / task_name / combo_id
                    combo_dir.mkdir(parents=True, exist_ok=True)
                    positive_margin = bool(spec["positive_margin"])
                    rows = [
                        {
                            "example_id": "1",
                            "gold_label": "SUPPORTS",
                            "predicted_label": "SUPPORTS" if positive_margin else "REFUTES",
                            "predicted_correct": positive_margin,
                            "task_score": float(spec["task_score"]),
                            "final_margin": 0.4 if positive_margin else -0.4,
                            "candidate_labels": ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"],
                            "candidate_texts": ["Supports", "Refutes", "Not enough info"],
                        }
                    ]
                    case_path = write_case_dump(combo_dir / "task_case_dump.jsonl", rows)
                    metrics = {
                        "task_name": task_name,
                        "benchmark_id": task_name,
                        "task_metric_name": "accuracy",
                        "best_adapt_task_score": float(spec["task_score"]),
                        "best_adapt_exact_match": float(spec["task_score"]),
                        "prefix_attention_mass_mean": float(spec["prefix_attention_mass"]),
                        "prefix_attention_mass_mean_by_layer": {
                            "0": float(spec["prefix_attention_mass"]),
                            "1": float(spec["prefix_attention_mass"]),
                        },
                        "projected_memory_effective_rank": 4.0,
                        "memory_long_common_mode_energy_ratio": float(spec["common_mode_ratio"]),
                        "train_final_support_state_effective_rank": float(spec["support_rank"]),
                        "train_final_memory_long_effective_rank": max(1.0, float(spec["support_rank"])),
                        "task_case_dump_path": str(case_path),
                        "pilot_train_steps": 100,
                    }
                    (combo_dir / "metrics.json").write_text(json.dumps(metrics) + "\n")
                    (combo_dir / "train_events.json").write_text(
                        json.dumps(
                            build_events(
                                writer_grad=float(spec["writer_grad"]),
                                task_only_grad=float(spec["task_only_grad"]),
                                aux_only_grad=float(spec["aux_only_grad"]),
                                loss_head=float(spec["loss_head"]),
                                loss_tail=float(spec["loss_tail"]),
                            )
                        )
                        + "\n"
                    )

            for task_name in ("fever", "gsm8k", "narrativeqa"):
                write_control(task_name)

            write_combo(
                "s3_multi_item_cross_attn_raw__c2_support_and_context_gated__l2_contrastive",
                task_specs={
                    "fever": {
                        "task_score": 1.0,
                        "support_rank": 2.0,
                        "common_mode_ratio": 0.95,
                        "positive_margin": True,
                        "writer_grad": 0.03,
                        "task_only_grad": 0.01,
                        "aux_only_grad": 0.002,
                        "loss_head": 4.0,
                        "loss_tail": 1.0,
                        "prefix_attention_mass": 0.01,
                    },
                    "gsm8k": {
                        "task_score": 1.0,
                        "support_rank": 2.0,
                        "common_mode_ratio": 0.95,
                        "positive_margin": True,
                        "writer_grad": 0.03,
                        "task_only_grad": 0.01,
                        "aux_only_grad": 0.002,
                        "loss_head": 4.0,
                        "loss_tail": 1.1,
                        "prefix_attention_mass": 0.01,
                    },
                    "narrativeqa": {
                        "task_score": 1.0,
                        "support_rank": 2.0,
                        "common_mode_ratio": 0.95,
                        "positive_margin": True,
                        "writer_grad": 0.03,
                        "task_only_grad": 0.01,
                        "aux_only_grad": 0.002,
                        "loss_head": 4.0,
                        "loss_tail": 1.2,
                        "prefix_attention_mass": 0.01,
                    },
                },
            )

            write_combo(
                "s3_multi_item_cross_attn_raw__c0_support_only__l2_contrastive",
                task_specs={
                    "fever": {
                        "task_score": 1.0,
                        "support_rank": 2.0,
                        "common_mode_ratio": 0.95,
                        "positive_margin": True,
                        "writer_grad": 0.03,
                        "task_only_grad": 0.01,
                        "aux_only_grad": 0.002,
                        "loss_head": 4.0,
                        "loss_tail": 1.0,
                        "prefix_attention_mass": 0.01,
                    },
                    "gsm8k": {
                        "task_score": 1.0,
                        "support_rank": 2.0,
                        "common_mode_ratio": 0.95,
                        "positive_margin": True,
                        "writer_grad": 0.03,
                        "task_only_grad": 0.01,
                        "aux_only_grad": 0.002,
                        "loss_head": 4.0,
                        "loss_tail": 1.1,
                        "prefix_attention_mass": 0.01,
                    },
                    "narrativeqa": {
                        "task_score": 0.0,
                        "support_rank": 2.0,
                        "common_mode_ratio": 0.95,
                        "positive_margin": False,
                        "writer_grad": 0.03,
                        "task_only_grad": 0.01,
                        "aux_only_grad": 0.002,
                        "loss_head": 4.0,
                        "loss_tail": 1.2,
                        "prefix_attention_mass": 0.01,
                    },
                },
            )

            write_combo(
                "s5_hybrid_pooled_plus_items__c2_support_and_context_gated__l5_orthogonality_coverage",
                task_specs={
                    "fever": {
                        "task_score": 1.0,
                        "support_rank": 1.8,
                        "common_mode_ratio": 0.97,
                        "positive_margin": True,
                        "writer_grad": 0.02,
                        "task_only_grad": 0.01,
                        "aux_only_grad": 0.001,
                        "loss_head": 4.0,
                        "loss_tail": 1.3,
                        "prefix_attention_mass": 0.01,
                    },
                    "gsm8k": {
                        "task_score": 1.0,
                        "support_rank": 1.8,
                        "common_mode_ratio": 0.97,
                        "positive_margin": True,
                        "writer_grad": 0.02,
                        "task_only_grad": 0.01,
                        "aux_only_grad": 0.001,
                        "loss_head": 4.0,
                        "loss_tail": 1.4,
                        "prefix_attention_mass": 0.01,
                    },
                    "narrativeqa": {
                        "task_score": 0.0,
                        "support_rank": 1.8,
                        "common_mode_ratio": 0.97,
                        "positive_margin": False,
                        "writer_grad": 0.00001,
                        "task_only_grad": 0.0,
                        "aux_only_grad": 0.0,
                        "loss_head": 4.0,
                        "loss_tail": 3.9,
                        "prefix_attention_mass": 0.0,
                    },
                },
            )

            write_combo(
                "s5_hybrid_pooled_plus_items__c0_support_only__l3_vicreg",
                task_specs={
                    "fever": {
                        "task_score": 1.0,
                        "support_rank": 1.6,
                        "common_mode_ratio": 0.98,
                        "positive_margin": True,
                        "writer_grad": 0.02,
                        "task_only_grad": 0.01,
                        "aux_only_grad": 0.001,
                        "loss_head": 4.0,
                        "loss_tail": 1.5,
                        "prefix_attention_mass": 0.01,
                    },
                    "gsm8k": {
                        "task_score": 0.0,
                        "support_rank": 1.6,
                        "common_mode_ratio": 0.98,
                        "positive_margin": False,
                        "writer_grad": 0.02,
                        "task_only_grad": 0.01,
                        "aux_only_grad": 0.001,
                        "loss_head": 4.0,
                        "loss_tail": 1.6,
                        "prefix_attention_mass": 0.01,
                    },
                    "narrativeqa": {
                        "task_score": 0.0,
                        "support_rank": 1.6,
                        "common_mode_ratio": 0.98,
                        "positive_margin": False,
                        "writer_grad": 0.02,
                        "task_only_grad": 0.01,
                        "aux_only_grad": 0.001,
                        "loss_head": 4.0,
                        "loss_tail": 1.7,
                        "prefix_attention_mass": 0.01,
                    },
                },
            )

            output_json = tmp_path / "v6-4-summary.json"
            output_report = tmp_path / "v6-4-summary.md"
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
            self.assertEqual(summary["comparison_conclusion"], "select_finalists")
            self.assertEqual(summary["recommended_next_step"], "open_v6_5_recipe_stabilization")
            self.assertEqual(
                summary["finalist_configs"][0],
                "s3_multi_item_cross_attn_raw__c2_support_and_context_gated__l2_contrastive",
            )
            self.assertIn("C2 support_and_context_gated", output_report.read_text())


if __name__ == "__main__":
    unittest.main()
