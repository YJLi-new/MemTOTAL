from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class PlanV6RecipeStabilizationSummaryTest(unittest.TestCase):
    def test_summary_selects_stabilized_recipe(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script_path = repo_root / "scripts" / "update_planv6_v6_5_recipe_stabilization_summary.py"
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            result_root = tmp_path / "results"

            def write_case_dump(path: Path, *, correct: bool) -> Path:
                path.parent.mkdir(parents=True, exist_ok=True)
                row = {
                    "example_id": "1",
                    "gold_label": "SUPPORTS",
                    "predicted_label": "SUPPORTS" if correct else "REFUTES",
                    "predicted_correct": correct,
                    "task_score": 1.0 if correct else 0.0,
                    "final_margin": 0.6 if correct else -0.4,
                    "candidate_labels": ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"],
                    "candidate_texts": ["Supports", "Refutes", "Not enough info"],
                }
                path.write_text(json.dumps(row) + "\n")
                return path

            def build_events(*, writer_grad: float, task_only_grad: float, loss_head: float, loss_tail: float) -> list[dict[str, object]]:
                events: list[dict[str, object]] = []
                for step in range(1, 101):
                    head_fraction = min(1.0, step / 50.0)
                    loss = loss_head - ((loss_head - loss_tail) * head_fraction)
                    events.append(
                        {
                            "step": step,
                            "loss": loss,
                            "grad_norm_writer": writer_grad,
                            "grad_norm_projector": 0.03,
                            "grad_norm_receiver_lora": 0.02,
                            "writer_frozen": False,
                            "gradient_probe_step_active": True,
                            "grad_probe_writer_task_only_norm": task_only_grad,
                            "grad_probe_writer_aux_only_norm": 0.002,
                            "grad_probe_writer_total_norm": max(task_only_grad + 0.002, writer_grad),
                            "grad_probe_writer_task_aux_cosine": 0.05,
                            "grad_probe_writer_task_total_cosine": 0.95,
                            "grad_probe_writer_aux_total_cosine": 0.2,
                            "was_grad_clipped_writer": False,
                            "was_grad_clipped_projector": False,
                            "was_grad_clipped_receiver_lora": False,
                        }
                    )
                return events

            def write_metrics(target_dir: Path, *, correct: bool, stable: bool, delta: float) -> None:
                case_path = write_case_dump(target_dir / "task_case_dump.jsonl", correct=correct)
                metrics = {
                    "task_name": "fever_real_smoke",
                    "benchmark_id": "fever",
                    "task_metric_name": "accuracy",
                    "best_adapt_task_score": 1.0 if correct else 0.0,
                    "best_adapt_exact_match": 1.0 if correct else 0.0,
                    "prefix_attention_mass_mean": 0.01,
                    "prefix_attention_mass_mean_by_layer": {"0": 0.01, "1": 0.01, "2": 0.01, "3": 0.01},
                    "projected_memory_effective_rank": 4.0,
                    "memory_long_common_mode_energy_ratio": 0.95,
                    "train_final_support_state_effective_rank": 2.0,
                    "train_final_memory_long_effective_rank": 2.0,
                    "task_case_dump_path": str(case_path),
                    "pilot_train_steps": 100,
                    "delta_answer_logprob": delta,
                }
                (target_dir / "metrics.json").write_text(json.dumps(metrics) + "\n")
                (target_dir / "suite_metrics.json").write_text(json.dumps({"accuracy": metrics["best_adapt_task_score"]}) + "\n")
                (target_dir / "train_events.json").write_text(
                    json.dumps(
                        build_events(
                            writer_grad=0.03 if stable else 0.015,
                            task_only_grad=0.01 if stable else 0.004,
                            loss_head=4.0,
                            loss_tail=1.0 if stable else 4.5,
                        )
                    )
                    + "\n"
                )

            manifest = {
                "finalists": [
                    {
                        "alias": "F1",
                        "combo_id": "s3_multi_item_cross_attn_raw__c2_support_and_context_gated__l5_orthogonality_coverage",
                    },
                    {
                        "alias": "F2",
                        "combo_id": "s3_multi_item_cross_attn_raw__c2_support_and_context_gated__l3_vicreg",
                    },
                ],
                "screen_recipes": [
                    {
                        "recipe_id": "F1__w10__clip_groupwise__plr5e5__acc1__layers_base",
                        "combo_id": "s3_multi_item_cross_attn_raw__c2_support_and_context_gated__l5_orthogonality_coverage",
                        "warmup_steps": 10,
                        "clipping_scheme": "groupwise",
                        "projector_learning_rate": 5.0e-5,
                        "accumulation_steps": 1,
                        "layer_variant": "layers_base",
                    },
                    {
                        "recipe_id": "F2__w20__clip_global__plr75e6__acc4__layers_additive",
                        "combo_id": "s3_multi_item_cross_attn_raw__c2_support_and_context_gated__l3_vicreg",
                        "warmup_steps": 20,
                        "clipping_scheme": "global",
                        "projector_learning_rate": 7.5e-5,
                        "accumulation_steps": 4,
                        "layer_variant": "layers_additive",
                    },
                ],
                "confirmation_recipe_ids": [
                    "F1__w10__clip_groupwise__plr5e5__acc1__layers_base",
                    "F2__w20__clip_global__plr75e6__acc4__layers_additive",
                ],
                "confirmation_seeds": [61109, 61110, 61111],
            }
            (result_root / "screen-manifest.json").parent.mkdir(parents=True, exist_ok=True)
            (result_root / "screen-manifest.json").write_text(json.dumps(manifest) + "\n")

            write_metrics(result_root / "control", correct=False, stable=True, delta=0.0)

            write_metrics(
                result_root / "screen" / "F1__w10__clip_groupwise__plr5e5__acc1__layers_base",
                correct=True,
                stable=True,
                delta=1.0,
            )
            write_metrics(
                result_root / "screen" / "F2__w20__clip_global__plr75e6__acc4__layers_additive",
                correct=True,
                stable=False,
                delta=0.1,
            )

            for seed in (61109, 61110, 61111):
                write_metrics(result_root / "confirm" / f"seed_{seed}" / "control", correct=False, stable=True, delta=0.0)
                write_metrics(
                    result_root / "confirm" / f"seed_{seed}" / "F1__w10__clip_groupwise__plr5e5__acc1__layers_base",
                    correct=True,
                    stable=True,
                    delta=1.0,
                )
                write_metrics(
                    result_root / "confirm" / f"seed_{seed}" / "F2__w20__clip_global__plr75e6__acc4__layers_additive",
                    correct=(seed != 61111),
                    stable=(seed != 61111),
                    delta=0.1 if seed != 61111 else -0.1,
                )

            output_json = tmp_path / "summary.json"
            output_md = tmp_path / "summary.md"
            subprocess.run(
                [
                    sys.executable,
                    str(script_path),
                    "--result-root",
                    str(result_root),
                    "--output-json",
                    str(output_json),
                    "--output-report",
                    str(output_md),
                ],
                check=True,
                cwd=repo_root,
            )

            payload = json.loads(output_json.read_text())
            self.assertEqual(payload["comparison_conclusion"], "select_stabilized_recipe")
            self.assertEqual(
                payload["screen_top_two_recipes"],
                [
                    "F1__w10__clip_groupwise__plr5e5__acc1__layers_base",
                    "F2__w20__clip_global__plr75e6__acc4__layers_additive",
                ],
            )
            self.assertEqual(
                payload["stabilized_recipes"],
                ["F1__w10__clip_groupwise__plr5e5__acc1__layers_base"],
            )


if __name__ == "__main__":
    unittest.main()
