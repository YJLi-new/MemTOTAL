from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class PlanV7V76MultiseedConfirmationSummaryTest(unittest.TestCase):
    def _write_json(self, path: Path, payload: dict[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    def _write_train_events(self, path: Path, *, writer_grad: float, projector_grad: float) -> None:
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

    def _write_generation_dump(self, path: Path, *, delta: float) -> None:
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

    def _write_classification_dump(self, path: Path, *, margin_delta: float, correct_flip: bool) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for idx in range(4):
            rows.append(
                {
                    "example_id": f"row-{idx}",
                    "predicted_label": "SUPPORTS" if correct_flip and idx == 0 else "REFUTES",
                    "predicted_correct": bool(correct_flip and idx == 0) or idx > 0,
                    "final_margin": float(idx) + margin_delta,
                }
            )
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    def _metrics_payload(
        self,
        *,
        benchmark_id: str,
        task_score: float,
        task_case_dump_path: str,
        delta_answer_logprob: float,
        writer_rank_fraction: float,
        writer_rank: float,
        common_mode_ratio: float,
        projector_manufactured_diversity: bool = False,
        task_metric_name: str = "exact_match",
        bridge_family: str = "B2",
        memory_path_variant: str = "two_level",
        projector_token_source: str = "short_slots",
        reader_queries: int = 16,
        short_slots: int = 16,
    ) -> dict[str, object]:
        return {
            "benchmark_id": benchmark_id,
            "task_name": benchmark_id,
            "task_metric_name": task_metric_name,
            "best_adapt_task_score": task_score,
            "best_adapt_exact_match": task_score if task_metric_name == "exact_match" else 0.0,
            "best_adapt_accuracy": task_score if task_metric_name == "accuracy" else 0.0,
            "delta_answer_logprob": delta_answer_logprob,
            "prefix_attention_mass_mean": 0.012,
            "prefix_attention_mass_mean_by_layer": {"12": 0.004, "13": 0.003, "14": 0.003, "15": 0.002},
            "projected_memory_effective_rank": 18.0,
            "memory_long_common_mode_energy_ratio": common_mode_ratio,
            "train_final_support_state_effective_rank": 2.0,
            "train_final_memory_long_effective_rank": writer_rank,
            "train_final_writer_slot_basis_pairwise_cosine_mean": 0.20,
            "writer_memory_slots": 64,
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
            "pilot_active_writer_family": "W3",
            "pilot_active_bridge_family": bridge_family,
            "pilot_active_projector_family": "P2",
            "writer_memory_not_collapsed_strict": writer_rank_fraction >= 0.125 and common_mode_ratio <= 0.985,
            "writer_rank_fraction": writer_rank_fraction,
            "writer_memory_slot_effective_rank": writer_rank,
            "slot_pairwise_cosine": 0.20,
            "slot_pairwise_cosine_present": True,
            "projector_manufactured_diversity": projector_manufactured_diversity,
        }

    def _write_suite(
        self,
        result_root: Path,
        variant_id: str,
        seed: int,
        task_name: str,
        *,
        task_score: float,
        delta: float,
        writer_rank_fraction: float,
        writer_rank: float,
        common_mode_ratio: float,
        task_metric_name: str = "exact_match",
        projector_manufactured_diversity: bool = False,
    ) -> None:
        task_dir = result_root / variant_id / f"seed_{seed}" / task_name
        if task_metric_name == "accuracy":
            case_dump = task_dir / "task_case_dump.jsonl"
            self._write_classification_dump(case_dump, margin_delta=delta, correct_flip=delta > 0.0)
        else:
            case_dump = task_dir / "task_case_dump.jsonl"
            self._write_generation_dump(case_dump, delta=delta)
        self._write_json(
            task_dir / "metrics.json",
            self._metrics_payload(
                benchmark_id=task_name,
                task_score=task_score,
                task_case_dump_path=str(case_dump),
                delta_answer_logprob=delta,
                writer_rank_fraction=writer_rank_fraction,
                writer_rank=writer_rank,
                common_mode_ratio=common_mode_ratio,
                projector_manufactured_diversity=projector_manufactured_diversity,
                task_metric_name=task_metric_name,
            ),
        )
        self._write_train_events(task_dir / "train_events.json", writer_grad=0.25, projector_grad=0.12)

    def _run_summary(
        self,
        *,
        scenario: str,
        a1_gsm_scores: list[float],
        a1_trivia_scores: list[float],
        a1_writer_rank_fraction: float,
        a1_common_mode_ratio: float,
        expect_conclusion: str,
    ) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script_path = repo_root / "scripts" / "update_planv7_v7_6_multiseed_confirmation_summary.py"
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            result_root = tmp_path / "results"
            v75_summary_path = tmp_path / "v7-5-summary.json"
            v70_summary_path = tmp_path / "v7-0-summary.json"
            seeds = [61109, 61110, 61111]

            self._write_json(
                result_root / "selection-manifest.json",
                {
                    "seeds": seeds,
                    "promoted_arms": ["a1_reconstruction", "a2_vicreg"],
                    "variants": [
                        "c0_frozen_no_memory",
                        "c1_additive_continuity",
                        "c2_best_direct",
                        "p1_a1_reconstruction",
                        "p2_a2_vicreg",
                    ],
                    "base_arm_id": "b_w3_q16",
                    "base_source_phase": "v7_3",
                    "control_source_arm_id": "b_w3_q16",
                    "direct_control_arm_id": "d_w1_shared",
                    "winning_depth": "D1",
                    "winning_depth_label": "mid4",
                    "winner_uses_bridge": True,
                },
            )
            self._write_json(
                v75_summary_path,
                {
                    "base_from_v7_4_arm_id": "b_w3_q16",
                    "base_from_v7_4_source_phase": "v7_3",
                    "control_source_arm_id": "b_w3_q16",
                    "direct_control_arm_id": "d_w1_shared",
                    "winning_depth": "D1",
                    "winning_depth_label": "mid4",
                    "baseline": {"memory_path_variant": "two_level"},
                    "aux_arm_ranking": [
                        {"arm_id": "a1_reconstruction"},
                        {"arm_id": "a2_vicreg"},
                    ],
                    "arms": {
                        "a1_reconstruction": {"acceptance_qualified": True},
                        "a2_vicreg": {"acceptance_qualified": True},
                    },
                },
            )
            self._write_json(v70_summary_path, {"all_oracles_flat_on_primary_tasks": True})

            for seed in seeds:
                for task_name in ("gsm8k", "triviaqa"):
                    base_score = 0.20 if task_name == "gsm8k" else 0.30
                    for variant_id in ("c0_frozen_no_memory", "c1_additive_continuity", "c2_best_direct", "p2_a2_vicreg"):
                        self._write_suite(
                            result_root,
                            variant_id,
                            seed,
                            task_name,
                            task_score=base_score,
                            delta=0.0,
                            writer_rank_fraction=0.05,
                            writer_rank=3.0,
                            common_mode_ratio=0.995,
                        )
                for variant_id in ("c0_frozen_no_memory", "c1_additive_continuity", "c2_best_direct", "p2_a2_vicreg"):
                    self._write_suite(
                        result_root,
                        variant_id,
                        seed,
                        "fever",
                        task_score=0.60,
                        delta=0.0,
                        writer_rank_fraction=0.05,
                        writer_rank=3.0,
                        common_mode_ratio=0.995,
                        task_metric_name="accuracy",
                    )

            for index, seed in enumerate(seeds):
                self._write_suite(
                    result_root,
                    "p1_a1_reconstruction",
                    seed,
                    "gsm8k",
                    task_score=a1_gsm_scores[index],
                    delta=0.30,
                    writer_rank_fraction=a1_writer_rank_fraction,
                    writer_rank=10.0,
                    common_mode_ratio=a1_common_mode_ratio,
                )
                self._write_suite(
                    result_root,
                    "p1_a1_reconstruction",
                    seed,
                    "triviaqa",
                    task_score=a1_trivia_scores[index],
                    delta=0.12,
                    writer_rank_fraction=a1_writer_rank_fraction,
                    writer_rank=10.0,
                    common_mode_ratio=a1_common_mode_ratio,
                )
                self._write_suite(
                    result_root,
                    "p1_a1_reconstruction",
                    seed,
                    "fever",
                    task_score=0.62,
                    delta=0.08,
                    writer_rank_fraction=a1_writer_rank_fraction,
                    writer_rank=10.0,
                    common_mode_ratio=a1_common_mode_ratio,
                    task_metric_name="accuracy",
                )

            output_json = tmp_path / f"{scenario}-summary.json"
            output_report = tmp_path / f"{scenario}-summary.md"
            subprocess.run(
                [
                    sys.executable,
                    str(script_path),
                    "--result_root",
                    str(result_root),
                    "--v75_summary",
                    str(v75_summary_path),
                    "--v70_summary",
                    str(v70_summary_path),
                    "--output_json",
                    str(output_json),
                    "--output_report",
                    str(output_report),
                ],
                check=True,
                cwd=repo_root,
            )
            summary = json.loads(output_json.read_text())
            self.assertEqual(summary["comparison_conclusion"], expect_conclusion)
            self.assertEqual(summary["best_confirmed_promoted_arm_id"], "a1_reconstruction")
            self.assertEqual(summary["branch_ranking"][0]["variant_id"], "p1_a1_reconstruction")

    def test_summary_selects_path_p_when_primary_gain_reproduces(self) -> None:
        self._run_summary(
            scenario="path-p",
            a1_gsm_scores=[0.26, 0.25, 0.27],
            a1_trivia_scores=[0.30, 0.30, 0.30],
            a1_writer_rank_fraction=0.16,
            a1_common_mode_ratio=0.980,
            expect_conclusion="path_p_external_writer_survives_main_thesis",
        )

    def test_summary_selects_path_q_when_writer_metrics_improve_but_scores_stay_flat(self) -> None:
        self._run_summary(
            scenario="path-q",
            a1_gsm_scores=[0.20, 0.20, 0.20],
            a1_trivia_scores=[0.30, 0.30, 0.30],
            a1_writer_rank_fraction=0.16,
            a1_common_mode_ratio=0.980,
            expect_conclusion="path_q_external_writer_unresolved_not_dead",
        )


if __name__ == "__main__":
    unittest.main()
