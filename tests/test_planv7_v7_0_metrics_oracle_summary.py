from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from memtotal.utils.config import load_config


class PlanV7V70MetricsOracleSummaryTest(unittest.TestCase):
    def test_mid4_and_additive_method_presets_match_planv7_depths(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        mid4_cfg = load_config(
            str(repo_root / "configs" / "method" / "writer_direct_deep_prefix_receiver_lora_r2_mid4.yaml")
        )
        additive_cfg = load_config(
            str(repo_root / "configs" / "method" / "writer_direct_deep_prefix_receiver_lora_r2_additive7.yaml")
        )
        self.assertEqual(mid4_cfg["runtime"]["pilot_deep_prefix_layers"], [12, 13, 14, 15])
        self.assertEqual(
            mid4_cfg["method"]["receiver_lora"]["target_layers"],
            [12, 13, 14, 15],
        )
        self.assertEqual(additive_cfg["runtime"]["pilot_deep_prefix_layers"], [0, 1, 2, 3, 4, 8, 14])
        self.assertEqual(
            additive_cfg["method"]["receiver_lora"]["target_layers"],
            [0, 1, 2, 3, 4],
        )

    def test_summary_prefers_mid4_when_mid_oracle_beats_early(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script_path = repo_root / "scripts" / "update_planv7_v7_0_metrics_oracle_summary.py"
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            result_root = tmp_path / "results"

            def write_json(path: Path, payload: dict[str, object]) -> None:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

            def write_events(path: Path, *, loss_start: float, loss_end: float) -> None:
                path.parent.mkdir(parents=True, exist_ok=True)
                events = []
                for step in range(1, 201):
                    alpha = float(step - 1) / 199.0
                    loss = loss_start + ((loss_end - loss_start) * alpha)
                    events.append(
                        {
                            "step": step,
                            "loss": loss,
                            "writer_frozen": False,
                            "grad_norm_writer": 0.02,
                            "grad_norm_projector": 0.05,
                            "grad_norm_receiver_lora": 0.02,
                            "grad_probe_writer_task_only_norm": 0.02,
                            "grad_probe_writer_aux_only_norm": 0.01,
                            "grad_probe_writer_total_norm": 0.05,
                            "grad_probe_writer_task_aux_cosine": 0.0,
                            "grad_probe_writer_task_total_cosine": 0.6,
                            "grad_probe_writer_aux_total_cosine": 0.1,
                            "was_grad_clipped_writer": False,
                            "was_grad_clipped_projector": False,
                            "was_grad_clipped_receiver_lora": False,
                        }
                    )
                path.write_text(json.dumps(events, indent=2) + "\n")

            def metrics(
                *,
                benchmark_id: str,
                task_score: float,
                metric_name: str = "exact_match",
                prefix_attention_by_layer: dict[str, float] | None = None,
                writer_rank: float = 2.5,
                projected_rank: float = 3.0,
                common_mode: float = 0.99,
                pairwise_cosine: float | None = 0.6,
                owner_metadata: bool = False,
            ) -> dict[str, object]:
                payload: dict[str, object] = {
                    "benchmark_id": benchmark_id,
                    "task_name": benchmark_id,
                    "task_metric_name": metric_name,
                    "best_adapt_task_score": task_score,
                    "best_adapt_exact_match": task_score,
                    "delta_answer_logprob": 0.0,
                    "prefix_attention_mass_mean": sum((prefix_attention_by_layer or {}).values())
                    / max(1, len(prefix_attention_by_layer or {})),
                    "prefix_attention_mass_mean_by_layer": prefix_attention_by_layer or {},
                    "projected_memory_effective_rank": projected_rank,
                    "memory_long_common_mode_energy_ratio": common_mode,
                    "train_final_support_state_effective_rank": 1.8,
                    "train_final_memory_long_effective_rank": writer_rank,
                    "train_final_writer_slot_basis_pairwise_cosine_mean": pairwise_cosine,
                    "writer_memory_slots": 8,
                    "memory_long_slot_norm_std": 0.2,
                    "memory_long_slot_norm_mean": 1.0,
                    "pilot_train_steps": 200,
                    "train_loss_steps_1_50_median": 8.0,
                    "train_loss_steps_451_500_median": 2.0,
                    "snapshot_metrics": [{"step": 0, "prefix_l2": 1.0}, {"step": 200, "prefix_l2": 2.0}],
                }
                if pairwise_cosine is None:
                    payload.pop("train_final_writer_slot_basis_pairwise_cosine_mean")
                if owner_metadata:
                    payload["owner_locked_projector_lr"] = 7.5e-6
                    payload["repo_confirmed_v65_projector_lr_reference"] = 7.5e-5
                    payload["owner_override_note"] = True
                return payload

            for task_name, task_score in (("gsm8k", 0.10), ("triviaqa", 0.20), ("fever", 0.60)):
                write_json(
                    result_root / "control" / task_name / "metrics.json",
                    metrics(
                        benchmark_id=task_name,
                        task_score=task_score,
                        metric_name="accuracy" if task_name == "fever" else "exact_match",
                        owner_metadata=(task_name == "gsm8k"),
                    ),
                )

            for arm_name in ("c_add", "c_early", "c_mid"):
                for task_name, base_score in (("gsm8k", 0.10), ("triviaqa", 0.20), ("fever", 0.60)):
                    task_score = base_score
                    writer_rank = 2.5
                    projected_rank = 3.0
                    common_mode = 0.99
                    pairwise = 0.6
                    if arm_name == "c_early":
                        writer_rank = 0.4
                        projected_rank = 3.2
                        common_mode = 0.994
                        pairwise = 0.92
                    if arm_name == "c_mid" and task_name == "triviaqa":
                        task_score = 0.25
                    write_json(
                        result_root / arm_name / task_name / "metrics.json",
                        metrics(
                            benchmark_id=task_name,
                            task_score=task_score,
                            metric_name="accuracy" if task_name == "fever" else "exact_match",
                            prefix_attention_by_layer={"12": 0.003, "13": 0.002}
                            if arm_name == "c_mid"
                            else {"0": 0.003, "1": 0.002},
                            writer_rank=writer_rank,
                            projected_rank=projected_rank,
                            common_mode=common_mode,
                            pairwise_cosine=pairwise,
                        ),
                    )
                    write_events(
                        result_root / arm_name / task_name / "train_events.json",
                        loss_start=8.0,
                        loss_end=2.0,
                    )

            oracle_scores = {
                "o_ctx_early": {"gsm8k": 0.10, "triviaqa": 0.20},
                "o_ctx_mid": {"gsm8k": 0.16, "triviaqa": 0.20},
                "o_sup_early": {"gsm8k": 0.10, "triviaqa": 0.20},
                "o_sup_mid": {"gsm8k": 0.12, "triviaqa": 0.21},
            }
            for arm_name, task_scores in oracle_scores.items():
                for task_name, task_score in task_scores.items():
                    write_json(
                        result_root / arm_name / task_name / "metrics.json",
                        metrics(
                            benchmark_id=task_name,
                            task_score=task_score,
                            prefix_attention_by_layer={"0": 0.004, "1": 0.003},
                            pairwise_cosine=None,
                        ),
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
            self.assertEqual(summary["comparison_conclusion"], "prefer_mid4_mainline")
            self.assertEqual(summary["preferred_depth"], "mid4")
            self.assertTrue(summary["early_vs_mid_oracle"]["context_echo"]["mid_beats_early"])
            self.assertEqual(
                summary["owner_lr_discrepancy_metadata"],
                {
                    "owner_locked_projector_lr": 7.5e-6,
                    "repo_confirmed_v65_projector_lr_reference": 7.5e-5,
                    "owner_override_note": True,
                },
            )
            c_early_gsm8k = summary["baseline_replay"]["c_early"]["tasks"]["gsm8k"]
            self.assertTrue(c_early_gsm8k["source_not_collapsed_old"])
            self.assertFalse(c_early_gsm8k["writer_memory_not_collapsed_strict"])
            self.assertTrue(c_early_gsm8k["projector_manufactured_diversity"])
            c_mid_triviaqa = summary["baseline_replay"]["c_mid"]["tasks"]["triviaqa"]
            self.assertTrue(c_mid_triviaqa["writer_memory_not_collapsed_strict"])
            self.assertTrue(c_mid_triviaqa["primary_task_improved"])


if __name__ == "__main__":
    unittest.main()
