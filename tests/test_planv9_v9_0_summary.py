from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.update_planv9_v9_0_summary import build_summary


class PlanV9V90SummaryTest(unittest.TestCase):
    def _write_v80_summary(self, path: Path) -> None:
        path.write_text(
            json.dumps(
                {
                    "selected_qwen34_baseline_scores": {
                        "gsm8k": 0.8125,
                    }
                }
            )
            + "\n"
        )

    def _write_selected_prompts(self, path: Path) -> None:
        path.write_text(
            json.dumps(
                {
                    "gsm8k": {
                        "selected_arm_id": "b0_q34_gsm8k_nonthink",
                        "selected_prompt_variant": "q3_gsm8k_nonthink",
                    }
                }
            )
            + "\n"
        )

    def _write_arm(
        self,
        root: Path,
        arm_id: str,
        *,
        task_score: float,
        delta_answer_logprob: float,
        attention_mass: float,
        cache_growth: int,
        malformed_rows: int = 0,
    ) -> None:
        arm_dir = root / arm_id
        arm_dir.mkdir(parents=True, exist_ok=True)
        metrics = {
            "benchmark_id": "gsm8k",
            "pilot_prompt_variant": "q3_gsm8k_nonthink",
            "best_adapt_task_score": task_score,
            "answer_logprob_with_memory": -1.0,
            "answer_logprob_without_memory": -1.0 - delta_answer_logprob,
            "delta_answer_logprob": delta_answer_logprob,
            "prefix_attention_mass_mean": attention_mass,
            "prefix_attention_nontrivial_layer_count": 2 if attention_mass > 0.0 else 0,
            "peak_device_memory_mib": 1024.0 + cache_growth,
            "prefix_artifact_stats": {
                "memory_tokens_count": cache_growth,
                "memory_tokens_l2": float(cache_growth * 10),
                "memory_tokens_slot_norm_mean": float(cache_growth),
                "memory_tokens_slot_norm_std": 0.5,
                "memory_tokens_slot_norm_max": float(cache_growth + 1),
            },
        }
        (arm_dir / "metrics.json").write_text(json.dumps(metrics) + "\n")
        (arm_dir / "profiling.json").write_text(json.dumps({"wall_time_sec": 100.0 + cache_growth}) + "\n")
        rows = []
        for index in range(4):
            malformed = index < malformed_rows
            rows.append(
                {
                    "predicted_text": "" if malformed else f"answer {index}",
                    "normalized_prediction": "" if malformed else str(index),
                }
            )
        (arm_dir / "task_case_dump.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n"
        )

    def test_build_summary_classifies_o0_safe_flat(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_root = root / "runs"
            v80_summary = root / "v8-0-summary.json"
            selected_prompts = root / "selected-prompts.json"
            self._write_v80_summary(v80_summary)
            self._write_selected_prompts(selected_prompts)
            self._write_arm(run_root, "a0_nomemory_control", task_score=0.8125, delta_answer_logprob=0.0, attention_mass=0.0, cache_growth=0)
            self._write_arm(run_root, "a1_legacy_prefix_oracle", task_score=0.796875, delta_answer_logprob=-0.01, attention_mass=0.001, cache_growth=16)
            self._write_arm(run_root, "a2_precache_latent_oracle", task_score=0.796875, delta_answer_logprob=-0.01, attention_mass=0.020, cache_growth=8, malformed_rows=1)
            self._write_arm(run_root, "a3_sequence_replay_oracle", task_score=0.28125, delta_answer_logprob=-0.20, attention_mass=0.090, cache_growth=8)

            summary = build_summary(
                run_root=run_root,
                v80_summary_path=v80_summary,
                selected_prompt_modes_path=selected_prompts,
            )

            self.assertEqual(summary["outcome_id"], "O0")
            self.assertEqual(summary["recommended_next_step"], "open_v9_1_longhorizon_benchmark_hardening")
            self.assertEqual(summary["mainline_consumer_candidate"], "C1_flash_style_soft_append_to_cache")
            self.assertAlmostEqual(
                summary["arm_summaries"]["a2_precache_latent_oracle"]["malformed_answer_rate"],
                0.25,
            )

    def test_build_summary_classifies_o1_positive_signal(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_root = root / "runs"
            v80_summary = root / "v8-0-summary.json"
            selected_prompts = root / "selected-prompts.json"
            self._write_v80_summary(v80_summary)
            self._write_selected_prompts(selected_prompts)
            self._write_arm(run_root, "a0_nomemory_control", task_score=0.8125, delta_answer_logprob=0.0, attention_mass=0.0, cache_growth=0)
            self._write_arm(run_root, "a1_legacy_prefix_oracle", task_score=0.78125, delta_answer_logprob=-0.01, attention_mass=0.001, cache_growth=16)
            self._write_arm(run_root, "a2_precache_latent_oracle", task_score=0.859375, delta_answer_logprob=0.03, attention_mass=0.030, cache_growth=8)
            self._write_arm(run_root, "a3_sequence_replay_oracle", task_score=0.265625, delta_answer_logprob=-0.20, attention_mass=0.090, cache_growth=8)

            summary = build_summary(
                run_root=run_root,
                v80_summary_path=v80_summary,
                selected_prompt_modes_path=selected_prompts,
            )

            self.assertEqual(summary["outcome_id"], "O1")
            self.assertEqual(
                summary["recommended_next_step"],
                "open_v9_1_longhorizon_benchmark_hardening_with_c1_mainline",
            )

    def test_build_summary_classifies_o2_robust_collapse(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_root = root / "runs"
            v80_summary = root / "v8-0-summary.json"
            selected_prompts = root / "selected-prompts.json"
            self._write_v80_summary(v80_summary)
            self._write_selected_prompts(selected_prompts)
            self._write_arm(run_root, "a0_nomemory_control", task_score=0.8125, delta_answer_logprob=0.0, attention_mass=0.0, cache_growth=0)
            self._write_arm(run_root, "a1_legacy_prefix_oracle", task_score=0.78125, delta_answer_logprob=-0.01, attention_mass=0.001, cache_growth=16)
            self._write_arm(run_root, "a2_precache_latent_oracle", task_score=0.28125, delta_answer_logprob=-0.22, attention_mass=0.080, cache_growth=8)
            self._write_arm(run_root, "a3_sequence_replay_oracle", task_score=0.265625, delta_answer_logprob=-0.21, attention_mass=0.090, cache_growth=8)

            summary = build_summary(
                run_root=run_root,
                v80_summary_path=v80_summary,
                selected_prompt_modes_path=selected_prompts,
            )

            self.assertEqual(summary["outcome_id"], "O2")
            self.assertEqual(summary["recommended_next_step"], "hard_fail_a2_shift_mainline_consumer_to_c0_or_c2")
            self.assertTrue(summary["hard_fail_a2"])

    def test_build_summary_classifies_o3_partial_reduction(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_root = root / "runs"
            v80_summary = root / "v8-0-summary.json"
            selected_prompts = root / "selected-prompts.json"
            self._write_v80_summary(v80_summary)
            self._write_selected_prompts(selected_prompts)
            self._write_arm(run_root, "a0_nomemory_control", task_score=0.8125, delta_answer_logprob=0.0, attention_mass=0.0, cache_growth=0)
            self._write_arm(run_root, "a1_legacy_prefix_oracle", task_score=0.78125, delta_answer_logprob=-0.01, attention_mass=0.001, cache_growth=16)
            self._write_arm(run_root, "a2_precache_latent_oracle", task_score=0.578125, delta_answer_logprob=-0.08, attention_mass=0.040, cache_growth=8)
            self._write_arm(run_root, "a3_sequence_replay_oracle", task_score=0.265625, delta_answer_logprob=-0.20, attention_mass=0.090, cache_growth=8)

            summary = build_summary(
                run_root=run_root,
                v80_summary_path=v80_summary,
                selected_prompt_modes_path=selected_prompts,
            )

            self.assertEqual(summary["outcome_id"], "O3")
            self.assertEqual(
                summary["recommended_next_step"],
                "open_v9_1_longhorizon_benchmark_hardening_with_c1_guardrails",
            )


if __name__ == "__main__":
    unittest.main()
