from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.update_planv8_v8_0_summary import build_summary


class PlanV8V80SummaryTest(unittest.TestCase):
    def _write_arm(
        self,
        root: Path,
        arm_id: str,
        *,
        benchmark_id: str,
        prompt_variant: str,
        task_score: float,
        macro_f1: float = 0.0,
        memory_tokens_count: int = 0,
        cross_attn_gate_open_fraction: float = 0.0,
        memory_token_attention_mass_mean: float = 0.0,
        train_events: list[dict[str, object]] | None = None,
    ) -> None:
        arm_dir = root / arm_id
        arm_dir.mkdir(parents=True, exist_ok=True)
        metrics = {
            "benchmark_id": benchmark_id,
            "pilot_prompt_variant": prompt_variant,
            "best_adapt_task_score": float(task_score),
            "best_adapt_macro_f1": float(macro_f1),
            "pilot_prefix_source_mode": "oracle_hidden_state_slots" if arm_id.startswith("o") else "",
            "pilot_memory_consumer_mode": (
                "reader_cross_attn"
                if arm_id.startswith("o4_")
                else ("reader_lora_sequence" if arm_id.startswith(("o2_", "o3_")) else "legacy_prefix")
            ),
            "cross_attn_gate_open_fraction": float(cross_attn_gate_open_fraction),
            "prefix_attention_nontrivial_layer_count": 1 if memory_tokens_count > 0 else 0,
            "prefix_artifact_stats": {
                "memory_tokens_count": int(memory_tokens_count),
                "memory_token_attention_mass_mean": float(memory_token_attention_mass_mean),
            },
        }
        (arm_dir / "metrics.json").write_text(json.dumps(metrics) + "\n")
        payload = {"events": train_events or []}
        (arm_dir / "train_events.json").write_text(json.dumps(payload) + "\n")

    def test_build_summary_selects_prompts_and_marks_smokes_alive(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_root = Path(temp_dir) / "runs"
            run_root.mkdir(parents=True, exist_ok=True)
            selected_prompt_path = Path(temp_dir) / "selected-prompt-modes.json"
            qwen25_reference_path = Path(temp_dir) / "qwen25-reference.json"

            self._write_arm(
                run_root,
                "b0_q3_gsm8k_nonthink",
                benchmark_id="gsm8k",
                prompt_variant="q3_gsm8k_nonthink",
                task_score=0.18,
            )
            self._write_arm(
                run_root,
                "b1_q3_gsm8k_think_boxed",
                benchmark_id="gsm8k",
                prompt_variant="q3_gsm8k_think_boxed",
                task_score=0.29,
            )
            self._write_arm(
                run_root,
                "b2_q3_trivia_nonthink",
                benchmark_id="triviaqa",
                prompt_variant="q3_trivia_nonthink",
                task_score=0.22,
            )
            self._write_arm(
                run_root,
                "b3_q3_trivia_think",
                benchmark_id="triviaqa",
                prompt_variant="q3_trivia_think",
                task_score=0.24,
            )
            self._write_arm(
                run_root,
                "b4_q3_fever_nonthink",
                benchmark_id="fever",
                prompt_variant="answer_slot_labels",
                task_score=0.71,
                macro_f1=0.69,
            )

            self._write_arm(
                run_root,
                "o0_q25_prefix_replay_gsm8k",
                benchmark_id="gsm8k",
                prompt_variant="q3_gsm8k_think_boxed",
                task_score=0.11,
            )
            self._write_arm(
                run_root,
                "o0_q25_prefix_replay_triviaqa",
                benchmark_id="triviaqa",
                prompt_variant="q3_trivia_think",
                task_score=0.14,
            )
            self._write_arm(
                run_root,
                "o1_q3_prefix_oracle_mid4_gsm8k",
                benchmark_id="gsm8k",
                prompt_variant="q3_gsm8k_think_boxed",
                task_score=0.21,
                memory_tokens_count=16,
            )
            self._write_arm(
                run_root,
                "o1_q3_prefix_oracle_mid4_triviaqa",
                benchmark_id="triviaqa",
                prompt_variant="q3_trivia_think",
                task_score=0.20,
                memory_tokens_count=16,
            )
            self._write_arm(
                run_root,
                "o2_q3_seq_oracle16_gsm8k",
                benchmark_id="gsm8k",
                prompt_variant="q3_gsm8k_think_boxed",
                task_score=0.23,
                memory_tokens_count=16,
            )
            self._write_arm(
                run_root,
                "o2_q3_seq_oracle16_triviaqa",
                benchmark_id="triviaqa",
                prompt_variant="q3_trivia_think",
                task_score=0.25,
                memory_tokens_count=16,
            )
            self._write_arm(
                run_root,
                "o3_q3_seq_oracle32_gsm8k",
                benchmark_id="gsm8k",
                prompt_variant="q3_gsm8k_think_boxed",
                task_score=0.26,
                memory_tokens_count=32,
            )
            self._write_arm(
                run_root,
                "o3_q3_seq_oracle32_triviaqa",
                benchmark_id="triviaqa",
                prompt_variant="q3_trivia_think",
                task_score=0.27,
                memory_tokens_count=32,
            )
            self._write_arm(
                run_root,
                "o4_q3_xattn_oracle_smoke_gsm8k",
                benchmark_id="gsm8k",
                prompt_variant="q3_gsm8k_think_boxed",
                task_score=0.28,
                memory_tokens_count=16,
                cross_attn_gate_open_fraction=0.43,
                memory_token_attention_mass_mean=0.07,
                train_events=[
                    {"grad_norm_reader_cross_attn": 0.0},
                    {"grad_norm_reader_cross_attn": 0.19},
                    {"grad_norm_reader_cross_attn": 0.31},
                ],
            )
            self._write_arm(
                run_root,
                "o4_q3_xattn_oracle_smoke_triviaqa",
                benchmark_id="triviaqa",
                prompt_variant="q3_trivia_think",
                task_score=0.29,
                memory_tokens_count=16,
                cross_attn_gate_open_fraction=0.37,
                memory_token_attention_mass_mean=0.05,
                train_events=[
                    {"grad_norm_reader_cross_attn": 0.0},
                    {"grad_norm_reader_cross_attn": 0.22},
                    {"grad_norm_reader_cross_attn": 0.33},
                ],
            )

            selected_prompt_path.write_text(
                json.dumps(
                    {
                        "gsm8k": {
                            "selected_arm_id": "b1_q3_gsm8k_think_boxed",
                            "selected_prompt_variant": "q3_gsm8k_think_boxed",
                        },
                        "triviaqa": {
                            "selected_arm_id": "b3_q3_trivia_think",
                            "selected_prompt_variant": "q3_trivia_think",
                        },
                        "fever": {
                            "selected_arm_id": "b4_q3_fever_nonthink",
                            "selected_prompt_variant": "answer_slot_labels",
                        },
                    }
                )
                + "\n"
            )
            qwen25_reference_path.write_text(
                json.dumps(
                    {
                        "baseline_replay": {
                            "c_add": {
                                "tasks": {
                                    "gsm8k": {"task_score": 0.07},
                                    "triviaqa": {"task_score": 0.10},
                                }
                            }
                        }
                    }
                )
                + "\n"
            )

            summary = build_summary(
                run_root=run_root,
                qwen25_reference_summary=qwen25_reference_path,
                selected_prompt_modes_path=selected_prompt_path,
            )

            self.assertEqual(summary["selected_prompt_modes_by_task"]["gsm8k"], "q3_gsm8k_think_boxed")
            self.assertEqual(summary["selected_prompt_modes_by_task"]["triviaqa"], "q3_trivia_think")
            self.assertTrue(summary["ri1_passed_basic_smoke"])
            self.assertTrue(summary["ri2_passed_basic_smoke"])
            self.assertEqual(summary["selected_primary_baseline_scores"]["gsm8k"], 0.29)
            self.assertEqual(summary["selected_qwen3_baseline_scores"]["triviaqa"], 0.24)
            self.assertTrue(summary["qwen3_primary_beats_q25_replay_on_any_task"])
            self.assertTrue(summary["qwen3_primary_beats_historical_q25_on_any_task"])
            self.assertTrue(summary["legacy_prefix_oracle_reproduced_or_bounded"])
            self.assertEqual(summary["recommended_next_step"], "open_v8_1_reader_interface_scout")
            self.assertEqual(
                summary["comparison_conclusion"],
                "qwen3_calibrated_interfaces_alive_open_v8_1",
            )

    def test_build_summary_supports_qwen34_parallel_track(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_root = Path(temp_dir) / "runs"
            run_root.mkdir(parents=True, exist_ok=True)
            selected_prompt_path = Path(temp_dir) / "selected-prompt-modes.json"
            qwen25_reference_path = Path(temp_dir) / "qwen25-reference.json"

            baseline_specs = [
                ("b0_q34_gsm8k_nonthink", "gsm8k", "q3_gsm8k_nonthink", 0.18, 0.0),
                ("b1_q34_gsm8k_think_boxed", "gsm8k", "q3_gsm8k_think_boxed", 0.29, 0.0),
                ("b2_q34_trivia_nonthink", "triviaqa", "q3_trivia_nonthink", 0.22, 0.0),
                ("b3_q34_trivia_think", "triviaqa", "q3_trivia_think", 0.24, 0.0),
                ("b4_q34_fever_nonthink", "fever", "answer_slot_labels", 0.71, 0.69),
            ]
            for arm_id, benchmark_id, prompt_variant, task_score, macro_f1 in baseline_specs:
                self._write_arm(
                    run_root,
                    arm_id,
                    benchmark_id=benchmark_id,
                    prompt_variant=prompt_variant,
                    task_score=task_score,
                    macro_f1=macro_f1,
                )

            oracle_specs = [
                ("o0_q25_prefix_replay_gsm8k", "gsm8k", "q3_gsm8k_think_boxed", 0.11, 16, 0.0, 0.0, None),
                ("o0_q25_prefix_replay_triviaqa", "triviaqa", "q3_trivia_think", 0.14, 16, 0.0, 0.0, None),
                ("o1_q34_prefix_oracle_mid4_gsm8k", "gsm8k", "q3_gsm8k_think_boxed", 0.21, 16, 0.0, 0.0, None),
                ("o1_q34_prefix_oracle_mid4_triviaqa", "triviaqa", "q3_trivia_think", 0.20, 16, 0.0, 0.0, None),
                ("o2_q34_seq_oracle16_gsm8k", "gsm8k", "q3_gsm8k_think_boxed", 0.23, 16, 0.0, 0.0, None),
                ("o2_q34_seq_oracle16_triviaqa", "triviaqa", "q3_trivia_think", 0.25, 16, 0.0, 0.0, None),
                ("o3_q34_seq_oracle32_gsm8k", "gsm8k", "q3_gsm8k_think_boxed", 0.26, 32, 0.0, 0.0, None),
                ("o3_q34_seq_oracle32_triviaqa", "triviaqa", "q3_trivia_think", 0.27, 32, 0.0, 0.0, None),
                (
                    "o4_q34_xattn_oracle_smoke_gsm8k",
                    "gsm8k",
                    "q3_gsm8k_think_boxed",
                    0.28,
                    16,
                    0.43,
                    0.07,
                    [
                        {"grad_norm_reader_cross_attn": 0.0},
                        {"grad_norm_reader_cross_attn": 0.19},
                        {"grad_norm_reader_cross_attn": 0.31},
                    ],
                ),
                (
                    "o4_q34_xattn_oracle_smoke_triviaqa",
                    "triviaqa",
                    "q3_trivia_think",
                    0.29,
                    16,
                    0.37,
                    0.05,
                    [
                        {"grad_norm_reader_cross_attn": 0.0},
                        {"grad_norm_reader_cross_attn": 0.22},
                        {"grad_norm_reader_cross_attn": 0.33},
                    ],
                ),
            ]
            for (
                arm_id,
                benchmark_id,
                prompt_variant,
                task_score,
                memory_tokens_count,
                cross_attn_gate_open_fraction,
                memory_token_attention_mass_mean,
                train_events,
            ) in oracle_specs:
                self._write_arm(
                    run_root,
                    arm_id,
                    benchmark_id=benchmark_id,
                    prompt_variant=prompt_variant,
                    task_score=task_score,
                    memory_tokens_count=memory_tokens_count,
                    cross_attn_gate_open_fraction=cross_attn_gate_open_fraction,
                    memory_token_attention_mass_mean=memory_token_attention_mass_mean,
                    train_events=train_events,
                )

            selected_prompt_path.write_text(
                json.dumps(
                    {
                        "gsm8k": {
                            "selected_arm_id": "b1_q34_gsm8k_think_boxed",
                            "selected_prompt_variant": "q3_gsm8k_think_boxed",
                        },
                        "triviaqa": {
                            "selected_arm_id": "b3_q34_trivia_think",
                            "selected_prompt_variant": "q3_trivia_think",
                        },
                        "fever": {
                            "selected_arm_id": "b4_q34_fever_nonthink",
                            "selected_prompt_variant": "answer_slot_labels",
                        },
                    }
                )
                + "\n"
            )
            qwen25_reference_path.write_text(
                json.dumps(
                    {
                        "baseline_replay": {
                            "c_add": {
                                "tasks": {
                                    "gsm8k": {"task_score": 0.07},
                                    "triviaqa": {"task_score": 0.10},
                                }
                            }
                        }
                    }
                )
                + "\n"
            )

            summary = build_summary(
                run_root=run_root,
                qwen25_reference_summary=qwen25_reference_path,
                selected_prompt_modes_path=selected_prompt_path,
                primary_backbone_key="qwen34",
                primary_backbone_label="Qwen3-4B",
                primary_arm_prefix="q34",
            )

            self.assertEqual(summary["primary_backbone_label"], "Qwen3-4B")
            self.assertEqual(summary["selected_baseline_arms_by_task"]["gsm8k"], "b1_q34_gsm8k_think_boxed")
            self.assertEqual(summary["selected_qwen34_baseline_scores"]["triviaqa"], 0.24)
            self.assertTrue(summary["qwen34_primary_beats_q25_replay_on_any_task"])
            self.assertTrue(summary["qwen34_primary_beats_historical_q25_on_any_task"])
            self.assertEqual(summary["comparison_conclusion"], "qwen34_calibrated_interfaces_alive_open_v8_1")


if __name__ == "__main__":
    unittest.main()
