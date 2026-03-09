from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from memtotal.training.m4_shared_injection import (
    _apply_learning_rate_schedule,
    _learning_rate_scale,
    _resolve_lr_schedule,
    _resolve_lr_warmup_steps,
)


class WriterDeepPrefixJointPEFTSummaryTest(unittest.TestCase):
    def test_lr_schedule_helpers_support_linear_warmup(self) -> None:
        config = {"runtime": {"pilot_lr_schedule": "constant_with_linear_warmup", "pilot_lr_warmup_steps": 4}}
        self.assertEqual(_resolve_lr_schedule(config), "constant_with_linear_warmup")
        self.assertEqual(_resolve_lr_warmup_steps(config), 4)
        self.assertAlmostEqual(
            _learning_rate_scale(schedule="constant_with_linear_warmup", step=1, warmup_steps=4),
            0.25,
            places=6,
        )
        self.assertAlmostEqual(
            _learning_rate_scale(schedule="constant_with_linear_warmup", step=4, warmup_steps=4),
            1.0,
            places=6,
        )
        self.assertAlmostEqual(
            _learning_rate_scale(schedule="constant", step=1, warmup_steps=4),
            1.0,
            places=6,
        )

    def test_lr_schedule_rejects_unknown_value(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported runtime.pilot_lr_schedule"):
            _resolve_lr_schedule({"runtime": {"pilot_lr_schedule": "cosine"}})

    def test_apply_learning_rate_schedule_warms_optimizer_groups_monotonically(self) -> None:
        param_a = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
        param_b = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
        optimizer = torch.optim.AdamW(
            [
                {"name": "projector", "params": [param_a], "lr": 0.2},
                {"name": "receiver_lora", "params": [param_b], "lr": 0.1},
            ]
        )
        base_lrs = [0.2, 0.1]
        step1 = _apply_learning_rate_schedule(
            optimizer,
            base_lrs=base_lrs,
            schedule="constant_with_linear_warmup",
            warmup_steps=4,
            step=1,
        )
        step2 = _apply_learning_rate_schedule(
            optimizer,
            base_lrs=base_lrs,
            schedule="constant_with_linear_warmup",
            warmup_steps=4,
            step=2,
        )
        step4 = _apply_learning_rate_schedule(
            optimizer,
            base_lrs=base_lrs,
            schedule="constant_with_linear_warmup",
            warmup_steps=4,
            step=4,
        )
        self.assertAlmostEqual(step1["projector"], 0.05, places=6)
        self.assertAlmostEqual(step1["receiver_lora"], 0.025, places=6)
        self.assertLess(step1["projector"], step2["projector"])
        self.assertLess(step2["projector"], step4["projector"] + 1e-12)
        self.assertAlmostEqual(step4["projector"], 0.2, places=6)
        self.assertAlmostEqual(step4["receiver_lora"], 0.1, places=6)

    def test_addendum_summary_moves_to_writer_direct_validation(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script_path = repo_root / "scripts" / "update_writer_circuit_opening_addendum_summary.py"
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            def write_json(name: str, payload: dict[str, object]) -> Path:
                path = tmp_path / name
                path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
                return path

            def write_events(name: str, losses: list[float]) -> Path:
                path = tmp_path / name
                payload = [{"step": index, "loss": loss} for index, loss in enumerate(losses, start=1)]
                path.write_text(json.dumps(payload, indent=2) + "\n")
                return path

            def metrics(
                *,
                benchmark_id: str,
                task_name: str,
                task_score: float,
                exact_match: float,
                delta_answer_logprob: float,
                source_grad: float,
                receiver_grad: float,
                prefix_attention_by_layer: dict[str, float],
            ) -> dict[str, object]:
                return {
                    "benchmark_id": benchmark_id,
                    "task_name": task_name,
                    "task_metric_name": "accuracy",
                    "best_adapt_task_score": task_score,
                    "best_adapt_exact_match": exact_match,
                    "delta_answer_logprob": delta_answer_logprob,
                    "train_grad_norm_source_stub_steps_1_4_median": source_grad,
                    "train_grad_norm_receiver_lora_steps_1_4_median": receiver_grad,
                    "prefix_attention_mass_mean": (
                        sum(prefix_attention_by_layer.values()) / max(1, len(prefix_attention_by_layer))
                    ),
                    "prefix_attention_mass_mean_by_layer": prefix_attention_by_layer,
                    "projected_memory_effective_rank": 8.0,
                    "memory_long_common_mode_energy_ratio": 0.72,
                }

            output_json = tmp_path / "addendum-summary.json"
            output_report = tmp_path / "addendum-summary.md"
            control_paths: dict[str, Path] = {}
            p1a_metric_paths: dict[str, Path] = {}
            p1a_event_paths: dict[str, Path] = {}
            p2a_metric_paths: dict[str, Path] = {}
            p2a_event_paths: dict[str, Path] = {}
            for task_name, benchmark_id in (
                ("gsm8k", "gsm8k"),
                ("narrativeqa", "narrativeqa"),
                ("fever", "fever"),
            ):
                control_paths[task_name] = write_json(
                    f"{task_name}-control.json",
                    metrics(
                        benchmark_id=benchmark_id,
                        task_name=task_name,
                        task_score=0.2 if benchmark_id != "fever" else 0.5,
                        exact_match=0.2 if benchmark_id != "fever" else 0.5,
                        delta_answer_logprob=0.0,
                        source_grad=0.0,
                        receiver_grad=0.0,
                        prefix_attention_by_layer={},
                    ),
                )
                p1a_metric_paths[task_name] = write_json(
                    f"{task_name}-p1a.json",
                    metrics(
                        benchmark_id=benchmark_id,
                        task_name=task_name,
                        task_score=0.2 if benchmark_id != "fever" else 0.5,
                        exact_match=0.2 if benchmark_id != "fever" else 0.5,
                        delta_answer_logprob=0.0,
                        source_grad=0.02 if benchmark_id != "fever" else 0.005,
                        receiver_grad=0.0,
                        prefix_attention_by_layer={"0": 0.002} if benchmark_id != "fever" else {"0": 0.0015},
                    ),
                )
                p1a_event_paths[task_name] = write_events(
                    f"{task_name}-p1a-events.json",
                    [8.0, 7.5, 7.0, 6.5, 6.0, 5.8, 5.4, 5.2, 5.0, 4.8, 4.7, 4.5, 4.4, 4.2, 4.1, 4.0],
                )
                p2a_metric_paths[task_name] = write_json(
                    f"{task_name}-p2a.json",
                    metrics(
                        benchmark_id=benchmark_id,
                        task_name=task_name,
                        task_score=0.2 if benchmark_id != "fever" else 0.5,
                        exact_match=0.2 if benchmark_id != "fever" else 0.5,
                        delta_answer_logprob=0.0,
                        source_grad=0.25 if benchmark_id != "fever" else 0.08,
                        receiver_grad=4.5 if benchmark_id != "fever" else 2.0,
                        prefix_attention_by_layer={"0": 0.008, "1": 0.004} if benchmark_id != "fever" else {"0": 0.003},
                    ),
                )
                p2a_event_paths[task_name] = write_events(
                    f"{task_name}-p2a-events.json",
                    [10.0, 9.0, 8.4, 8.0, 7.5, 7.0, 6.8, 6.2, 6.0, 5.8, 5.5, 5.3, 5.1, 4.9, 4.7, 4.5],
                )

            subprocess.run(
                [
                    sys.executable,
                    str(script_path),
                    "--gsm8k_control_metrics_json",
                    str(control_paths["gsm8k"]),
                    "--gsm8k_p1a_metrics_json",
                    str(p1a_metric_paths["gsm8k"]),
                    "--gsm8k_p1a_train_events_json",
                    str(p1a_event_paths["gsm8k"]),
                    "--gsm8k_p2a_metrics_json",
                    str(p2a_metric_paths["gsm8k"]),
                    "--gsm8k_p2a_train_events_json",
                    str(p2a_event_paths["gsm8k"]),
                    "--narrativeqa_control_metrics_json",
                    str(control_paths["narrativeqa"]),
                    "--narrativeqa_p1a_metrics_json",
                    str(p1a_metric_paths["narrativeqa"]),
                    "--narrativeqa_p1a_train_events_json",
                    str(p1a_event_paths["narrativeqa"]),
                    "--narrativeqa_p2a_metrics_json",
                    str(p2a_metric_paths["narrativeqa"]),
                    "--narrativeqa_p2a_train_events_json",
                    str(p2a_event_paths["narrativeqa"]),
                    "--fever_control_metrics_json",
                    str(control_paths["fever"]),
                    "--fever_p1a_metrics_json",
                    str(p1a_metric_paths["fever"]),
                    "--fever_p1a_train_events_json",
                    str(p1a_event_paths["fever"]),
                    "--fever_p2a_metrics_json",
                    str(p2a_metric_paths["fever"]),
                    "--fever_p2a_train_events_json",
                    str(p2a_event_paths["fever"]),
                    "--output_json",
                    str(output_json),
                    "--output_report",
                    str(output_report),
                ],
                cwd=repo_root,
                check=True,
            )

            summary = json.loads(output_json.read_text())
            report = output_report.read_text()
            self.assertEqual(summary["comparison_conclusion"], "move_to_writer_direct_validation")
            self.assertEqual(summary["recommended_substrate"], "p2a_source_stub_receiver_lora_r2")
            self.assertTrue(summary["move_to_writer_direct_validation"])
            self.assertFalse(summary["stop_after_p2a"])
            self.assertTrue(summary["p2a_route_live_any_nonfever"])
            self.assertFalse(summary["p2a_usefulness_positive_any_nonfever"])
            self.assertIn("comparison_conclusion: move_to_writer_direct_validation", report)

    def test_jointpeft_summary_moves_to_layer_expansion_when_attention_is_weak(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script_path = repo_root / "scripts" / "update_writer_deep_prefix_jointpeft_summary.py"
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            def write_json(name: str, payload: dict[str, object]) -> Path:
                path = tmp_path / name
                path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
                return path

            def write_events(name: str, losses: list[float]) -> Path:
                path = tmp_path / name
                payload = [{"step": index, "loss": loss} for index, loss in enumerate(losses, start=1)]
                path.write_text(json.dumps(payload, indent=2) + "\n")
                return path

            def write_case_dump(name: str, deltas: list[float]) -> Path:
                path = tmp_path / name
                rows = [{"example_id": str(index), "delta_answer_logprob": delta} for index, delta in enumerate(deltas, start=1)]
                path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
                return path

            def control_metrics(*, benchmark_id: str, task_name: str, task_score: float) -> dict[str, object]:
                return {
                    "benchmark_id": benchmark_id,
                    "task_name": task_name,
                    "task_metric_name": "accuracy",
                    "best_adapt_task_score": task_score,
                    "best_adapt_exact_match": task_score,
                }

            def writer_metrics(
                *,
                benchmark_id: str,
                task_name: str,
                case_dump_path: Path,
                task_score: float,
                exact_match: float,
                delta_answer_logprob: float,
                prefix_attention_mass_mean: float,
            ) -> dict[str, object]:
                return {
                    "benchmark_id": benchmark_id,
                    "task_name": task_name,
                    "task_metric_name": "accuracy",
                    "best_adapt_task_score": task_score,
                    "best_adapt_exact_match": exact_match,
                    "delta_answer_logprob": delta_answer_logprob,
                    "prefix_attention_mass_mean": prefix_attention_mass_mean,
                    "prefix_attention_mass_mean_by_layer": {"0": prefix_attention_mass_mean},
                    "projected_memory_effective_rank": 10.0,
                    "memory_long_common_mode_energy_ratio": 0.6,
                    "train_loss_steps_1_50_median": 9.0,
                    "train_loss_steps_451_500_median": 3.0,
                    "train_grad_norm_writer_steps_1_50_median": 0.02,
                    "train_grad_norm_writer_steps_451_500_median": 0.01,
                    "train_grad_norm_projector_steps_1_50_median": 0.05,
                    "train_grad_norm_projector_steps_451_500_median": 0.02,
                    "train_grad_norm_receiver_lora_steps_1_50_median": 0.03,
                    "train_grad_norm_receiver_lora_steps_451_500_median": 0.01,
                    "task_case_dump_path": str(case_dump_path.resolve()),
                }

            source_stub_metrics = write_json(
                "source-stub-health-metrics.json",
                {
                    "delta_answer_logprob": 0.0,
                    "prefix_attention_mass_mean": 0.004,
                    "prefix_attention_mass_mean_by_layer": {"0": 0.004},
                    "train_grad_norm_source_stub_steps_1_4_median": 0.02,
                    "train_grad_norm_receiver_lora_steps_1_4_median": 0.5,
                    "train_loss_steps_1_50_median": 7.5,
                    "train_loss_tail_50_steps_median": 5.0,
                },
            )
            source_stub_events = write_events(
                "source-stub-health-events.json",
                [8.0, 7.6, 7.2, 6.9, 6.5, 6.1, 5.8, 5.4, 5.2, 5.0, 4.9, 4.8, 4.7, 4.7, 4.6, 4.6],
            )

            gsm8k_case_dump = write_case_dump("gsm8k-case-dump.jsonl", [0.2, -0.2, 0.0])
            narrativeqa_case_dump = write_case_dump("narrativeqa-case-dump.jsonl", [0.1, -0.1])
            fever_case_dump = write_case_dump("fever-case-dump.jsonl", [0.0, 0.0])

            gsm8k_control = write_json(
                "gsm8k-control.json",
                control_metrics(benchmark_id="gsm8k", task_name="gsm8k", task_score=0.2),
            )
            gsm8k_writer = write_json(
                "gsm8k-writer.json",
                writer_metrics(
                    benchmark_id="gsm8k",
                    task_name="gsm8k",
                    case_dump_path=gsm8k_case_dump,
                    task_score=0.2,
                    exact_match=0.2,
                    delta_answer_logprob=0.0,
                    prefix_attention_mass_mean=0.005,
                ),
            )
            narrativeqa_control = write_json(
                "narrativeqa-control.json",
                control_metrics(benchmark_id="narrativeqa", task_name="narrativeqa", task_score=0.2),
            )
            narrativeqa_writer = write_json(
                "narrativeqa-writer.json",
                writer_metrics(
                    benchmark_id="narrativeqa",
                    task_name="narrativeqa",
                    case_dump_path=narrativeqa_case_dump,
                    task_score=0.21,
                    exact_match=0.21,
                    delta_answer_logprob=0.0,
                    prefix_attention_mass_mean=0.004,
                ),
            )
            fever_control = write_json(
                "fever-control.json",
                control_metrics(benchmark_id="fever", task_name="fever", task_score=0.5),
            )
            fever_writer = write_json(
                "fever-writer.json",
                writer_metrics(
                    benchmark_id="fever",
                    task_name="fever",
                    case_dump_path=fever_case_dump,
                    task_score=0.5,
                    exact_match=0.5,
                    delta_answer_logprob=0.0,
                    prefix_attention_mass_mean=0.006,
                ),
            )

            output_json = tmp_path / "jointpeft-summary.json"
            output_report = tmp_path / "jointpeft-summary.md"
            subprocess.run(
                [
                    sys.executable,
                    str(script_path),
                    "--source_stub_health_metrics_json",
                    str(source_stub_metrics),
                    "--source_stub_health_train_events_json",
                    str(source_stub_events),
                    "--gsm8k_control_metrics_json",
                    str(gsm8k_control),
                    "--gsm8k_writer_metrics_json",
                    str(gsm8k_writer),
                    "--narrativeqa_control_metrics_json",
                    str(narrativeqa_control),
                    "--narrativeqa_writer_metrics_json",
                    str(narrativeqa_writer),
                    "--fever_control_metrics_json",
                    str(fever_control),
                    "--fever_writer_metrics_json",
                    str(fever_writer),
                    "--output_json",
                    str(output_json),
                    "--output_report",
                    str(output_report),
                ],
                cwd=repo_root,
                check=True,
            )

            summary = json.loads(output_json.read_text())
            report = output_report.read_text()
            self.assertEqual(summary["comparison_conclusion"], "move_to_layer_expansion_comparator")
            self.assertEqual(summary["recommended_next_step"], "run_deeper_layer_comparator")
            self.assertTrue(summary["move_to_layer_expansion_comparator"])
            self.assertFalse(summary["any_nonfever_usefulness_positive"])
            self.assertTrue(summary["source_stub_health"]["route_live"])
            self.assertEqual(summary["source_stub_health"]["delta_answer_logprob"], 0.0)
            self.assertIn("comparison_conclusion: move_to_layer_expansion_comparator", report)
