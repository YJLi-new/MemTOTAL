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
    _mean_numeric_payloads,
    _resolve_gradient_accumulation_steps,
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

    def test_gradient_accumulation_helper_clamps_to_positive_integer(self) -> None:
        self.assertEqual(
            _resolve_gradient_accumulation_steps({"runtime": {"pilot_gradient_accumulation_steps": 4}}),
            4,
        )
        self.assertEqual(
            _resolve_gradient_accumulation_steps({"runtime": {"pilot_gradient_accumulation_steps": 0}}),
            1,
        )

    def test_mean_numeric_payloads_averages_nested_metrics(self) -> None:
        payload = _mean_numeric_payloads(
            [
                {
                    "loss": 8.0,
                    "delta_answer_logprob": 1.0,
                    "prefix_attention_mass_mean_by_layer": {"0": 0.1, "1": 0.3},
                },
                {
                    "loss": 4.0,
                    "delta_answer_logprob": 3.0,
                    "prefix_attention_mass_mean_by_layer": {"0": 0.3, "1": 0.5},
                },
            ]
        )
        self.assertAlmostEqual(payload["loss"], 6.0, places=6)
        self.assertAlmostEqual(payload["delta_answer_logprob"], 2.0, places=6)
        self.assertAlmostEqual(payload["prefix_attention_mass_mean_by_layer"]["0"], 0.2, places=6)
        self.assertAlmostEqual(payload["prefix_attention_mass_mean_by_layer"]["1"], 0.4, places=6)

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

    def test_jointpeft_summary_uses_post_unfreeze_windows(self) -> None:
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

            def write_writer_events(name: str, *, task_loss_start: float, task_loss_end: float) -> Path:
                path = tmp_path / name
                payload: list[dict[str, object]] = []
                for step in range(1, 121):
                    frozen = step <= 50
                    step_fraction = (step - 1) / 119.0
                    loss = task_loss_start + ((task_loss_end - task_loss_start) * step_fraction)
                    payload.append(
                        {
                            "step": step,
                            "loss": loss,
                            "writer_frozen": frozen,
                            "gradient_probe_step_active": not frozen,
                            "grad_norm_writer": 0.0 if frozen else 0.02,
                            "grad_norm_projector": 0.0 if frozen else 0.05,
                            "grad_norm_receiver_lora": 0.0 if frozen else 0.03,
                            "grad_probe_writer_task_only_norm": 0.0 if frozen else 0.015,
                            "grad_probe_writer_aux_only_norm": 0.0 if frozen else 0.01,
                            "grad_probe_writer_total_norm": 0.0 if frozen else 0.05,
                            "grad_probe_writer_task_aux_cosine": 0.0 if frozen else 0.1,
                            "grad_probe_writer_task_total_cosine": 0.0 if frozen else 0.4,
                            "grad_probe_writer_aux_total_cosine": 0.0 if frozen else 0.3,
                            "was_grad_clipped_writer": False,
                            "was_grad_clipped_projector": False,
                            "was_grad_clipped_receiver_lora": False,
                        }
                    )
                path.write_text(json.dumps({"events": payload}, indent=2) + "\n")
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
                    "prefix_attention_mass_mean_by_layer": {
                        "0": prefix_attention_mass_mean,
                        "1": prefix_attention_mass_mean * 0.8,
                    },
                    "projected_memory_effective_rank": 10.0,
                    "memory_long_common_mode_energy_ratio": 0.999999,
                    "train_loss_steps_1_50_median": 9.0,
                    "train_loss_steps_451_500_median": 3.0,
                    "train_grad_norm_writer_steps_1_50_median": 0.0,
                    "train_grad_norm_writer_steps_451_500_median": 0.01,
                    "train_grad_norm_projector_steps_1_50_median": 0.0,
                    "train_grad_norm_projector_steps_451_500_median": 0.02,
                    "train_grad_norm_receiver_lora_steps_1_50_median": 0.0,
                    "train_grad_norm_receiver_lora_steps_451_500_median": 0.01,
                    "train_final_support_state_effective_rank": 1.0,
                    "train_final_memory_long_effective_rank": 1.0,
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
            gsm8k_writer_events = write_writer_events(
                "gsm8k-writer-events.json",
                task_loss_start=9.0,
                task_loss_end=3.0,
            )
            narrativeqa_writer_events = write_writer_events(
                "narrativeqa-writer-events.json",
                task_loss_start=10.0,
                task_loss_end=4.0,
            )
            fever_writer_events = write_writer_events(
                "fever-writer-events.json",
                task_loss_start=6.0,
                task_loss_end=2.0,
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
                    "--gsm8k_writer_train_events_json",
                    str(gsm8k_writer_events),
                    "--narrativeqa_control_metrics_json",
                    str(narrativeqa_control),
                    "--narrativeqa_writer_metrics_json",
                    str(narrativeqa_writer),
                    "--narrativeqa_writer_train_events_json",
                    str(narrativeqa_writer_events),
                    "--fever_control_metrics_json",
                    str(fever_control),
                    "--fever_writer_metrics_json",
                    str(fever_writer),
                    "--fever_writer_train_events_json",
                    str(fever_writer_events),
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
            self.assertEqual(summary["comparison_conclusion"], "move_to_support_screening")
            self.assertEqual(summary["recommended_next_step"], "run_support_interface_screen")
            self.assertTrue(summary["move_to_support_screening"])
            self.assertFalse(summary["any_nonfever_usefulness_positive"])
            self.assertTrue(summary["source_stub_health"]["route_live"])
            self.assertEqual(summary["source_stub_health"]["delta_answer_logprob"], 0.0)
            self.assertTrue(summary["gsm8k"]["route_live_post_unfreeze"])
            self.assertTrue(summary["gsm8k"]["writer_task_supervision_live"])
            self.assertIn("comparison_conclusion: move_to_support_screening", report)

    def test_jointpeft_summary_classification_usefulness_does_not_require_positive_mean_delta(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script_path = repo_root / "scripts" / "update_writer_deep_prefix_jointpeft_summary.py"
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            def write_json(name: str, payload: dict[str, object]) -> Path:
                path = tmp_path / name
                path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
                return path

            def write_events(name: str) -> Path:
                path = tmp_path / name
                payload = {
                    "events": [
                        {
                            "step": step,
                            "loss": 6.0 - (0.04 * step),
                            "writer_frozen": step <= 10,
                            "gradient_probe_step_active": step > 10,
                            "grad_norm_writer": 0.0 if step <= 10 else 0.02,
                            "grad_norm_projector": 0.0 if step <= 10 else 0.05,
                            "grad_norm_receiver_lora": 0.0 if step <= 10 else 0.03,
                            "grad_probe_writer_task_only_norm": 0.0 if step <= 10 else 0.02,
                            "grad_probe_writer_aux_only_norm": 0.0 if step <= 10 else 0.01,
                            "grad_probe_writer_total_norm": 0.0 if step <= 10 else 0.04,
                            "grad_probe_writer_task_aux_cosine": 0.0 if step <= 10 else 0.2,
                            "grad_probe_writer_task_total_cosine": 0.0 if step <= 10 else 0.5,
                            "grad_probe_writer_aux_total_cosine": 0.0 if step <= 10 else 0.4,
                            "was_grad_clipped_writer": False,
                            "was_grad_clipped_projector": False,
                            "was_grad_clipped_receiver_lora": False,
                        }
                        for step in range(1, 61)
                    ]
                }
                path.write_text(json.dumps(payload, indent=2) + "\n")
                return path

            def write_rows(name: str, rows: list[dict[str, object]]) -> Path:
                path = tmp_path / name
                path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
                return path

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
            source_stub_events = write_json(
                "source-stub-health-events.json",
                {"events": [{"step": 1, "loss": 7.5}, {"step": 2, "loss": 5.0}]},
            )
            control_rows = write_rows(
                "fever-control.jsonl",
                [
                    {
                        "example_id": "1",
                        "predicted_label": "REFUTES",
                        "gold_label": "SUPPORTS",
                        "predicted_correct": False,
                        "final_margin": -0.5,
                    },
                    {
                        "example_id": "2",
                        "predicted_label": "REFUTES",
                        "gold_label": "REFUTES",
                        "predicted_correct": True,
                        "final_margin": 0.1,
                    },
                ],
            )
            writer_rows = write_rows(
                "fever-writer.jsonl",
                [
                    {
                        "example_id": "1",
                        "predicted_label": "SUPPORTS",
                        "gold_label": "SUPPORTS",
                        "predicted_correct": True,
                        "final_margin": 0.3,
                        "delta_answer_logprob": 0.0,
                    },
                    {
                        "example_id": "2",
                        "predicted_label": "REFUTES",
                        "gold_label": "REFUTES",
                        "predicted_correct": True,
                        "final_margin": 0.2,
                        "delta_answer_logprob": 0.0,
                    },
                ],
            )

            def metrics(*, task_score: float, case_dump_path: Path) -> dict[str, object]:
                return {
                    "benchmark_id": "fever",
                    "task_name": "fever",
                    "task_metric_name": "accuracy",
                    "best_adapt_task_score": task_score,
                    "best_adapt_exact_match": task_score,
                    "delta_answer_logprob": 0.0,
                    "prefix_attention_mass_mean": 0.02,
                    "prefix_attention_mass_mean_by_layer": {"0": 0.02, "1": 0.02},
                    "projected_memory_effective_rank": 6.0,
                    "memory_long_common_mode_energy_ratio": 0.95,
                    "train_final_support_state_effective_rank": 1.4,
                    "train_final_memory_long_effective_rank": 1.8,
                    "task_case_dump_path": str(case_dump_path.resolve()),
                }

            fever_control = write_json("fever-control.json", metrics(task_score=0.5, case_dump_path=control_rows))
            fever_writer = write_json("fever-writer.json", metrics(task_score=1.0, case_dump_path=writer_rows))
            writer_events = write_events("fever-writer-events.json")

            shared_control = write_json("shared-control.json", metrics(task_score=0.0, case_dump_path=writer_rows))
            shared_writer = write_json("shared-writer.json", metrics(task_score=0.0, case_dump_path=writer_rows))

            output_json = tmp_path / "summary.json"
            output_report = tmp_path / "summary.md"
            subprocess.run(
                [
                    sys.executable,
                    str(script_path),
                    "--source_stub_health_metrics_json",
                    str(source_stub_metrics),
                    "--source_stub_health_train_events_json",
                    str(source_stub_events),
                    "--gsm8k_control_metrics_json",
                    str(shared_control),
                    "--gsm8k_writer_metrics_json",
                    str(shared_writer),
                    "--gsm8k_writer_train_events_json",
                    str(writer_events),
                    "--narrativeqa_control_metrics_json",
                    str(shared_control),
                    "--narrativeqa_writer_metrics_json",
                    str(shared_writer),
                    "--narrativeqa_writer_train_events_json",
                    str(writer_events),
                    "--fever_control_metrics_json",
                    str(fever_control),
                    "--fever_writer_metrics_json",
                    str(fever_writer),
                    "--fever_writer_train_events_json",
                    str(writer_events),
                    "--post_unfreeze_window",
                    "20",
                    "--tail_window",
                    "20",
                    "--output_json",
                    str(output_json),
                    "--output_report",
                    str(output_report),
                ],
                cwd=repo_root,
                check=True,
            )

            summary = json.loads(output_json.read_text())
            self.assertTrue(summary["fever"]["usefulness_positive_v6"])
            self.assertEqual(summary["fever"]["delta_answer_logprob"], 0.0)
            self.assertGreater(summary["fever"]["margin_delta_mean"], 0.0)
            self.assertGreater(summary["fever"]["correct_flip_count"], 0.0)

    def test_jointpeft_summary_uses_only_active_probe_steps_for_task_supervision(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script_path = repo_root / "scripts" / "update_writer_deep_prefix_jointpeft_summary.py"
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            def write_json(name: str, payload: dict[str, object]) -> Path:
                path = tmp_path / name
                path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
                return path

            def write_jsonl(name: str, rows: list[dict[str, object]]) -> Path:
                path = tmp_path / name
                path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
                return path

            def write_events(name: str) -> Path:
                path = tmp_path / name
                payload = {"events": []}
                for step in range(1, 61):
                    frozen = step <= 10
                    probe_active = (step % 5 == 0) and not frozen
                    payload["events"].append(
                        {
                            "step": step,
                            "loss": 8.0 if step <= 50 else 3.0,
                            "writer_frozen": frozen,
                            "grad_norm_writer": 0.0 if frozen else 2.0,
                            "grad_norm_projector": 0.0 if frozen else 5.0,
                            "grad_norm_receiver_lora": 0.0 if frozen else 1.0,
                            "gradient_probe_step_active": probe_active,
                            "grad_probe_writer_task_only_norm": 2.5 if probe_active else 0.0,
                            "grad_probe_writer_aux_only_norm": 0.0,
                            "grad_probe_writer_total_norm": 2.5 if probe_active else 0.0,
                            "grad_probe_writer_task_aux_cosine": 0.0,
                            "grad_probe_writer_task_total_cosine": 1.0 if probe_active else 0.0,
                            "grad_probe_writer_aux_total_cosine": 0.0,
                            "was_grad_clipped_writer": False,
                            "was_grad_clipped_projector": False,
                            "was_grad_clipped_receiver_lora": False,
                        }
                    )
                path.write_text(json.dumps(payload, indent=2) + "\n")
                return path

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
            source_stub_events = write_json(
                "source-stub-health-events.json",
                {"events": [{"step": 1, "loss": 7.5}, {"step": 2, "loss": 5.0}]},
            )
            control_rows = write_jsonl(
                "control-case-dump.jsonl",
                [{"example_id": "1", "predicted_text": "a", "delta_answer_logprob": 0.0}],
            )
            writer_rows = write_jsonl(
                "writer-case-dump.jsonl",
                [{"example_id": "1", "predicted_text": "b", "delta_answer_logprob": 1.0}],
            )

            def metrics(task_name: str) -> tuple[Path, Path]:
                metric_name = "accuracy" if task_name == "fever" else "exact_match"
                control = write_json(
                    f"{task_name}-control.json",
                    {
                        "benchmark_id": task_name,
                        "task_name": task_name,
                        "task_metric_name": metric_name,
                        "best_adapt_task_score": 0.0,
                        "best_adapt_exact_match": 0.0,
                        "task_case_dump_path": str(control_rows.resolve()),
                    },
                )
                writer = write_json(
                    f"{task_name}-writer.json",
                    {
                        "benchmark_id": task_name,
                        "task_name": task_name,
                        "task_metric_name": metric_name,
                        "best_adapt_task_score": 0.0,
                        "best_adapt_exact_match": 0.0,
                        "delta_answer_logprob": 0.0,
                        "prefix_attention_mass_mean": 0.004,
                        "prefix_attention_mass_mean_by_layer": {
                            "0": 0.004,
                            "1": 0.003,
                            "2": 0.002,
                            "3": 0.0015,
                        },
                        "projected_memory_effective_rank": 12.0,
                        "memory_long_common_mode_energy_ratio": 0.9995,
                        "train_final_support_state_effective_rank": 1.0,
                        "train_final_memory_long_effective_rank": 1.1,
                        "task_case_dump_path": str(writer_rows.resolve()),
                    },
                )
                return control, writer

            gsm8k_control, gsm8k_writer = metrics("gsm8k")
            narrativeqa_control, narrativeqa_writer = metrics("narrativeqa")
            fever_control, fever_writer = metrics("fever")
            writer_events = write_events("writer-events.json")
            output_json = tmp_path / "summary.json"
            output_report = tmp_path / "summary.md"

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
                    "--gsm8k_writer_train_events_json",
                    str(writer_events),
                    "--narrativeqa_control_metrics_json",
                    str(narrativeqa_control),
                    "--narrativeqa_writer_metrics_json",
                    str(narrativeqa_writer),
                    "--narrativeqa_writer_train_events_json",
                    str(writer_events),
                    "--fever_control_metrics_json",
                    str(fever_control),
                    "--fever_writer_metrics_json",
                    str(fever_writer),
                    "--fever_writer_train_events_json",
                    str(writer_events),
                    "--post_unfreeze_window",
                    "50",
                    "--output_json",
                    str(output_json),
                    "--output_report",
                    str(output_report),
                ],
                cwd=repo_root,
                check=True,
            )

            summary = json.loads(output_json.read_text())
            self.assertTrue(summary["gsm8k"]["writer_task_supervision_live"])
            self.assertGreater(summary["gsm8k"]["writer_task_only_grad_norm_post_unfreeze_median"], 0.0)
            self.assertEqual(
                summary["gsm8k"]["writer_task_only_grad_norm_post_unfreeze_median"],
                summary["gsm8k"]["writer_total_grad_norm_post_unfreeze_median"],
            )
