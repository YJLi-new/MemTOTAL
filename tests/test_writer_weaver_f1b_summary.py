from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class WriterWeaverF1BSummaryTest(unittest.TestCase):
    def test_summary_moves_to_f2_on_geometry_only_signal(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script_path = repo_root / "scripts" / "update_writer_weaver_f1b_summary.py"
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            def write_json(name: str, payload: dict[str, object]) -> Path:
                path = tmp_path / name
                path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
                return path

            def metrics(
                *,
                benchmark_id: str,
                task_name: str,
                metric_name: str,
                task_score: float,
                exact_match: float,
                delta_answer_logprob: float,
                common_mode_energy_ratio: float,
                top1_top2_ratio: float,
                centered_effective_rank: float,
            ) -> dict[str, object]:
                return {
                    "benchmark_id": benchmark_id,
                    "task_name": task_name,
                    "task_metric_name": metric_name,
                    "best_adapt_task_score": task_score,
                    "best_adapt_exact_match": exact_match,
                    "delta_answer_logprob": delta_answer_logprob,
                    "memory_long_common_mode_energy_ratio": common_mode_energy_ratio,
                    "memory_long_top1_top2_ratio": top1_top2_ratio,
                    "memory_long_centered_effective_rank": centered_effective_rank,
                    "pilot_writer_adapter_enabled": True,
                    "pilot_writer_adapter_target_modules": [
                        "conditioning_out_proj",
                        "encoder_self_attn_out_proj",
                    ],
                    "pilot_writer_adapter_rank": 2,
                    "pilot_writer_adapter_alpha": 4.0,
                    "pilot_writer_adapter_trainable_params": 2048,
                    "train_grad_norm_writer_adapter_steps_1_4_median": 0.03,
                    "projected_memory_effective_rank": 3.6,
                    "memory_long_pairwise_cosine_mean": 0.35,
                }

            w0_summary = write_json("w0-summary.json", {"comparison_conclusion": "plumbing_only"})
            f1a_summary = write_json("f1a-summary.json", {"comparison_conclusion": "move_to_f1b"})
            gsm8k_control = write_json(
                "gsm8k-control.json",
                metrics(
                    benchmark_id="gsm8k",
                    task_name="gsm8k",
                    metric_name="exact_match",
                    task_score=0.0,
                    exact_match=0.0,
                    delta_answer_logprob=0.0,
                    common_mode_energy_ratio=0.0,
                    top1_top2_ratio=0.0,
                    centered_effective_rank=0.0,
                ),
            )
            gsm8k_w0 = write_json(
                "gsm8k-w0.json",
                metrics(
                    benchmark_id="gsm8k",
                    task_name="gsm8k",
                    metric_name="exact_match",
                    task_score=0.0,
                    exact_match=0.0,
                    delta_answer_logprob=0.0,
                    common_mode_energy_ratio=0.999995,
                    top1_top2_ratio=615.0,
                    centered_effective_rank=5.3,
                ),
            )
            gsm8k_f1a = write_json(
                "gsm8k-f1a.json",
                metrics(
                    benchmark_id="gsm8k",
                    task_name="gsm8k",
                    metric_name="exact_match",
                    task_score=0.0,
                    exact_match=0.0,
                    delta_answer_logprob=0.0,
                    common_mode_energy_ratio=1.0,
                    top1_top2_ratio=2.1e7,
                    centered_effective_rank=7.3,
                ),
            )
            gsm8k_f1b = write_json(
                "gsm8k-f1b.json",
                metrics(
                    benchmark_id="gsm8k",
                    task_name="gsm8k",
                    metric_name="exact_match",
                    task_score=0.0,
                    exact_match=0.0,
                    delta_answer_logprob=0.0,
                    common_mode_energy_ratio=0.995,
                    top1_top2_ratio=300.0,
                    centered_effective_rank=6.0,
                ),
            )
            narrativeqa_control = write_json(
                "narrativeqa-control.json",
                metrics(
                    benchmark_id="narrativeqa",
                    task_name="narrativeqa",
                    metric_name="f1",
                    task_score=0.0,
                    exact_match=0.0,
                    delta_answer_logprob=0.0,
                    common_mode_energy_ratio=0.0,
                    top1_top2_ratio=0.0,
                    centered_effective_rank=0.0,
                ),
            )
            narrativeqa_w0 = write_json(
                "narrativeqa-w0.json",
                metrics(
                    benchmark_id="narrativeqa",
                    task_name="narrativeqa",
                    metric_name="f1",
                    task_score=0.0,
                    exact_match=0.0,
                    delta_answer_logprob=0.0,
                    common_mode_energy_ratio=0.999995,
                    top1_top2_ratio=620.0,
                    centered_effective_rank=5.2,
                ),
            )
            narrativeqa_f1a = write_json(
                "narrativeqa-f1a.json",
                metrics(
                    benchmark_id="narrativeqa",
                    task_name="narrativeqa",
                    metric_name="f1",
                    task_score=0.0,
                    exact_match=0.0,
                    delta_answer_logprob=0.0,
                    common_mode_energy_ratio=1.0,
                    top1_top2_ratio=2.4e7,
                    centered_effective_rank=7.4,
                ),
            )
            narrativeqa_f1b = write_json(
                "narrativeqa-f1b.json",
                metrics(
                    benchmark_id="narrativeqa",
                    task_name="narrativeqa",
                    metric_name="f1",
                    task_score=0.0,
                    exact_match=0.0,
                    delta_answer_logprob=0.0,
                    common_mode_energy_ratio=0.996,
                    top1_top2_ratio=280.0,
                    centered_effective_rank=5.8,
                ),
            )
            fever_control = write_json(
                "fever-control.json",
                metrics(
                    benchmark_id="fever",
                    task_name="fever",
                    metric_name="accuracy",
                    task_score=0.5,
                    exact_match=0.5,
                    delta_answer_logprob=0.0,
                    common_mode_energy_ratio=0.0,
                    top1_top2_ratio=0.0,
                    centered_effective_rank=0.0,
                ),
            )
            fever_w0 = write_json(
                "fever-w0.json",
                metrics(
                    benchmark_id="fever",
                    task_name="fever",
                    metric_name="accuracy",
                    task_score=0.5,
                    exact_match=0.5,
                    delta_answer_logprob=0.0,
                    common_mode_energy_ratio=0.99998,
                    top1_top2_ratio=283.0,
                    centered_effective_rank=3.9,
                ),
            )
            fever_f1a = write_json(
                "fever-f1a.json",
                metrics(
                    benchmark_id="fever",
                    task_name="fever",
                    metric_name="accuracy",
                    task_score=0.5,
                    exact_match=0.5,
                    delta_answer_logprob=0.0,
                    common_mode_energy_ratio=1.0,
                    top1_top2_ratio=2.6e7,
                    centered_effective_rank=7.4,
                ),
            )
            fever_f1b = write_json(
                "fever-f1b.json",
                metrics(
                    benchmark_id="fever",
                    task_name="fever",
                    metric_name="accuracy",
                    task_score=0.5,
                    exact_match=0.5,
                    delta_answer_logprob=0.0,
                    common_mode_energy_ratio=0.997,
                    top1_top2_ratio=200.0,
                    centered_effective_rank=4.8,
                ),
            )

            output_json = tmp_path / "f1b-summary.json"
            output_report = tmp_path / "f1b-summary.md"
            subprocess.run(
                [
                    sys.executable,
                    str(script_path),
                    "--w0_summary_json",
                    str(w0_summary),
                    "--f1a_summary_json",
                    str(f1a_summary),
                    "--gsm8k_control_metrics_json",
                    str(gsm8k_control),
                    "--gsm8k_w0_metrics_json",
                    str(gsm8k_w0),
                    "--gsm8k_f1a_metrics_json",
                    str(gsm8k_f1a),
                    "--gsm8k_f1b_metrics_json",
                    str(gsm8k_f1b),
                    "--narrativeqa_control_metrics_json",
                    str(narrativeqa_control),
                    "--narrativeqa_w0_metrics_json",
                    str(narrativeqa_w0),
                    "--narrativeqa_f1a_metrics_json",
                    str(narrativeqa_f1a),
                    "--narrativeqa_f1b_metrics_json",
                    str(narrativeqa_f1b),
                    "--fever_control_metrics_json",
                    str(fever_control),
                    "--fever_w0_metrics_json",
                    str(fever_w0),
                    "--fever_f1a_metrics_json",
                    str(fever_f1a),
                    "--fever_f1b_metrics_json",
                    str(fever_f1b),
                    "--output_json",
                    str(output_json),
                    "--output_report",
                    str(output_report),
                ],
                cwd=repo_root,
                check=True,
            )

            summary = json.loads(output_json.read_text())
            self.assertEqual(summary["comparison_conclusion"], "move_to_f2")
            self.assertFalse(summary["move_to_w2"])
            self.assertTrue(summary["move_to_f2"])
            self.assertFalse(summary["stop_after_f1b"])
            self.assertIn("comparison_conclusion: move_to_f2", output_report.read_text())


if __name__ == "__main__":
    unittest.main()
