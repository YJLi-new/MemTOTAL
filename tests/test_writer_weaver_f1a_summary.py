from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class WriterWeaverF1ASummaryTest(unittest.TestCase):
    def test_summary_moves_to_w2_on_weak_success(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script_path = repo_root / "scripts" / "update_writer_weaver_f1a_summary.py"
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
                conditioning_layers: int,
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
                    "pilot_writer_conditioning_layers": conditioning_layers,
                    "projected_memory_effective_rank": 4.2,
                    "memory_long_pairwise_cosine_mean": 0.4,
                }

            w0_summary = write_json("w0-summary.json", {"comparison_conclusion": "plumbing_only"})
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
                    conditioning_layers=1,
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
                    conditioning_layers=1,
                ),
            )
            gsm8k_f1a = write_json(
                "gsm8k-f1a.json",
                metrics(
                    benchmark_id="gsm8k",
                    task_name="gsm8k",
                    metric_name="exact_match",
                    task_score=0.1,
                    exact_match=0.1,
                    delta_answer_logprob=0.2,
                    common_mode_energy_ratio=0.95,
                    top1_top2_ratio=20.0,
                    centered_effective_rank=4.2,
                    conditioning_layers=3,
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
                    conditioning_layers=1,
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
                    conditioning_layers=1,
                ),
            )
            narrativeqa_f1a = write_json(
                "narrativeqa-f1a.json",
                metrics(
                    benchmark_id="narrativeqa",
                    task_name="narrativeqa",
                    metric_name="f1",
                    task_score=0.05,
                    exact_match=0.0,
                    delta_answer_logprob=0.05,
                    common_mode_energy_ratio=0.96,
                    top1_top2_ratio=25.0,
                    centered_effective_rank=3.5,
                    conditioning_layers=3,
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
                    conditioning_layers=1,
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
                    conditioning_layers=1,
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
                    common_mode_energy_ratio=0.98,
                    top1_top2_ratio=40.0,
                    centered_effective_rank=4.0,
                    conditioning_layers=3,
                ),
            )

            output_json = tmp_path / "f1a-summary.json"
            output_report = tmp_path / "f1a-summary.md"
            subprocess.run(
                [
                    sys.executable,
                    str(script_path),
                    "--w0_summary_json",
                    str(w0_summary),
                    "--gsm8k_control_metrics_json",
                    str(gsm8k_control),
                    "--gsm8k_w0_metrics_json",
                    str(gsm8k_w0),
                    "--gsm8k_f1a_metrics_json",
                    str(gsm8k_f1a),
                    "--narrativeqa_control_metrics_json",
                    str(narrativeqa_control),
                    "--narrativeqa_w0_metrics_json",
                    str(narrativeqa_w0),
                    "--narrativeqa_f1a_metrics_json",
                    str(narrativeqa_f1a),
                    "--fever_control_metrics_json",
                    str(fever_control),
                    "--fever_w0_metrics_json",
                    str(fever_w0),
                    "--fever_f1a_metrics_json",
                    str(fever_f1a),
                    "--output_json",
                    str(output_json),
                    "--output_report",
                    str(output_report),
                ],
                cwd=repo_root,
                check=True,
            )

            summary = json.loads(output_json.read_text())
            self.assertEqual(summary["comparison_conclusion"], "move_to_w2")
            self.assertTrue(summary["move_to_w2"])
            self.assertFalse(summary["move_to_f1b"])
            self.assertIn("comparison_conclusion: move_to_w2", output_report.read_text())


if __name__ == "__main__":
    unittest.main()
