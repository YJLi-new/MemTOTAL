from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


class WriterDeepPrefixJointPEFTMonitorTest(unittest.TestCase):
    def test_monitor_script_renders_live_gradient_and_loss_report(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script_path = repo_root / "scripts" / "monitor_writer_deep_prefix_jointpeft.py"
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            run_root = tmp_path / "runs"
            log_path = tmp_path / "tmux.log"
            log_path.write_text("step 1 starting\nstep 8 snapshot complete\n")
            arm_dir = run_root / "gsm8k-writer" / "pilot-I-writer-direct"
            _write_jsonl(
                arm_dir / "train_trace.live.jsonl",
                [
                    {
                        "step": 1,
                        "loss": 8.0,
                        "delta_answer_logprob": -0.1,
                        "grad_norm_source_stub": 0.0,
                        "grad_norm_writer": 0.02,
                        "grad_norm_projector": 0.10,
                        "grad_norm_receiver_lora": 0.03,
                        "total_grad_norm_pre_clip": 0.15,
                        "was_grad_clipped": False,
                        "optimizer_lr_by_group": {"writer_base": 2.0e-06, "prefix_projector": 3.0e-06},
                    },
                    {
                        "step": 2,
                        "loss": 6.5,
                        "delta_answer_logprob": 0.0,
                        "grad_norm_source_stub": 0.0,
                        "grad_norm_writer": 0.03,
                        "grad_norm_projector": 0.12,
                        "grad_norm_receiver_lora": 0.04,
                        "total_grad_norm_pre_clip": 1.5,
                        "was_grad_clipped": True,
                        "optimizer_lr_by_group": {"writer_base": 4.0e-06, "prefix_projector": 6.0e-06},
                    },
                ],
            )
            _write_jsonl(
                arm_dir / "snapshot_metrics.live.jsonl",
                [
                    {
                        "step": 0,
                        "accuracy": 0.0,
                        "macro_f1": 0.0,
                        "mean_margin": -0.2,
                        "prefix_l2": 4.0,
                    },
                    {
                        "step": 10,
                        "accuracy": 0.1,
                        "macro_f1": 0.1,
                        "mean_margin": 0.05,
                        "prefix_l2": 3.5,
                    },
                ],
            )
            (run_root / "gsm8k-control").mkdir(parents=True, exist_ok=True)
            _write_json(run_root / "gsm8k-control" / "suite_metrics.json", {"rows": []})

            output_report = tmp_path / "monitor.md"
            subprocess.run(
                [
                    sys.executable,
                    str(script_path),
                    "--run_root",
                    str(run_root),
                    "--output_report",
                    str(output_report),
                    "--session_name",
                    "definitely-not-a-live-session",
                    "--log_path",
                    str(log_path),
                    "--base_seed",
                    "60917",
                    "--once",
                ],
                cwd=repo_root,
                check=True,
            )

            report = output_report.read_text()
            self.assertIn("# Writer Deep-Prefix JointPEFT Live Monitor", report)
            self.assertIn("completed_suites: 1/7", report)
            self.assertIn("## writer_gsm8k", report)
            self.assertIn("latest_step: 2", report)
            self.assertIn("recent_loss_trace: [8.0, 6.5]", report)
            self.assertIn("recent_writer_grad_trace: [0.02, 0.03]", report)
            self.assertIn("recent_clipped_steps: 1", report)
            self.assertIn("latest_snapshot_step: 10", report)
            self.assertIn("step 8 snapshot complete", report)
