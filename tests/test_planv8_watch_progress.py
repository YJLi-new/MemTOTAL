from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.planv8_watch_progress import build_progress_summary


class PlanV8WatchProgressTest(unittest.TestCase):
    def test_build_progress_summary_ignores_materialization_dirs_and_reports_latest_step(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_root = Path(temp_dir)
            (run_root / "materialized-configs").mkdir()
            active_arm = run_root / "r0_mid8_r32_lr5e5-gsm8k"
            active_arm.mkdir()
            (active_arm / ".suite.lock").write_text("")
            (active_arm / "train_trace.live.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"step": 25, "loss": 10.0}),
                        json.dumps({"step": 50, "loss": 9.0}),
                    ]
                )
                + "\n"
            )
            (active_arm / "snapshot_evals" / "step_0050").mkdir(parents=True)
            ((active_arm / "snapshot_evals" / "step_0050" / "metrics.json")).write_text("{}\n")

            finished_arm = run_root / "r1_mid8_r64_lr1e4-triviaqa"
            finished_arm.mkdir()
            (finished_arm / ".suite.lock").write_text("")
            (finished_arm / "metrics.json").write_text("{}\n")

            summary = build_progress_summary(run_root)

            self.assertEqual(summary["final_metrics_count"], 1)
            self.assertEqual(summary["snapshot_metrics_count"], 2)
            self.assertEqual(summary["active_arm_count"], 1)
            self.assertEqual(summary["latest_arm"], "r0_mid8_r32_lr5e5-gsm8k")
            self.assertEqual(summary["latest_step"], 50)


if __name__ == "__main__":
    unittest.main()
