from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.update_planv9_v9_1_summary import build_summary


class PlanV9V91SummaryTest(unittest.TestCase):
    def _write_manifest(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload) + "\n")

    def _write_metrics(self, run_root: Path, benchmark_name: str, baseline_id: str, **metrics: float) -> None:
        target = run_root / benchmark_name / baseline_id
        target.mkdir(parents=True, exist_ok=True)
        payload = {"task_score": metrics.get("task_score", 0.0), "examples_evaluated": int(metrics.get("examples_evaluated", 100))}
        payload.update(metrics)
        (target / "metrics.json").write_text(json.dumps(payload) + "\n")

    def test_summary_opens_v92_when_all_required_baselines_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_root = root / "runs"
            manifest_root = run_root / "materialized-manifests"
            self._write_manifest(manifest_root / "memoryagentbench-pilot.json", {"selected": 100})
            self._write_manifest(manifest_root / "longmemeval-pilot.json", {"selected": 100})
            self._write_manifest(manifest_root / "alfworld-pilot.json", {"selected": 120})
            v90_summary = root / "v9-0-summary.json"
            v90_summary.write_text(
                json.dumps(
                    {
                        "outcome_id": "O2",
                        "recommended_next_step": "hard_fail_a2_shift_mainline_consumer_to_c0_or_c2",
                    }
                )
                + "\n"
            )
            for benchmark_name in ("memoryagentbench", "longmemeval"):
                for baseline_id in ("b0_short_window", "b1_full_history", "b2_text_summary", "b3_text_rag"):
                    self._write_metrics(run_root, benchmark_name, baseline_id, task_score=0.42, examples_evaluated=100)
            for baseline_id in ("b0_short_window", "b1_full_history", "b2_text_summary", "b3_text_rag"):
                self._write_metrics(
                    run_root,
                    "alfworld",
                    baseline_id,
                    task_score=0.10,
                    success_rate=0.10,
                    examples_evaluated=120,
                    mean_steps_executed=11.0,
                )

            summary = build_summary(run_root=run_root, v90_summary_path=v90_summary)

            self.assertTrue(summary["benchmark_hardening_complete"])
            self.assertEqual(summary["recommended_next_step"], "open_v9_2_withinsession_sharedkv_scout_c0_c2")
            self.assertTrue(summary["acceptance"]["memoryagentbench_ready"])
            self.assertTrue(summary["acceptance"]["longmemeval_ready"])
            self.assertTrue(summary["acceptance"]["alfworld_ready"])

    def test_summary_holds_when_a_required_baseline_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_root = root / "runs"
            manifest_root = run_root / "materialized-manifests"
            self._write_manifest(manifest_root / "memoryagentbench-pilot.json", {"selected": 100})
            self._write_manifest(manifest_root / "longmemeval-pilot.json", {"selected": 100})
            self._write_manifest(manifest_root / "alfworld-pilot.json", {"selected": 120})
            v90_summary = root / "v9-0-summary.json"
            v90_summary.write_text(
                json.dumps(
                    {
                        "outcome_id": "O2",
                        "recommended_next_step": "hard_fail_a2_shift_mainline_consumer_to_c0_or_c2",
                    }
                )
                + "\n"
            )
            for baseline_id in ("b0_short_window", "b1_full_history", "b2_text_summary", "b3_text_rag"):
                self._write_metrics(run_root, "memoryagentbench", baseline_id, task_score=0.5, examples_evaluated=100)
                self._write_metrics(run_root, "alfworld", baseline_id, task_score=0.1, success_rate=0.1, examples_evaluated=120)
            for baseline_id in ("b0_short_window", "b1_full_history", "b2_text_summary"):
                self._write_metrics(run_root, "longmemeval", baseline_id, task_score=0.4, examples_evaluated=100)

            summary = build_summary(run_root=run_root, v90_summary_path=v90_summary)

            self.assertFalse(summary["benchmark_hardening_complete"])
            self.assertEqual(summary["recommended_next_step"], "hold_v9_1_repair_and_rerun")
            self.assertFalse(summary["acceptance"]["longmemeval_ready"])


if __name__ == "__main__":
    unittest.main()
