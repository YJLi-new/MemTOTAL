from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from memtotal.analysis.run_analysis import main as analysis_main
from memtotal.training.run_train import main as train_main


class FailureChecksTest(unittest.TestCase):
    def test_m3_failure_checks_write_expected_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            stage_a_dir = root / "stage_a"
            stage_b_dir = root / "stage_b"
            analysis_dir = root / "failure_checks"

            self.assertEqual(
                train_main(
                    [
                        "--config",
                        str(ROOT / "configs/exp/m3_stage_a_qwen25_smoke.yaml"),
                        "--seed",
                        "301",
                        "--output_dir",
                        str(stage_a_dir),
                    ]
                ),
                0,
            )
            self.assertEqual(
                train_main(
                    [
                        "--config",
                        str(ROOT / "configs/exp/m3_stage_b_qwen25_smoke.yaml"),
                        "--seed",
                        "303",
                        "--output_dir",
                        str(stage_b_dir),
                        "--resume",
                        str(stage_a_dir),
                    ]
                ),
                0,
            )
            self.assertEqual(
                analysis_main(
                    [
                        "--config",
                        str(ROOT / "configs/exp/m3_failure_checks_qwen25_smoke.yaml"),
                        "--seed",
                        "307",
                        "--output_dir",
                        str(analysis_dir),
                        "--resume",
                        str(stage_b_dir),
                    ]
                ),
                0,
            )
            self.assertTrue(analysis_dir.joinpath("failure_checks.json").exists())
            self.assertTrue(analysis_dir.joinpath("failure_ablation_summary.csv").exists())
            self.assertTrue(analysis_dir.joinpath("failure_ablation_summary.svg").exists())
            metrics = json.loads(analysis_dir.joinpath("metrics.json").read_text())
            self.assertEqual(metrics["analysis_mode"], "m3_failure_checks")
            self.assertEqual(metrics["query_learning_mode"], "meta_trained")
            self.assertEqual(metrics["checks_total"], 3)
            self.assertGreaterEqual(metrics["checks_passed"], 2)
            checks_payload = json.loads(analysis_dir.joinpath("failure_checks.json").read_text())
            self.assertEqual(set(checks_payload["checks"].keys()), {
                "reader_uses_memory",
                "writer_beats_noise",
                "fuser_avoids_collapse",
            })
            self.assertEqual(
                [row["variant"] for row in checks_payload["variants"]],
                ["base", "zero_memory", "writer_noise", "collapsed_fuser"],
            )


if __name__ == "__main__":
    unittest.main()
