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


class M3GradientAuditTest(unittest.TestCase):
    def test_gradient_audit_writes_module_gradients(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            stage_a_dir = root / "stage_a"
            stage_b_dir = root / "stage_b"
            audit_dir = root / "audit"

            self.assertEqual(
                train_main(
                    [
                        "--config",
                        str(ROOT / "configs/exp/m3_stage_a_core4_qwen25_smoke.yaml"),
                        "--seed",
                        "71",
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
                        str(ROOT / "configs/exp/m3_stage_b_core4_qwen25_smoke.yaml"),
                        "--seed",
                        "73",
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
                        str(ROOT / "configs/exp/m3_stage_c_gradient_audit_qwen25.yaml"),
                        "--seed",
                        "79",
                        "--output_dir",
                        str(audit_dir),
                        "--resume",
                        str(stage_b_dir),
                    ]
                ),
                0,
            )

            metrics = json.loads(audit_dir.joinpath("metrics.json").read_text())
            self.assertEqual(metrics["analysis_mode"], "m3_stage_c_gradient_audit")
            self.assertEqual(metrics["backbone"], "Qwen2.5-1.5B-Instruct")
            self.assertEqual(metrics["adaptation_target"], "q_only")
            self.assertEqual(metrics["trainable_module"], "reader.queries")
            self.assertIn("queries_grad_norm", metrics)
            self.assertIn("fuser_grad_norm", metrics)
            self.assertIn("writer_grad_norm", metrics)
            self.assertIn("query_to_fuser_grad_ratio", metrics)
            self.assertTrue(audit_dir.joinpath("gradient_audit.csv").exists())
            self.assertTrue(audit_dir.joinpath("gradient_audit.svg").exists())


if __name__ == "__main__":
    unittest.main()
