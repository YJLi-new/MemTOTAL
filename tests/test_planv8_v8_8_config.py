from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.planv8_v8_8_config import materialize_planv8_v8_8_config


class PlanV8V88ConfigTest(unittest.TestCase):
    def test_materialize_config_sets_confirmation_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            base_config = tmp / "base.json"
            output_config = tmp / "out.json"
            base_config.write_text(
                json.dumps(
                    {
                        "experiment": {"name": "old", "stage": "V8-6"},
                        "backbone": {"name": "old-backbone", "model_id": "old/model"},
                        "runtime": {"pilot_train_steps": 300, "pilot_snapshot_steps": [0, 300]},
                        "task": {"benchmark_id": "gsm8k"},
                    }
                )
                + "\n"
            )

            config = materialize_planv8_v8_8_config(
                base_config_path=base_config,
                output_config=output_config,
                variant_id="c2_best_writer_route",
                source_phase="V8-6",
                source_arm_id="a4_writer_opd_ansctx",
                primary_model_dir="/models/Qwen3-4B",
                primary_backbone_name="Qwen3-4B",
                train_steps=400,
            )

            self.assertEqual(config["experiment"]["stage"], "V8-8")
            self.assertEqual(config["experiment"]["method_variant"], "c2_best_writer_route")
            self.assertEqual(config["backbone"]["model_id"], "/models/Qwen3-4B")
            self.assertEqual(config["backbone"]["name"], "Qwen3-4B")
            self.assertEqual(config["runtime"]["pilot_confirmation_source_phase"], "V8-6")
            self.assertEqual(config["runtime"]["pilot_confirmation_source_arm_id"], "a4_writer_opd_ansctx")
            self.assertEqual(config["runtime"]["pilot_train_steps"], 400)
            self.assertIn(400, config["runtime"]["pilot_snapshot_steps"])


if __name__ == "__main__":
    unittest.main()
