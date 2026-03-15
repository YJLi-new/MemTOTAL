from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.planv8_v8_9_config import materialize_planv8_v8_9_config


class PlanV8V89ConfigTest(unittest.TestCase):
    def _write_base_config(self, path: Path) -> None:
        payload = {
            "experiment": {
                "name": "MemTOTAL_source_arm_gsm8k",
                "stage": "V8-6",
                "method_variant": "a4_writer_opd_ansctx",
            },
            "backbone": {
                "name": "Qwen3-4B",
                "model_id": "/tmp/source-model",
            },
            "task": {
                "name": "gsm8k_real_smoke",
                "benchmark_id": "gsm8k",
                "dataset_path": "/tmp/source/eval.jsonl",
                "support_dataset_path": "/tmp/source/support.jsonl",
                "train_dataset_path": "/tmp/source/train.jsonl",
                "train_support_dataset_path": "/tmp/source/support.jsonl",
                "metric_name": "exact_match",
                "evaluator": {"type": "exact_match", "normalizer": "gsm8k_final_answer"},
            },
            "runtime": {
                "pilot_prompt_variant": "q3_gsm8k_nonthink",
                "pilot_init_checkpoint_path": "/tmp/source/checkpoint.pt",
                "pilot_init_checkpoint_mode": "full",
                "pilot_checkpoint_path": "",
                "pilot_train_steps": 300,
                "pilot_snapshot_steps": [0, 10, 25, 50, 100, 150, 200, 250, 300],
            },
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    def test_materialize_training_config_retargets_cdmi_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            base_config = root / "base.json"
            output_config = root / "c2_joint_math.json"
            self._write_base_config(base_config)

            payload = materialize_planv8_v8_9_config(
                base_config_path=base_config,
                output_config=output_config,
                condition_id="c2_joint_math",
                eval_task_name="gsm8k",
                support_path=str(root / "joint_support.jsonl"),
                train_path=str(root / "joint_train.jsonl"),
                eval_path=str(root / "gsm8k_eval.jsonl"),
                source_variant_id="c2_best_writer_route",
                source_phase="V8-6",
                source_arm_id="a4_writer_opd_ansctx",
                source_interface_family="ri2_cross_attn",
                source_bridge_family="BR2",
                source_auxiliary_family="writer_opd_answer_plus_context",
                source_prompt_variant="q3_gsm8k_nonthink",
                primary_model_dir="/tmp/qwen34",
                primary_backbone_name="Qwen3-4B",
                train_steps=400,
            )

            self.assertEqual(payload["experiment"]["stage"], "V8-9")
            self.assertEqual(payload["experiment"]["method_variant"], "c2_joint_math")
            self.assertEqual(payload["backbone"]["model_id"], "/tmp/qwen34")
            self.assertEqual(payload["runtime"]["pilot_prompt_variant"], "task_native")
            self.assertEqual(payload["runtime"]["pilot_cdmi_source_bridge_family"], "BR2")
            self.assertEqual(payload["runtime"]["pilot_train_steps"], 400)
            self.assertEqual(payload["runtime"]["pilot_checkpoint_path"], "")
            self.assertEqual(
                payload["runtime"]["pilot_init_checkpoint_path"],
                "/tmp/source/checkpoint.pt",
            )
            self.assertEqual(
                payload["task"]["support_dataset_path"],
                str((root / "joint_support.jsonl").resolve()),
            )
            self.assertEqual(
                payload["task"]["dataset_path"],
                str((root / "gsm8k_eval.jsonl").resolve()),
            )

    def test_materialize_eval_only_config_switches_to_checkpoint_replay(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            base_config = root / "base.json"
            output_config = root / "c3_joint_trivia.json"
            self._write_base_config(base_config)

            payload = materialize_planv8_v8_9_config(
                base_config_path=base_config,
                output_config=output_config,
                condition_id="c3_joint_trivia",
                eval_task_name="triviaqa",
                support_path=str(root / "joint_support.jsonl"),
                train_path=str(root / "joint_train.jsonl"),
                eval_path=str(root / "trivia_eval.jsonl"),
                source_variant_id="c2_best_writer_route",
                source_phase="V8-6",
                source_arm_id="a4_writer_opd_ansctx",
                source_interface_family="ri2_cross_attn",
                source_bridge_family="BR2",
                source_auxiliary_family="writer_opd_answer_plus_context",
                source_prompt_variant="q3_trivia_nonthink",
                checkpoint_path=str(root / "joint_checkpoint.pt"),
                primary_model_dir="/tmp/qwen34",
                primary_backbone_name="Qwen3-4B",
                train_steps=400,
            )

            self.assertEqual(payload["runtime"]["pilot_prompt_variant"], "task_native")
            self.assertEqual(payload["runtime"]["pilot_train_steps"], 0)
            self.assertEqual(payload["runtime"]["pilot_snapshot_steps"], [0])
            self.assertFalse(payload["runtime"]["pilot_gradient_probe_enabled"])
            self.assertEqual(
                payload["runtime"]["pilot_checkpoint_path"],
                str((root / "joint_checkpoint.pt").resolve()),
            )
            self.assertEqual(payload["runtime"]["pilot_init_checkpoint_path"], "")


if __name__ == "__main__":
    unittest.main()
