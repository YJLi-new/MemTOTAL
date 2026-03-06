from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from memtotal.baselines.run_memgen import (
    _baseline_metadata,
    _build_memgen_options,
    _memgen_runtime_env,
    _preflight_failure,
    _resolve_load_model_path,
    _translate_conversations_txt,
)


class MemGenAdapterTest(unittest.TestCase):
    def test_translate_dynamic_conversations(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            conversations_path = temp_root / "conversations.txt"
            output_dir = temp_root / "translated"
            output_dir.mkdir()

            conversations_path.write_text(
                "DynamicEvalRecorder Log\n\n"
                "Conversation:\n<|im_start|>user\nQuestion?<|im_end|>\n"
                "Reward: 1.0000\n"
                "----------------------------------------\n"
                "Conversation:\n<|im_start|>user\nAnother?<|im_end|>\n"
                "Reward: 0.0000\n"
                "----------------------------------------\n"
                "\nFinal Results\n========================================\nAverage Reward: 0.5000\n"
            )

            metrics = _translate_conversations_txt(conversations_path, output_dir)
            self.assertEqual(metrics["num_predictions"], 2)
            self.assertEqual(metrics["compute_reward"], 0.5)

            rows = [
                json.loads(line)
                for line in (output_dir / "predictions.jsonl").read_text().splitlines()
                if line.strip()
            ]
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["metric_compute_reward"], 1.0)

    def test_preflight_failure_for_gated_gpqa_without_token(self) -> None:
        with patch("memtotal.baselines.run_memgen._has_hf_auth_token", return_value=False):
            message = _preflight_failure({"baseline": {"task_name": "gpqa"}})
        self.assertIsNotNone(message)
        self.assertIn("huggingface-cli login", message)

    def test_resolve_relative_load_model_path_from_repo_root(self) -> None:
        config = {
            "baseline": {
                "repo_root": "MemGen-master",
                "load_model_path": "results/train/example/trigger",
            }
        }
        resolved = _resolve_load_model_path(config)
        self.assertEqual(
            resolved,
            (ROOT / "MemGen-master" / "results/train/example/trigger").resolve(),
        )

    def test_preflight_requires_trained_checkpoint(self) -> None:
        message = _preflight_failure(
            {
                "baseline": {
                    "task_name": "gsm8k",
                    "repo_root": "MemGen-master",
                    "load_model_path": None,
                    "requires_trained_checkpoint": True,
                }
            }
        )
        self.assertIsNotNone(message)
        self.assertIn("requires a trained checkpoint", message)

    def test_build_options_includes_explicit_trigger_active(self) -> None:
        options = _build_memgen_options(
            {
                "baseline": {
                    "repo_root": "MemGen-master",
                    "task_name": "gsm8k",
                    "memgen_run_mode": "evaluate",
                    "trigger_active": True,
                    "max_prompt_aug_num": 1,
                    "max_inference_aug_num": 1,
                    "load_model_path": None,
                    "extra_options": [],
                },
                "backbone": {
                    "model_id": "/tmp/qwen",
                },
            },
            seed=7,
        )
        self.assertIn("model.trigger.active=True", options)

    def test_baseline_metadata_and_runtime_env(self) -> None:
        metadata = _baseline_metadata(
            {
                "baseline": {
                    "repo_root": "MemGen-master",
                    "task_name": "gsm8k",
                    "trigger_active": False,
                    "max_prompt_aug_num": 1,
                    "max_inference_aug_num": 2,
                    "requires_trained_checkpoint": False,
                    "insertion_profile": "single_turn_smoke",
                    "load_model_path": None,
                }
            }
        )
        self.assertEqual(metadata["insertion_profile"], "single_turn_smoke")
        self.assertEqual(metadata["max_inference_aug_num"], 2)

        env = _memgen_runtime_env()
        self.assertEqual(env["TOKENIZERS_PARALLELISM"], "false")


if __name__ == "__main__":
    unittest.main()
