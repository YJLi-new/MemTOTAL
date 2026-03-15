from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.planv8_v8_7_config import (
    materialize_planv8_v8_7_memgen_config,
    materialize_planv8_v8_7_rag_config,
)


class PlanV8V87ConfigTest(unittest.TestCase):
    def test_materialize_rag_config_uses_real_qwen34_eval_settings(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_config = Path(tmpdir) / "gsm8k-rag.json"
            payload = materialize_planv8_v8_7_rag_config(
                task_name="gsm8k",
                output_config=output_config,
                eval_path="/tmp/gsm8k-eval.jsonl",
                support_path="/tmp/gsm8k-support.jsonl",
                primary_model_dir="/tmp/qwen34",
                primary_backbone_name="Qwen3-4B",
            )

            self.assertEqual(payload["experiment"]["stage"], "V8-7")
            self.assertEqual(payload["baseline"]["family"], "rag")
            self.assertEqual(payload["baseline"]["support_examples"], 8)
            self.assertEqual(payload["backbone"]["use_chat_template"], True)
            self.assertEqual(payload["backbone"]["chat_template_enable_thinking"], False)
            self.assertEqual(payload["task"]["support_dataset_path"], "/tmp/gsm8k-support.jsonl")
            self.assertEqual(payload["runtime"]["eval_examples"], 64)

    def test_materialize_memgen_config_uses_trivia_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_config = Path(tmpdir) / "triviaqa-memgen.json"
            payload = materialize_planv8_v8_7_memgen_config(
                task_name="triviaqa",
                output_config=output_config,
                primary_model_dir="/tmp/qwen34",
                primary_backbone_name="Qwen3-4B",
            )

            self.assertEqual(payload["experiment"]["stage"], "V8-7")
            self.assertEqual(payload["baseline"]["task_name"], "triviaqa")
            self.assertEqual(payload["baseline"]["insertion_profile"], "dynamic_search_smoke")
            self.assertEqual(payload["baseline"]["max_inference_aug_num"], 0)
            self.assertIn("dataset.sft.max_test_samples=8", payload["baseline"]["extra_options"])

    def test_materialize_memgen_config_rejects_unsupported_task(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_config = Path(tmpdir) / "fever-memgen.json"
            with self.assertRaises(ValueError):
                materialize_planv8_v8_7_memgen_config(
                    task_name="fever",
                    output_config=output_config,
                    primary_model_dir="/tmp/qwen34",
                    primary_backbone_name="Qwen3-4B",
                )


if __name__ == "__main__":
    unittest.main()
