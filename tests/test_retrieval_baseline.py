from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from memtotal.baselines.retrieval import RetrievalBaselineRuntime


class RetrievalBaselineRuntimeTest(unittest.TestCase):
    def test_forwards_chat_template_backbone_options(self) -> None:
        seen_kwargs = {}

        class DummyBackbone:
            def __init__(self, **kwargs):
                seen_kwargs.update(kwargs)

        config = {
            "backbone": {
                "name": "Qwen3-4B",
                "model_id": "/tmp/qwen34",
                "load_mode": "hf_causal_lm",
                "dtype": "bfloat16",
                "cache_dir": "/tmp/hf-cache",
                "max_new_tokens": 64,
                "attn_implementation": "sdpa",
                "gradient_checkpointing": False,
                "use_chat_template": True,
                "chat_template_enable_thinking": False,
            },
            "baseline": {
                "family": "rag",
                "mode": "retrieval_augmented",
                "rag": {
                    "retriever": "lexical_overlap",
                },
            },
            "runtime": {
                "device": "cuda",
            },
        }

        with patch("memtotal.baselines.retrieval.BackboneWrapper", DummyBackbone):
            RetrievalBaselineRuntime(config=config, seed=7)

        self.assertEqual(seen_kwargs["attn_implementation"], "sdpa")
        self.assertEqual(seen_kwargs["use_chat_template"], True)
        self.assertEqual(seen_kwargs["chat_template_enable_thinking"], False)
        self.assertEqual(seen_kwargs["max_new_tokens"], 64)


if __name__ == "__main__":
    unittest.main()
