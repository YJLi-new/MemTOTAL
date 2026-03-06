from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from memtotal.eval.run_eval import main as eval_main


class BenchmarkEvalTest(unittest.TestCase):
    def test_exact_match_benchmark_eval_writes_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "gsm8k_eval"
            self.assertEqual(
                eval_main(
                    [
                        "--config",
                        str(ROOT / "configs/exp/benchmark_gsm8k_qwen25_smoke.yaml"),
                        "--seed",
                        "501",
                        "--output_dir",
                        str(output_dir),
                    ]
                ),
                0,
            )
            metrics = json.loads((output_dir / "metrics.json").read_text())
            predictions = [json.loads(line) for line in (output_dir / "predictions.jsonl").read_text().splitlines()]
            run_info = json.loads((output_dir / "run_info.json").read_text())
            self.assertEqual(metrics["benchmark_id"], "gsm8k")
            self.assertEqual(metrics["task_domain"], "math")
            self.assertEqual(metrics["evaluator_type"], "exact_match")
            self.assertEqual(run_info["smoke_subset"], "local_contract_v1")
            self.assertEqual(len(predictions), 2)
            self.assertEqual(predictions[0]["benchmark_id"], "gsm8k")
            self.assertIn("predicted_text", predictions[0])

    def test_multiple_choice_benchmark_eval_writes_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "gpqa_eval"
            self.assertEqual(
                eval_main(
                    [
                        "--config",
                        str(ROOT / "configs/exp/benchmark_gpqa_qwen25_smoke.yaml"),
                        "--seed",
                        "503",
                        "--output_dir",
                        str(output_dir),
                    ]
                ),
                0,
            )
            metrics = json.loads((output_dir / "metrics.json").read_text())
            predictions = [json.loads(line) for line in (output_dir / "predictions.jsonl").read_text().splitlines()]
            self.assertEqual(metrics["benchmark_id"], "gpqa")
            self.assertEqual(metrics["evaluator_type"], "multiple_choice")
            self.assertEqual(metrics["smoke_subset"], "local_contract_v1")
            self.assertEqual(len(predictions), 2)
            self.assertIn("predicted_label", predictions[0])
            self.assertIn("score", predictions[0])

    def test_memoryagentbench_eval_writes_capability_breakdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            dataset_path = tmp / "memoryagentbench.jsonl"
            dataset_path.write_text(
                json.dumps(
                    {
                        "id": "memoryagentbench-ar-000",
                        "question": "In what country is Normandy located?",
                        "answer": "France",
                        "aliases": ["France"],
                        "context": "Document 1: Normandy is located in France.",
                        "capability": "AR",
                        "capability_name": "Accurate Retrieval",
                        "memoryagent_source": "ruler_qa1_197K",
                        "question_index": 0,
                        "capability_metric_name": "exact_match",
                        "context_token_budget": 512,
                        "context_tokens_total": 7,
                        "context_tokens_used": 7,
                        "context_was_truncated": False,
                        "full_context_chars": 40,
                    }
                )
                + "\n"
            )
            config = {
                "experiment": {
                    "name": "benchmark_memoryagentbench_test",
                    "stage": "M4",
                    "method_variant": "ours-benchmark-real-smoke",
                },
                "task": {
                    "name": "memoryagentbench_real_smoke",
                    "benchmark_id": "memoryagentbench",
                    "domain": "agent",
                    "split": "eval",
                    "smoke_subset": "hf_real_smoke4_trunc512",
                    "dataset_path": str(dataset_path),
                    "metric_name": "memoryagent_score",
                    "evaluator": {"type": "memoryagentbench"},
                },
                "backbone": {
                    "name": "Qwen2.5-1.5B-Instruct",
                    "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
                    "load_mode": "stub",
                    "stub_hidden_size": 64,
                },
                "method": {
                    "embed_dim": 64,
                    "segmenter": {"mode": "delimiter", "delimiter": "||"},
                    "writer": {"memory_slots": 4, "arch": "mlp"},
                    "reader": {
                        "num_queries": 4,
                        "use_query_gating": False,
                        "gating_mode": "off",
                        "num_heads": 4,
                        "condition_on_context": True,
                        "conditioning": {"domain_key": "domain", "include_task_name": True},
                    },
                    "fuser": {"short_slots": 2, "arch": "linear"},
                    "injector": {"mode": "prefix", "enabled": True, "position": "segment"},
                },
                "runtime": {
                    "train_steps": 2,
                    "eval_examples": 1,
                    "learning_rate": 0.01,
                    "device": "cpu",
                },
            }
            config_path = tmp / "memoryagentbench.yaml"
            config_path.write_text(yaml.safe_dump(config, sort_keys=False))
            output_dir = tmp / "memoryagentbench_eval"
            self.assertEqual(
                eval_main(
                    [
                        "--config",
                        str(config_path),
                        "--seed",
                        "509",
                        "--output_dir",
                        str(output_dir),
                    ]
                ),
                0,
            )
            metrics = json.loads((output_dir / "metrics.json").read_text())
            predictions = [json.loads(line) for line in (output_dir / "predictions.jsonl").read_text().splitlines()]
            self.assertEqual(metrics["benchmark_id"], "memoryagentbench")
            self.assertIn("capability_scores", metrics)
            self.assertIn("AR", metrics["capability_scores"])
            self.assertEqual(predictions[0]["capability"], "AR")
            self.assertIn("extra_metrics", predictions[0])

    def test_narrativeqa_eval_writes_f1_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            dataset_path = tmp / "narrativeqa.jsonl"
            dataset_path.write_text(
                json.dumps(
                    {
                        "id": "narrativeqa-000",
                        "story": "Alice travels to Paris and solves a mystery.",
                        "question": "Where does Alice travel?",
                        "answer": "Paris",
                        "aliases": ["Paris", "She travels to Paris"],
                        "summary_title": "Alice Story",
                        "document_kind": "movie",
                        "story_chars": 44,
                        "story_word_count": 8,
                        "narrativeqa_view": "summary_only",
                    }
                )
                + "\n"
            )
            config = {
                "experiment": {
                    "name": "benchmark_narrativeqa_test",
                    "stage": "M4",
                    "method_variant": "ours-benchmark-real-smoke",
                },
                "task": {
                    "name": "narrativeqa_real_smoke",
                    "benchmark_id": "narrativeqa",
                    "domain": "narrative",
                    "split": "eval",
                    "smoke_subset": "hf_real_smoke4_summary_only",
                    "dataset_path": str(dataset_path),
                    "metric_name": "f1",
                    "evaluator": {"type": "qa_f1"},
                },
                "backbone": {
                    "name": "Qwen2.5-1.5B-Instruct",
                    "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
                    "load_mode": "stub",
                    "stub_hidden_size": 64,
                },
                "method": {
                    "embed_dim": 64,
                    "segmenter": {"mode": "delimiter", "delimiter": "||"},
                    "writer": {"memory_slots": 4, "arch": "mlp"},
                    "reader": {
                        "num_queries": 4,
                        "use_query_gating": False,
                        "gating_mode": "off",
                        "num_heads": 4,
                        "condition_on_context": True,
                        "conditioning": {"domain_key": "domain", "include_task_name": True},
                    },
                    "fuser": {"short_slots": 2, "arch": "linear"},
                    "injector": {"mode": "prefix", "enabled": True, "position": "segment"},
                },
                "runtime": {
                    "train_steps": 2,
                    "eval_examples": 1,
                    "learning_rate": 0.01,
                    "device": "cpu",
                },
            }
            config_path = tmp / "narrativeqa.yaml"
            config_path.write_text(yaml.safe_dump(config, sort_keys=False))
            output_dir = tmp / "narrativeqa_eval"
            self.assertEqual(
                eval_main(
                    [
                        "--config",
                        str(config_path),
                        "--seed",
                        "511",
                        "--output_dir",
                        str(output_dir),
                    ]
                ),
                0,
            )
            metrics = json.loads((output_dir / "metrics.json").read_text())
            predictions = [json.loads(line) for line in (output_dir / "predictions.jsonl").read_text().splitlines()]
            self.assertEqual(metrics["benchmark_id"], "narrativeqa")
            self.assertEqual(metrics["metric_name"], "f1")
            self.assertIn("extra_metrics", predictions[0])
            self.assertIn("f1", predictions[0]["extra_metrics"])


if __name__ == "__main__":
    unittest.main()
