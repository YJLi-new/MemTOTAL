from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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
            with patch("memtotal.models.backbone.BackboneWrapper.generate", return_value=["France"]):
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
            self.assertEqual(metrics["capability_scores"]["AR"], 1.0)
            self.assertEqual(predictions[0]["capability"], "AR")
            self.assertIn("extra_metrics", predictions[0])

    def test_prompt_baseline_multiple_choice_eval_writes_baseline_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            dataset_path = tmp / "story_cloze.jsonl"
            dataset_path.write_text(
                json.dumps(
                    {
                        "id": "story-cloze-000",
                        "story": "Alice packed her bag and hurried to the station.",
                        "choices": [
                            {"label": "A", "text": "She boarded the train before sunset."},
                            {"label": "B", "text": "She stayed home and ignored the ticket."},
                        ],
                        "label": "A",
                        "answer": "She boarded the train before sunset.",
                    }
                )
                + "\n"
            )
            config = {
                "experiment": {
                    "name": "baseline_vanilla_story_cloze_test",
                    "stage": "M5",
                    "method_variant": "baseline-vanilla",
                },
                "task": {
                    "name": "story_cloze_smoke",
                    "benchmark_id": "story_cloze",
                    "domain": "narrative",
                    "split": "eval",
                    "smoke_subset": "local_contract_v1",
                    "dataset_path": str(dataset_path),
                    "metric_name": "accuracy",
                    "evaluator": {"type": "multiple_choice"},
                },
                "backbone": {
                    "name": "Qwen2.5-1.5B-Instruct",
                    "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
                    "load_mode": "stub",
                    "stub_hidden_size": 64,
                },
                "baseline": {
                    "family": "prompting",
                    "mode": "vanilla",
                },
                "runtime": {
                    "eval_examples": 1,
                    "device": "cpu",
                },
            }
            config_path = tmp / "baseline_story_cloze.yaml"
            config_path.write_text(yaml.safe_dump(config, sort_keys=False))
            output_dir = tmp / "baseline_story_cloze_eval"
            self.assertEqual(
                eval_main(
                    [
                        "--config",
                        str(config_path),
                        "--seed",
                        "521",
                        "--output_dir",
                        str(output_dir),
                    ]
                ),
                0,
            )
            metrics = json.loads((output_dir / "metrics.json").read_text())
            predictions = [json.loads(line) for line in (output_dir / "predictions.jsonl").read_text().splitlines()]
            self.assertEqual(metrics["mode"], "eval_baseline")
            self.assertEqual(metrics["baseline_family"], "prompting")
            self.assertEqual(metrics["baseline_mode"], "vanilla")
            self.assertEqual(metrics["support_examples"], 0)
            self.assertEqual(metrics["train_steps"], 0)
            self.assertEqual(metrics["trainable_parameter_count"], 0)
            self.assertEqual(predictions[0]["baseline_mode"], "vanilla")
            self.assertIn("Story:", predictions[0]["baseline_prompt"])
            self.assertIsNotNone(predictions[0]["candidate_scores"])

    def test_prompt_baseline_cot_eval_uses_cot_suffix(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            dataset_path = tmp / "gsm8k.jsonl"
            dataset_path.write_text(
                json.dumps(
                    {
                        "id": "gsm8k-000",
                        "question": "What is 2 + 2?",
                        "answer": "4",
                    }
                )
                + "\n"
            )
            config = {
                "experiment": {
                    "name": "baseline_cot_gsm8k_test",
                    "stage": "M5",
                    "method_variant": "baseline-cot",
                },
                "task": {
                    "name": "gsm8k_smoke",
                    "benchmark_id": "gsm8k",
                    "domain": "math",
                    "split": "eval",
                    "smoke_subset": "local_contract_v1",
                    "dataset_path": str(dataset_path),
                    "metric_name": "exact_match",
                    "evaluator": {"type": "exact_match"},
                },
                "backbone": {
                    "name": "Qwen2.5-1.5B-Instruct",
                    "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
                    "load_mode": "stub",
                    "stub_hidden_size": 64,
                },
                "baseline": {
                    "family": "prompting",
                    "mode": "cot",
                    "cot_suffix": "Think step by step and then answer.",
                },
                "runtime": {
                    "eval_examples": 1,
                    "device": "cpu",
                },
            }
            config_path = tmp / "baseline_gsm8k.yaml"
            config_path.write_text(yaml.safe_dump(config, sort_keys=False))
            output_dir = tmp / "baseline_gsm8k_eval"
            with patch("memtotal.models.backbone.BackboneWrapper.generate", return_value=["4"]):
                self.assertEqual(
                    eval_main(
                        [
                            "--config",
                            str(config_path),
                            "--seed",
                            "523",
                            "--output_dir",
                            str(output_dir),
                        ]
                    ),
                    0,
                )
            metrics = json.loads((output_dir / "metrics.json").read_text())
            predictions = [json.loads(line) for line in (output_dir / "predictions.jsonl").read_text().splitlines()]
            self.assertEqual(metrics["mode"], "eval_baseline")
            self.assertEqual(metrics["baseline_mode"], "cot")
            self.assertEqual(metrics["support_examples"], 0)
            self.assertEqual(metrics["train_steps"], 0)
            self.assertEqual(metrics["trainable_parameter_count"], 0)
            self.assertEqual(metrics["exact_match"], 1.0)
            self.assertIn("Think step by step and then answer.", predictions[0]["baseline_prompt"])
            self.assertEqual(predictions[0]["generated_text"], "4")

    def test_meta_prompting_eval_writes_meta_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            dataset_path = tmp / "story_cloze_meta.jsonl"
            dataset_path.write_text(
                json.dumps(
                    {
                        "id": "story-cloze-meta-000",
                        "story": "Ava rehearsed her violin solo every evening.",
                        "choices": [
                            {"label": "A", "text": "She performed confidently at the recital."},
                            {"label": "B", "text": "She forgot her violin at home and quit music forever."},
                        ],
                        "label": "A",
                        "answer": "She performed confidently at the recital.",
                    }
                )
                + "\n"
            )
            config = {
                "experiment": {
                    "name": "baseline_metaprompting_story_cloze_test",
                    "stage": "M5",
                    "method_variant": "baseline-metaprompting",
                },
                "task": {
                    "name": "story_cloze_smoke",
                    "benchmark_id": "story_cloze",
                    "domain": "narrative",
                    "split": "eval",
                    "smoke_subset": "local_contract_v1",
                    "dataset_path": str(dataset_path),
                    "metric_name": "accuracy",
                    "evaluator": {"type": "multiple_choice"},
                },
                "backbone": {
                    "name": "Qwen2.5-1.5B-Instruct",
                    "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
                    "load_mode": "stub",
                    "stub_hidden_size": 64,
                },
                "baseline": {
                    "family": "meta_prompting",
                    "mode": "planner_critic",
                },
                "runtime": {
                    "eval_examples": 1,
                    "device": "cpu",
                },
            }
            config_path = tmp / "metaprompting_story_cloze.yaml"
            config_path.write_text(yaml.safe_dump(config, sort_keys=False))
            output_dir = tmp / "metaprompting_story_cloze_eval"
            self.assertEqual(
                eval_main(
                    [
                        "--config",
                        str(config_path),
                        "--seed",
                        "531",
                        "--output_dir",
                        str(output_dir),
                    ]
                ),
                0,
            )
            metrics = json.loads((output_dir / "metrics.json").read_text())
            predictions = [json.loads(line) for line in (output_dir / "predictions.jsonl").read_text().splitlines()]
            self.assertEqual(metrics["baseline_family"], "meta_prompting")
            self.assertEqual(metrics["baseline_mode"], "planner_critic")
            self.assertEqual(metrics["support_examples"], 0)
            self.assertEqual(metrics["train_steps"], 0)
            self.assertEqual(metrics["trainable_parameter_count"], 0)
            self.assertIn("Planner:", predictions[0]["baseline_prompt"])
            self.assertIn("Critic:", predictions[0]["baseline_prompt"])

    def test_prompt_baseline_support_examples_are_injected_into_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            dataset_path = tmp / "story_cloze_support.jsonl"
            dataset_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "id": "story-cloze-support-000",
                                "story": "Mila practiced free throws all month.",
                                "choices": [
                                    {"label": "A", "text": "She scored confidently during the game."},
                                    {"label": "B", "text": "She refused to touch a basketball again."},
                                ],
                                "label": "A",
                                "answer": "She scored confidently during the game.",
                            }
                        ),
                        json.dumps(
                            {
                                "id": "story-cloze-support-001",
                                "story": "Owen packed snacks before the trip.",
                                "choices": [
                                    {"label": "A", "text": "The group had food during the drive."},
                                    {"label": "B", "text": "Everyone went hungry by choice."},
                                ],
                                "label": "A",
                                "answer": "The group had food during the drive.",
                            }
                        ),
                    ]
                )
                + "\n"
            )
            config = {
                "experiment": {
                    "name": "baseline_vanilla_story_cloze_support_test",
                    "stage": "M5",
                    "method_variant": "baseline-vanilla-support",
                },
                "task": {
                    "name": "story_cloze_smoke",
                    "benchmark_id": "story_cloze",
                    "domain": "narrative",
                    "split": "eval",
                    "smoke_subset": "local_contract_v1",
                    "dataset_path": str(dataset_path),
                    "metric_name": "accuracy",
                    "evaluator": {"type": "multiple_choice"},
                },
                "backbone": {
                    "name": "Qwen2.5-1.5B-Instruct",
                    "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
                    "load_mode": "stub",
                    "stub_hidden_size": 64,
                },
                "baseline": {
                    "family": "prompting",
                    "mode": "vanilla",
                    "support_examples": 1,
                },
                "runtime": {
                    "eval_examples": 2,
                    "device": "cpu",
                },
            }
            config_path = tmp / "baseline_story_support.yaml"
            config_path.write_text(yaml.safe_dump(config, sort_keys=False))
            output_dir = tmp / "baseline_story_support_eval"
            self.assertEqual(
                eval_main(
                    [
                        "--config",
                        str(config_path),
                        "--seed",
                        "541",
                        "--output_dir",
                        str(output_dir),
                    ]
                ),
                0,
            )
            metrics = json.loads((output_dir / "metrics.json").read_text())
            predictions = [json.loads(line) for line in (output_dir / "predictions.jsonl").read_text().splitlines()]
            self.assertEqual(metrics["support_examples"], 1)
            self.assertIn("Demo 1 Input:", predictions[0]["baseline_prompt"])
            self.assertEqual(len(predictions[0]["baseline_support_ids"]), 1)

    def test_rag_baseline_eval_records_retrieval_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            dataset_path = tmp / "story_cloze_rag.jsonl"
            dataset_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "id": "story-cloze-rag-000",
                                "story": "Mila practiced her free throws every night before the tournament.",
                                "choices": [
                                    {"label": "A", "text": "She sank key shots during the final game."},
                                    {"label": "B", "text": "She avoided touching a basketball forever."},
                                ],
                                "label": "A",
                                "answer": "She sank key shots during the final game.",
                            }
                        ),
                        json.dumps(
                            {
                                "id": "story-cloze-rag-001",
                                "story": "Jordan packed extra batteries before the concert recording.",
                                "choices": [
                                    {"label": "A", "text": "The camera kept working for the whole show."},
                                    {"label": "B", "text": "The equipment disappeared into the ocean."},
                                ],
                                "label": "A",
                                "answer": "The camera kept working for the whole show.",
                            }
                        ),
                    ]
                )
                + "\n"
            )
            config = {
                "experiment": {
                    "name": "baseline_rag_story_cloze_test",
                    "stage": "M5",
                    "method_variant": "baseline-rag",
                },
                "task": {
                    "name": "story_cloze_smoke",
                    "benchmark_id": "story_cloze",
                    "domain": "narrative",
                    "split": "eval",
                    "smoke_subset": "local_contract_v1",
                    "dataset_path": str(dataset_path),
                    "metric_name": "accuracy",
                    "evaluator": {"type": "multiple_choice"},
                },
                "backbone": {
                    "name": "Qwen2.5-1.5B-Instruct",
                    "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
                    "load_mode": "stub",
                    "stub_hidden_size": 64,
                },
                "baseline": {
                    "family": "rag",
                    "mode": "retrieval_augmented",
                    "support_examples": 1,
                    "rag": {"retriever": "lexical_overlap"},
                },
                "runtime": {
                    "eval_examples": 2,
                    "device": "cpu",
                },
            }
            config_path = tmp / "rag_story_cloze.yaml"
            config_path.write_text(yaml.safe_dump(config, sort_keys=False))
            output_dir = tmp / "rag_story_cloze_eval"
            self.assertEqual(
                eval_main(
                    [
                        "--config",
                        str(config_path),
                        "--seed",
                        "547",
                        "--output_dir",
                        str(output_dir),
                    ]
                ),
                0,
            )
            metrics = json.loads((output_dir / "metrics.json").read_text())
            predictions = [json.loads(line) for line in (output_dir / "predictions.jsonl").read_text().splitlines()]
            self.assertEqual(metrics["baseline_family"], "rag")
            self.assertEqual(metrics["baseline_mode"], "retrieval_augmented")
            self.assertEqual(metrics["baseline_retriever"], "lexical_overlap")
            self.assertEqual(metrics["support_examples"], 1)
            self.assertEqual(metrics["train_steps"], 0)
            self.assertEqual(metrics["trainable_parameter_count"], 0)
            self.assertIn("Retrieved memory 1 Input:", predictions[0]["baseline_prompt"])
            self.assertEqual(predictions[0]["baseline_retriever"], "lexical_overlap")
            self.assertEqual(len(predictions[0]["baseline_support_ids"]), 1)
            self.assertEqual(len(predictions[0]["baseline_support_scores"]), 1)

    def test_lightthinker_eval_records_thought_sketch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            dataset_path = tmp / "story_cloze_lightthinker.jsonl"
            dataset_path.write_text(
                json.dumps(
                    {
                        "id": "story-cloze-lightthinker-000",
                        "story": "Ava packed extra markers before the workshop.",
                        "choices": [
                            {"label": "A", "text": "She had enough supplies for the students."},
                            {"label": "B", "text": "The classroom floated into the sky."},
                        ],
                        "label": "A",
                        "answer": "She had enough supplies for the students.",
                    }
                )
                + "\n"
            )
            config = {
                "experiment": {
                    "name": "baseline_lightthinker_story_cloze_test",
                    "stage": "M5",
                    "method_variant": "baseline-lightthinker",
                },
                "task": {
                    "name": "story_cloze_smoke",
                    "benchmark_id": "story_cloze",
                    "domain": "narrative",
                    "split": "eval",
                    "smoke_subset": "local_contract_v1",
                    "dataset_path": str(dataset_path),
                    "metric_name": "accuracy",
                    "evaluator": {"type": "multiple_choice"},
                },
                "backbone": {
                    "name": "Qwen2.5-1.5B-Instruct",
                    "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
                    "load_mode": "stub",
                    "stub_hidden_size": 64,
                },
                "baseline": {
                    "family": "lightthinker",
                    "mode": "compress_then_answer",
                    "lightthinker": {"max_sketch_tokens": 8},
                },
                "runtime": {
                    "eval_examples": 1,
                    "device": "cpu",
                },
            }
            config_path = tmp / "lightthinker_story_cloze.yaml"
            config_path.write_text(yaml.safe_dump(config, sort_keys=False))
            output_dir = tmp / "lightthinker_story_cloze_eval"
            self.assertEqual(
                eval_main(
                    [
                        "--config",
                        str(config_path),
                        "--seed",
                        "549",
                        "--output_dir",
                        str(output_dir),
                    ]
                ),
                0,
            )
            metrics = json.loads((output_dir / "metrics.json").read_text())
            predictions = [json.loads(line) for line in (output_dir / "predictions.jsonl").read_text().splitlines()]
            self.assertEqual(metrics["baseline_family"], "lightthinker")
            self.assertEqual(metrics["baseline_mode"], "compress_then_answer")
            self.assertEqual(metrics["budget_scope"], "compressed_reasoning_prompt")
            self.assertGreater(metrics["mean_thought_sketch_tokens"], 0.0)
            self.assertIn("Think through the task", predictions[0]["lightthinker_compression_prompt"])
            self.assertTrue(predictions[0]["lightthinker_thought_sketch"])

    def test_narrativeqa_eval_writes_f1_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            dataset_path = tmp / "narrativeqa.jsonl"
            dataset_path.write_text(
                json.dumps(
                    {
                        "id": "narrativeqa-000",
                        "story": "Alice travels to Paris and solves a mystery.",
                        "story_segments": [
                            "Alice leaves London.",
                            "She travels to Paris and solves a mystery.",
                        ],
                        "story_chunk_pool": [
                            "Alice leaves London.",
                            "She waits at the station in silence.",
                            "She travels to Paris and solves a mystery.",
                            "The detective reviews the case notes.",
                        ],
                        "question": "Where does Alice travel?",
                        "answer": "Paris",
                        "aliases": ["Paris", "She travels to Paris"],
                        "summary_title": "Alice Story",
                        "document_kind": "movie",
                        "story_chars": 44,
                        "story_word_count": 8,
                        "story_excerpt_chars": 44,
                        "story_segment_words": 160,
                        "story_chunk_pool_size": 4,
                        "story_segments_materialized": 2,
                        "story_total_segments": 4,
                        "story_selection_strategy": "evenly_spaced_chunks",
                        "story_selected_indexes": [0, 2],
                        "story_truncated_for_smoke": False,
                        "narrativeqa_view": "full_text_segmented",
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
                    "smoke_subset": "hf_real_smoke4_full_text_segmented",
                    "dataset_path": str(dataset_path),
                    "metric_name": "f1",
                    "narrativeqa_runtime": {"selector": "question_aware", "segment_budget": 2},
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
            with patch("memtotal.models.backbone.BackboneWrapper.generate", return_value=["Paris"]):
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
            self.assertEqual(metrics["f1"], 1.0)
            self.assertEqual(metrics["story_runtime_selector"], "question_aware")
            self.assertEqual(metrics["story_runtime_segment_budget"], 2)
            self.assertIn("extra_metrics", predictions[0])
            self.assertIn("f1", predictions[0]["extra_metrics"])
            self.assertEqual(predictions[0]["benchmark_metadata"]["story_runtime_segment_budget"], 2)
            self.assertEqual(predictions[0]["benchmark_metadata"]["story_runtime_selector"], "question_aware")


if __name__ == "__main__":
    unittest.main()
