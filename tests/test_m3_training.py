from __future__ import annotations

import csv
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

from memtotal.data import (
    EpisodeSampler,
    build_meta_manifest,
    load_domain_dataset,
    load_meta_grouped_examples,
    validate_meta_split,
)
from memtotal.pipeline import MemoryRuntime
from memtotal.training.m3 import _continuation_retrieval_loss, _resolve_episode_support_weights
from memtotal.training.run_train import main as train_main
from memtotal.utils.config import load_config


class M3TrainingTest(unittest.TestCase):
    def _core4_task_config(self) -> dict[str, object]:
        return {
            "name": "core4_transfer_smoke_test",
            "domain": "cross_domain_meta",
            "split": "meta",
            "meta": {
                "general_domains": ["math", "code", "qa", "narrative"],
                "source_domains": ["math", "code", "qa"],
                "target_domain": "narrative",
                "support_size": 3,
                "query_size": 3,
                "sampling_policy": "uniform_examples",
                "dataset_sources": [
                    {
                        "benchmark_id": "gsm8k",
                        "dataset_path": str(ROOT / "data/benchmarks/materialized/gsm8k/eval-real-smoke8.jsonl"),
                        "domain": "math",
                        "smoke_subset": "hf_real_smoke8",
                    },
                    {
                        "benchmark_id": "kodcode",
                        "dataset_path": str(ROOT / "data/benchmarks/materialized/kodcode/eval-real-smoke8.jsonl"),
                        "domain": "code",
                        "smoke_subset": "hf_real_smoke8",
                    },
                    {
                        "benchmark_id": "gpqa",
                        "dataset_path": str(ROOT / "data/benchmarks/materialized/gpqa/eval-real-smoke8.jsonl"),
                        "domain": "qa",
                        "smoke_subset": "hf_real_smoke8",
                    },
                    {
                        "benchmark_id": "story_cloze",
                        "dataset_path": str(ROOT / "data/benchmarks/materialized/story_cloze/eval-real-smoke8.jsonl"),
                        "domain": "narrative",
                        "smoke_subset": "hf_real_smoke8",
                    },
                ],
            },
        }

    def _write_stage_b_override(self, root: Path, query_learning_mode: str) -> Path:
        override_path = root / f"stage_b_{query_learning_mode}.yaml"
        override_path.write_text(
            yaml.safe_dump(
                {
                    "includes": [str(ROOT / "configs/exp/m3_stage_b_qwen25_smoke.yaml")],
                    "experiment": {
                        "name": f"m3_stage_b_{query_learning_mode}_test",
                        "method_variant": f"ours-stage-b-{query_learning_mode}-test",
                    },
                    "runtime": {
                        "query_learning_mode": query_learning_mode,
                    },
                },
                sort_keys=False,
            )
        )
        return override_path

    def _write_stage_c_override(
        self,
        root: Path,
        adaptation_target: str,
        *,
        expected_query_learning_mode: str = "meta_trained",
    ) -> Path:
        override_path = root / f"stage_c_{adaptation_target}_{expected_query_learning_mode}.yaml"
        override_path.write_text(
            yaml.safe_dump(
                {
                    "includes": [str(ROOT / "configs/exp/m3_stage_c_qwen25_smoke.yaml")],
                    "experiment": {
                        "name": f"m3_stage_c_{adaptation_target}_{expected_query_learning_mode}_test",
                        "method_variant": (
                            f"ours-stage-c-{adaptation_target}-{expected_query_learning_mode}-test"
                        ),
                    },
                    "runtime": {
                        "adaptation_target": adaptation_target,
                        "expected_query_learning_mode": expected_query_learning_mode,
                    },
                },
                sort_keys=False,
            )
        )
        return override_path

    def test_episode_sampler_respects_meta_split(self) -> None:
        grouped = load_domain_dataset(ROOT / "data/toy/meta_samples.jsonl")
        validate_meta_split(
            grouped,
            general_domains=["math", "code", "qa", "narrative"],
            source_domains=["math", "code", "qa"],
            target_domain="narrative",
            support_size=2,
            query_size=2,
        )
        sampler = EpisodeSampler(
            grouped,
            source_domains=["math", "code", "qa"],
            support_size=2,
            query_size=2,
            seed=17,
        )
        episode = sampler.sample_episode()
        self.assertIn(episode.domain, {"math", "code", "qa"})
        self.assertEqual(len(episode.support_examples), 2)
        self.assertEqual(len(episode.query_examples), 2)
        self.assertEqual(
            {row["label"] for row in episode.support_examples},
            {row["label"] for row in episode.query_examples},
        )

    def test_benchmark_meta_loader_builds_multisource_manifest(self) -> None:
        task_cfg = self._core4_task_config()
        grouped = load_meta_grouped_examples(task_cfg)
        validate_meta_split(
            grouped,
            general_domains=["math", "code", "qa", "narrative"],
            source_domains=["math", "code", "qa"],
            target_domain="narrative",
            support_size=3,
            query_size=3,
            sampling_policy="uniform_examples",
        )
        sampler = EpisodeSampler(
            grouped,
            source_domains=["math", "code", "qa"],
            support_size=3,
            query_size=3,
            seed=23,
            sampling_policy="uniform_examples",
        )
        episode = sampler.sample_episode()
        self.assertIn(episode.domain, {"math", "code", "qa"})
        self.assertEqual(len(episode.support_examples), 3)
        self.assertEqual(len(episode.query_examples), 3)
        self.assertTrue(all(row["benchmark_id"] == "story_cloze" for row in grouped["narrative"]))
        manifest = build_meta_manifest(
            dataset_sources=task_cfg["meta"]["dataset_sources"],
            grouped_examples=grouped,
            general_domains=["math", "code", "qa", "narrative"],
            source_domains=["math", "code", "qa"],
            target_domain="narrative",
            support_size=3,
            query_size=3,
            sampling_policy="uniform_examples",
        )
        self.assertEqual(manifest["sampling_policy"], "uniform_examples")
        self.assertEqual(manifest["benchmarks_by_domain"]["math"], "gsm8k")
        self.assertIn("dataset_sources", manifest)
        self.assertIn("math:gsm8k", manifest["dataset_sha256s"])

    def test_continuation_retrieval_loss_supports_exact_match_and_multiple_choice(self) -> None:
        config = load_config(ROOT / "configs/exp/m3_stage_b_core4_qwen25_smoke.yaml")
        runtime = MemoryRuntime(config=config, seed=29)
        grouped = load_meta_grouped_examples(config["task"])
        for domain in ("math", "narrative"):
            example = grouped[domain][0]
            loss, accuracy = _continuation_retrieval_loss(
                runtime,
                example,
                candidate_pool=grouped[domain],
                negative_count=7,
            )
            self.assertTrue(loss.requires_grad)
            self.assertGreaterEqual(accuracy, 0.0)
            self.assertLessEqual(accuracy, 1.0)

    def test_resolve_episode_support_weights_supports_weighting_modes(self) -> None:
        self.assertEqual(
            _resolve_episode_support_weights([0.2, 0.4, 0.6], weighting="uniform"),
            [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
        )
        self.assertEqual(
            _resolve_episode_support_weights([0.2, 0.9, 0.6], weighting="proxy_top1"),
            [0.0, 1.0, 0.0],
        )
        softmax_weights = _resolve_episode_support_weights([0.1, 0.2, 0.4], weighting="proxy_softmax")
        self.assertAlmostEqual(sum(softmax_weights), 1.0)
        self.assertGreater(softmax_weights[2], softmax_weights[1])
        self.assertGreater(softmax_weights[1], softmax_weights[0])

    def test_m3_stage_sequence_writes_expected_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            stage_a_dir = root / "stage_a"

            self.assertEqual(
                train_main(
                    [
                        "--config",
                        str(ROOT / "configs/exp/m3_stage_a_qwen25_smoke.yaml"),
                        "--seed",
                        "41",
                        "--output_dir",
                        str(stage_a_dir),
                    ]
                ),
                0,
            )
            self.assertTrue(stage_a_dir.joinpath("writer.ckpt").exists())
            stage_a_metrics = json.loads(stage_a_dir.joinpath("metrics.json").read_text())
            self.assertEqual(stage_a_metrics["training_stage"], "stage_a")

            stage_b_expectations = [
                (
                    ROOT / "configs/exp/m3_stage_b_qwen25_smoke.yaml",
                    "meta_trained",
                ),
                (
                    self._write_stage_b_override(root, "non_meta_multitask"),
                    "non_meta_multitask",
                ),
                (
                    self._write_stage_b_override(root, "random"),
                    "random",
                ),
            ]
            stage_b_dirs: dict[str, Path] = {}
            for index, (config_path, query_learning_mode) in enumerate(stage_b_expectations):
                stage_b_dir = root / f"stage_b_{query_learning_mode}"
                self.assertEqual(
                    train_main(
                        [
                            "--config",
                            str(config_path),
                            "--seed",
                            str(43 + index),
                            "--output_dir",
                            str(stage_b_dir),
                            "--resume",
                            str(stage_a_dir),
                        ]
                    ),
                    0,
                )
                self.assertTrue(stage_b_dir.joinpath("queries_meta_init.pt").exists())
                stage_b_metrics = json.loads(stage_b_dir.joinpath("metrics.json").read_text())
                self.assertEqual(stage_b_metrics["training_stage"], "stage_b")
                self.assertEqual(stage_b_metrics["query_learning_mode"], query_learning_mode)
                stage_b_dirs[query_learning_mode] = stage_b_dir

            stage_c_mode_expectations = [
                (
                    ROOT / "configs/exp/m3_stage_c_qwen25_smoke.yaml",
                    "meta_trained",
                    "q_only",
                    "reader.queries",
                    False,
                ),
                (
                    self._write_stage_c_override(
                        root,
                        "q_only",
                        expected_query_learning_mode="non_meta_multitask",
                    ),
                    "non_meta_multitask",
                    "q_only",
                    "reader.queries",
                    False,
                ),
                (
                    self._write_stage_c_override(
                        root,
                        "q_only",
                        expected_query_learning_mode="random",
                    ),
                    "random",
                    "q_only",
                    "reader.queries",
                    False,
                ),
            ]
            for index, (
                config_path,
                query_learning_mode,
                adaptation_target,
                trainable_module,
                expects_writer,
            ) in enumerate(stage_c_mode_expectations):
                stage_c_dir = root / f"stage_c_{query_learning_mode}"
                self.assertEqual(
                    train_main(
                        [
                            "--config",
                            str(config_path),
                            "--seed",
                            str(51 + index),
                            "--output_dir",
                            str(stage_c_dir),
                            "--resume",
                            str(stage_b_dirs[query_learning_mode]),
                        ]
                    ),
                    0,
                )
                self.assertTrue(stage_c_dir.joinpath("queries_adapted.pt").exists())
                self.assertTrue(stage_c_dir.joinpath("adapt_curve.csv").exists())
                stage_c_metrics = json.loads(stage_c_dir.joinpath("metrics.json").read_text())
                self.assertEqual(stage_c_metrics["query_learning_mode"], query_learning_mode)
                self.assertEqual(stage_c_metrics["adaptation_target"], adaptation_target)
                self.assertEqual(stage_c_metrics["trainable_module"], trainable_module)
                if expects_writer:
                    self.assertTrue(stage_c_dir.joinpath("writer_adapted.ckpt").exists())
                else:
                    self.assertFalse(stage_c_dir.joinpath("writer_adapted.ckpt").exists())
                with stage_c_dir.joinpath("adapt_curve.csv").open() as handle:
                    rows = list(csv.DictReader(handle))
                self.assertGreaterEqual(len(rows), 2)
                self.assertEqual({row["query_learning_mode"] for row in rows}, {query_learning_mode})

            adaptation_expectations = [
                (
                    ROOT / "configs/exp/m3_stage_c_qwen25_smoke.yaml",
                    "q_only",
                    "reader.queries",
                    False,
                ),
                (
                    self._write_stage_c_override(root, "w_only"),
                    "w_only",
                    "writer",
                    True,
                ),
                (
                    self._write_stage_c_override(root, "w_plus_q"),
                    "w_plus_q",
                    "writer+reader.queries",
                    True,
                ),
            ]

            for index, (config_path, adaptation_target, trainable_module, expects_writer) in enumerate(adaptation_expectations):
                stage_c_dir = root / f"stage_c_{adaptation_target}"
                self.assertEqual(
                    train_main(
                        [
                            "--config",
                            str(config_path),
                            "--seed",
                            str(61 + index),
                            "--output_dir",
                            str(stage_c_dir),
                            "--resume",
                            str(stage_b_dirs["meta_trained"]),
                        ]
                    ),
                    0,
                )
                self.assertTrue(stage_c_dir.joinpath("queries_adapted.pt").exists())
                self.assertTrue(stage_c_dir.joinpath("adapt_curve.csv").exists())
                self.assertTrue(stage_c_dir.joinpath("adapt_cost.json").exists())
                stage_c_metrics = json.loads(stage_c_dir.joinpath("metrics.json").read_text())
                self.assertEqual(stage_c_metrics["training_stage"], "stage_c")
                self.assertEqual(stage_c_metrics["adaptation_target"], adaptation_target)
                self.assertEqual(stage_c_metrics["trainable_module"], trainable_module)
                if expects_writer:
                    self.assertTrue(stage_c_dir.joinpath("writer_adapted.ckpt").exists())
                else:
                    self.assertFalse(stage_c_dir.joinpath("writer_adapted.ckpt").exists())
                with stage_c_dir.joinpath("adapt_curve.csv").open() as handle:
                    rows = list(csv.DictReader(handle))
                self.assertGreaterEqual(len(rows), 2)
                self.assertEqual({row["query_learning_mode"] for row in rows}, {"meta_trained"})
                self.assertEqual({row["adaptation_target"] for row in rows}, {adaptation_target})
                self.assertEqual({row["trainable_module"] for row in rows}, {trainable_module})

    def test_core4_stage_sequence_writes_task_score_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            stage_a_dir = root / "core4_stage_a"
            stage_b_dir = root / "core4_stage_b"
            stage_c_dir = root / "core4_stage_c"

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
                train_main(
                    [
                        "--config",
                        str(ROOT / "configs/exp/m3_stage_c_core4_qwen25_smoke.yaml"),
                        "--seed",
                        "79",
                        "--output_dir",
                        str(stage_c_dir),
                        "--resume",
                        str(stage_b_dir),
                    ]
                ),
                0,
            )

            stage_b_manifest = json.loads(stage_b_dir.joinpath("meta_data_manifest.json").read_text())
            self.assertEqual(stage_b_manifest["sampling_policy"], "uniform_examples")
            self.assertEqual(stage_b_manifest["benchmarks_by_domain"]["narrative"], "story_cloze")

            stage_b_metrics = json.loads(stage_b_dir.joinpath("metrics.json").read_text())
            self.assertEqual(stage_b_metrics["query_objective"], "continuation_retrieval")
            self.assertIn("source_eval_task_score", stage_b_metrics)
            self.assertEqual(stage_b_metrics["source_eval_metric_name"], "mean_score")
            self.assertIn("source_eval_task_proxy_score", stage_b_metrics)
            self.assertIn("source_eval_task_proxy_name", stage_b_metrics)
            self.assertEqual(stage_b_metrics["stage_b_trainable_target"], "queries_plus_fuser")
            self.assertEqual(stage_b_metrics["trainable_module"], "reader.queries+fuser")
            self.assertEqual(stage_b_metrics["query_candidate_pool_policy"], "exclude_support_for_query_eval")
            self.assertEqual(stage_b_metrics["support_candidate_pool_policy"], "support_only_for_inner_loop")
            self.assertEqual(stage_b_metrics["retrieval_negative_count"], 7)
            self.assertEqual(stage_b_metrics["meta_episodes"], 16)
            self.assertEqual(stage_b_metrics["inner_steps"], 1)
            self.assertEqual(stage_b_metrics["meta_learning_rate"], 0.05)

            stage_c_metrics = json.loads(stage_c_dir.joinpath("metrics.json").read_text())
            self.assertEqual(stage_c_metrics["query_objective"], "continuation_retrieval")
            self.assertIn("zero_shot_task_score", stage_c_metrics)
            self.assertIn("best_adapt_task_score", stage_c_metrics)
            self.assertEqual(stage_c_metrics["task_metric_name"], "accuracy")
            self.assertIn("zero_shot_task_proxy_score", stage_c_metrics)
            self.assertIn("best_adapt_task_proxy_score", stage_c_metrics)
            self.assertEqual(stage_c_metrics["task_proxy_name"], "gold_choice_probability")
            self.assertEqual(stage_c_metrics["retrieval_negative_count"], 7)
            self.assertEqual(stage_c_metrics["adapt_learning_rate"], 0.2)
            self.assertEqual(stage_c_metrics["adapt_steps"], 3)
            self.assertEqual(stage_c_metrics["adapt_shots"], [0, 3])
            self.assertEqual(stage_c_metrics["target_eval_repeats"], 3)
            self.assertEqual(stage_c_metrics["target_episode_repeats"], 3)
            self.assertEqual(stage_c_metrics["target_episode_policy"], "aggregate_support")
            self.assertEqual(stage_c_metrics["target_support_weighting"], "uniform")
            self.assertEqual(stage_c_metrics["target_split_policy"], "proxy_bottomk_support")
            self.assertEqual(stage_c_metrics["checkpoint_target_episode_policy"], "shared_aggregate")
            self.assertIn("mean_support_grad_norm", stage_c_metrics)
            self.assertIn("max_support_update_max_abs", stage_c_metrics)
            self.assertIn("adaptation_effective_threshold", stage_c_metrics)
            self.assertIn("adaptation_effective", stage_c_metrics)

            with stage_c_dir.joinpath("adapt_curve.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertGreaterEqual(len(rows), 2)
            self.assertIn("task_score", rows[0])
            self.assertIn("objective_loss", rows[0])
            self.assertIn("task_proxy_score", rows[0])
            self.assertIn("task_proxy_name", rows[0])
            self.assertIn("task_margin", rows[0])
            self.assertIn("target_eval_repeats", rows[0])
            self.assertIn("target_episode_repeats", rows[0])
            self.assertIn("target_episode_policy", rows[0])
            self.assertIn("target_support_weighting", rows[0])
            self.assertIn("target_split_policy", rows[0])
            self.assertIn("evaluated_target_episodes", rows[0])
            self.assertIn("evaluated_query_examples", rows[0])
            self.assertIn("query_candidate_pool_size", rows[0])
            self.assertIn("support_candidate_pool_size", rows[0])
            self.assertIn("preceding_support_grad_norm", rows[0])
            self.assertIn("preceding_support_update_max_abs", rows[0])
            self.assertEqual({row["query_objective"] for row in rows}, {"continuation_retrieval"})
            zero_row = next(row for row in rows if row["shot"] == "0" and row["step"] == "0")
            shot_row = next(row for row in rows if row["shot"] == "3" and row["step"] == "0")
            self.assertAlmostEqual(float(zero_row["task_score"]), float(shot_row["task_score"]))
            self.assertEqual(zero_row["query_candidate_pool_size"], shot_row["query_candidate_pool_size"])
            self.assertEqual(zero_row["support_candidate_pool_size"], shot_row["support_candidate_pool_size"])


if __name__ == "__main__":
    unittest.main()
