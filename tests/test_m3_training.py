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

from memtotal.data import EpisodeSampler, load_domain_dataset, validate_meta_split
from memtotal.training.run_train import main as train_main


class M3TrainingTest(unittest.TestCase):
    def _write_stage_c_override(self, root: Path, adaptation_target: str) -> Path:
        override_path = root / f"stage_c_{adaptation_target}.yaml"
        override_path.write_text(
            yaml.safe_dump(
                {
                    "includes": [str(ROOT / "configs/exp/m3_stage_c_qwen25_smoke.yaml")],
                    "experiment": {
                        "name": f"m3_stage_c_{adaptation_target}_test",
                        "method_variant": f"ours-stage-c-{adaptation_target}-test",
                    },
                    "runtime": {
                        "adaptation_target": adaptation_target,
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

    def test_m3_stage_sequence_writes_expected_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            stage_a_dir = root / "stage_a"
            stage_b_dir = root / "stage_b"

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

            self.assertEqual(
                train_main(
                    [
                        "--config",
                        str(ROOT / "configs/exp/m3_stage_b_qwen25_smoke.yaml"),
                        "--seed",
                        "43",
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

            expectations = [
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

            for index, (config_path, adaptation_target, trainable_module, expects_writer) in enumerate(expectations):
                stage_c_dir = root / f"stage_c_{adaptation_target}"
                self.assertEqual(
                    train_main(
                        [
                            "--config",
                            str(config_path),
                            "--seed",
                            str(47 + index),
                            "--output_dir",
                            str(stage_c_dir),
                            "--resume",
                            str(stage_b_dir),
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
                self.assertEqual({row["adaptation_target"] for row in rows}, {adaptation_target})
                self.assertEqual({row["trainable_module"] for row in rows}, {trainable_module})


if __name__ == "__main__":
    unittest.main()
