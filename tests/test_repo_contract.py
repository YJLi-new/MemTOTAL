from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from memtotal.analysis.run_analysis import main as analysis_main
from memtotal.baselines.run_memgen import main as memgen_main
from memtotal.training.run_train import main as train_main
from memtotal.eval.run_eval import main as eval_main
from memtotal.utils.config import load_config
from memtotal.utils.repro import SUPPORTED_BACKBONES


class RepoContractTest(unittest.TestCase):
    def test_governing_docs_exist(self) -> None:
        required = [
            ROOT / "AGENTS.md",
            ROOT / "docs/TODO_LIST.md",
            ROOT / "docs/MAIN_IDEA.md",
            ROOT / "docs/EXPERIMENTS_INFO.md",
        ]
        for path in required:
            self.assertTrue(path.is_file(), msg=f"Missing governing doc: {path}")

    def test_only_supported_backbones_in_exp_configs(self) -> None:
        for config_path in sorted((ROOT / "configs/exp").glob("*.yaml")):
            config = load_config(config_path)
            self.assertIn(config["backbone"]["name"], SUPPORTED_BACKBONES)

    def test_smoke_entrypoints_write_required_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            train_dir = temp_root / "train"
            eval_dir = temp_root / "eval"
            analysis_dir = temp_root / "analysis"
            stage_a_dir = temp_root / "stage_a"
            stage_b_dir = temp_root / "stage_b"
            failure_dir = temp_root / "failure_checks"
            config_path = ROOT / "configs/exp/smoke_qwen25.yaml"

            self.assertEqual(
                train_main(
                    [
                        "--config",
                        str(config_path),
                        "--seed",
                        "123",
                        "--output_dir",
                        str(train_dir),
                        "--dry-run",
                    ]
                ),
                0,
            )
            self.assertEqual(
                eval_main(
                    [
                        "--config",
                        str(config_path),
                        "--seed",
                        "123",
                        "--output_dir",
                        str(eval_dir),
                        "--checkpoint",
                        str(train_dir / "checkpoint.pt"),
                        "--dry-run",
                    ]
                ),
                0,
            )
            self.assertEqual(
                analysis_main(
                    [
                        "--config",
                        str(config_path),
                        "--seed",
                        "123",
                        "--output_dir",
                        str(analysis_dir),
                        "--input_root",
                        str(temp_root),
                        "--dry-run",
                    ]
                ),
                0,
            )
            self.assertEqual(
                train_main(
                    [
                        "--config",
                        str(ROOT / "configs/exp/m3_stage_a_qwen25_smoke.yaml"),
                        "--seed",
                        "301",
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
                        str(ROOT / "configs/exp/m3_stage_b_qwen25_smoke.yaml"),
                        "--seed",
                        "303",
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
                        str(ROOT / "configs/exp/m3_failure_checks_qwen25_smoke.yaml"),
                        "--seed",
                        "307",
                        "--output_dir",
                        str(failure_dir),
                        "--resume",
                        str(stage_b_dir),
                    ]
                ),
                0,
            )

            self.assertTrue((train_dir / "config.snapshot.yaml").is_file())
            self.assertTrue((train_dir / "run_info.json").is_file())
            self.assertTrue((train_dir / "metrics.json").is_file())
            self.assertTrue((train_dir / "checkpoint.pt").is_file())
            self.assertTrue((train_dir / "profiling.json").is_file())
            self.assertTrue((train_dir / "profiling.csv").is_file())
            self.assertTrue((eval_dir / "predictions.jsonl").is_file())
            self.assertTrue((eval_dir / "profiling.json").is_file())
            self.assertTrue((analysis_dir / "summary.csv").is_file())
            self.assertTrue((analysis_dir / "summary.svg").is_file())
            self.assertTrue((analysis_dir / "profiling.json").is_file())
            self.assertTrue((failure_dir / "failure_checks.json").is_file())
            self.assertTrue((failure_dir / "failure_ablation_summary.csv").is_file())
            self.assertTrue((failure_dir / "failure_ablation_summary.svg").is_file())

            run_info = json.loads((train_dir / "run_info.json").read_text())
            self.assertIn("git_hash", run_info)
            self.assertEqual(run_info["backbone"], "Qwen2.5-1.5B-Instruct")

    def test_collect_artifacts_script_copies_managed_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            source_dir = temp_root / "source"
            dest_dir = temp_root / "dest"
            source_dir.mkdir()
            for name in [
                "config.snapshot.yaml",
                "run_info.json",
                "metrics.json",
                "profiling.json",
                "profiling.csv",
                "summary.csv",
                "summary.svg",
            ]:
                (source_dir / name).write_text(name)

            subprocess.run(
                [
                    str(ROOT / "scripts/collect_artifacts.sh"),
                    str(source_dir),
                    str(dest_dir),
                ],
                check=True,
                cwd=ROOT,
            )

            for name in [
                "config.snapshot.yaml",
                "run_info.json",
                "metrics.json",
                "profiling.json",
                "profiling.csv",
                "summary.csv",
                "summary.svg",
            ]:
                self.assertTrue((dest_dir / name).is_file(), msg=f"Missing copied artifact: {name}")

    def test_memgen_adapter_dry_run_writes_launch_plan(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "memgen-dry-run"
            config_path = ROOT / "configs/exp/memgen_gsm8k_qwen25_eval.yaml"
            self.assertEqual(
                memgen_main(
                    [
                        "--config",
                        str(config_path),
                        "--seed",
                        "11",
                        "--output_dir",
                        str(output_dir),
                        "--dry-run",
                    ]
                ),
                0,
            )
            self.assertTrue((output_dir / "memgen_launch.sh").is_file())
            self.assertTrue((output_dir / "memgen_launch.json").is_file())
            metrics = json.loads((output_dir / "metrics.json").read_text())
            self.assertTrue(metrics["dry_run"])
            self.assertEqual(metrics["backbone"], "Qwen2.5-1.5B-Instruct")
            self.assertEqual(metrics["baseline_family"], "memgen")
            self.assertEqual(metrics["train_steps"], 0)
            self.assertEqual(metrics["trainable_parameter_count"], 0)

    def test_github_review_snapshot_builder_stays_under_budget(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "review-snapshot"
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/build_github_review_snapshot.py"),
                    "--output-root",
                    str(output_dir),
                    "--source-commit",
                    "test-commit",
                ],
                check=True,
                cwd=ROOT,
            )
            manifest = json.loads((output_dir / "REVIEW_SNAPSHOT_MANIFEST.json").read_text())
            self.assertLessEqual(manifest["total_size_bytes"], 31 * 1024 * 1024)
            archive_path = Path(temp_dir) / "review-snapshot.zip"
            shutil.make_archive(str(archive_path.with_suffix("")), "zip", root_dir=output_dir)
            self.assertLessEqual(archive_path.stat().st_size, 31 * 1024 * 1024)
            for relative_path in [
                "README.md",
                "AGENTS.md",
                "PLANv6.md",
                "docs/MAIN_IDEA.md",
                "docs/EXPERIMENTS_INFO.md",
                "docs/GITHUB_REVIEW_EXPORT.md",
                "results/generated/review/planv6-v6-4-mixed-matrix-qwen25/v6-4-summary.json",
            ]:
                self.assertTrue(
                    (output_dir / relative_path).is_file(),
                    msg=f"Missing snapshot artifact: {relative_path}",
                )


if __name__ == "__main__":
    unittest.main()
