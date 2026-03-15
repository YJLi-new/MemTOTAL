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

from memtotal.analysis.baseline_budget import audit_baseline_budget_rows, collect_baseline_budget_rows
from memtotal.analysis.run_analysis import main as analysis_main


class BaselineBudgetAuditTest(unittest.TestCase):
    def test_collect_baseline_budget_rows_backfills_budget_from_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            run_dir = temp_root / "qwen25-story"
            run_dir.mkdir()
            (run_dir / "run_info.json").write_text(
                json.dumps(
                    {
                        "backbone": "Qwen2.5-1.5B-Instruct",
                        "task_name": "story_cloze_real_smoke",
                        "smoke_subset": "hf_real_smoke4",
                    }
                )
            )
            (run_dir / "metrics.json").write_text(
                json.dumps(
                    {
                        "mode": "eval_baseline",
                        "baseline_family": "meta_prompting",
                        "baseline_mode": "planner_critic",
                        "accuracy": 0.75,
                    }
                )
            )
            (run_dir / "config.snapshot.yaml").write_text(
                yaml.safe_dump(
                    {
                        "baseline": {
                            "family": "meta_prompting",
                            "mode": "planner_critic",
                        },
                        "runtime": {},
                    },
                    sort_keys=False,
                )
            )
            rows = collect_baseline_budget_rows(temp_root)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["support_examples"], 0)
            self.assertEqual(rows[0]["train_steps"], 0)
            self.assertEqual(rows[0]["trainable_parameter_count"], 0)
            self.assertEqual(rows[0]["budget_scope"], "zero_shot_prompt")

    def test_collect_baseline_budget_rows_backfills_rag_scope(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            run_dir = temp_root / "rag-qwen25-story"
            run_dir.mkdir()
            (run_dir / "run_info.json").write_text(
                json.dumps(
                    {
                        "backbone": "Qwen2.5-1.5B-Instruct",
                        "task_name": "story_cloze_real_smoke",
                        "smoke_subset": "hf_real_smoke4",
                    }
                )
            )
            (run_dir / "metrics.json").write_text(
                json.dumps(
                    {
                        "mode": "eval_baseline",
                        "baseline_family": "rag",
                        "baseline_mode": "retrieval_augmented",
                        "accuracy": 0.5,
                        "support_examples": 2,
                    }
                )
            )
            (run_dir / "config.snapshot.yaml").write_text(
                yaml.safe_dump(
                    {
                        "baseline": {
                            "family": "rag",
                            "mode": "retrieval_augmented",
                            "support_examples": 2,
                            "rag": {"retriever": "lexical_overlap"},
                        },
                        "runtime": {},
                    },
                    sort_keys=False,
                )
            )
            rows = collect_baseline_budget_rows(temp_root)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["support_examples"], 2)
            self.assertEqual(rows[0]["train_steps"], 0)
            self.assertEqual(rows[0]["trainable_parameter_count"], 0)
            self.assertEqual(rows[0]["budget_scope"], "external_memory_prompt")

    def test_collect_baseline_budget_rows_backfills_lightthinker_scope(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            run_dir = temp_root / "lightthinker-qwen25-story"
            run_dir.mkdir()
            (run_dir / "run_info.json").write_text(
                json.dumps(
                    {
                        "backbone": "Qwen2.5-1.5B-Instruct",
                        "task_name": "story_cloze_real_smoke",
                        "smoke_subset": "hf_real_smoke4",
                    }
                )
            )
            (run_dir / "metrics.json").write_text(
                json.dumps(
                    {
                        "mode": "eval_baseline",
                        "baseline_family": "lightthinker",
                        "baseline_mode": "compress_then_answer",
                        "accuracy": 0.5,
                    }
                )
            )
            (run_dir / "config.snapshot.yaml").write_text(
                yaml.safe_dump(
                    {
                        "baseline": {
                            "family": "lightthinker",
                            "mode": "compress_then_answer",
                            "lightthinker": {"max_sketch_tokens": 16},
                        },
                        "runtime": {},
                    },
                    sort_keys=False,
                )
            )
            rows = collect_baseline_budget_rows(temp_root)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["support_examples"], 0)
            self.assertEqual(rows[0]["train_steps"], 0)
            self.assertEqual(rows[0]["trainable_parameter_count"], 0)
            self.assertEqual(rows[0]["budget_scope"], "compressed_reasoning_prompt")

    def test_collect_baseline_budget_rows_backfills_memory_bank_scope(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            run_dir = temp_root / "memory-bank-qwen25-story"
            run_dir.mkdir()
            (run_dir / "run_info.json").write_text(
                json.dumps(
                    {
                        "backbone": "Qwen2.5-1.5B-Instruct",
                        "task_name": "story_cloze_real_smoke",
                        "smoke_subset": "hf_real_smoke4",
                    }
                )
            )
            (run_dir / "metrics.json").write_text(
                json.dumps(
                    {
                        "mode": "eval_baseline",
                        "baseline_family": "memory_bank",
                        "baseline_mode": "episodic_bank",
                        "accuracy": 0.5,
                        "support_examples": 4,
                    }
                )
            )
            (run_dir / "config.snapshot.yaml").write_text(
                yaml.safe_dump(
                    {
                        "baseline": {
                            "family": "memory_bank",
                            "mode": "episodic_bank",
                            "support_examples": 4,
                            "memory_bank": {"bank_capacity": 2},
                        },
                        "runtime": {},
                    },
                    sort_keys=False,
                )
            )
            rows = collect_baseline_budget_rows(temp_root)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["support_examples"], 4)
            self.assertEqual(rows[0]["train_steps"], 0)
            self.assertEqual(rows[0]["trainable_parameter_count"], 0)
            self.assertEqual(rows[0]["budget_scope"], "external_memory_bank_prompt")

    def test_audit_baseline_budget_rows_flags_missing_backbone_pair(self) -> None:
        rows = [
            {
                "run_dir": "/tmp/qwen25",
                "mode": "eval_baseline",
                "backbone": "Qwen2.5-1.5B-Instruct",
                "task_name": "story_cloze_real_smoke",
                "smoke_subset": "hf_real_smoke4",
                "baseline_family": "prompting",
                "baseline_mode": "vanilla",
                "support_examples": 0,
                "train_steps": 0,
                "trainable_parameter_count": 0,
            }
        ]
        audited_rows, issues = audit_baseline_budget_rows(rows)
        self.assertEqual(audited_rows[0]["budget_ok"], 0.0)
        self.assertIn("missing_backbone_pair:Qwen3-8B", audited_rows[0]["budget_issues"])
        self.assertEqual(len(issues), 1)

    def test_collect_baseline_budget_rows_infers_adapter_parameter_count(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            run_dir = temp_root / "adapter-eval"
            run_dir.mkdir()
            (run_dir / "run_info.json").write_text(
                json.dumps(
                    {
                        "backbone": "Qwen2.5-1.5B-Instruct",
                        "task_name": "story_cloze_smoke",
                        "smoke_subset": "local_contract_v1",
                    }
                )
            )
            (run_dir / "metrics.json").write_text(
                json.dumps(
                    {
                        "mode": "eval_baseline",
                        "baseline_family": "adapter",
                        "baseline_mode": "lora",
                        "support_examples": 1,
                        "train_steps": 4,
                    }
                )
            )
            (run_dir / "config.snapshot.yaml").write_text(
                yaml.safe_dump(
                    {
                        "backbone": {"stub_hidden_size": 64},
                        "baseline": {
                            "family": "adapter",
                            "mode": "lora",
                            "support_examples": 1,
                            "lora": {"rank": 4, "alpha": 8.0},
                        },
                        "runtime": {"train_steps": 4},
                    },
                    sort_keys=False,
                )
            )
            rows = collect_baseline_budget_rows(temp_root)
            self.assertEqual(rows[0]["trainable_parameter_count"], 512)

    def test_collect_baseline_budget_rows_infers_ia3_parameter_count(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            run_dir = temp_root / "adapter-ia3-eval"
            run_dir.mkdir()
            (run_dir / "run_info.json").write_text(
                json.dumps(
                    {
                        "backbone": "Qwen2.5-1.5B-Instruct",
                        "task_name": "story_cloze_smoke",
                        "smoke_subset": "local_contract_v1",
                    }
                )
            )
            (run_dir / "metrics.json").write_text(
                json.dumps(
                    {
                        "mode": "eval_baseline",
                        "baseline_family": "adapter",
                        "baseline_mode": "ia3",
                        "support_examples": 1,
                        "train_steps": 4,
                    }
                )
            )
            (run_dir / "config.snapshot.yaml").write_text(
                yaml.safe_dump(
                    {
                        "backbone": {"stub_hidden_size": 64},
                        "baseline": {
                            "family": "adapter",
                            "mode": "ia3",
                            "support_examples": 1,
                            "ia3": {"init_scale": 1.0},
                        },
                        "runtime": {"train_steps": 4},
                    },
                    sort_keys=False,
                )
            )
            rows = collect_baseline_budget_rows(temp_root)
            self.assertEqual(rows[0]["trainable_parameter_count"], 64)

    def test_collect_baseline_budget_rows_infers_prefix_tuning_parameter_count(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            run_dir = temp_root / "adapter-prefix-eval"
            run_dir.mkdir()
            (run_dir / "run_info.json").write_text(
                json.dumps(
                    {
                        "backbone": "Qwen2.5-1.5B-Instruct",
                        "task_name": "story_cloze_smoke",
                        "smoke_subset": "local_contract_v1",
                    }
                )
            )
            (run_dir / "metrics.json").write_text(
                json.dumps(
                    {
                        "mode": "eval_baseline",
                        "baseline_family": "adapter",
                        "baseline_mode": "prefix_tuning",
                        "support_examples": 1,
                        "train_steps": 4,
                    }
                )
            )
            (run_dir / "config.snapshot.yaml").write_text(
                yaml.safe_dump(
                    {
                        "backbone": {"stub_hidden_size": 64},
                        "baseline": {
                            "family": "adapter",
                            "mode": "prefix_tuning",
                            "support_examples": 1,
                            "prefix_tuning": {"prefix_tokens": 4},
                        },
                        "runtime": {"train_steps": 4},
                    },
                    sort_keys=False,
                )
            )
            rows = collect_baseline_budget_rows(temp_root)
            self.assertEqual(rows[0]["trainable_parameter_count"], 4416)

    def test_analysis_mode_writes_budget_report(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            input_root = temp_root / "runs"
            input_root.mkdir()
            for backbone in ["Qwen2.5-1.5B-Instruct", "Qwen3-8B"]:
                run_dir = input_root / backbone
                run_dir.mkdir()
                (run_dir / "run_info.json").write_text(
                    json.dumps(
                        {
                            "backbone": backbone,
                            "task_name": "story_cloze_real_smoke",
                            "smoke_subset": "hf_real_smoke4",
                        }
                    )
                )
                (run_dir / "metrics.json").write_text(
                    json.dumps(
                        {
                            "mode": "eval_baseline",
                            "baseline_family": "meta_prompting",
                            "baseline_mode": "planner_critic",
                            "support_examples": 0,
                            "train_steps": 0,
                            "trainable_parameter_count": 0,
                            "accuracy": 1.0,
                        }
                    )
                )
                (run_dir / "config.snapshot.yaml").write_text(
                    yaml.safe_dump(
                        {
                            "baseline": {
                                "family": "meta_prompting",
                                "mode": "planner_critic",
                            },
                            "runtime": {},
                        },
                        sort_keys=False,
                    )
                )
            output_dir = temp_root / "analysis"
            self.assertEqual(
                analysis_main(
                    [
                        "--config",
                        str(ROOT / "configs/exp/m5_baseline_budget_audit.yaml"),
                        "--seed",
                        "961",
                        "--output_dir",
                        str(output_dir),
                        "--input_root",
                        str(input_root),
                    ]
                ),
                0,
            )
            metrics = json.loads((output_dir / "metrics.json").read_text())
            report = json.loads((output_dir / "baseline_budget_report.json").read_text())
            self.assertEqual(metrics["mode"], "analysis")
            self.assertEqual(report["checks_pass_rate"], 1.0)
            self.assertTrue((output_dir / "summary.csv").is_file())
            self.assertTrue((output_dir / "summary.svg").is_file())


if __name__ == "__main__":
    unittest.main()
