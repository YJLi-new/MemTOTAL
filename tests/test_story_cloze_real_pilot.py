from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from memtotal.analysis.story_cloze_real_pilot import (
    _pick_oracle_alpha_case,
    run_fever_real_fixed_set_builder,
    run_fever_real_pilot_split,
    run_stage_c_real_pilot_compare,
    run_stage_c_real_pilot_oracle_audit,
    run_story_cloze_real_fixed_set_builder,
    run_story_cloze_real_pilot_split,
)
from memtotal.training.m3_real_pilot import (
    _pick_residual_calibration_row,
    _resolve_residual_calibration_alpha_grid,
)


def _story_row(example_id: str, label: str, story: str, good: str, bad: str) -> dict[str, object]:
    choices = [{"label": "A", "text": good}, {"label": "B", "text": bad}]
    if label == "B":
        choices = [{"label": "A", "text": bad}, {"label": "B", "text": good}]
    answer = next(choice["text"] for choice in choices if choice["label"] == label)
    return {
        "id": example_id,
        "story": story,
        "choices": choices,
        "label": label,
        "answer": answer,
    }


class StoryClozeRealPilotAnalysisTest(unittest.TestCase):
    def test_residual_calibration_alpha_grid_preserves_configured_scale(self):
        config = {"runtime": {"stage_c_residual_calibration_alpha_grid": [0.0, 5.0, 10.0]}}
        grid = _resolve_residual_calibration_alpha_grid(config, configured_residual_scale=1.0)
        self.assertEqual(grid, [1.0, 0.0, 5.0, 10.0])

    def test_pick_residual_calibration_row_uses_task_then_proxy_then_margin(self):
        rows = [
            {"alpha": 1.0, "task_score": 0.5, "task_proxy_score": 0.60, "task_margin": -0.2},
            {"alpha": 40.0, "task_score": 0.5, "task_proxy_score": 0.60, "task_margin": -0.1},
            {"alpha": 5.0, "task_score": 0.6, "task_proxy_score": 0.55, "task_margin": -0.3},
        ]
        best = _pick_residual_calibration_row(rows, configured_residual_scale=1.0)
        self.assertEqual(best["alpha"], 5.0)

    def test_pick_residual_calibration_row_breaks_exact_tie_toward_configured_scale(self):
        rows = [
            {"alpha": 1.0, "task_score": 0.5, "task_proxy_score": 0.60, "task_margin": -0.1},
            {"alpha": 40.0, "task_score": 0.5, "task_proxy_score": 0.60, "task_margin": -0.1},
        ]
        best = _pick_residual_calibration_row(rows, configured_residual_scale=1.0)
        self.assertEqual(best["alpha"], 1.0)

    def test_split_builder_writes_support_and_eval_subsets(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source_path = root / "source.jsonl"
            rows = [
                _story_row(f"id-{index}", "A" if index % 2 == 0 else "B", f"story {index}", "good ending", "bad ending")
                for index in range(12)
            ]
            source_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
            config = {
                "task": {"dataset_path": str(source_path)},
                "runtime": {
                    "screen_source_examples": 12,
                    "pilot_support_examples": 4,
                    "support_selection_seed": 11,
                    "pilot_output_root": str(root / "pilots"),
                },
            }
            output_dir = root / "analysis"
            output_dir.mkdir()
            run_story_cloze_real_pilot_split(config=config, output_dir=output_dir, dry_run=False)
            support_rows = [json.loads(line) for line in (root / "pilots" / "pilot-support4.jsonl").read_text().splitlines()]
            eval_rows = [json.loads(line) for line in (root / "pilots" / "screen-eval8.jsonl").read_text().splitlines()]
            self.assertEqual(len(support_rows), 4)
            self.assertEqual(len(eval_rows), 8)
            self.assertTrue(all("shuffled_memory_example_id" in row for row in support_rows))
            self.assertTrue(all(row["screening_split"] == "screen_eval" for row in eval_rows))

    def test_fever_split_builder_reuses_generic_split_logic(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source_path = root / "source.jsonl"
            rows = []
            for index, label in enumerate(["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"] * 4):
                rows.append(
                    {
                        "id": f"fever-{index}",
                        "claim": f"claim {index}",
                        "evidence": f"evidence {index}",
                        "choices": [
                            {"label": "SUPPORTS", "text": "Supports"},
                            {"label": "REFUTES", "text": "Refutes"},
                            {"label": "NOT_ENOUGH_INFO", "text": "Not enough info"},
                        ],
                        "label": label,
                        "answer": label.replace("_", " ").title(),
                    }
                )
            source_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
            config = {
                "task": {"dataset_path": str(source_path), "benchmark_id": "fever"},
                "runtime": {
                    "screen_source_examples": 12,
                    "pilot_support_examples": 4,
                    "support_selection_seed": 7,
                    "pilot_output_root": str(root / "pilots"),
                },
            }
            output_dir = root / "analysis"
            output_dir.mkdir()
            run_fever_real_pilot_split(config=config, output_dir=output_dir, dry_run=False)
            support_rows = [json.loads(line) for line in (root / "pilots" / "pilot-support4.jsonl").read_text().splitlines()]
            eval_rows = [json.loads(line) for line in (root / "pilots" / "screen-eval8.jsonl").read_text().splitlines()]
            self.assertEqual(len(support_rows), 4)
            self.assertEqual(len(eval_rows), 8)
            self.assertTrue(all("shuffled_memory_example_id" in row for row in eval_rows))

    def test_fixed_set_builder_writes_fixed_dataset_with_buckets(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source_rows = [
                _story_row(f"id-{index}", "A", f"story {index}", "good outcome", "bad outcome")
                for index in range(10)
            ]
            source_path = root / "screen-eval.jsonl"
            source_path.write_text("\n".join(json.dumps(row) for row in source_rows) + "\n")
            input_root = root / "runs"
            for alias, decision_mode in [("base", "base_only"), ("shared", "shared_summary_late_fusion")]:
                run_dir = input_root / alias
                run_dir.mkdir(parents=True)
                metrics = {
                    "training_stage": "stage_c_real_pilot",
                    "decision_mode": decision_mode,
                    "memory_control_mode": "real",
                    "choice_objective": "continuation_retrieval",
                    "best_adapt_step": 0 if alias == "base" else 1,
                }
                (run_dir / "metrics.json").write_text(json.dumps(metrics))
                rows = []
                for index, source_row in enumerate(source_rows):
                    base_margin = -0.05 if index < 3 else (-0.2 if index < 7 else 0.2)
                    final_margin = base_margin + (0.04 if alias == "shared" and index < 7 else 0.0)
                    predicted_correct = final_margin > 0
                    rows.append(
                        {
                            "step": metrics["best_adapt_step"],
                            "example_id": source_row["id"],
                            "predicted_correct": predicted_correct,
                            "final_margin": final_margin,
                            "task_score": float(predicted_correct),
                            "task_proxy_score": 0.5 + final_margin / 2.0,
                        }
                    )
                (run_dir / "task_case_dump.jsonl").write_text(
                    "\n".join(json.dumps(row) for row in rows) + "\n"
                )
            config = {
                "task": {"dataset_path": str(source_path)},
                "runtime": {"pilot_output_root": str(root / "pilots")},
            }
            output_dir = root / "analysis"
            output_dir.mkdir()
            run_story_cloze_real_fixed_set_builder(
                config=config,
                output_dir=output_dir,
                input_root=input_root,
                dry_run=False,
            )
            fixed_rows = [json.loads(line) for line in (root / "pilots" / "fixed10.jsonl").read_text().splitlines()]
            self.assertEqual(len(fixed_rows), 10)
            self.assertTrue(all("screening_bucket" in row for row in fixed_rows))
            self.assertTrue(all("shuffled_memory_example_id" in row for row in fixed_rows))
            self.assertTrue((root / "pilots" / "calibration-hard0.jsonl").exists())

    def test_fever_fixed_set_builder_writes_fixed_dataset_with_backfill(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source_rows = []
            labels = ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"]
            for index in range(18):
                source_rows.append(
                    {
                        "id": f"fever-{index}",
                        "claim": f"claim {index}",
                        "evidence": f"evidence {index}",
                        "choices": [
                            {"label": "SUPPORTS", "text": "Supports"},
                            {"label": "REFUTES", "text": "Refutes"},
                            {"label": "NOT_ENOUGH_INFO", "text": "Not enough info"},
                        ],
                        "label": labels[index % len(labels)],
                        "answer": labels[index % len(labels)].replace("_", " ").title(),
                    }
                )
            source_path = root / "screen-eval248.jsonl"
            source_path.write_text("\n".join(json.dumps(row) for row in source_rows) + "\n")
            input_root = root / "runs"
            for alias, decision_mode in [("base", "base_only"), ("shared", "shared_summary_late_fusion")]:
                run_dir = input_root / alias
                run_dir.mkdir(parents=True)
                metrics = {
                    "training_stage": "stage_c_real_pilot",
                    "decision_mode": decision_mode,
                    "memory_control_mode": "real",
                    "choice_objective": "continuation_retrieval",
                    "pilot_split": "screen_eval248",
                    "best_adapt_step": 0,
                }
                (run_dir / "metrics.json").write_text(json.dumps(metrics))
                rows = []
                for index, source_row in enumerate(source_rows):
                    base_margin = -0.05 if index < 4 else (-0.3 if index < 12 else 0.4)
                    final_margin = base_margin + (0.08 if alias == "shared" and index < 12 else 0.0)
                    predicted_correct = final_margin > 0
                    rows.append(
                        {
                            "step": 0,
                            "example_id": source_row["id"],
                            "predicted_correct": predicted_correct,
                            "final_margin": final_margin,
                            "task_score": float(predicted_correct),
                            "task_proxy_score": 0.5 + final_margin / 4.0,
                        }
                    )
                (run_dir / "task_case_dump.jsonl").write_text(
                    "\n".join(json.dumps(row) for row in rows) + "\n"
                )
            config = {
                "task": {"dataset_path": str(source_path)},
                "runtime": {"pilot_output_root": str(root / "pilots")},
            }
            output_dir = root / "analysis"
            output_dir.mkdir()
            run_fever_real_fixed_set_builder(
                config=config,
                output_dir=output_dir,
                input_root=input_root,
                dry_run=True,
            )
            fixed_rows = [json.loads(line) for line in (root / "pilots" / "fixed8.jsonl").read_text().splitlines()]
            self.assertEqual(len(fixed_rows), 8)
            self.assertTrue(all("screening_bucket" in row for row in fixed_rows))
            self.assertTrue(any(row["screening_bucket"] == "stubborn_wrong" for row in fixed_rows))

    def test_compare_analysis_writes_pairwise_summary(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_root = root / "runs"
            arm_specs = {
                "A": ("base_only", "real", "continuation_retrieval", 0.0),
                "B": ("shared_summary_late_fusion", "real", "continuation_retrieval", 0.0),
                "C": ("candidate_conditioned_late_fusion", "real", "continuation_retrieval", 1.0),
                "D": ("candidate_conditioned_late_fusion", "shuffled", "continuation_retrieval", 0.0),
                "E": ("candidate_conditioned_late_fusion", "real", "choice_ce_plus_margin", 1.0),
            }
            for alias, (decision_mode, memory_control, choice_objective, score) in arm_specs.items():
                run_dir = input_root / alias
                run_dir.mkdir(parents=True)
                (run_dir / "metrics.json").write_text(
                    json.dumps(
                        {
                            "training_stage": "stage_c_real_pilot",
                            "decision_mode": decision_mode,
                            "memory_control_mode": memory_control,
                            "choice_objective": choice_objective,
                            "pilot_split": "fixed100",
                            "eval_dataset_path": str(root / "fixed100.jsonl"),
                            "best_adapt_step": 0,
                            "best_adapt_task_score": score,
                            "zero_shot_task_score": 0.0,
                            "best_adapt_task_proxy_score": 0.5 + score / 2.0,
                            "best_adapt_task_margin": score / 2.0,
                            "task_case_dump_path": str(run_dir / "task_case_dump.jsonl"),
                        }
                    )
                )
                case_rows = [
                    {
                        "step": 0,
                        "example_id": "id-1",
                        "predicted_correct": bool(score > 0),
                        "task_score": float(score > 0),
                        "task_proxy_score": 0.5 + score / 2.0,
                        "final_margin": score / 2.0,
                        "story": "story",
                        "gold_text": "good",
                        "final_predicted_label": "A" if score > 0 else "B",
                        "screening_bucket": "near_threshold_bad",
                    }
                ]
                (run_dir / "task_case_dump.jsonl").write_text(
                    "\n".join(json.dumps(row) for row in case_rows) + "\n"
                )
            for alias, decision_mode in [("screen-A", "base_only"), ("screen-B", "shared_summary_late_fusion")]:
                run_dir = input_root / alias
                run_dir.mkdir(parents=True)
                (run_dir / "metrics.json").write_text(
                    json.dumps(
                        {
                            "training_stage": "stage_c_real_pilot",
                            "decision_mode": decision_mode,
                            "memory_control_mode": "real",
                            "choice_objective": "continuation_retrieval",
                            "pilot_split": "screen_eval248",
                            "eval_dataset_path": str(root / "screen-eval248.jsonl"),
                            "best_adapt_step": 0,
                            "best_adapt_task_score": 0.0,
                            "zero_shot_task_score": 0.0,
                            "best_adapt_task_proxy_score": 0.5,
                            "best_adapt_task_margin": 0.0,
                            "task_case_dump_path": str(run_dir / "task_case_dump.jsonl"),
                        }
                    )
                )
                (run_dir / "task_case_dump.jsonl").write_text(
                    json.dumps(
                        {
                            "step": 0,
                            "example_id": "screen-id",
                            "predicted_correct": False,
                            "task_score": 0.0,
                            "task_proxy_score": 0.5,
                            "final_margin": 0.0,
                            "story": "screen story",
                            "gold_text": "screen gold",
                            "final_predicted_label": "B",
                            "screening_bucket": "stubborn_wrong_other",
                        }
                    )
                    + "\n"
                )
            output_dir = root / "analysis"
            output_dir.mkdir()
            run_stage_c_real_pilot_compare(
                output_dir=output_dir,
                input_root=input_root,
                dry_run=False,
            )
            with (output_dir / "arm_pairwise_compare.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 4)
            with (output_dir / "arm_summary.csv").open() as handle:
                summary_rows = list(csv.DictReader(handle))
            self.assertTrue(all(row["run_dir"].endswith(("/A", "/B", "/C", "/D", "/E")) for row in summary_rows))
            self.assertTrue((output_dir / "summary.svg").exists())

    def test_compare_analysis_respects_configured_aliases(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_root = root / "runs"
            arm_specs = {
                "A": ("base_only", "real", "continuation_retrieval", 0.0),
                "B": ("shared_summary_late_fusion", "real", "continuation_retrieval", 0.0),
                "C": ("candidate_conditioned_late_fusion", "real", "continuation_retrieval", 1.0),
                "D": ("candidate_conditioned_late_fusion", "shuffled", "continuation_retrieval", 0.0),
            }
            for alias, (decision_mode, memory_control, choice_objective, score) in arm_specs.items():
                run_dir = input_root / alias
                run_dir.mkdir(parents=True)
                (run_dir / "metrics.json").write_text(
                    json.dumps(
                        {
                            "training_stage": "stage_c_real_pilot",
                            "decision_mode": decision_mode,
                            "memory_control_mode": memory_control,
                            "choice_objective": choice_objective,
                            "pilot_split": "fixed64",
                            "eval_dataset_path": str(root / "fixed64.jsonl"),
                            "best_adapt_step": 0,
                            "best_adapt_task_score": score,
                            "zero_shot_task_score": 0.0,
                            "best_adapt_task_proxy_score": 0.5 + score / 2.0,
                            "best_adapt_task_margin": score / 2.0,
                            "task_case_dump_path": str(run_dir / "task_case_dump.jsonl"),
                        }
                    )
                )
                (run_dir / "task_case_dump.jsonl").write_text(
                    json.dumps(
                        {
                            "step": 0,
                            "example_id": "id-1",
                            "predicted_correct": bool(score > 0),
                            "task_score": float(score > 0),
                            "task_proxy_score": 0.5 + score / 2.0,
                            "final_margin": score / 2.0,
                            "segment": "claim and evidence",
                            "gold_text": "supports",
                            "final_predicted_label": "SUPPORTS" if score > 0 else "REFUTES",
                            "screening_bucket": "near_threshold_bad",
                        }
                    )
                    + "\n"
                )
            output_dir = root / "analysis"
            output_dir.mkdir()
            run_stage_c_real_pilot_compare(
                config={
                    "task": {"dataset_path": str(root / "fixed64.jsonl")},
                    "runtime": {
                        "pilot_split_name": "fixed64",
                        "required_arm_aliases": ["A", "B", "C", "D"],
                        "pair_specs": ["A->B", "A->C", "C->D"],
                    },
                },
                output_dir=output_dir,
                input_root=input_root,
                dry_run=False,
            )
            with (output_dir / "arm_pairwise_compare.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 3)

    def test_pick_oracle_alpha_case_prefers_smallest_successful_abs_scale(self):
        row = {
            "base_choice_scores": [0.0, 0.4],
            "memory_residual_scores": [1.0, 0.0],
            "choices": [
                {"label": "A", "text": "good"},
                {"label": "B", "text": "bad"},
            ],
            "gold_label": "A",
        }
        best = _pick_oracle_alpha_case(row, alpha_grid=[0.25, 0.5, 1.0, 2.0])
        self.assertEqual(best["alpha"], 0.5)
        self.assertTrue(best["predicted_correct"])

    def test_oracle_audit_writes_summary_and_case_deltas(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_root = root / "runs"
            arm_specs = {
                "A": ("base_only", "real", "continuation_retrieval", [0.0, 1.1], [0.0, 0.0], 0.0),
                "C": ("candidate_conditioned_late_fusion", "real", "continuation_retrieval", [0.0, 1.1], [0.05, 0.0], 0.0),
            }
            for alias, (decision_mode, memory_control, choice_objective, base_scores, residual_scores, score) in arm_specs.items():
                run_dir = input_root / alias
                run_dir.mkdir(parents=True)
                final_scores = [
                    base_score + residual_score
                    for base_score, residual_score in zip(base_scores, residual_scores, strict=True)
                ]
                predicted_index = 0 if final_scores[0] > final_scores[1] else 1
                predicted_label = ["A", "B"][predicted_index]
                (run_dir / "metrics.json").write_text(
                    json.dumps(
                        {
                            "training_stage": "stage_c_real_pilot",
                            "decision_mode": decision_mode,
                            "memory_control_mode": memory_control,
                            "choice_objective": choice_objective,
                            "pilot_split": "fixed100",
                            "eval_dataset_path": str(root / "fixed100.jsonl"),
                            "best_adapt_step": 0,
                            "best_adapt_task_score": score,
                            "zero_shot_task_score": 0.0,
                            "best_adapt_task_proxy_score": 0.5,
                            "best_adapt_task_margin": final_scores[0] - final_scores[1],
                            "task_case_dump_path": str(run_dir / "task_case_dump.jsonl"),
                        }
                    )
                )
                (run_dir / "task_case_dump.jsonl").write_text(
                    json.dumps(
                        {
                            "step": 0,
                            "example_id": "id-1",
                            "predicted_correct": predicted_label == "A",
                            "task_score": float(predicted_label == "A"),
                            "task_proxy_score": 0.25,
                            "final_margin": final_scores[0] - final_scores[1],
                            "final_choice_scores": final_scores,
                            "base_choice_scores": base_scores,
                            "memory_residual_scores": residual_scores,
                            "gold_label": "A",
                            "final_predicted_label": predicted_label,
                            "choices": [
                                {"label": "A", "text": "good"},
                                {"label": "B", "text": "bad"},
                            ],
                            "story": "story",
                            "gold_text": "good",
                            "screening_bucket": "near_threshold_bad",
                        }
                    )
                    + "\n"
                )
            output_dir = root / "analysis"
            output_dir.mkdir()
            run_stage_c_real_pilot_oracle_audit(
                config={"task": {"dataset_path": str(root / "fixed100.jsonl")}, "runtime": {}},
                output_dir=output_dir,
                input_root=input_root,
                dry_run=False,
            )
            with (output_dir / "oracle_summary.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            by_alias = {row["alias"]: row for row in rows}
            self.assertEqual(float(by_alias["A"]["task_score"]), 0.0)
            self.assertEqual(float(by_alias["oracle_per_case_alpha"]["task_score"]), 1.0)
            self.assertTrue((output_dir / "oracle_case_deltas.csv").exists())


if __name__ == "__main__":
    unittest.main()
