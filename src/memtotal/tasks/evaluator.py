from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from memtotal.tasks.registry import get_task_spec


def _normalize_text(text: str) -> str:
    compact = re.sub(r"\s+", " ", text.strip().lower())
    compact = re.sub(r"^[a-z ]*:\s*", "", compact)
    return compact


def _normalize_code(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.strip().splitlines())


def _normalize_action(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _apply_normalizer(text: str, normalizer: str) -> str:
    if normalizer == "code":
        return _normalize_code(text)
    if normalizer == "action":
        return _normalize_action(text)
    return _normalize_text(text)


@dataclass(frozen=True)
class TaskEvaluator:
    evaluator_type: str
    metric_name: str
    normalizer: str = "text"
    benchmark_id: str | None = None

    def evaluate_prediction(self, prediction: dict[str, Any], example: dict[str, Any]) -> dict[str, Any]:
        if self.evaluator_type == "multiple_choice":
            predicted_label = str(prediction.get("label", ""))
            predicted_text = str(prediction.get("text", ""))
            gold_label = str(example["label"])
            gold_text = str(example["continuation"])
            is_correct = (
                _apply_normalizer(predicted_label, "text") == _apply_normalizer(gold_label, "text")
                or _apply_normalizer(predicted_text, self.normalizer)
                == _apply_normalizer(gold_text, self.normalizer)
            )
            return {
                "correct": is_correct,
                "score": float(is_correct),
                "normalized_prediction": _apply_normalizer(predicted_label or predicted_text, self.normalizer),
                "normalized_reference": _apply_normalizer(gold_label, self.normalizer),
            }

        if self.evaluator_type == "exact_match":
            predicted_text = str(prediction.get("text", ""))
            normalized_prediction = _apply_normalizer(predicted_text, self.normalizer)
            references = [str(example.get("gold_answer", example["continuation"]))]
            references.extend(str(alias) for alias in example.get("aliases", []))
            normalized_references = [_apply_normalizer(reference, self.normalizer) for reference in references]
            is_correct = normalized_prediction in normalized_references
            return {
                "correct": is_correct,
                "score": float(is_correct),
                "normalized_prediction": normalized_prediction,
                "normalized_reference": normalized_references[0],
            }

        predicted_label = str(prediction.get("label", prediction.get("text", "")))
        gold_label = str(example["label"])
        is_correct = _apply_normalizer(predicted_label, self.normalizer) == _apply_normalizer(
            gold_label, self.normalizer
        )
        return {
            "correct": is_correct,
            "score": float(is_correct),
            "normalized_prediction": _apply_normalizer(predicted_label, self.normalizer),
            "normalized_reference": _apply_normalizer(gold_label, self.normalizer),
        }


def build_task_evaluator(config: dict[str, Any]) -> TaskEvaluator:
    task_cfg = config["task"]
    evaluator_cfg = task_cfg.get("evaluator", {})
    benchmark_id = task_cfg.get("benchmark_id")
    if benchmark_id is not None:
        spec = get_task_spec(str(benchmark_id))
        evaluator_type = str(evaluator_cfg.get("type", spec.evaluator_type))
        metric_name = str(task_cfg.get("metric_name", spec.metric_name))
        normalizer = str(evaluator_cfg.get("normalizer", spec.normalizer))
        return TaskEvaluator(
            evaluator_type=evaluator_type,
            metric_name=metric_name,
            normalizer=normalizer,
            benchmark_id=str(benchmark_id),
        )

    evaluator_type = str(evaluator_cfg.get("type", "dataset_label_classification"))
    metric_name = str(task_cfg.get("metric_name", "accuracy"))
    normalizer = str(evaluator_cfg.get("normalizer", "text"))
    return TaskEvaluator(
        evaluator_type=evaluator_type,
        metric_name=metric_name,
        normalizer=normalizer,
        benchmark_id=None,
    )
