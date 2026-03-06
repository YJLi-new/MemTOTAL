from __future__ import annotations

import re
import string
from collections import Counter
from dataclasses import dataclass
from typing import Any

from rouge_score import rouge_scorer

from memtotal.tasks.registry import get_task_spec


_MEMORYAGENTBENCH_ROUGE = rouge_scorer.RougeScorer(["rougeL", "rougeLsum"], use_stemmer=True)


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


def _normalize_qa_text(text: str) -> str:
    lowered = text.lower()
    no_punctuation = "".join(char for char in lowered if char not in string.punctuation)
    no_articles = re.sub(r"\b(a|an|the)\b", " ", no_punctuation)
    return " ".join(no_articles.split())


def _memoryagentbench_f1_score(prediction: str, ground_truth: str) -> float:
    normalized_prediction = _normalize_qa_text(prediction)
    normalized_ground_truth = _normalize_qa_text(ground_truth)
    special_answers = {"yes", "no", "noanswer"}
    if (
        normalized_prediction in special_answers or normalized_ground_truth in special_answers
    ) and normalized_prediction != normalized_ground_truth:
        return 0.0
    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common_tokens = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    overlap = sum(common_tokens.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(ground_truth_tokens)
    return (2 * precision * recall) / (precision + recall)


def _flatten_references(values: object) -> list[str]:
    if isinstance(values, str):
        text = values.strip()
        return [text] if text else []
    if isinstance(values, list):
        flattened: list[str] = []
        for value in values:
            flattened.extend(_flatten_references(value))
        return flattened
    if values is None:
        return []
    text = str(values).strip()
    return [text] if text else []


def _metric_max_over_references(prediction: str, references: list[str], metric_name: str) -> float:
    if not references:
        return 0.0
    if metric_name == "exact_match":
        return float(max(_normalize_qa_text(prediction) == _normalize_qa_text(reference) for reference in references))
    if metric_name == "f1":
        return max(_memoryagentbench_f1_score(prediction, reference) for reference in references)
    if metric_name == "substring_exact_match":
        normalized_prediction = _normalize_qa_text(prediction)
        return float(max(_normalize_qa_text(reference) in normalized_prediction for reference in references))
    if metric_name in {"rougeL_f1", "rougeL_recall", "rougeLsum_f1", "rougeLsum_recall"}:
        rouge_name, field_name = metric_name.rsplit("_", maxsplit=1)
        best_score = 0.0
        for reference in references:
            score = _MEMORYAGENTBENCH_ROUGE.score(target=reference, prediction=prediction)[rouge_name]
            best_score = max(best_score, score.fmeasure if field_name == "f1" else score.recall)
        return best_score
    raise ValueError(f"Unsupported MemoryAgentBench metric: {metric_name}")


def _memoryagentbench_keypoint_recall(prediction: str, keypoints: list[str]) -> float:
    if not keypoints:
        return 0.0
    normalized_prediction = _normalize_qa_text(prediction)
    hits = 0
    for keypoint in keypoints:
        normalized_keypoint = _normalize_qa_text(keypoint)
        if normalized_keypoint and normalized_keypoint in normalized_prediction:
            hits += 1
    return hits / len(keypoints)


def _calculate_memoryagentbench_metrics(
    prediction: str,
    references: list[str],
    *,
    keypoints: list[str] | None,
) -> dict[str, float]:
    metrics = {
        "exact_match": _metric_max_over_references(prediction, references, "exact_match"),
        "f1": _metric_max_over_references(prediction, references, "f1"),
        "substring_exact_match": _metric_max_over_references(prediction, references, "substring_exact_match"),
        "rougeL_f1": _metric_max_over_references(prediction, references, "rougeL_f1"),
        "rougeL_recall": _metric_max_over_references(prediction, references, "rougeL_recall"),
        "rougeLsum_f1": _metric_max_over_references(prediction, references, "rougeLsum_f1"),
        "rougeLsum_recall": _metric_max_over_references(prediction, references, "rougeLsum_recall"),
    }
    if keypoints:
        metrics["keypoint_recall"] = _memoryagentbench_keypoint_recall(prediction, keypoints)
    return metrics


@dataclass(frozen=True)
class TaskEvaluator:
    evaluator_type: str
    metric_name: str
    normalizer: str = "text"
    benchmark_id: str | None = None

    def evaluate_prediction(self, prediction: dict[str, Any], example: dict[str, Any]) -> dict[str, Any]:
        if self.evaluator_type == "qa_f1":
            predicted_text = str(prediction.get("text", ""))
            references = _flatten_references(example.get("aliases", example.get("gold_answer", example["continuation"])))
            exact_match = _metric_max_over_references(predicted_text, references, "exact_match")
            f1_score = _metric_max_over_references(predicted_text, references, "f1")
            substring_match = _metric_max_over_references(predicted_text, references, "substring_exact_match")
            return {
                "correct": bool(exact_match),
                "score": f1_score,
                "normalized_prediction": _normalize_qa_text(predicted_text),
                "normalized_reference": _normalize_qa_text(references[0] if references else ""),
                "extra_metrics": {
                    "exact_match": exact_match,
                    "f1": f1_score,
                    "substring_exact_match": substring_match,
                },
            }

        if self.evaluator_type == "memoryagentbench":
            predicted_text = str(prediction.get("text", ""))
            references = _flatten_references(example.get("aliases", example.get("gold_answer", example["continuation"])))
            keypoints = [str(item) for item in example.get("keypoints", [])]
            metrics = _calculate_memoryagentbench_metrics(predicted_text, references, keypoints=keypoints)
            primary_metric = str(example.get("capability_metric_name", "exact_match"))
            score = float(metrics.get(primary_metric, metrics["exact_match"]))
            return {
                "correct": bool(metrics["exact_match"]),
                "score": score,
                "normalized_prediction": _normalize_qa_text(predicted_text),
                "normalized_reference": _normalize_qa_text(references[0] if references else ""),
                "extra_metrics": metrics,
                "capability": str(example.get("capability", "")),
                "capability_metric_name": primary_metric,
            }

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
