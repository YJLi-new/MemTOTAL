#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


TASK_DEFAULTS: dict[str, dict[str, Any]] = {
    "gsm8k": {
        "benchmark_id": "gsm8k",
        "domain": "math",
        "metric_name": "exact_match",
        "task_name": "gsm8k_real_smoke",
        "evaluator": {"type": "exact_match", "normalizer": "gsm8k_final_answer"},
        "eval_examples": 64,
    },
    "triviaqa": {
        "benchmark_id": "triviaqa",
        "domain": "qa",
        "metric_name": "exact_match",
        "task_name": "triviaqa_real_smoke",
        "evaluator": {"type": "exact_match", "normalizer": "text"},
        "eval_examples": 64,
    },
    "fever": {
        "benchmark_id": "fever",
        "domain": "qa",
        "metric_name": "accuracy",
        "task_name": "fever_real_smoke",
        "evaluator": {"type": "multiple_choice", "normalizer": "text"},
        "eval_examples": 48,
    },
}

MEMGEN_TASK_DEFAULTS: dict[str, dict[str, Any]] = {
    "gsm8k": {
        "memgen_config_path": "configs/latent_memory/gsm8k.yaml",
        "insertion_profile": "single_turn_smoke",
        "max_prompt_aug_num": 1,
        "max_inference_aug_num": 1,
        "extra_options": [
            "model.attn_implementation=sdpa",
            "dataset.mode=sft",
            "dataset.sft.val_ratio=0.01",
            "dataset.sft.num_workers=1",
            "dataset.sft.max_train_samples=8",
            "dataset.sft.max_valid_samples=4",
            "dataset.sft.max_test_samples=8",
            "run.interaction.batch_size=1",
            "run.interaction.do_sample=False",
            "run.interaction.max_response_length=256",
        ],
    },
    "triviaqa": {
        "memgen_config_path": "configs/latent_memory/triviaqa.yaml",
        "insertion_profile": "dynamic_search_smoke",
        "max_prompt_aug_num": 1,
        "max_inference_aug_num": 0,
        "extra_options": [
            "model.attn_implementation=sdpa",
            "dataset.mode=sft",
            "dataset.sft.valid_ratio=0.01",
            "dataset.sft.num_workers=1",
            "dataset.sft.max_train_samples=8",
            "dataset.sft.max_valid_samples=4",
            "dataset.sft.max_test_samples=8",
            "run.generation.eval_batch_size=1",
            "run.generation.max_response_length=256",
            "run.generation.max_turns=2",
        ],
    },
}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def materialize_planv8_v8_7_rag_config(
    *,
    task_name: str,
    output_config: Path,
    eval_path: str,
    support_path: str,
    primary_model_dir: str,
    primary_backbone_name: str,
    hf_cache_dir: str = "/root/autodl-tmp/hf-cache",
) -> dict[str, Any]:
    if task_name not in TASK_DEFAULTS:
        raise ValueError(f"Unsupported V8-7 RAG task: {task_name}")
    task_defaults = TASK_DEFAULTS[task_name]
    payload = {
        "experiment": {
            "name": f"planv8_v8_7_m1_text_rag_{task_name}",
            "stage": "V8-7",
            "method_variant": "m1_text_rag",
        },
        "backbone": {
            "name": primary_backbone_name,
            "model_id": primary_model_dir,
            "load_mode": "hf_causal_lm",
            "dtype": "bfloat16",
            "cache_dir": hf_cache_dir,
            "attn_implementation": "sdpa",
            "max_new_tokens": 192,
            "gradient_checkpointing": False,
            "use_chat_template": True,
            "chat_template_enable_thinking": False,
        },
        "baseline": {
            "family": "rag",
            "mode": "retrieval_augmented",
            "support_examples": 8,
            "rag": {
                "retriever": "lexical_overlap",
                "include_answer_in_memory": True,
                "memory_prefix": "Retrieved memory",
            },
        },
        "runtime": {
            "device": "cuda",
            "eval_examples": int(task_defaults["eval_examples"]),
        },
        "task": {
            "name": str(task_defaults["task_name"]),
            "benchmark_id": str(task_defaults["benchmark_id"]),
            "domain": str(task_defaults["domain"]),
            "metric_name": str(task_defaults["metric_name"]),
            "evaluator": dict(task_defaults["evaluator"]),
            "dataset_path": str(eval_path),
            "support_dataset_path": str(support_path),
        },
    }
    _write_json(output_config, payload)
    return payload


def materialize_planv8_v8_7_memgen_config(
    *,
    task_name: str,
    output_config: Path,
    primary_model_dir: str,
    primary_backbone_name: str,
    repo_root: str = "MemGen-master",
) -> dict[str, Any]:
    if task_name not in MEMGEN_TASK_DEFAULTS:
        raise ValueError(f"Unsupported V8-7 MemGen task: {task_name}")
    task_defaults = MEMGEN_TASK_DEFAULTS[task_name]
    payload = {
        "experiment": {
            "name": f"planv8_v8_7_m2_memgen_{task_name}",
            "stage": "V8-7",
            "method_variant": "m2_memgen_context",
        },
        "task": {
            "name": task_name,
        },
        "backbone": {
            "name": primary_backbone_name,
            "model_id": primary_model_dir,
        },
        "baseline": {
            "name": "memgen",
            "repo_root": repo_root,
            "task_name": task_name,
            "memgen_run_mode": "evaluate",
            "memgen_config_path": str(task_defaults["memgen_config_path"]),
            "trigger_active": False,
            "insertion_profile": str(task_defaults["insertion_profile"]),
            "load_model_path": None,
            "max_prompt_aug_num": int(task_defaults["max_prompt_aug_num"]),
            "max_inference_aug_num": int(task_defaults["max_inference_aug_num"]),
            "min_free_disk_gb": 8.0,
            "extra_options": list(task_defaults["extra_options"]),
        },
    }
    _write_json(output_config, payload)
    return payload


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Materialize PLANv8 V8-7 comparator configs.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    rag = subparsers.add_parser("rag")
    rag.add_argument("--task_name", required=True)
    rag.add_argument("--output_config", required=True)
    rag.add_argument("--eval_path", required=True)
    rag.add_argument("--support_path", required=True)
    rag.add_argument("--primary_model_dir", required=True)
    rag.add_argument("--primary_backbone_name", required=True)
    rag.add_argument("--hf_cache_dir", default="/root/autodl-tmp/hf-cache")

    memgen = subparsers.add_parser("memgen")
    memgen.add_argument("--task_name", required=True)
    memgen.add_argument("--output_config", required=True)
    memgen.add_argument("--primary_model_dir", required=True)
    memgen.add_argument("--primary_backbone_name", required=True)
    memgen.add_argument("--repo_root", default="MemGen-master")
    return parser


def main() -> int:
    args = _build_arg_parser().parse_args()
    if args.command == "rag":
        materialize_planv8_v8_7_rag_config(
            task_name=args.task_name,
            output_config=Path(args.output_config),
            eval_path=args.eval_path,
            support_path=args.support_path,
            primary_model_dir=args.primary_model_dir,
            primary_backbone_name=args.primary_backbone_name,
            hf_cache_dir=args.hf_cache_dir,
        )
        return 0
    materialize_planv8_v8_7_memgen_config(
        task_name=args.task_name,
        output_config=Path(args.output_config),
        primary_model_dir=args.primary_model_dir,
        primary_backbone_name=args.primary_backbone_name,
        repo_root=args.repo_root,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
