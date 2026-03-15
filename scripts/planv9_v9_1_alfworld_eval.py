#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import textworld
import textworld.gym
from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredExpert, AlfredExpertType, AlfredInfos

from memtotal.models import BackboneWrapper
from memtotal.tasks.alfworld_env import extract_alfworld_goal
from memtotal.utils.io import initialize_run_artifacts, write_json, write_jsonl


@dataclass(frozen=True)
class StepState:
    index: int
    observation: str
    expert_action: str
    admissible_commands: list[str]


def _normalize_action_text(text: str) -> str:
    compact = str(text).strip().splitlines()[0] if str(text).strip() else ""
    compact = re.sub(r"^[a-z ]*:\s*", "", compact.strip(), flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", compact.lower()).strip()


def _score_candidate_action(predicted_text: str, candidate_action: str) -> float:
    normalized_prediction = _normalize_action_text(predicted_text)
    normalized_candidate = _normalize_action_text(candidate_action)
    if normalized_prediction == normalized_candidate:
        return 10.0
    if not normalized_prediction or not normalized_candidate:
        return 0.0
    overlap = set(normalized_prediction.split()) & set(normalized_candidate.split())
    ratio = SequenceMatcher(a=normalized_prediction, b=normalized_candidate).ratio()
    return float(len(overlap)) + ratio


def _resolve_action(predicted_text: str, admissible_commands: list[str]) -> tuple[str, str]:
    if not admissible_commands:
        return _normalize_action_text(predicted_text), "raw_no_admissible"
    normalized_prediction = _normalize_action_text(predicted_text)
    normalized_lookup = {_normalize_action_text(command): command for command in admissible_commands}
    if normalized_prediction in normalized_lookup:
        return normalized_lookup[normalized_prediction], "exact"
    ranked = sorted(
        (
            (_score_candidate_action(predicted_text, command), command)
            for command in admissible_commands
        ),
        key=lambda item: (-item[0], item[1]),
    )
    return ranked[0][1], "fuzzy"


def _summarize_history_entries(history_entries: list[dict[str, Any]], *, word_budget: int = 120) -> str:
    lines = [
        f"Step {entry['step']}: action={entry['action']} -> {entry['observation']}"
        for entry in history_entries
    ]
    words = " || ".join(lines).split()
    return " ".join(words[:word_budget]).strip()


def _select_retrieved_history(history_entries: list[dict[str, Any]], query_text: str, *, top_k: int = 4) -> list[dict[str, Any]]:
    query_tokens = set(_normalize_action_text(query_text).split())
    scored: list[tuple[float, dict[str, Any]]] = []
    for entry in history_entries:
        candidate_text = f"{entry['action']} {entry['observation']}"
        candidate_tokens = set(_normalize_action_text(candidate_text).split())
        overlap = len(query_tokens & candidate_tokens)
        ratio = SequenceMatcher(a=_normalize_action_text(query_text), b=_normalize_action_text(candidate_text)).ratio()
        scored.append((float(overlap) + ratio, entry))
    scored.sort(key=lambda item: (-item[0], int(item[1]["step"])))
    return [entry for _score, entry in scored[:top_k]]


def _build_prompt(
    *,
    baseline_id: str,
    goal: str,
    current_observation: str,
    history_entries: list[dict[str, Any]],
) -> str:
    query = f"Goal: {goal} || Current observation: {current_observation} || Return the next action only."
    if baseline_id == "b0_short_window":
        return query
    if baseline_id == "b1_full_history":
        transcript = " || ".join(
            f"History step {entry['step']}: action={entry['action']} || observation={entry['observation']}"
            for entry in history_entries
        )
        return f"{transcript} || {query}" if transcript else query
    if baseline_id == "b2_text_summary":
        summary = _summarize_history_entries(history_entries)
        return f"History summary: {summary} || {query}" if summary else query
    if baseline_id == "b3_text_rag":
        retrieved = _select_retrieved_history(history_entries, f"{goal} {current_observation}")
        blocks = [
            f"Retrieved step {entry['step']}: action={entry['action']} || observation={entry['observation']}"
            for entry in retrieved
        ]
        return f"{' || '.join(blocks)} || {query}" if blocks else query
    raise ValueError(f"Unsupported ALFWorld baseline_id: {baseline_id}")


def _register_env(game_file: str, *, max_episode_steps: int) -> tuple[Any, str]:
    request_infos = textworld.EnvInfos(won=True, admissible_commands=True, extras=["gamefile", "expert_plan"])
    wrappers = [AlfredDemangler(shuffle=False), AlfredInfos, AlfredExpert(AlfredExpertType.HANDCODED)]
    env_id = textworld.gym.register_games(
        [game_file],
        request_infos,
        batch_size=1,
        asynchronous=False,
        max_episode_steps=max_episode_steps,
        wrappers=wrappers,
    )
    env = textworld.gym.make(env_id)
    return env, env_id


def _run_episode(
    *,
    backbone: BackboneWrapper,
    episode_payload: dict[str, Any],
    baseline_id: str,
    max_steps: int,
) -> dict[str, Any]:
    game_file = str(episode_payload["game_file"])
    task_family = str(episode_payload["task_family"])
    split = str(episode_payload["split"])
    env, _env_id = _register_env(game_file, max_episode_steps=max_steps + 2)
    history_entries: list[dict[str, Any]] = []
    invalid_resolution_count = 0
    try:
        initial_observations, initial_infos = env.reset()
        initial_observation = str(initial_observations[0]).strip()
        _prefix, goal = extract_alfworld_goal(initial_observation)
        bootstrap_plan = initial_infos["extra.expert_plan"][0]
        if not bootstrap_plan:
            raise RuntimeError(f"No bootstrap expert plan for {game_file}.")
        bootstrap_action = str(bootstrap_plan[0]).strip()
        next_observations, _reward, _done, next_infos = env.step([bootstrap_action])
        current_observation = str(next_observations[0]).strip()
        done = bool(next_infos["won"][0])
        step_rows: list[dict[str, Any]] = [
            {
                "step": 0,
                "action": bootstrap_action,
                "observation": current_observation,
                "action_resolution": "expert_bootstrap",
            }
        ]
        history_entries.append(
            {
                "step": 0,
                "action": bootstrap_action,
                "observation": current_observation,
            }
        )

        steps_executed = 0
        while not done and steps_executed < max_steps:
            expert_plan = next_infos["extra.expert_plan"][0]
            expert_action = str(expert_plan[0]).strip() if expert_plan else ""
            admissible_commands = [str(command) for command in next_infos["admissible_commands"][0]]
            prompt = _build_prompt(
                baseline_id=baseline_id,
                goal=goal,
                current_observation=current_observation,
                history_entries=history_entries,
            )
            predicted_text = backbone.generate([prompt])[0]
            resolved_action, resolution_mode = _resolve_action(predicted_text, admissible_commands)
            if resolution_mode != "exact":
                invalid_resolution_count += 1
            next_observations, _reward, done_flags, next_infos = env.step([resolved_action])
            current_observation = str(next_observations[0]).strip()
            done = bool(done_flags[0]) or bool(next_infos["won"][0])
            steps_executed += 1
            step_rows.append(
                {
                    "step": steps_executed,
                    "prompt": prompt,
                    "predicted_text": predicted_text,
                    "action": resolved_action,
                    "action_resolution": resolution_mode,
                    "expert_action": expert_action,
                    "observation": current_observation,
                    "done": done,
                }
            )
            history_entries.append(
                {
                    "step": steps_executed,
                    "action": resolved_action,
                    "observation": current_observation,
                }
            )
        return {
            "episode_id": f"{task_family}-{Path(game_file).parent.name}",
            "game_file": game_file,
            "split": split,
            "task_family": task_family,
            "goal": goal,
            "success": bool(done),
            "steps_executed": int(steps_executed),
            "invalid_resolution_count": int(invalid_resolution_count),
            "history_entries": history_entries,
            "step_rows": step_rows,
        }
    finally:
        env.close()


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PLANv9 V9-1 ALFWorld episode-level eval.")
    parser.add_argument("--manifest_path", required=True)
    parser.add_argument("--baseline_id", required=True)
    parser.add_argument("--primary_model_dir", required=True)
    parser.add_argument("--primary_backbone_name", default="Qwen3-4B")
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--eval_episodes", type=int, default=120)
    parser.add_argument("--max_steps", type=int, default=12)
    parser.add_argument("--hf_cache_dir", default="/root/autodl-tmp/hf-cache")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    manifest = json.loads(Path(args.manifest_path).read_text())
    selected_episodes = list(manifest.get("selected_episodes", []))[: int(args.eval_episodes)]
    output_dir = Path(args.output_dir).resolve()
    config = {
        "experiment": {
            "name": f"planv9_v9_1_alfworld_{args.baseline_id}",
            "stage": "V9-1",
            "method_variant": args.baseline_id,
        },
        "runtime": {
            "device": "cuda",
            "eval_examples": len(selected_episodes),
            "max_steps": int(args.max_steps),
        },
        "backbone": {
            "name": str(args.primary_backbone_name),
            "load_mode": "hf_causal_lm",
            "model_id": str(args.primary_model_dir),
            "dtype": "bfloat16",
            "cache_dir": str(args.hf_cache_dir),
            "attn_implementation": "sdpa",
            "use_chat_template": True,
            "chat_template_enable_thinking": False,
            "max_new_tokens": 64,
        },
        "task": {
            "name": "planv9_v9_1_alfworld",
            "metric_name": "success_rate",
            "evaluator": {"type": "exact_match", "normalizer": "action"},
        },
        "baseline": {
            "family": "prompting",
            "mode": "vanilla",
            "baseline_id": args.baseline_id,
        },
    }
    initialize_run_artifacts(
        output_dir=output_dir,
        config=config,
        seed=args.seed,
        argv=sys.argv if argv is None else ["planv9_v9_1_alfworld_eval", *argv],
    )
    backbone = BackboneWrapper(
        name=str(args.primary_backbone_name),
        load_mode="hf_causal_lm",
        hidden_size=None,
        seed=args.seed,
        model_id=str(args.primary_model_dir),
        device="cuda",
        dtype="bfloat16",
        cache_dir=str(args.hf_cache_dir),
        attn_implementation="sdpa",
        max_new_tokens=64,
        use_chat_template=True,
        chat_template_enable_thinking=False,
    )

    episode_rows = [
        _run_episode(
            backbone=backbone,
            episode_payload=episode_payload,
            baseline_id=args.baseline_id,
            max_steps=int(args.max_steps),
        )
        for episode_payload in selected_episodes
    ]
    success_count = sum(1 for row in episode_rows if row["success"])
    metrics = {
        "mode": "eval_alfworld_longhorizon",
        "benchmark_id": "alfworld",
        "baseline_id": args.baseline_id,
        "examples_evaluated": len(episode_rows),
        "success_rate": success_count / max(1, len(episode_rows)),
        "task_score": success_count / max(1, len(episode_rows)),
        "mean_steps_executed": (
            sum(float(row["steps_executed"]) for row in episode_rows) / len(episode_rows)
            if episode_rows
            else 0.0
        ),
        "mean_invalid_resolution_count": (
            sum(float(row["invalid_resolution_count"]) for row in episode_rows) / len(episode_rows)
            if episode_rows
            else 0.0
        ),
        "step_budget": int(args.max_steps),
        "backbone": str(args.primary_backbone_name),
    }
    write_json(output_dir / "metrics.json", metrics)
    write_jsonl(output_dir / "episode_rollouts.jsonl", episode_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
