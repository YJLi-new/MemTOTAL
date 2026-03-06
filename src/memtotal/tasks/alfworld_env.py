from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path
from typing import Any
from urllib.request import urlopen


ALFWORLD_JSON_FILES_URL = "https://github.com/alfworld/alfworld/releases/download/0.2.2/json_2.1.1_json.zip"
ALFWORLD_PDDL_FILES_URL = "https://github.com/alfworld/alfworld/releases/download/0.2.2/json_2.1.1_pddl.zip"
ALFWORLD_TW_PDDL_FILES_URL = (
    "https://github.com/alfworld/alfworld/releases/download/0.4.0/json_2.1.2_tw-pddl.zip"
)
ALFWORLD_TEXTWORLD_SPLIT_ROOT = "json_2.1.1"


def _download_file(url: str, download_dir: Path) -> Path:
    download_dir.mkdir(parents=True, exist_ok=True)
    destination = download_dir / url.rsplit("/", 1)[-1]
    if destination.exists():
        return destination
    with urlopen(url) as response:
        destination.write_bytes(response.read())
    return destination


def _unzip_file(archive_path: Path, destination_dir: Path) -> None:
    with zipfile.ZipFile(archive_path) as zipped_file:
        zipped_file.extractall(destination_dir)


def ensure_alfworld_textworld_assets(asset_root: str | Path) -> dict[str, str]:
    try:
        from alfworld.info import ALFRED_PDDL_PATH, ALFRED_TWL2_PATH
    except ImportError as exc:
        raise RuntimeError("ALFWorld is not installed. Run `python -m pip install alfworld` first.") from exc

    asset_root = Path(asset_root).resolve()
    split_root = asset_root / ALFWORLD_TEXTWORLD_SPLIT_ROOT
    marker_game = next(split_root.rglob("game.tw-pddl"), None) if split_root.exists() else None
    if marker_game is None:
        download_dir = asset_root / "_downloads"
        for url in (ALFWORLD_JSON_FILES_URL, ALFWORLD_PDDL_FILES_URL, ALFWORLD_TW_PDDL_FILES_URL):
            archive_path = _download_file(url, download_dir)
            _unzip_file(archive_path, asset_root)

    marker_game = next(split_root.rglob("game.tw-pddl"), None) if split_root.exists() else None
    if marker_game is None:
        raise RuntimeError(f"ALFWorld TextWorld assets are still missing under {split_root}.")

    logic_root = asset_root / "logic"
    logic_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ALFRED_PDDL_PATH, logic_root / "alfred.pddl")
    shutil.copy2(ALFRED_TWL2_PATH, logic_root / "alfred.twl2")
    return {
        "asset_root": str(asset_root),
        "split_root": str(split_root),
        "logic_root": str(logic_root),
    }


def _task_family_from_game_file(game_file: Path) -> str:
    return game_file.parent.parent.name.split("-", 1)[0]


def list_alfworld_game_files(asset_root: str | Path, *, split: str) -> list[Path]:
    split_root = Path(asset_root).resolve() / ALFWORLD_TEXTWORLD_SPLIT_ROOT / split
    return sorted(split_root.rglob("game.tw-pddl"))


def select_alfworld_smoke_game_files(game_files: list[Path], max_examples: int) -> list[Path]:
    if max_examples <= 0:
        return []

    selected: list[Path] = []
    seen_task_families: set[str] = set()
    for game_file in game_files:
        task_family = _task_family_from_game_file(game_file)
        if task_family in seen_task_families:
            continue
        selected.append(game_file)
        seen_task_families.add(task_family)
        if len(selected) == max_examples:
            return selected

    selected_set = {path.resolve() for path in selected}
    for game_file in game_files:
        if game_file.resolve() in selected_set:
            continue
        selected.append(game_file)
        if len(selected) == max_examples:
            break
    return selected


def extract_alfworld_goal(initial_observation: str) -> tuple[str, str]:
    marker = "Your task is to:"
    if marker not in initial_observation:
        return initial_observation.strip(), ""
    prefix, goal = initial_observation.split(marker, 1)
    return prefix.strip(), goal.strip()


def build_alfworld_transition_example(game_file: str | Path) -> dict[str, Any]:
    try:
        import textworld
        import textworld.gym
        from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredExpert, AlfredExpertType, AlfredInfos
    except ImportError as exc:
        raise RuntimeError(
            "ALFWorld TextWorld dependencies are missing. Ensure `alfworld` and `textworld` are installed."
        ) from exc

    game_file = Path(game_file).resolve()
    traj_data = json.loads((game_file.parent / "traj_data.json").read_text())
    request_infos = textworld.EnvInfos(won=True, admissible_commands=True, extras=["gamefile", "expert_plan"])
    wrappers = [AlfredDemangler(shuffle=False), AlfredInfos, AlfredExpert(AlfredExpertType.HANDCODED)]
    env_id = textworld.gym.register_games(
        [str(game_file)],
        request_infos,
        batch_size=1,
        asynchronous=False,
        max_episode_steps=20,
        wrappers=wrappers,
    )
    env = textworld.gym.make(env_id)

    try:
        initial_observations, initial_infos = env.reset()
        initial_observation = str(initial_observations[0]).strip()
        _, goal = extract_alfworld_goal(initial_observation)
        first_expert_plan = initial_infos["extra.expert_plan"][0]
        if not first_expert_plan:
            raise RuntimeError(f"No initial expert action for {game_file}.")
        previous_action = str(first_expert_plan[0])

        next_observations, _, _, next_infos = env.step([previous_action])
        next_observation = str(next_observations[0]).strip()
        next_expert_plan = next_infos["extra.expert_plan"][0]
        if not next_expert_plan:
            raise RuntimeError(f"No follow-up expert action for {game_file}.")
        next_action = str(next_expert_plan[0])
        admissible_commands = [str(command) for command in next_infos["admissible_commands"][0]]
    finally:
        env.close()

    task_descriptor = game_file.parent.parent.name
    return {
        "id": f"alfworld-{game_file.parent.name}",
        "observation": next_observation,
        "goal": goal,
        "answer": next_action,
        "previous_action": previous_action,
        "task_type": str(traj_data.get("task_type", _task_family_from_game_file(game_file))),
        "task_desc": str(traj_data.get("turk_annotations", {}).get("anns", [{}])[0].get("task_desc", goal)),
        "game_file": str(game_file),
        "admissible_commands": admissible_commands,
    }


def materialize_alfworld_textworld_smoke(
    *,
    asset_root: str | Path,
    max_examples: int,
    split: str = "valid_seen",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    asset_info = ensure_alfworld_textworld_assets(asset_root)
    candidate_games = list_alfworld_game_files(asset_root, split=split)
    if not candidate_games:
        raise RuntimeError(f"No ALFWorld games found for split={split} under {asset_root}.")
    selected_games = select_alfworld_smoke_game_files(candidate_games, max_examples=max_examples)
    rows = [build_alfworld_transition_example(game_file) for game_file in selected_games]
    metadata = {
        **asset_info,
        "split": split,
        "selected_games": [str(game_file) for game_file in selected_games],
        "selected_task_types": [str(row["task_type"]) for row in rows],
        "transition_protocol": "Execute the first official hand-coded expert action, then predict the next expert action.",
    }
    return rows, metadata
