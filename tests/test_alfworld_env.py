from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from memtotal.tasks.alfworld_env import extract_alfworld_goal, select_alfworld_smoke_game_files


class AlfworldEnvTest(unittest.TestCase):
    def test_extract_goal_from_initial_observation(self) -> None:
        observation = (
            "-= Welcome to TextWorld, ALFRED! =-\n\n"
            "You are in the kitchen.\n\n"
            "Your task is to: pick up the apple."
        )
        prefix, goal = extract_alfworld_goal(observation)
        self.assertIn("You are in the kitchen.", prefix)
        self.assertEqual(goal, "pick up the apple.")

    def test_select_smoke_games_prefers_distinct_task_families(self) -> None:
        game_files = [
            Path("/tmp/valid_seen/look_at_obj_in_light-A/trial_1/game.tw-pddl"),
            Path("/tmp/valid_seen/look_at_obj_in_light-B/trial_2/game.tw-pddl"),
            Path("/tmp/valid_seen/pick_and_place_simple-A/trial_3/game.tw-pddl"),
            Path("/tmp/valid_seen/pick_heat_then_place_in_recep-A/trial_4/game.tw-pddl"),
            Path("/tmp/valid_seen/pick_two_obj_and_place-A/trial_5/game.tw-pddl"),
        ]
        selected = select_alfworld_smoke_game_files(game_files, max_examples=4)
        families = {path.parent.parent.name.split("-", 1)[0] for path in selected}
        self.assertEqual(len(selected), 4)
        self.assertEqual(len(families), 4)


if __name__ == "__main__":
    unittest.main()
