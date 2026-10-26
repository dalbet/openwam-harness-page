"""Contract tests for drawer binding and completed-step context."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from common import config  # noqa: E402
from common import plan_contract  # noqa: E402
from common.plan_contract import validate_plan  # noqa: E402


def snapshot() -> dict:
    return {
        "scene_epoch": 1,
        "sim_action_count": 0,
        "observation_id": "query-test:1:0:1",
        "sim": {
            "objects": {
                "akita_black_bowl_1": {"pos": [-0.1, 0.1, 0.9]},
                "wine_bottle_1": {"pos": [0.1, 0.1, 0.9]},
                "plate_1": {"pos": [0.0, -0.1, 0.9]},
            }
        },
        "named_poses": {
            "sites": {
                "wooden_cabinet_1_top_region": [0.0, 0.3, 0.95],
                "wooden_cabinet_1_middle_region": [0.0, 0.3, 0.9],
                "wine_rack_1_top_region": [0.2, 0.2, 1.0],
                "flat_stove_1_cook_region": [0.0, 0.0, 0.9],
                "main_table_stove_front_region": [0.0, -0.1, 0.9],
            },
            "site_geometry": {
                "wooden_cabinet_1_top_region": {"size": [0.06, 0.06, 0.03], "mat": [1, 0, 0, 0, 1, 0, 0, 0, 1]},
                "wine_rack_1_top_region": {"size": [0.05, 0.05, 0.01], "mat": [1, 0, 0, 0, 1, 0, 0, 0, 1]},
            },
        },
        "joints": {
            "wooden_cabinet_1_top_level": 0.0,
            "wooden_cabinet_1_middle_level": 0.0,
            "flat_stove_1_button": 0.0,
        },
    }


SKILLS = config.skill_config()
TASK9 = config.task_for(9)
TASK3 = config.task_for(3)


def plan(*steps: dict) -> dict:
    return {"interpretation": "test", "success_condition": "test", "steps": list(steps)}


class DrawerBindingTest(unittest.TestCase):
    def test_wine_rack_top_region_is_not_a_drawer(self) -> None:
        _, errors = validate_plan(
            plan(
                {
                    "skill": "open_drawer",
                    "object_n": None,
                    "target_n": "wine_rack_1_top_region",
                    "relation": "open",
                    "prompt": "open the wine rack drawer",
                }
            ),
            instruction=TASK9["instruction"],
            task=TASK9,
            snapshot=snapshot(),
            skills=SKILLS,
        )
        self.assertEqual([row["code"] for row in errors], ["drawer_joint_missing"])

    def test_wine_rack_placement_does_not_require_a_drawer_step(self) -> None:
        checked, errors = validate_plan(
            plan(
                {
                    "skill": "pick_place",
                    "object_n": "wine_bottle_1",
                    "target_n": "wine_rack_1_top_region",
                    "relation": "on",
                    "prompt": "put the wine bottle on the rack",
                }
            ),
            instruction=TASK9["instruction"],
            task=TASK9,
            snapshot=snapshot(),
            skills=SKILLS,
        )
        self.assertEqual(errors, [])
        self.assertIsNotNone(checked)

    def test_cabinet_drawer_placement_requires_an_open_step(self) -> None:
        _, errors = validate_plan(
            plan(
                {
                    "skill": "pick_place",
                    "object_n": "akita_black_bowl_1",
                    "target_n": "wooden_cabinet_1_top_region",
                    "relation": "in",
                    "prompt": "put the black bowl in the top drawer",
                }
            ),
            instruction=TASK3["instruction"],
            task=TASK3,
            snapshot=snapshot(),
            skills=SKILLS,
        )
        self.assertIn("closed_drawer_requires_open_step", [row["code"] for row in errors])

    def test_replan_accepts_drawer_opened_by_a_completed_step(self) -> None:
        checked, errors = validate_plan(
            plan(
                {
                    "skill": "pick_place",
                    "object_n": "akita_black_bowl_1",
                    "target_n": "wooden_cabinet_1_top_region",
                    "relation": "in",
                    "prompt": "put the black bowl in the top drawer",
                }
            ),
            instruction=TASK3["instruction"],
            task=TASK3,
            snapshot=snapshot(),
            skills=SKILLS,
            prior_steps=[
                {
                    "skill": "open_drawer",
                    "object_n": None,
                    "target_n": "wooden_cabinet_1_top_region",
                    "relation": "open",
                }
            ],
        )
        self.assertEqual(errors, [])
        self.assertIsNotNone(checked)


if __name__ == "__main__":
    unittest.main()


class GoalPreservationTest(unittest.TestCase):
    def test_replan_keeping_goal_is_accepted(self):
        goal = {"skill": "pick_place", "object_n": "bowl_1", "target_n": "cabinet_top_region", "relation": "in"}
        steps = [
            {"skill": "open_drawer", "object_n": None, "target_n": "cabinet_top_region", "relation": "open"},
            {"skill": "pick_place", "object_n": "bowl_1", "target_n": "cabinet_top_region", "relation": "in"},
        ]
        self.assertTrue(plan_contract.preserves_goal(steps, goal))

    def test_replan_that_drops_the_goal_is_rejected(self):
        goal = {"skill": "pick_place", "object_n": "bowl_1", "target_n": "cabinet_top_region", "relation": "in"}
        steps = [
            {"skill": "pick_place", "object_n": "bowl_1", "target_n": "main_table_cabinet_region", "relation": "on"},
        ]
        self.assertFalse(plan_contract.preserves_goal(steps, goal))
