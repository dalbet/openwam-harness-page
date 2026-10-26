"""Completion-evidence tests for baseline displacement and drawer ownership."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from common import config  # noqa: E402
from engine import completion  # noqa: E402


def site_snapshot(plate_x: float) -> dict:
    return {
        "scene_epoch": 1,
        "sim_action_count": 100,
        "observation_id": f"query-test:1:100:{plate_x}",
        "sim": {
            "objects": {
                "plate_1": {"pos": [plate_x, 0.0, 0.9025]},
            },
            "gripper_opening_m": 0.08,
        },
        "named_poses": {
            "sites": {
                "main_table_stove_front_region": [0.05, 0.0, 0.895],
            },
            "site_geometry": {
                "main_table_stove_front_region": {
                    "size": [0.04, 0.04, 0.007],
                    "mat": [1, 0, 0, 0, 1, 0, 0, 0, 1],
                }
            },
        },
        "joints": {},
    }


class PushCompletionTest(unittest.TestCase):
    def test_baseline_is_read_from_nested_sim_objects(self) -> None:
        task = config.task_for(5)
        step = {
            "skill": "push_object",
            "object_n": "plate_1",
            "target_n": "main_table_stove_front_region",
            "relation": "on",
        }
        baseline = site_snapshot(-0.10)
        current = site_snapshot(0.05)
        state = completion.goal_state(current, step, task, baseline)
        self.assertAlmostEqual(state["displacement_m"], 0.15, places=3)
        self.assertTrue(state["ok"])

    def test_displacement_uses_the_episode_baseline_not_zero(self) -> None:
        task = config.task_for(5)
        step = {
            "skill": "push_object",
            "object_n": "plate_1",
            "target_n": "main_table_stove_front_region",
            "relation": "on",
        }
        baseline = site_snapshot(0.045)
        current = site_snapshot(0.05)
        state = completion.goal_state(current, step, task, baseline)
        self.assertAlmostEqual(state["displacement_m"], 0.005, places=3)
        self.assertFalse(state["ok"])


class DrawerCompletionTest(unittest.TestCase):
    def test_drawer_joint_is_owned_by_the_target_body(self) -> None:
        snapshot = site_snapshot(0.0)
        snapshot["joints"] = {
            "wooden_cabinet_1_top_level": -0.2,
            "wine_rack_1_joint0": 0.0,
        }
        name, qpos = completion.drawer_joint(snapshot, "top", "wooden_cabinet_1_top_region")
        self.assertEqual(name, "wooden_cabinet_1_top_level")
        self.assertAlmostEqual(qpos, -0.2)

    def test_wine_rack_has_no_drawer_joint(self) -> None:
        snapshot = site_snapshot(0.0)
        snapshot["joints"] = {"wooden_cabinet_1_top_level": -0.2}
        name, qpos = completion.drawer_joint(snapshot, "top", "wine_rack_1_top_region")
        self.assertIsNone(name)
        self.assertIsNone(qpos)


if __name__ == "__main__":
    unittest.main()
