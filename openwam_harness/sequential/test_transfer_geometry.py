"""Geometry of the three transport targets. No model and no env.step."""

from __future__ import annotations

import unittest

from free_task import judge_user_goal
from placement import resolve_motion_step
from transfer_geometry import motion_target, relation_check


SITE = [-0.2634, 0.2179, 0.905]
HOLD = [-0.0137, -0.0383, -0.0318]


def _obs(obj, eef):
    return {
        "observation_id": "geom",
        "scene_epoch": 1,
        "named_poses": {"sites": {"flat_stove_1_cook_region": list(SITE)}},
        "sim": {
            "objects": {"akita_black_bowl_1": {"pos": obj, "identified": True}},
            "eef_pos": eef,
            "gripper_opening_m": 0.02,
        },
    }


class TransferGeometryTest(unittest.TestCase):
    def test_stove_travel_keeps_region_clearance_and_descend_uses_the_site_top(self):
        bowl = [SITE[0], SITE[1], 1.02]
        eef = [bowl[i] - HOLD[i] for i in range(3)]
        plan = {"objects": {
            "akita_black_bowl_1": {"sim_name": "akita_black_bowl_1"},
            "flat_stove_1": {"sim_name": "flat_stove_1", "place_site": "flat_stove_1_cook_region"},
        }}
        travel = resolve_motion_step(
            {"object_ref": "akita_black_bowl_1", "target_ref": "flat_stove_1", "args": {
                "xyz_from": "segment:stove", "z_offset": 0.12, "gripper": 1,
                "object_ref": "akita_black_bowl_1", "hold_offset": "akita_black_bowl_1",
                "placement": "support_region", "motion": "travel",
            }},
            {},
            _obs(bowl, eef),
            plan,
        )
        descend = resolve_motion_step(
            {"object_ref": "akita_black_bowl_1", "target_ref": "flat_stove_1", "args": {
                "xyz_from": "segment:stove", "z_offset": 0.03, "gripper": 1,
                "object_ref": "akita_black_bowl_1", "hold_offset": "akita_black_bowl_1",
                "placement": "support_region", "motion": "vertical",
            }},
            {},
            _obs(bowl, eef),
            plan,
        )
        self.assertTrue(travel["ok"])
        self.assertAlmostEqual(travel["object_origin"][2], round(SITE[2] + 0.12, 4), places=3)
        self.assertTrue(descend["ok"])
        self.assertAlmostEqual(descend["object_origin"][2], SITE[2] + 0.0025 - (-0.012), places=3)
        self.assertNotAlmostEqual(descend["object_origin"][2], SITE[2] + 0.03, places=3)
        self.assertEqual(descend["placement_kind"], "support_site")

    def test_cheese_near_the_bowl_is_not_inside_it(self):
        bowl = [0.0, 0.0, 0.9]
        near = relation_check([0.08, 0.0, 0.9], bowl, "cream_cheese_1", "akita_black_bowl_1")
        self.assertFalse(near["ok"])
        self.assertEqual(near["predicate"], "bowl_interior")
        self.assertEqual(near["contact_status"], "not_provided")
        inside = relation_check([0.01, 0.0, 0.9 + 0.0286], bowl, "cream_cheese_1", "akita_black_bowl_1")
        self.assertTrue(inside["ok"])
        goal = judge_user_goal(
            [{"object_ref": "cream_cheese_1", "target_ref": "akita_black_bowl_1", "relation": "on"}],
            {"objects": {
                "cream_cheese_1": {"pos": [0.08, 0.0, 0.9]},
                "akita_black_bowl_1": {"pos": bowl},
            }},
        )
        self.assertFalse(goal["user_goal_success"])

    def test_plate_uses_the_support_plane_and_official_xy_not_body_center(self):
        plate = [0.1, -0.2, 0.9]
        far = relation_check([0.1 + 0.05, -0.2, 0.9], plate, "akita_black_bowl_1", "plate_1")
        self.assertFalse(far["ok"])
        self.assertEqual(far["xy_limit_m"], 0.03)
        landed_z = 0.9 + 0.00535 - (-0.012)
        close = relation_check([0.11, -0.2, landed_z], plate, "akita_black_bowl_1", "plate_1")
        self.assertTrue(close["ok"])
        self.assertEqual(close["predicate"], "plate_support")

    def test_cabinet_top_clearance_differs_for_the_bowl_and_the_bottle(self):
        surface = [0.0, 0.2, 1.0]
        bowl = motion_target("akita_black_bowl_1", "wooden_cabinet_1", surface, motion="vertical", clearance=0.03)
        bottle = motion_target("wine_bottle_1", "wooden_cabinet_1", surface, motion="vertical", clearance=0.03)
        self.assertAlmostEqual(bowl["object_z"], 1.0 + 0.00147 - (-0.012))
        self.assertAlmostEqual(bottle["object_z"], 1.0 + 0.00147 - 0.00111)
        self.assertGreater(abs(bowl["object_z"] - bottle["object_z"]), 0.01)


if __name__ == "__main__":
    unittest.main()
