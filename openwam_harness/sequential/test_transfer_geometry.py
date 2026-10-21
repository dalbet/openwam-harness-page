"""Geometry of the three transport targets. No model and no env.step."""

from __future__ import annotations

import unittest

from candidates import resolve_plan_bindings, scene_graph
from free_task import judge_user_goal
from placement import resolve_motion_step
from suites import _catalog, objects_for_names
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

    def test_official_cabinet_sentence_selects_top_side_only(self):
        from pathlib import Path

        bddl = Path(
            "/root/Page_dmh/openwam_harness/envs/rpent/lib/python3.10/site-packages/"
            "liberopro/liberopro/bddl_files/libero_goal/put_the_bowl_on_the_stove.bddl"
        )
        catalog = _catalog(str(bddl))
        names = [item["name"] for item in catalog["instances"] + catalog["fixtures"]]
        objects = objects_for_names(names, catalog)
        graph = scene_graph(objects, {}, scene_epoch=1, observation_id="cabinet")
        cabinet = next(item["n"] for item in graph["nodes"] if item["object_ref"] == "wooden_cabinet_1")
        bottle = next(item["n"] for item in graph["nodes"] if item["object_ref"] == "wine_bottle_1")
        plan = {
            "goal": "把酒瓶放到柜子顶上。",
            "steps": [{
                "step_id": "step1",
                "skill": "transfer",
                "object_n": bottle,
                "target_n": cabinet,
                "relation": "on",
                "prompt": "put the wine bottle on top of the cabinet",
                "depends_on": [],
                "preconditions": [],
                "completion_check": "object_placed",
                "budget": {"max_s": 180, "max_attempts": 1},
                "on_failure": "stop",
            }],
        }
        resolved, bindings = resolve_plan_bindings(plan, graph)
        self.assertEqual(bindings[1]["query_key"], "wooden_cabinet_1_top_side")
        self.assertEqual(resolved["steps"][0]["target_region"], "wooden_cabinet_1_top_side")
        generic = dict(plan, goal="把酒瓶放到柜子上")
        with self.assertRaises(Exception):
            resolve_plan_bindings(generic, graph)

    def test_fixture_target_site_reaches_localize_and_transport_checks(self):
        """A cabinet has place_sites, not place_site. The resolved region is the pose."""
        from pathlib import Path

        from executor import ToolOutcome
        from schema import prepare_plan
        from verify import _target_reading, verify_step

        bddl = Path(
            "/root/Page_dmh/openwam_harness/envs/rpent/lib/python3.10/site-packages/"
            "liberopro/liberopro/bddl_files/libero_goal/put_the_bowl_on_the_stove.bddl"
        )
        catalog = _catalog(str(bddl))
        names = [item["name"] for item in catalog["instances"] + catalog["fixtures"]]
        objects = objects_for_names(names, catalog)
        self.assertNotIn("place_site", objects["wooden_cabinet_1"])
        self.assertIn("wooden_cabinet_1_top_side", objects["wooden_cabinet_1"]["place_sites"])
        graph = scene_graph(objects, {}, scene_epoch=1, observation_id="cabinet")
        cabinet = next(item["n"] for item in graph["nodes"] if item["object_ref"] == "wooden_cabinet_1")
        bottle = next(item["n"] for item in graph["nodes"] if item["object_ref"] == "wine_bottle_1")
        plan = {
            "plan_id": "wine-top",
            "goal": "把酒瓶放到柜子顶上。",
            "candidate_set_version": graph["candidate_set_version"],
            "scene_epoch": graph["scene_epoch"],
            "steps": [{
                "step_id": "step1",
                "skill": "transfer",
                "object_n": bottle,
                "target_n": cabinet,
                "relation": "on",
                "prompt": "put the wine bottle on top of the cabinet",
                "depends_on": [],
                "preconditions": [],
                "completion_check": "object_placed",
                "budget": {"max_s": 180, "max_attempts": 1},
                "on_failure": "stop",
            }],
        }
        resolved, _bindings = resolve_plan_bindings(plan, graph)
        prepared = prepare_plan(resolved, scene_objects=objects, execution_mode="free")
        steps = {step["step_id"]: step for step in prepared["steps"]}
        for key in ("move_step1", "descend_step1", "place_step1"):
            self.assertEqual(steps[key]["target_region"], "wooden_cabinet_1_top_side", key)

        site_xyz = [0.0, 0.2, 1.1]
        obs = {
            "observation_id": "cabinet-obs",
            "scene_epoch": 1,
            "env_step": 4,
            "sim_action_count": 9,
            "step_clock": "tool_record",
            "timestamp": 4,
            "cameras": {"agentview": {"id": "a"}, "wrist": {"id": "w"}},
            "named_poses": {"sites": {"wooden_cabinet_1_top_side": list(site_xyz)}},
            "sim": {
                "objects": {"wine_bottle_1": {"pos": [-0.18, -0.05, 0.9], "identified": True}},
                "eef_pos": [0.0, 0.0, 1.0],
                "gripper_opening_m": 0.02,
                "evidence_source": "sim_state",
            },
        }
        seg = steps["seg_wooden_cabinet_1"]
        self.assertEqual(seg["query_key"], "wooden_cabinet_1_top_side")
        outcome = ToolOutcome(executed=True, env_step=4, xyz=[0.01, 0.19, 1.12], raw={"found": True})
        verdict = verify_step(
            seg,
            outcome,
            [obs],
            scene_epoch=1,
            min_step=None,
            objects=prepared["objects"],
            support_z={},
            goal_relation="on",
        )
        self.assertEqual(verdict["status"], "success", verdict)
        self.assertEqual(verdict["reason_code"], "OBJECT_LOCALIZED")
        self.assertEqual(verdict["evidence_source"], "mujoco_site")
        self.assertEqual(verdict["metrics"]["query_key"], "wooden_cabinet_1_top_side")

        target = _target_reading(steps["move_step1"], prepared["objects"], obs)
        self.assertEqual(target["status"], "ok")
        self.assertEqual(target["xyz"], site_xyz)
        self.assertEqual(target["query_key"], "wooden_cabinet_1_top_side")
        self.assertEqual(target["evidence_source"], "mujoco_site")


if __name__ == "__main__":
    unittest.main()
