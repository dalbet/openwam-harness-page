"""Replay the stove transport pause and the release gate. No model and no motion."""

from __future__ import annotations

import unittest

from placement import arrival, preplace_gate, resolve_motion_step
from verify import verify_step


SITE = [-0.2633796387096554, 0.21791066565670916, 0.905]
HOLD = [-0.0137, -0.0383, -0.0318]
OBJECTS = {
    "akita_black_bowl_1": {
        "sim_name": "akita_black_bowl_1",
        "role": "movable",
    },
    "flat_stove_1": {
        "sim_name": "flat_stove_1",
        "role": "support",
        "place_site": "flat_stove_1_cook_region",
    },
}


def _obs(name, obj, eef, version, opening=0.0047, named=None, epoch=2):
    return {
        "observation_id": name,
        "scene_epoch": epoch,
        "env_step": version,
        "timestamp": version,
        "cameras": {"agentview": {"id": name}, "wrist": {"id": name}},
        "named_poses": named if named is not None else {"sites": {"flat_stove_1_cook_region": list(SITE)}},
        "sim": {
            "objects": {"akita_black_bowl_1": {"pos": obj, "identified": True}},
            "eef_pos": eef,
            "gripper_opening_m": opening,
            "contact": None,
            "evidence_source": "sim_state",
        },
    }


class TransferReleaseTest(unittest.TestCase):
    def test_archived_travel_is_not_a_generic_confirmation(self):
        bowl = [-0.1164, -0.0254, 0.9559]
        eef = [-0.1026, 0.0128, 0.9877]
        before = _obs("grasp-end", bowl, eef, 154)
        moved_bowl = [-0.2539, 0.202, 1.0244]
        moved_eef = [-0.2402, 0.2403, 1.0562]
        after = _obs("above-stove", moved_bowl, moved_eef, 192)
        step = {
            "step_id": "move_bowl_to_stove",
            "subtask": "bowl_to_stove",
            "skill": "move_to",
            "completion_check": "object_transported",
            "object_ref": "akita_black_bowl_1",
            "target_ref": "flat_stove_1",
            "args": {"motion": "travel"},
            "transfer_phase": "transporting",
        }
        outcome = type("O", (), {"executed": True, "grasp_signal": None, "xyz": None, "error": None})()
        verdict = verify_step(
            step, outcome, [after], scene_epoch=2, min_step=None, objects=OBJECTS,
            support_z={}, goal_relation="on", before=before,
        )
        self.assertEqual(verdict["status"], "success")
        self.assertEqual(verdict["reason_code"], "TRANSPORTED")
        self.assertNotEqual(verdict["reason_code"], "INSUFFICIENT_EVIDENCE")
        self.assertFalse(verdict["metrics"]["placed_required"])
        self.assertEqual(verdict["metrics"]["evidence_sources"]["object"], "sim_state")
        self.assertEqual(verdict["metrics"]["evidence_sources"]["target_region"], "mujoco_site")
        self.assertGreater(verdict["metrics"]["origin_to_eef_m"], 0.03)
        self.assertLess(verdict["metrics"]["relative_delta_m"], 0.03)

    def test_missing_site_names_the_field_and_does_not_ask_a_person(self):
        before = _obs("a", [-0.12, -0.03, 0.96], [-0.10, 0.01, 0.99], 10, named={"sites": {}})
        after = _obs("b", [-0.20, 0.10, 1.00], [-0.18, 0.14, 1.03], 20, named={"sites": {}})
        step = {
            "completion_check": "object_transported",
            "object_ref": "akita_black_bowl_1",
            "target_ref": "flat_stove_1",
            "args": {"motion": "travel"},
        }
        outcome = type("O", (), {"executed": True, "error": None})()
        verdict = verify_step(
            step, outcome, [after], scene_epoch=2, min_step=None, objects=OBJECTS,
            support_z={}, goal_relation="on", before=before,
        )
        self.assertEqual(verdict["status"], "failure")
        self.assertEqual(verdict["reason_code"], "pose_not_provided")
        self.assertIn("target_region_pose", verdict["metrics"]["missing"])
        self.assertIsNone(verdict["metrics"]["actual"]["target_region_pose"])
        self.assertIn("observation_id", verdict["metrics"]["actual"])
        self.assertEqual(verdict["validator_version"], "sequential-verify-2")
        self.assertNotEqual(verdict["evidence_source"], "mujoco_site")

    def test_slip_read_failure_and_stale_observation_stay_distinct(self):
        before = _obs("a", [-0.12, -0.03, 0.96], [-0.1063, 0.0083, 0.9918], 10)
        slipped = _obs("b", [-0.12, -0.03, 0.90], [-0.05, 0.10, 1.05], 20)
        step = {
            "completion_check": "object_transported",
            "object_ref": "akita_black_bowl_1",
            "target_ref": "flat_stove_1",
            "args": {"motion": "travel"},
        }
        outcome = type("O", (), {"executed": True, "error": None})()
        lost = verify_step(step, outcome, [slipped], scene_epoch=2, min_step=None, objects=OBJECTS, support_z={}, goal_relation="on", before=before)
        self.assertEqual(lost["reason_code"], "GRASP_LOST")
        broken = _obs("c", [-0.20, 0.10, 1.02], [-0.1863, 0.1383, 1.0518], 30, named={"error": "worker missing", "sites": {}})
        read = verify_step(step, outcome, [broken], scene_epoch=2, min_step=None, objects=OBJECTS, support_z={}, goal_relation="on", before=before)
        self.assertEqual(read["reason_code"], "state_service_error")
        stale = _obs("d", [-0.20, 0.10, 1.02], [-0.1863, 0.1383, 1.0518], 40, epoch=1)
        old = verify_step(step, outcome, [stale], scene_epoch=2, min_step=None, objects=OBJECTS, support_z={}, goal_relation="on", before=before)
        self.assertEqual(old["status"], "failure")
        self.assertEqual(old["reason_code"], "STALE_OBSERVATION")

    def test_descent_does_not_require_release_and_release_is_not_from_travel_height(self):
        bowl = [-0.25, 0.18, 1.0244]
        eef = [bowl[i] - HOLD[i] for i in range(3)]
        before = _obs("travel-before", [-0.12, -0.03, 0.96], [-0.1063, 0.0083, 0.9918], 10)
        above = _obs("travel", bowl, eef, 20, opening=0.0047)
        step = {
            "completion_check": "object_transported",
            "skill": "move_to",
            "object_ref": "akita_black_bowl_1",
            "target_ref": "flat_stove_1",
            "args": {"motion": "vertical", "gripper": 1},
        }
        outcome = type("O", (), {"executed": True, "error": None})()
        descended_check = verify_step(
            step, outcome, [above], scene_epoch=2, min_step=None, objects=OBJECTS,
            support_z={}, goal_relation="on", before=before,
        )
        self.assertEqual(descended_check["reason_code"], "TRANSPORTED")
        self.assertFalse(descended_check["metrics"]["released_required"])
        bindings = {"hold:akita_black_bowl_1": HOLD}
        plan = {"objects": OBJECTS}
        release = {
            "skill": "set_gripper",
            "completion_check": "object_placed",
            "object_ref": "akita_black_bowl_1",
            "target_ref": "flat_stove_1",
            "args": {"gripper": -1, "object_ref": "akita_black_bowl_1"},
        }
        blocked = preplace_gate(release, above, plan, bindings)
        self.assertFalse(blocked["ok"])
        self.assertFalse(blocked["released_required"])
        self.assertFalse(blocked["placed_required"])
        self.assertIn("释放高度", blocked["unmet"][0] if "释放" in blocked["unmet"][0] or "object_z" in blocked["unmet"][0] else blocked["summary"])
        low_bowl = [SITE[0], SITE[1], SITE[2] + 0.03]
        low_eef = [low_bowl[i] - HOLD[i] for i in range(3)]
        low = _obs("descend", low_bowl, low_eef, 30, opening=0.0047)
        ready = preplace_gate(release, low, plan, bindings)
        self.assertTrue(ready["ok"])
        self.assertNotEqual(low_eef, SITE)

    def test_site_origin_is_not_the_eef_target(self):
        bowl = [SITE[0] + 0.01, SITE[1] - 0.01, 1.02]
        eef = [bowl[i] - HOLD[i] for i in range(3)]
        obs = _obs("now", bowl, eef, 20)
        step = {
            "skill": "move_to",
            "object_ref": "akita_black_bowl_1",
            "target_ref": "flat_stove_1",
            "relation": "on",
            "args": {
                "xyz_from": "segment:stove",
                "z_offset": 0.12,
                "gripper": 1,
                "object_ref": "akita_black_bowl_1",
                "hold_offset": "akita_black_bowl_1",
                "placement": "support_region",
                "motion": "travel",
            },
        }
        resolved = resolve_motion_step(
            step,
            {"segment:stove": [-0.2229, 0.2078, 0.9258]},
            obs,
            {"objects": OBJECTS},
        )
        self.assertTrue(resolved["ok"])
        self.assertEqual(resolved["region_source"], "mujoco_site")
        self.assertNotEqual(resolved["args"]["xyz"], [round(value, 4) for value in SITE])
        self.assertAlmostEqual(resolved["object_origin"][2], round(SITE[2] + 0.12, 4), places=3)
        self.assertAlmostEqual(resolved["args"]["xyz"][2], round(SITE[2] + 0.12 - HOLD[2], 4), places=3)
        missed = arrival(resolved["args"]["xyz"], [-0.24, 0.24, 1.2])
        self.assertFalse(missed["ok"])
        self.assertEqual(missed["code"], "MOTION_NOT_ARRIVED")
        reached = arrival([-0.2402, 0.2403, 1.0562], [-0.2389, 0.2487, 1.0568])
        self.assertTrue(reached["ok"])

    def test_motion_rereads_when_the_current_observation_has_no_site(self):
        from placement import region_pose_ready

        step = {"target_ref": "flat_stove_1", "skill": "move_to"}
        plan = {"objects": OBJECTS}
        self.assertFalse(region_pose_ready(step, {"sim": {}, "named_poses": {"sites": {}}}, plan))
        self.assertTrue(region_pose_ready(step, {"named_poses": {"sites": {"flat_stove_1_cook_region": list(SITE)}}}, plan))


if __name__ == "__main__":
    unittest.main()
