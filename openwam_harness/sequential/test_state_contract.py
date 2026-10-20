"""Saved clocks and constructed snapshots. No env.step and no model."""

from __future__ import annotations

import unittest

from episode_service import classify_step, publish_mode, read_mode, read_task_scope
from episode_service import _worker_reading, _wrap as wrap_facade
from verify import _snapshot_version
from state_contract import (
    distinct_physical_samples,
    post_action_freshness,
    same_physical_state,
    static_region_valid,
    stamp_snapshot,
)


def _snap(count: int, *, epoch=2, generation=1, service="svc-a", query=1, terminated=False, truncated=False):
    return stamp_snapshot(
        {"sim": {"eef_pos": [0, 0, 1], "objects": {}}},
        service_instance_id=service,
        scene_epoch=epoch,
        scene_generation=generation,
        sim_action_count=count,
        query_seq=query,
        terminated=terminated,
        truncated=truncated,
        source="action_return",
    )


class StateContractTest(unittest.TestCase):
    def test_pre_action_count_is_rejected_and_post_action_count_is_kept(self):
        before = _snap(10, query=1)
        early = _snap(9, query=2)
        late = _snap(11, query=3)
        refused = post_action_freshness(before, early, completed_action_count=11)
        self.assertFalse(refused["ok"])
        self.assertEqual(refused["reason_code"], "pre_action_state")
        self.assertEqual(refused["expected"], 11)
        self.assertEqual(refused["received"], 9)
        accepted = post_action_freshness(before, late, completed_action_count=11)
        self.assertTrue(accepted["ok"])
        self.assertEqual(accepted["received"], 11)

    def test_same_readonly_count_is_current_but_not_a_second_stable_sample(self):
        before = _snap(20, query=1)
        again = _snap(20, query=2)
        again["state_source"] = "readonly_query"
        verdict = post_action_freshness(before, again, completed_action_count=20)
        self.assertTrue(verdict["ok"])
        self.assertTrue(verdict["same_physical_state"])
        self.assertTrue(same_physical_state(before, again))
        self.assertNotEqual(before["observation_id"], again["observation_id"])
        self.assertEqual(len(distinct_physical_samples([before, again])), 1)

    def test_saved_hold_versus_tool_record_is_a_missing_count_not_a_blanket_stale_label(self):
        """The recorded failure compared hold-636 with live-38. Those numbers are different clocks."""
        before = {
            "observation_id": "hold-636",
            "scene_epoch": 10,
            "env_step": 636,
            "step_clock": "env_action_count",
            "sim_action_count": 636,
            "service_instance_id": "svc-a",
            "scene_generation": 1,
        }
        after = {
            "observation_id": "live-38-1792484223025563721",
            "scene_epoch": 10,
            "env_step": 38,
            "step_clock": "tool_record",
            "record_index": 38,
        }
        verdict = post_action_freshness(before, after, completed_action_count=636)
        self.assertFalse(verdict["ok"])
        self.assertEqual(verdict["reason_code"], "sim_action_count_missing")
        self.assertEqual(verdict["field"], "after.sim_action_count")
        self.assertEqual(verdict["expected"], 636)
        self.assertEqual(verdict["received"], 38)
        self.assertNotEqual(verdict["reason_code"], "stale_observation")

    def test_cross_reset_and_cross_service_are_rejected(self):
        before = _snap(4)
        other_boot = _snap(8, service="svc-b")
        other_reset = _snap(8, generation=2)
        self.assertEqual(post_action_freshness(before, other_boot, completed_action_count=4)["reason_code"], "cross_service_instance")
        self.assertEqual(post_action_freshness(before, other_reset, completed_action_count=4)["reason_code"], "cross_reset")

    def test_static_site_survives_action_count_and_drawer_region_does_not(self):
        cached = {"motion_class": "rigid", "scene_epoch": 1, "service_instance_id": "svc-a", "owner_xyz": [0.1, 0.2, 0.9], "sim_action_count": 3}
        later = dict(cached, sim_action_count=40)
        self.assertTrue(static_region_valid(cached, later)["ok"])
        drawer = dict(cached, motion_class="articulated")
        self.assertEqual(static_region_valid(drawer, later)["reason_code"], "articulated_region")
        moved = dict(later, owner_xyz=[0.2, 0.2, 0.9])
        self.assertEqual(static_region_valid(cached, moved)["reason_code"], "facility_moved")

    def test_self_stamped_cover_cannot_hide_an_earlier_count(self):
        before = _snap(10, query=1)
        stale = _snap(9, query=2)
        stale["covers_action_count"] = 9
        verdict = _snapshot_version(before, stale)
        self.assertIsNotNone(verdict)
        self.assertEqual(verdict["reason_code"], "pre_action_state")
        current = _snap(12, query=3)
        current["covers_action_count"] = 12
        self.assertIsNone(_snapshot_version(before, current))

    def test_worker_reading_does_not_step(self):
        class Env:
            def __init__(self):
                self.steps = 0

            def step(self, action):
                self.steps += 1

            def _get_observations(self, force_update=False):
                self.force_update = force_update
                return {
                    "akita_black_bowl_1_pos": [0.1, 0.2, 0.8],
                    "robot0_eef_pos": [0.0, 0.0, 1.0],
                    "robot0_gripper_qpos": [0.02, -0.02],
                    "agentview_image": [1, 2, 3],
                }

            @property
            def sim(self):
                class Model:
                    njnt = 1
                    joint_names = ["wooden_cabinet_1_middle_joint"]

                class Data:
                    qpos = [0.0]

                class Sim:
                    model = Model()
                    data = Data()

                return Sim()

        env = Env()
        reading = _worker_reading(env, {})
        self.assertEqual(env.steps, 0)
        self.assertFalse(env.force_update)
        self.assertEqual(reading["sim"]["objects"]["akita_black_bowl_1"]["pos"], [0.1, 0.2, 0.8])
        self.assertEqual(reading["sim"]["eef_pos"], [0.0, 0.0, 1.0])
        self.assertAlmostEqual(reading["sim"]["gripper_opening_m"], 0.04)
        self.assertEqual(reading["joints"]["wooden_cabinet_1_middle_joint"], 0.0)
        self.assertNotIn("agentview_image", reading["sim"])

    def test_official_single_refuses_another_step_after_termination(self):
        publish_mode("free", task_scope="official_single")
        self.assertEqual(read_mode(), "free")
        self.assertEqual(read_task_scope(), "official_single")
        decision = classify_step(True, False, "free", task_scope="official_single")
        self.assertTrue(decision["terminated"])
        self.assertFalse(decision["continuable"])
        self.assertFalse(decision["clear_terminated"])

        class Facade:
            def __init__(self):
                self.steps = 0
                self._rpc = {}
                self._readonly_methods = set()

            def step(self, action):
                self.steps += 1
                return {"obs": action}, 0.0, True, False, {"solved": True}

            def chunk_step(self, actions):
                self.steps += len(actions)
                return {"obs": actions}, 0.0, [True], [False], [{"solved": True}]

            def reset(self):
                self.steps = 0
                return "reset"

            def _register_rpc(self):
                return None

        wrap_facade(Facade)
        env = Facade()
        env._register_rpc()
        _obs, _reward, term, trunc, info = env.step([0.0])
        self.assertTrue(term)
        self.assertFalse(trunc)
        self.assertEqual(info["sim_action_count"], 1)
        self.assertEqual(env.steps, 1)
        with self.assertRaises(RuntimeError) as raised:
            env.step([0.0])
        self.assertIn("env.step refused", str(raised.exception))
        self.assertEqual(env.steps, 1)
        snap = env.state_snapshot()
        self.assertEqual(snap["sim_action_count"], 1)
        self.assertEqual(snap["observation_id_meaning"], "readonly_or_action_query")
        self.assertGreater(snap["query_seq"], 0)
        self.assertNotIn("step", str(type(snap)))


if __name__ == "__main__":
    unittest.main()
