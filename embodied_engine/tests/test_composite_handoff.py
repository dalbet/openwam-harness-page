"""The step handoff: bounded, evidenced, and never a disguised reset.

A fake env is enough here: the handoff's job is to move the arm a bounded
amount, refuse to continue when the step-1 goal is not true, clear the policy
buffer, and report exactly what happened. These tests pin those four
behaviours without a simulator.
"""

from __future__ import annotations

import copy
import sys
import threading
import unittest
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from engine import composite  # noqa: E402
from engine.skills import Stopped  # noqa: E402


def drawer_snapshot(*, qpos: float = -0.21, eef=(0.0, -0.30, 1.10), count: int = 40) -> dict:
    return {
        "scene_epoch": 7,
        "observation_id": f"obs-{count}",
        "sim_action_count": count,
        "native_goal": False,
        "joints": {"wooden_cabinet_1_middle_level": qpos},
        "sim": {
            "eef_pos": list(eef),
            "gripper_opening_m": 0.075,
            "objects": {"akita_black_bowl_1": {"pos": [0.1, 0.2, 1.0]}},
        },
        "named_poses": {"sites": {"wooden_cabinet_1_middle_region": [-0.1, 0.3, 0.9]}},
    }


class FakeEnv:
    def __init__(self, snapshot: dict, *, epoch_change: bool = False, close_drawer: bool = False):
        self.snapshot = copy.deepcopy(snapshot)
        self.count = int(snapshot["sim_action_count"])
        self.query_seq = int(str(snapshot["observation_id"]).split("-")[-1])
        self.actions: list[list[float]] = []
        self.epoch_change = epoch_change
        self.close_drawer = close_drawer
        self.obs_calls = 0

    def step(self, action):
        self.actions.append([float(v) for v in action])
        self.count += 1
        sim = self.snapshot["sim"]
        eef = list(sim["eef_pos"])
        for index in range(3):
            eef[index] = round(eef[index] + float(action[index]) * 0.05, 6)
        sim["eef_pos"] = eef
        self.snapshot["sim_action_count"] = self.count
        if self.close_drawer:
            self.snapshot["joints"]["wooden_cabinet_1_middle_level"] = 0.0
        if self.epoch_change:
            self.snapshot["scene_epoch"] = int(self.snapshot["scene_epoch"]) + 1
        return {}, 0.0, False, False, {}

    def state_snapshot(self):
        # The real env stamps every read with a new query sequence number, so a
        # fresh observation is a new read, not a new action.
        self.query_seq += 1
        snap = copy.deepcopy(self.snapshot)
        snap["observation_id"] = f"obs-{self.query_seq}"
        self.snapshot["observation_id"] = snap["observation_id"]
        return snap

    def raw_obs(self):
        self.obs_calls += 1
        return {"main_images": None, "states": [0.0] * 8}


class FakeInference:
    def __init__(self):
        self.resets: list[str] = []

    def reset(self, reason: str) -> str:
        self.resets.append(reason)
        return "tok-handoff-1"


class HandoffTest(unittest.TestCase):
    def setUp(self):
        self.step = {
            "step_id": "c1",
            "step_index": 1,
            "step_task_id": 0,
            "step_task_key": "open_middle_drawer",
            "goal_kind": "drawer_open",
            "completion": {
                "kind": "drawer_open",
                "predicate": "drawer_open",
                "skill": "open_drawer",
                "target_n": "wooden_cabinet_1_middle_region",
            },
        }
        from common import config

        self.step_task = config.task_for(0)
        self.loaded_task = config.task_for(8)
        self.cfg = {
            "composite": {
                "handoff": {
                    "retreat_step_m": 0.04,
                    "retreat_max_actions": 12,
                    "tolerance_m": 0.02,
                }
            }
        }

    def _handoff(self, env, inference, stop_event=None, home=(0.0, -0.30, 1.10)):
        return composite.Handoff(
            env=env,
            inference=inference,
            stop_event=stop_event or threading.Event(),
            cfg=self.cfg,
            episode_id="ep-test",
            step=self.step,
            step_task=self.step_task,
            task=self.loaded_task,
            home_eef=list(home),
        )

    def test_refuses_when_step_1_goal_is_not_true(self):
        env = FakeEnv(drawer_snapshot(qpos=-0.05))
        inference = FakeInference()
        block, _obs, snapshot, token = self._handoff(env, inference).run(env.state_snapshot())
        self.assertEqual(block["status"], "step_1_unverified")
        self.assertEqual(env.actions, [])
        self.assertEqual(inference.resets, [])
        self.assertEqual(token, "")
        self.assertEqual(snapshot["sim_action_count"], 40)

    def test_ready_handoff_moves_bounded_resets_buffer_and_keeps_epoch(self):
        env = FakeEnv(drawer_snapshot(eef=(0.0, -0.20, 1.04)))
        inference = FakeInference()
        block, obs, snapshot, token = self._handoff(env, inference).run(env.state_snapshot())
        self.assertEqual(block["status"], "ready", block)
        self.assertTrue(block["scene_epoch_same"])
        self.assertEqual(block["pre"]["scene_epoch"], block["post_reset"]["scene_epoch"])
        self.assertEqual(token, "tok-handoff-1")
        self.assertEqual(len(inference.resets), 1)
        self.assertIn("handoff:ep-test:c1", inference.resets[0])
        self.assertTrue(block["post_reset"]["fresh_observation"])
        self.assertGreater(block["sim_actions"], 0)
        self.assertEqual(block["sim_actions"], len(env.actions))
        self.assertFalse(block["requires_release"])
        self.assertEqual(block["release"]["required"], False)
        self.assertIsNotNone(obs)

    def test_retreat_is_bounded_and_needs_no_actions_when_already_home(self):
        env = FakeEnv(drawer_snapshot(eef=(0.0, -0.30, 1.10)))
        block, _obs, _snapshot, _token = self._handoff(env, FakeInference()).run(env.state_snapshot())
        self.assertEqual(block["retreat"]["actions"], 0)
        self.assertEqual(block["retreat"]["reason"], "already_home")
        self.assertEqual(block["status"], "ready")

    def test_retreat_is_capped_by_config(self):
        env = FakeEnv(drawer_snapshot(eef=(3.0, 3.0, 3.0)))
        block, _obs, _snapshot, _token = self._handoff(env, FakeInference(), home=(0.0, 0.0, 0.0)).run(
            env.state_snapshot()
        )
        self.assertLessEqual(block["retreat"]["actions"], 12)
        self.assertEqual(block["retreat"]["reason"], "bounded_actions_exhausted")
        # The drawer is still open, so the handoff may still start step 2.
        self.assertEqual(block["status"], "ready")

    def test_scene_epoch_change_blocks_the_handoff(self):
        env = FakeEnv(drawer_snapshot(eef=(3.0, 3.0, 3.0)), epoch_change=True)
        block, _obs, _snapshot, _token = self._handoff(env, FakeInference(), home=(0.0, 0.0, 0.0)).run(
            env.state_snapshot()
        )
        self.assertFalse(block["scene_epoch_same"])
        self.assertEqual(block["status"], "not_ready")
        self.assertEqual(block["reason_code"], "handoff_not_ready")

    def test_retreat_that_closes_the_drawer_blocks_the_handoff(self):
        env = FakeEnv(drawer_snapshot(), close_drawer=True)
        block, _obs, _snapshot, _token = self._handoff(env, FakeInference(), home=(0.0, 0.0, 0.0)).run(
            env.state_snapshot()
        )
        self.assertEqual(block["status"], "not_ready")
        self.assertEqual(block["reason_code"], "retreat_lost_step_1_goal")
        self.assertFalse(block["post_retreat"]["retreat_preserved_goal"])

    def test_stop_before_the_first_handoff_action_raises_stopped(self):
        env = FakeEnv(drawer_snapshot(eef=(3.0, 3.0, 3.0)))
        event = threading.Event()
        event.set()
        with self.assertRaises(Stopped):
            self._handoff(env, FakeInference(), stop_event=event, home=(0.0, 0.0, 0.0)).run(
                env.state_snapshot()
            )
        self.assertEqual(env.actions, [])

    def test_placement_step_task_requires_release(self):
        self.assertTrue(composite.release_required(self.loaded_task))
        self.assertFalse(composite.release_required(self.step_task))

    def test_state_check_uses_native_goal_only_for_the_anchor(self):
        snapshot = drawer_snapshot()
        snapshot["native_goal"] = True
        early = composite.state_check(snapshot, self.step, self.loaded_task)
        self.assertTrue(early["ok"])  # drawer really is open
        self.assertEqual(early["predicate"], "drawer_open")
        anchor_step = {"completion": {"kind": "native_goal"}}
        anchor = composite.state_check(snapshot, anchor_step, self.loaded_task)
        self.assertEqual(anchor["predicate"], "native_libero_goal")
        self.assertTrue(anchor["ok"])
        snapshot["native_goal"] = False
        self.assertFalse(composite.state_check(snapshot, anchor_step, self.loaded_task)["ok"])
        # The native flag never decides the early step.
        self.assertTrue(composite.state_check(snapshot, self.step, self.loaded_task)["ok"])


if __name__ == "__main__":
    unittest.main()


def _quat_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return [
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ]


def _quat_from_axis_angle(axis, angle):
    import math

    norm = math.sqrt(sum(v * v for v in axis)) or 1.0
    half = angle / 2.0
    scale = math.sin(half) / norm
    return [axis[0] * scale, axis[1] * scale, axis[2] * scale, math.cos(half)]


class QuatFakeEnv(FakeEnv):
    """FakeEnv that also integrates a scaled axis-angle rotation command.

    The real LIBERO OSC controller turns a unit rotation command into a
    fraction of a radian, so 1.0 is not a full radian. The fake uses the same
    shape (command * 0.05 rad) to prove the handoff closes the loop on the
    measured orientation instead of trusting one open-loop command.
    """

    ROT_SCALE = 0.05

    def __init__(self, snapshot: dict, *, quat):
        super().__init__(snapshot)
        self.snapshot["sim"]["eef_quat"] = list(quat)

    def step(self, action):
        _, reward, term, trunc, info = super().step(action)
        delta = _quat_from_axis_angle(
            [float(action[3]), float(action[4]), float(action[5])],
            float((action[3] ** 2 + action[4] ** 2 + action[5] ** 2) ** 0.5) * self.ROT_SCALE,
        )
        self.snapshot["sim"]["eef_quat"] = _quat_mul(delta, self.snapshot["sim"]["eef_quat"])
        return {}, reward, term, trunc, info


class OrientationRetreatTest(unittest.TestCase):
    def setUp(self):
        self.step = {
            "step_id": "c1",
            "step_index": 1,
            "step_task_id": 0,
            "step_task_key": "open_middle_drawer",
            "goal_kind": "drawer_open",
            "completion": {
                "kind": "drawer_open",
                "predicate": "drawer_open",
                "skill": "open_drawer",
                "target_n": "wooden_cabinet_1_middle_region",
            },
        }
        from common import config

        self.step_task = config.task_for(0)
        self.loaded_task = config.task_for(8)
        self.cfg = {
            "composite": {
                "handoff": {
                    "retreat_step_m": 0.04,
                    "retreat_max_actions": 60,
                    "tolerance_m": 0.01,
                    "retreat_rot_step_rad": 1.0,
                    "rot_tolerance_rad": 0.05,
                }
            }
        }

    def _run(self, env):
        handoff = composite.Handoff(
            env=env,
            inference=FakeInference(),
            stop_event=threading.Event(),
            cfg=self.cfg,
            episode_id="ep-test",
            step=self.step,
            step_task=self.step_task,
            task=self.loaded_task,
            home_eef=[0.0, -0.30, 1.10],
            home_quat=[0.0, 0.0, 0.0, 1.0],
        )
        return handoff.run(env.state_snapshot())[0]

    def test_orientation_is_restored_even_when_position_is_home(self):
        # 90 degrees about x, position already at home: only rotation is wrong.
        start = _quat_from_axis_angle([1, 0, 0], 1.5707963)
        env = QuatFakeEnv(drawer_snapshot(eef=(0.0, -0.30, 1.10)), quat=start)
        block = self._run(env)
        self.assertEqual(block["status"], "ready", block)
        retreat = block["retreat"]
        self.assertTrue(retreat["applied"])
        self.assertTrue(retreat.get("rotation_applied"))
        self.assertGreater(retreat["start_angle_deg"], 80.0)
        self.assertLess(retreat["end_angle_deg"], 10.0)
        self.assertTrue(any(any(abs(v) > 0 for v in action[3:6]) for action in env.actions))

    def test_rotation_commands_are_bounded_per_action(self):
        start = _quat_from_axis_angle([0, 0, 1], 3.0)
        env = QuatFakeEnv(drawer_snapshot(eef=(0.0, -0.30, 1.10)), quat=start)
        block = self._run(env)
        for action in env.actions:
            magnitude = sum(float(v) ** 2 for v in action[3:6]) ** 0.5
            self.assertLessEqual(magnitude, 1.0001)
        self.assertLessEqual(block["retreat"]["actions"], 60)

    def test_missing_orientation_keeps_the_position_only_retreat(self):
        env = FakeEnv(drawer_snapshot(eef=(0.0, -0.20, 1.04)))
        block = self._run(env)
        self.assertEqual(block["status"], "ready")
        self.assertNotIn("start_angle_deg", block["retreat"])
        self.assertFalse(block["retreat"].get("rotation_applied", False))
