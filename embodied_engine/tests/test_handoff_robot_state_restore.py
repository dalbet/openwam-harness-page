"""``robot_state_restore`` as a step handoff: refusal paths and accounting.

The env service is faked here. What this file pins is the engine's decision
logic: a restore that was not verified may never return ``ready``, a placement
step without a measured release may not move the arm at all, settle actions
are counted separately from policy actions, and the teleport flags are stated
rather than implied.
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

DRAWER_JOINT = "wooden_cabinet_1_middle_level"
HOME_EEF = [0.0, -0.30, 1.10]


def drawer_snapshot(*, qpos: float = -0.21, eef=None, count: int = 40) -> dict:
    return {
        "scene_epoch": 7,
        "observation_id": f"obs-{count}",
        "sim_action_count": count,
        "native_goal": False,
        "joints": {DRAWER_JOINT: qpos},
        "sim": {
            "eef_pos": list(eef or HOME_EEF),
            "gripper_opening_m": 0.075,
            "objects": {"akita_black_bowl_1": {"pos": [0.1, 0.2, 1.0]}},
        },
        "named_poses": {"sites": {"wooden_cabinet_1_middle_region": [-0.1, 0.3, 0.9]}},
    }


def capture_row(*, ok: bool = True, eef=None) -> dict:
    """The stamped row ``env.robot_state_capture`` returns at episode init."""
    state = {
        "robot": {"joints": ["robot0_joint1", "robot0_joint2", "gripper0_right_finger_joint1"]},
        "non_robot": {"digest": "abc"},
    }
    return {
        "ok": ok,
        "reason_code": None if ok else "sim_unavailable",
        "reason": None if ok else "env has no sim",
        "capture_id": "svc:rs1",
        "observation_id": "obs-1",
        "sim_action_count": 5,
        "scene_epoch": 7,
        "robot": {"ok": ok, "state": state, "capture": {"eef_pos": list(eef or HOME_EEF)}},
    }


class RestoreFakeEnv:
    """Snapshot bookkeeping plus a controllable robot_state_restore reply."""

    def __init__(
        self,
        snapshot: dict,
        *,
        reply: dict | None = None,
        settle_effect=None,
        eef_after_restore=None,
        contacts: list[list[dict]] | None = None,
        inspect: str = "ok",
    ):
        self.snapshot = copy.deepcopy(snapshot)
        self.count = int(snapshot["sim_action_count"])
        self.query_seq = int(str(snapshot["observation_id"]).split("-")[-1])
        self.actions: list[list[float]] = []
        self.restore_calls: list[dict] = []
        self.reply = reply if reply is not None else self.ok_reply()
        self.settle_effect = settle_effect
        self.eef_after_restore = eef_after_restore
        self.obs_calls = 0
        # Each inspect call consumes one entry; the last entry repeats.
        self.contact_plan = list(contacts or [])
        self.inspect = inspect
        self.inspect_calls = 0

    def robot_state_inspect(self):
        self.inspect_calls += 1
        if self.inspect == "raise":
            raise RuntimeError("inspect unavailable")
        if self.inspect == "not_ok":
            return {"ok": False, "reason_code": "sim_unavailable"}
        if not self.contact_plan:
            contacts: list[dict] = []
        elif len(self.contact_plan) == 1:
            contacts = self.contact_plan[0]
        else:
            contacts = self.contact_plan.pop(0)
        return {
            "ok": True,
            "robot": {"state": {"contacts": {"gripper_vs_nonrobot": contacts}}},
        }

    @staticmethod
    def ok_reply() -> dict:
        return {
            "ok": True,
            "reason_code": "",
            "reason": "",
            "service_instance_id": "svc",
            "sim_action_count": 40,
            "robot": {
                "block": {
                    "verification": {
                        "robot_qpos_error": 0.0,
                        "robot_qvel_abs_max": 0.0,
                        "non_robot_preserved": True,
                        "non_robot_digest_before": "abc",
                        "non_robot_digest_after": "abc",
                        "scene_time_unchanged": True,
                    },
                    "candidate": {"checked": True, "penetration_max_m": 0.0, "refused": False},
                    "controller_sync": {"robot": "robot0", "parts": {"arm": {"update": True}}},
                    "gripper_contacts": [],
                    "before": {"qpos": [0.1]},
                    "after_write": {"qpos": [0.0]},
                },
                "capture": {
                    "capture_id": "svc:rs1",
                    "qpos": [0.0] * 9,
                    "joints": ["robot0_joint1"],
                    "eef_pos": list(HOME_EEF),
                    "eef_quat": [0.0, 0.0, 0.0, 1.0],
                    "gripper_opening_m": 0.075,
                    "sim_time": 3.5,
                },
            },
        }

    def step(self, action):
        self.actions.append([float(v) for v in action])
        self.count += 1
        self.snapshot["sim_action_count"] = self.count
        eef = list(self.snapshot["sim"]["eef_pos"])
        for index in range(3):
            eef[index] = round(eef[index] + float(action[index]) * 0.05, 6)
        self.snapshot["sim"]["eef_pos"] = eef
        if self.settle_effect is not None:
            self.settle_effect(self.snapshot, len(self.actions))
        return {}, 0.0, False, False, {}

    def state_snapshot(self):
        self.query_seq += 1
        snap = copy.deepcopy(self.snapshot)
        snap["observation_id"] = f"obs-{self.query_seq}"
        self.snapshot["observation_id"] = snap["observation_id"]
        return snap

    def raw_obs(self):
        self.obs_calls += 1
        return {"main_images": None, "states": [0.0] * 8}

    def robot_state_restore(self, request):
        self.restore_calls.append(dict(request))
        if self.reply.get("ok"):
            # The write really does put the arm at the captured pose.
            target = self.eef_after_restore
            if target is None:
                target = (self.reply.get("robot") or {}).get("capture", {}).get("eef_pos")
            if target is not None:
                self.snapshot["sim"]["eef_pos"] = list(target)
        return copy.deepcopy(self.reply)


class FakeInference:
    def __init__(self):
        self.resets: list[str] = []

    def reset(self, reason: str) -> str:
        self.resets.append(reason)
        return "tok-handoff-1"


class RestoreHandoffTest(unittest.TestCase):
    def setUp(self):
        from common import config

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
        self.placement_step = {
            "step_id": "c2",
            "step_index": 2,
            "step_task_id": 8,
            "step_task_key": "put_bowl_on_plate",
            "goal_kind": "native_goal",
            "completion": {"kind": "native_goal", "predicate": "native_libero_goal"},
        }
        self.step_task = config.task_for(0)
        self.loaded_task = config.task_for(8)
        self.cfg = {
            "composite": {
                "handoff": {
                    "mode": "robot_state_restore",
                    "settle_actions": 2,
                    "settle_tol_m": 0.01,
                    "settle_tol_rot_rad": 0.05,
                    "restore_penetration_tol_m": 0.002,
                }
            }
        }

    def _handoff(
        self,
        env,
        *,
        stop_event=None,
        capture=None,
        step=None,
        prior_step=None,
        cfg=None,
    ):
        return composite.Handoff(
            env=env,
            inference=FakeInference(),
            stop_event=stop_event or threading.Event(),
            cfg=cfg or self.cfg,
            episode_id="ep-test",
            step=step or self.step,
            step_task=self.step_task,
            task=self.loaded_task,
            home_eef=list(HOME_EEF),
            robot_state=capture if capture is not None else capture_row(),
            prior_step=prior_step,
        )

    def test_mode_is_read_from_config_and_unknown_modes_fall_back_to_retreat(self):
        self.assertEqual(composite.handoff_mode(self.cfg), "robot_state_restore")
        self.assertEqual(composite.handoff_mode({}), "action_retreat")
        bogus = {"composite": {"handoff": {"mode": "teleport_please"}}}
        opts = composite.handoff_options(bogus)
        self.assertEqual(opts["mode"], "action_retreat")
        self.assertEqual(opts["mode_requested"], "teleport_please")
        self.assertTrue(opts["mode_fallback"])

    def test_ready_restore_reports_teleport_and_counts_settle_separately(self):
        env = RestoreFakeEnv(drawer_snapshot(eef=[0.2, -0.1, 1.0]))
        block, obs, snapshot, token = self._handoff(env).run(env.state_snapshot())
        self.assertEqual(block["status"], "ready", block)
        self.assertTrue(block["robot_state_restore"])
        self.assertTrue(block["robot_teleport"])
        self.assertFalse(block["physical_continuity"])
        self.assertTrue(block["scene_objects_preserved"])
        self.assertEqual(block["counts"]["retreat_actions"], 0)
        self.assertEqual(block["counts"]["settle_actions"], 2)
        self.assertEqual(block["counts"]["policy_actions"], 0)
        self.assertEqual(block["counts"]["restore_writes"], 1)
        self.assertEqual(len(env.restore_calls), 1)
        self.assertEqual(env.restore_calls[0]["capture_id"], "svc:rs1")
        self.assertEqual(len(env.actions), 2)
        # The settle actions are ordinary holds with the fingers open.
        for action in env.actions:
            self.assertEqual(action[:6], [0.0] * 6)
            self.assertEqual(action[6], -1.0)
        self.assertEqual(token, "tok-handoff-1")
        self.assertTrue(block["post_reset"]["fresh_observation"])
        self.assertNotEqual(snapshot["observation_id"], "obs-40")

    def test_missing_capture_never_returns_ready(self):
        env = RestoreFakeEnv(drawer_snapshot())
        block, _obs, _snap, token = self._handoff(env, capture=capture_row(ok=False)).run(
            env.state_snapshot()
        )
        self.assertEqual(block["status"], "not_ready")
        self.assertEqual(block["reason_code"], "robot_state_capture_missing")
        self.assertEqual(env.restore_calls, [])
        self.assertEqual(env.actions, [])
        self.assertEqual(token, "")

    def test_unverified_write_never_returns_ready(self):
        reply = RestoreFakeEnv.ok_reply()
        reply["robot"]["block"]["verification"]["non_robot_preserved"] = False
        env = RestoreFakeEnv(drawer_snapshot(), reply=reply)
        block, _obs, _snap, _token = self._handoff(env).run(env.state_snapshot())
        self.assertEqual(block["status"], "not_ready")
        self.assertEqual(block["reason_code"], "restore_write_unverified")
        self.assertFalse(block["restore"]["verified"])
        self.assertEqual(env.actions, [])  # no settle after a failed write

    def test_refused_write_is_reported_with_the_service_reason(self):
        env = RestoreFakeEnv(
            drawer_snapshot(),
            reply={"ok": False, "reason_code": "gripper_contact", "reason": "fingers on a handle"},
        )
        block, _obs, _snap, _token = self._handoff(env).run(env.state_snapshot())
        self.assertEqual(block["status"], "not_ready")
        self.assertEqual(block["reason_code"], "gripper_contact")
        self.assertEqual(env.actions, [])

    def test_settle_that_moves_the_drawer_blocks_the_handoff(self):
        def close(snapshot, _count):
            snapshot["joints"][DRAWER_JOINT] = -0.05

        env = RestoreFakeEnv(drawer_snapshot(), settle_effect=close)
        block, _obs, _snap, _token = self._handoff(env).run(env.state_snapshot())
        self.assertEqual(block["status"], "not_ready")
        self.assertEqual(block["reason_code"], "settle_moved_the_scene")
        self.assertFalse(block["scene_objects_preserved"])

    def test_settle_that_loses_the_step_1_goal_blocks_the_handoff(self):
        def shut(snapshot, _count):
            # Drawer closed: the step-1 predicate no longer holds, but nothing
            # is "moving" in the diff sense the previous test uses.
            snapshot["joints"][DRAWER_JOINT] = 0.0

        env = RestoreFakeEnv(drawer_snapshot(), settle_effect=shut)
        block, _obs, _snap, _token = self._handoff(env).run(env.state_snapshot())
        self.assertEqual(block["status"], "not_ready")
        self.assertIn(
            block["reason_code"],
            {"settle_moved_the_scene", "restore_lost_step_1_goal"},
        )
        self.assertFalse(block["post_reset"]["goal"]["ok"])

    def test_millimetre_settle_drift_is_recorded_not_called_a_scene_change(self):
        # The drawer keeps swinging a few millimetres after release: real
        # physics, not a leaked write, so it is an evidence row, not a refusal.
        def nudge(snapshot, _count):
            snapshot["joints"][DRAWER_JOINT] = -0.2068  # 3.2 mm from -0.21

        env = RestoreFakeEnv(drawer_snapshot(), settle_effect=nudge)
        block, _obs, _snap, _token = self._handoff(env).run(env.state_snapshot())
        self.assertEqual(block["status"], "ready", block)
        self.assertAlmostEqual(block["scene_check"]["joint_max_delta"], 0.0032, places=4)
        self.assertTrue(block["scene_check"]["preserved"])
        self.assertEqual(block["scene_check"]["joint_tol"], 0.01)

    def test_object_moving_in_settle_blocks_the_handoff(self):
        def shove(snapshot, _count):
            snapshot["sim"]["objects"]["akita_black_bowl_1"]["pos"] = [0.12, 0.2, 1.0]

        env = RestoreFakeEnv(drawer_snapshot(), settle_effect=shove)
        block, _obs, _snap, _token = self._handoff(env).run(env.state_snapshot())
        self.assertEqual(block["status"], "not_ready")
        self.assertEqual(block["reason_code"], "settle_moved_the_scene")
        self.assertAlmostEqual(block["scene_check"]["object_max_delta"], 0.02, places=4)

    def test_arm_drifting_from_the_restored_pose_blocks_the_handoff(self):
        # The write succeeded, but the controller still pulls the arm away.
        def drift(snapshot, _count):
            snapshot["sim"]["eef_pos"] = [0.0, -0.10, 1.10]

        env = RestoreFakeEnv(drawer_snapshot(), settle_effect=drift)
        block, _obs, _snap, _token = self._handoff(env).run(env.state_snapshot())
        self.assertEqual(block["status"], "not_ready")
        self.assertEqual(block["reason_code"], "restore_drift")

    def test_placement_step_without_a_measured_release_is_refused_before_the_write(self):
        snapshot = drawer_snapshot()
        snapshot["native_goal"] = True  # the anchor step is judged by the native goal
        env = RestoreFakeEnv(snapshot)
        handoff = composite.Handoff(
            env=env,
            inference=FakeInference(),
            stop_event=threading.Event(),
            cfg=self.cfg,
            episode_id="ep-test",
            step=self.placement_step,
            step_task=self.loaded_task,
            task=self.loaded_task,
            home_eef=list(HOME_EEF),
            robot_state=capture_row(),
            prior_step={"placement": {"placement_verified": False}},
        )
        block, _obs, _snap, _token = handoff.run(env.state_snapshot())
        self.assertEqual(block["status"], "not_ready")
        self.assertEqual(block["reason_code"], "release_not_verified")
        self.assertEqual(env.restore_calls, [])
        self.assertEqual(env.actions, [])

    def test_stop_set_before_the_write_raises_stopped_and_writes_nothing(self):
        env = RestoreFakeEnv(drawer_snapshot())
        event = threading.Event()
        event.set()
        with self.assertRaises(Stopped):
            self._handoff(env, stop_event=event).run(env.state_snapshot())
        self.assertEqual(env.restore_calls, [])

    def test_stop_during_settle_stops_after_the_written_state(self):
        env = RestoreFakeEnv(drawer_snapshot())
        event = threading.Event()

        def stop_after_first(snapshot, count):
            if count == 1:
                event.set()

        env.settle_effect = stop_after_first
        with self.assertRaises(Stopped):
            self._handoff(env, stop_event=event).run(env.state_snapshot())
        self.assertEqual(len(env.restore_calls), 1)
        self.assertEqual(len(env.actions), 1)

    def test_episode_goal_must_already_hold_before_any_restore(self):
        env = RestoreFakeEnv(drawer_snapshot(qpos=-0.02))
        block, _obs, _snap, token = self._handoff(env).run(env.state_snapshot())
        self.assertEqual(block["status"], "step_1_unverified")
        self.assertEqual(env.restore_calls, [])
        self.assertEqual(token, "")


class DetachTest(unittest.TestCase):
    """The fingers must really be off the handle before the arm is written."""

    def setUp(self):
        from common import config

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
        self.step_task = config.task_for(0)
        self.loaded_task = config.task_for(8)
        self.cfg = {
            "composite": {
                "handoff": {
                    "mode": "robot_state_restore",
                    "settle_actions": 1,
                    "detach_max_actions": 3,
                    "detach_step_m": 0.04,
                }
            }
        }
        self.contact = {
            "geom1": "wooden_cabinet_1_g22",
            "geom2": "gripper0_right_finger2_collision",
            "dist": -0.0001,
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
            home_eef=list(HOME_EEF),
            robot_state=capture_row(),
        )
        return handoff.run(env.state_snapshot())

    def test_contact_is_measured_and_detached_before_the_write(self):
        env = RestoreFakeEnv(
            drawer_snapshot(eef=[0.1, -0.2, 1.05]),
            contacts=[[self.contact], [self.contact], []],
        )
        block, _obs, _snap, _token = self._run(env)
        self.assertTrue(block["detach"]["required"])
        self.assertEqual(block["detach"]["reason"], None)
        self.assertGreaterEqual(block["detach"]["actions"], 1)
        self.assertEqual(block["detach"]["contacts_after"], [])
        self.assertEqual(block["counts"]["detach_actions"], block["detach"]["actions"])
        # The detach runs before the write, not after it.
        self.assertGreaterEqual(env.inspect_calls, 2)
        self.assertEqual(block["status"], "ready", block)
        self.assertEqual(len(env.restore_calls), 1)
        for action in env.actions[: block["detach"]["actions"]]:
            self.assertEqual(action[6], -1.0)  # fingers open while backing off

    def test_detach_that_never_clears_refuses_the_handoff(self):
        env = RestoreFakeEnv(drawer_snapshot(eef=[0.1, -0.2, 1.05]), contacts=[[self.contact]])
        block, _obs, _snap, _token = self._run(env)
        self.assertEqual(block["status"], "not_ready")
        self.assertEqual(block["reason_code"], "gripper_contact_after_detach")
        self.assertEqual(block["detach"]["actions"], 3)  # the whole budget was spent
        self.assertEqual(env.restore_calls, [])  # and nothing was written

    def test_no_contact_needs_no_detach(self):
        env = RestoreFakeEnv(drawer_snapshot(eef=[0.1, -0.2, 1.05]), contacts=[[]])
        block, _obs, _snap, _token = self._run(env)
        self.assertFalse(block["detach"]["required"])
        self.assertEqual(block["detach"]["reason"], "already_clear")
        self.assertEqual(block["detach"]["actions"], 0)
        self.assertEqual(block["status"], "ready", block)

    def test_unreadable_contact_falls_through_to_the_write_check(self):
        env = RestoreFakeEnv(drawer_snapshot(eef=[0.1, -0.2, 1.05]), inspect="raise")
        block, _obs, _snap, _token = self._run(env)
        self.assertEqual(block["detach"]["reason"], "contact_read_unavailable")
        self.assertEqual(block["detach"]["actions"], 0)
        self.assertEqual(len(env.restore_calls), 1)  # the service check still ran


class RetreatReadyTest(unittest.TestCase):
    """The original mode may only report ready when the pose really arrived."""

    def setUp(self):
        from common import config

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
        self.step_task = config.task_for(0)
        self.loaded_task = config.task_for(8)
        self.cfg = {
            "composite": {
                "handoff": {
                    "mode": "action_retreat",
                    "retreat_step_m": 0.04,
                    "retreat_max_actions": 4,
                    "tolerance_m": 0.01,
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
            home_eef=list(HOME_EEF),
        )
        return handoff.run(env.state_snapshot())[0]

    def test_exhausted_budget_is_not_ready_even_though_the_goal_still_holds(self):
        env = RestoreFakeEnv(drawer_snapshot(eef=[0.5, 0.5, 1.5]))
        block = self._run(env)
        self.assertEqual(block["retreat"]["actions"], 4)
        self.assertEqual(block["retreat"]["reason"], "bounded_actions_exhausted")
        self.assertFalse(block["retreat"]["converged"])
        self.assertTrue(block["post_retreat"]["goal"]["ok"])  # goal still holds
        self.assertEqual(block["status"], "not_ready")
        self.assertEqual(block["reason_code"], "retreat_not_converged")
        self.assertFalse(block["robot_state_restore"])
        self.assertFalse(block["robot_teleport"])
        self.assertFalse(block["physical_continuity"])

    def test_converged_retreat_is_ready(self):
        env = RestoreFakeEnv(drawer_snapshot(eef=[0.02, -0.29, 1.09]))
        block = self._run(env)
        self.assertTrue(block["retreat"]["converged"], block["retreat"])
        self.assertEqual(block["status"], "ready", block)
        self.assertEqual(block["counts"]["settle_actions"], 0)


if __name__ == "__main__":
    unittest.main()
