"""Step-2 tracing of an earlier step's goal joint (diagnostics only).

The composite executes both steps through ``WholeTaskRunner``; the trace must
exist there, not only in the skill runner, or a report has no action index for
the moment a previously opened drawer starts to close again. These tests pin
that the trace is read-only, that the baseline is the handoff's own state, and
that the contact read names the geom pair at the first movement.
"""

from __future__ import annotations

import sys
import threading
import unittest
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from engine import engine_service  # noqa: E402
from engine.openwam_rollout import WholeTaskRunner  # noqa: E402

DRAWER_JOINT = "wooden_cabinet_1_middle_level"


class FakeEnv:
    def __init__(self, contacts=None):
        self.inspect_calls = 0
        self._contacts = contacts

    def robot_state_inspect(self):
        self.inspect_calls += 1
        if self._contacts is None:
            return {"ok": False, "reason_code": "unavailable"}
        return {"ok": True, "robot": {"state": {"contacts": self._contacts}}}


def runner(watch_joints, env=None) -> WholeTaskRunner:
    return WholeTaskRunner(
        env=env or FakeEnv(),
        inference=None,
        token="tok",
        episode_id="ep-1",
        task={"task_id": 8},
        step={"step_id": "c2", "prompt": "put the bowl on the plate"},
        obs={},
        snapshot={"scene_epoch": 2},
        baseline={"scene_epoch": 2},
        stop_event=threading.Event(),
        watch_joints=watch_joints,
    )


def snap(qpos: float, count: int) -> dict:
    return {
        "scene_epoch": 2,
        "sim_action_count": count,
        "observation_id": f"obs-{count}",
        "native_goal": False,
        "joints": {DRAWER_JOINT: qpos},
        "sim": {"eef_pos": [0.1, 0.2, 1.0], "gripper_opening_m": 0.075},
    }


class WatchTraceTest(unittest.TestCase):
    def test_without_watch_joints_nothing_is_read(self):
        env = FakeEnv()
        row = runner([], env=env)
        self.assertIsNone(row._watch(snap(-0.15, 1)))
        self.assertEqual(env.inspect_calls, 0)
        self.assertEqual(row.watch_events, [])

    def test_baseline_is_the_handoff_state_not_the_episode_start(self):
        row = runner([DRAWER_JOINT])
        row._watch(snap(-0.21, 500))
        self.assertEqual(row.watch_baseline, {DRAWER_JOINT: -0.21})
        self.assertEqual(row.watch_events, [])

    def test_first_movement_records_action_index_and_delta(self):
        env = FakeEnv({"count": 2, "gripper_vs_nonrobot": [["gripper0_finger1", "handle"]]})
        row = runner([DRAWER_JOINT], env=env)
        row._watch(snap(-0.21, 100))
        row.sim_actions = 7
        row._watch(snap(-0.2100009, 107))  # below the 1e-4 threshold
        self.assertEqual(row.watch_events, [])
        row.sim_actions = 8
        row._watch(snap(-0.206, 108))
        self.assertEqual(len(row.watch_events), 1)
        event = row.watch_events[0]
        self.assertEqual(event["joint"], DRAWER_JOINT)
        self.assertEqual(event["sim_action_count"], 108)
        self.assertEqual(event["skill_sim_actions"], 8)
        self.assertAlmostEqual(event["delta"], 0.004, places=6)
        self.assertEqual(event["contacts"]["count"], 2)
        self.assertEqual(env.inspect_calls, 1)

    def test_only_the_first_movement_is_recorded(self):
        row = runner([DRAWER_JOINT])
        row._watch(snap(-0.21, 10))
        row._watch(snap(-0.20, 11))
        row._watch(snap(-0.19, 12))
        self.assertEqual(len(row.watch_events), 1)

    def test_unavailable_contact_read_does_not_break_the_step(self):
        row = runner([DRAWER_JOINT], env=FakeEnv(None))
        row._watch(snap(-0.21, 10))
        row._watch(snap(-0.20, 11))
        self.assertEqual(row.watch_events[0]["contacts"], {"error": "unavailable"})

    def test_mismatched_joint_names_are_ignored(self):
        row = runner(["not_a_joint"])
        self.assertIsNone(row._watch(snap(-0.21, 10)))
        self.assertEqual(row.watch_baseline, {})


class WatchedJointsResolutionTest(unittest.TestCase):
    def setUp(self):
        self.steps = [
            {
                "step_id": "c1",
                "completion": {
                    "kind": "drawer_open",
                    "skill": "open_drawer",
                    "target_n": "wooden_cabinet_1_middle_region",
                },
            },
            {"step_id": "c2", "completion": {"kind": "native_goal"}},
        ]
        self.snapshot = {
            "joints": {
                DRAWER_JOINT: -0.15,
                "wooden_cabinet_1_top_level": 0.0,
                "akita_black_bowl_1_joint0": 0.1,
            }
        }

    def test_first_step_watches_nothing(self):
        self.assertEqual(engine_service._watched_joints(self.steps, 0, self.snapshot, {}), [])

    def test_second_step_watches_the_drawer_joint_only(self):
        names = engine_service._watched_joints(self.steps, 1, self.snapshot, {})
        self.assertEqual(names, [DRAWER_JOINT])

    def test_fallback_by_level_suffix_when_owner_prefix_differs(self):
        snapshot = {"joints": {"different_cabinet_middle_level": -0.2}}
        names = engine_service._watched_joints(self.steps, 1, snapshot, {})
        self.assertEqual(names, ["different_cabinet_middle_level"])

    def test_no_joint_in_snapshot_yields_no_watch(self):
        names = engine_service._watched_joints(self.steps, 1, {"joints": {}}, {})
        self.assertEqual(names, [])


if __name__ == "__main__":
    unittest.main()
