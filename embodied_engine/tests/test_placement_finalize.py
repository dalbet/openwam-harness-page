"""Completion / release / finalizer contracts.

The point of these tests is the difference between three facts that used to be
collapsed into one number:

* the native LIBERO goal holds (a placement predicate),
* the gripper was measured opening (a release command with a measured effect),
* the object stays on the target on fresh post-release samples.

A closed gripper does not prove the object is held, and an open gripper does
not prove the object stayed put.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest

from engine import completion, finalize, grasp_geometry
from engine.openwam_rollout import WholeTaskRunner
from engine.skills import Stopped

OBJECT = "akita_black_bowl_1"
TARGET = "wooden_cabinet_1"
TASK = {
    "task_id": 1,
    "kind": "pick_place",
    "instruction": "put the bowl on the cabinet",
    "objects": [OBJECT, TARGET],
    "targets": [TARGET],
    "semantics": {"operation": "pick_place", "source": OBJECT, "target": TARGET, "relation": "on"},
}
STEP = {
    "skill": "pick_place",
    "object_n": OBJECT,
    "target_n": TARGET,
    "relation": "on",
}


def snapshot(
    *,
    count: int,
    opening: float,
    eef: tuple[float, float, float] = (0.0, 0.0, 1.0),
    obj: tuple[float, float, float] = (-0.20, 0.20, 0.90),
    target: tuple[float, float, float] = (-0.20, 0.20, 0.90),
    goal: bool = True,
    continuable: bool = True,
    obs_id: str | None = None,
    sep: float | None = None,
) -> dict:
    separation = opening if sep is None else sep
    return {
        "scene_epoch": 1,
        "sim_action_count": count,
        "observation_id": obs_id if obs_id is not None else f"obs-{count}",
        "native_goal": goal,
        "continuable": continuable,
        "sim": {
            "objects": {
                OBJECT: {"pos": list(obj)},
                TARGET: {"pos": list(target)},
            },
            "eef_pos": list(eef),
            "gripper_opening_m": opening,
        },
        "named_poses": {
            "bodies": {
                "gripper0_right_leftfinger": [
                    eef[0] - separation / 2.0,
                    eef[1],
                    eef[2],
                ],
                "gripper0_right_rightfinger": [
                    eef[0] + separation / 2.0,
                    eef[1],
                    eef[2],
                ],
            },
            "sites": {},
            "site_geometry": {},
        },
    }


class FakeEnv:
    """Deterministic simulator stub: opening follows the command, arm obeys dz."""

    def __init__(self, *, relation_after_release: bool = True, arm_moves: bool = True, opening_step: float = 0.02):
        self.count = 0
        self.opening = 0.01
        self.eef = [0.0, 0.0, 1.0]
        self.obj = [-0.20, 0.20, 0.90]
        self.target = [-0.20, 0.20, 0.90]
        self.held = True
        self.relation_after_release = relation_after_release
        self.arm_moves = arm_moves
        self.opening_step = opening_step
        self.actions: list[np.ndarray] = []
        self.released = False

    def step(self, action):
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        self.actions.append(action.copy())
        self.count += 1
        grip = float(action[6])
        if grip < 0:
            self.opening = min(0.08, self.opening + self.opening_step)
        else:
            self.opening = max(0.0, self.opening - self.opening_step)
        if self.opening >= 0.05 and self.held:
            self.held = False
            self.released = True
            if not self.relation_after_release:
                self.target = [-0.20, 0.60, 0.90]
        dz = float(action[2]) * 0.05 if self.arm_moves else 0.0
        if dz:
            self.eef[2] += dz
            if self.held:
                self.obj[2] += dz
                self.target[2] += dz
        return {"image": None}, 0.0, False, False, {}

    def state_snapshot(self):
        return snapshot(
            count=self.count,
            opening=round(self.opening, 5),
            eef=tuple(round(v, 5) for v in self.eef),
            obj=tuple(round(v, 5) for v in self.obj),
            target=tuple(round(v, 5) for v in self.target),
            goal=True,
        )


CFG = {
    "release_steps": 8,
    "retreat_m": 0.12,
    "move_action_scale_m": 0.05,
    "move_step_clip_m": 0.025,
    "retreat_step_m": 0.02,
    "stable_samples": 3,
    "release_open_min_m": 0.05,
}
POST = {"detach_probe_m": 0.03, "object_motion_tol_m": 0.005, "finalizer_max_actions": 24}


def run_finalizer(env: FakeEnv, *, snapshot_row: dict | None = None, reset_policy=None, stop_event=None):
    finalizer = finalize.PlacementFinalizer(
        env=env,
        task=TASK,
        relation_step=STEP,
        execution_cfg=CFG,
        post_goal_cfg=POST,
        stop_event=stop_event or threading.Event(),
        episode_id="ep-test",
        reset_policy=reset_policy,
    )
    start = snapshot_row if snapshot_row is not None else env.state_snapshot()
    return finalizer.run(start)


# ---------------------------------------------------------------- policy path


def test_native_goal_with_closed_gripper_is_unknown():
    """Goal true + fingers closed + no measured release is not a placement."""
    block = finalize.policy_release_block(
        required=True,
        native_goal=True,
        post_goal={"release": {"verified": False}},
        snapshot=snapshot(count=80, opening=0.005),
        object_name=OBJECT,
    )
    assert block["native_goal"] is True
    assert block["release_executed"] is False
    assert block["release_verified"] is False
    assert block["placement_verified"] is False
    assert block["execution_status"] == "unknown"
    assert block["reason_code"] == "release_not_verified"


def test_closed_gripper_after_measured_release_is_not_held():
    """A policy may open, release, then close again: the final opening is not the criterion."""
    block = finalize.policy_release_block(
        required=True,
        native_goal=True,
        post_goal={
            "release": {
                "verified": True,
                "samples": 2,
                "opening_m": 0.058,
                "opening_max_m": 0.058,
                "sim_action_count": 91,
                "observation_id": "obs-91",
            }
        },
        snapshot=snapshot(count=96, opening=0.004),
        object_name=OBJECT,
    )
    assert block["release_verified"] is True
    assert block["detach"]["final_gripper_opening_m"] == pytest.approx(0.004)


def test_finger_geometry_reports_raw_numbers():
    frame = grasp_geometry.finger_frame(snapshot(count=3, opening=0.06), OBJECT)
    assert frame["separation_m"] == pytest.approx(0.06)
    assert frame["object_gap_m"] is not None
    assert grasp_geometry.opening_m(snapshot(count=3, opening=0.012)) == pytest.approx(0.012)


# ------------------------------------------------------------- finalizer path


def test_finalizer_releases_and_verifies_placement():
    env = FakeEnv()
    block = run_finalizer(env)
    assert block["attempted"] is True
    assert block["release_executed"] is True
    assert block["release_verified"] is True
    assert block["placement_verified"] is True
    assert block["execution_status"] == "completed"
    assert block["retreat_status"] == "completed"
    assert block["evidence_source"] == "execution_finalizer"
    assert block["policy_owned"] is False
    assert block["extra_actions"] > 0
    # Release came before any retreat action, and every sample follows a step.
    phases = [row["phase"] for row in block["action_rows"]]
    assert phases.index("release") < phases.index("retreat")


def test_finalizer_relation_lost_after_release_is_not_success():
    env = FakeEnv(relation_after_release=False)
    block = run_finalizer(env)
    assert block["release_verified"] is True
    assert block["placement_verified"] is False
    assert block["execution_status"] == "unknown"
    assert block["reason_code"] in {"relation_lost_after_release", "object_follows_gripper"}


def test_finalizer_stale_samples_are_unknown():
    class StaleEnv(FakeEnv):
        def state_snapshot(self):
            row = super().state_snapshot()
            row["observation_id"] = "obs-frozen"
            return row

    block = run_finalizer(StaleEnv())
    assert block["fresh_samples"] is False
    assert block["placement_verified"] is False
    assert block["reason_code"] == "stale_samples"


def test_finalizer_refuses_when_episode_is_not_continuable():
    env = FakeEnv()
    block = run_finalizer(env, snapshot_row=snapshot(count=10, opening=0.01, continuable=False))
    assert block["attempted"] is False
    assert block["reason_code"] == "episode_not_continuable"
    assert env.actions == []


def test_finalizer_requires_relation_candidate():
    env = FakeEnv()
    far = snapshot(count=10, opening=0.01, target=(-0.20, 0.60, 0.90))
    block = run_finalizer(env, snapshot_row=far)
    assert block["attempted"] is False
    assert block["reason_code"] == "not_at_release_candidate"
    assert env.actions == []


def test_finalizer_clears_policy_buffer_before_first_action():
    order: list[str] = []
    env = FakeEnv()

    def reset(reason: str) -> str:
        order.append("reset")
        return "tok-new"

    original_step = env.step

    def step(action):
        order.append("step")
        return original_step(action)

    env.step = step  # type: ignore[method-assign]
    block = run_finalizer(env, reset_policy=reset)
    assert order[0] == "reset", order
    assert order.count("reset") == 1
    assert block["buffer_cleared"] is True
    assert block["token"] == "tok-new"


def test_finalizer_retreat_failure_keeps_placement():
    env = FakeEnv(arm_moves=False)
    block = run_finalizer(env)
    assert block["retreat_status"] == "failed"
    assert block["release_verified"] is True
    # The placement itself is still verified; the retreat stays a separate fact.
    assert block["placement_verified"] is True
    assert block["execution_status"] == "completed"


def test_finalizer_stop_issues_no_further_action():
    env = FakeEnv()
    stop = threading.Event()
    original_step = env.step
    seen: list[int] = []

    def step(action):
        seen.append(len(seen))
        if len(seen) == 3:
            stop.set()
        return original_step(action)

    env.step = step  # type: ignore[method-assign]
    block = run_finalizer(env, stop_event=stop)
    assert block["execution_status"] == "stopped"
    assert block["release_verified"] is False
    assert len(env.actions) == 3


# ------------------------------------------------------------- rollout window


class FakeInference:
    def __init__(self, horizon: int = 10, open_from: int = 5):
        self.horizon = horizon
        self.open_from = open_from
        self.calls = 0

    def infer(self, request, obs, timeout_s=180.0):
        index = self.calls % self.horizon
        self.calls += 1
        grip = 1.0 if index >= self.open_from else -1.0
        action = np.zeros(10, dtype=np.float32)
        action[9] = grip
        return {
            "status": "ok",
            "actions": action.reshape(1, -1),
            "inference_s": 0.0,
            "chunk": {
                "chunk_generation": index == 0,
                "cache_hit": index != 0,
                "chunk_id": self.calls // self.horizon,
                "buffer_index": index,
                "generated_actions": self.horizon if index == 0 else 0,
                "consumed_actions": index + 1,
                "execute_horizon": self.horizon,
            },
        }


class ChunkEnv:
    """Goal latches while the object is still gripped, like the real episodes."""

    def __init__(self, goal_at: int = 5, opening_step: float = 0.03):
        self.count = 0
        self.opening = 0.01
        self.goal_at = goal_at
        self.opening_step = opening_step
        self.eef = [0.0, 0.0, 1.0]

    def raw_obs(self):
        return {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float64)
        self.count += 1
        if float(action[6]) < 0:
            self.opening = min(0.08, self.opening + self.opening_step)
        return {"image": None}, 0.0, False, False, {}

    def state_snapshot(self):
        return snapshot(
            count=self.count,
            opening=round(self.opening, 5),
            goal=self.count >= self.goal_at,
            obs_id=f"obs-{self.count}",
        )


def make_runner(env, inference, window_actions: int) -> WholeTaskRunner:
    runner = WholeTaskRunner(
        env=env,
        inference=inference,
        token="tok",
        episode_id="ep-window",
        task=TASK,
        step={"skill": "whole_task", "prompt": "put the bowl on the cabinet"},
        obs={},
        snapshot=env.state_snapshot(),
        baseline=env.state_snapshot(),
        stop_event=threading.Event(),
        max_steps=40,
        timeout_s=30,
        post_goal={"window_actions": window_actions, "release_open_m": 0.05, "release_confirm_samples": 2},
    )
    runner._observe_model_input = lambda: {}  # images are irrelevant to this contract
    return runner


def test_window_lets_policy_finish_its_own_release():
    env = ChunkEnv()
    inference = FakeInference()
    result = make_runner(env, inference, window_actions=31).run()
    assert result["verification"]["ok"] is True
    assert result["stop_reason"] == "policy_release"
    post = result["post_goal"]
    assert post["release"]["verified"] is True
    assert post["release"]["opening_m"] >= 0.05
    assert post["actions_used"] >= 2
    # The window must not ask the model for a new plan.
    assert result["chunk_stats"]["chunk_generations"] == 1
    assert inference.calls == env.count


def test_window_off_preserves_legacy_stop():
    env = ChunkEnv()
    inference = FakeInference()
    result = make_runner(env, inference, window_actions=0).run()
    assert result["stop_reason"] == "native_goal"
    assert result["post_goal"]["release"]["verified"] is False
    assert result["sim_actions"] == 5


def test_window_is_bounded_by_the_buffered_chunk():
    env = ChunkEnv(goal_at=5)
    inference = FakeInference(horizon=10, open_from=99)  # never opens
    result = make_runner(env, inference, window_actions=31).run()
    assert result["stop_reason"] == "chunk_exhausted"
    assert result["chunk_stats"]["chunk_generations"] == 1
    assert result["sim_actions"] == 10
    assert result["post_goal"]["release"]["verified"] is False


def test_window_stops_when_the_relation_breaks():
    env = ChunkEnv(goal_at=3)
    base = env.state_snapshot

    def drifting():
        row = base()
        if env.count > env.goal_at:
            row["sim"]["objects"][OBJECT]["pos"] = [-0.20 + 0.4 * (env.count - env.goal_at), 0.20, 0.90]
        return row

    env.state_snapshot = drifting  # type: ignore[assignment]
    result = make_runner(env, FakeInference(open_from=99), window_actions=31).run()
    assert result["stop_reason"] == "relation_lost_in_window"


def test_completion_verification_still_requires_release_for_pick_place():
    samples = [snapshot(count=90 + i, opening=0.07) for i in range(3)]
    verify = completion.verification(samples, STEP, TASK, samples[0], release={"commanded": False})
    assert verify["ok"] is False
    assert verify["reason_code"] == "release_evidence_failed"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
