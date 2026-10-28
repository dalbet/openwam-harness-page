"""Execution side of a same-scene two-step composite.

The catalog contract (which ordered pairs exist, what each step's goal is,
plan validation) lives in :mod:`common.composite_plan` so the Agent layer can
use it without importing the engine. This module adds only what needs the
simulator:

* :class:`Handoff` — the bounded transition between step 1 and step 2;
* :func:`state_check` — one snapshot evaluated against a step's own goal;
* :func:`step_summary` — the per-step evidence row.

Nothing here decides success. It only produces evidence.
"""

from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np

from common.composite_plan import (  # noqa: F401 - re-exported for the engine
    DEFAULT_PAIRS,
    SCHEMA,
    build_plan,
    completion_for,
    options,
    release_required,
    step_budget,
    step_goal,
    support,
    validate_plan,
)
from engine import completion

DEFAULT_HANDOFF = {
    "retreat_m": 0.18,
    "retreat_step_m": 0.04,
    "retreat_max_actions": 30,
    "tolerance_m": 0.02,
    "retreat_rot_step_rad": 1.0,
    "rot_tolerance_rad": 0.139626,
    "min_clear_m": 0.02,
    "reset_policy_buffer": True,
    "gripper_open_command": -1.0,
}
RETREAT_SCALE_M = 0.05
"""Same OSC convention as ``engine.skills``: 1.0 of the action means 5 cm."""


def _quat_from_obs(obs: dict[str, Any] | None) -> list[float] | None:
    value = (obs or {}).get("robot0_eef_quat")
    if value is None:
        return None
    try:
        quat = [float(v) for v in np.asarray(value).reshape(-1)[:4]]
    except (TypeError, ValueError):
        return None
    norm = float(np.linalg.norm(quat))
    if norm < 1e-9:
        return None
    return [value / norm for value in quat]


def _quat_of_snapshot(snapshot: dict[str, Any] | None) -> list[float] | None:
    value = ((snapshot or {}).get("sim") or {}).get("eef_quat")
    if value is None:
        return None
    try:
        quat = [float(v) for v in np.asarray(value).reshape(-1)[:4]]
    except (TypeError, ValueError):
        return None
    if len(quat) != 4:
        return None
    norm = float(np.linalg.norm(quat))
    if norm < 1e-9:
        return None
    return [value / norm for value in quat]


def _quat_conj(quat: list[float]) -> list[float]:
    x, y, z, w = quat
    return [-x, -y, -z, w]


def _quat_mul(a: list[float], b: list[float]) -> list[float]:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return [
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ]


def _axis_angle(q_from: list[float], q_to: list[float]) -> np.ndarray:
    """World-frame rotation vector that takes ``q_from`` to ``q_to`` (xyzw)."""
    delta = _quat_mul(q_to, _quat_conj(q_from))
    if delta[3] < 0:
        delta = [-value for value in delta]
    vec = np.asarray(delta[:3], dtype=np.float64)
    sin_half = float(np.linalg.norm(vec))
    if sin_half < 1e-9:
        return np.zeros(3)
    angle = 2.0 * float(np.arctan2(sin_half, max(-1.0, min(1.0, delta[3]))))
    return vec / sin_half * angle


def handoff_options(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    block = dict((cfg or {}).get("composite") or {})
    return {**DEFAULT_HANDOFF, **{k: v for k, v in (block.get("handoff") or {}).items()}}


def state_check(snapshot: dict[str, Any], step: dict[str, Any], task: dict[str, Any]) -> dict[str, Any]:
    """One snapshot evaluated against this step's own goal."""
    goal = step_goal(step)
    if goal is None:
        return {
            "predicate": "native_libero_goal",
            "ok": bool(snapshot.get("native_goal")),
            "native_goal": bool(snapshot.get("native_goal")),
            "observation_id": snapshot.get("observation_id"),
            "sim_action_count": snapshot.get("sim_action_count"),
        }
    return completion.goal_state(snapshot, goal, task)


class Handoff:
    """The bounded transition between step 1 and step 2 of one episode."""

    def __init__(
        self,
        *,
        env,
        inference,
        stop_event,
        cfg: dict[str, Any] | None,
        episode_id: str,
        step: dict[str, Any],
        step_task: dict[str, Any],
        task: dict[str, Any],
        home_eef: list[float] | None,
        home_quat: list[float] | None = None,
        journal: Callable[[str, dict[str, Any]], None] | None = None,
        frame_sink: Callable[[dict[str, Any], dict[str, Any]], None] | None = None,
    ):
        self.env = env
        self.inference = inference
        self.stop_event = stop_event
        self.opts = handoff_options(cfg)
        self.episode_id = episode_id
        self.step = step
        self.step_task = step_task
        self.task = task
        self.home_eef = home_eef
        self.home_quat = [float(v) for v in home_quat] if home_quat else None
        self.journal = journal or (lambda _stage, _row: None)
        self.frame_sink = frame_sink
        self._last_snapshot: dict[str, Any] | None = None
        self._last_obs: dict[str, Any] | None = None

    def _check_stop(self) -> None:
        from engine.skills import Stopped

        if self.stop_event.is_set():
            raise Stopped("stop requested during step handoff")

    def _eef(self, snapshot: dict[str, Any]) -> list[float] | None:
        value = (snapshot.get("sim") or {}).get("eef_pos")
        if isinstance(value, list) and len(value) >= 3:
            return [float(v) for v in value[:3]]
        return None

    def run(self, snapshot: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], str]:
        started = time.perf_counter()
        block: dict[str, Any] = {
            "schema": f"{SCHEMA}.handoff",
            "step_id": self.step.get("step_id"),
            "step_task_key": self.step_task.get("key"),
            "requires_release": release_required(self.step_task),
            "status": "started",
        }
        pre = snapshot
        pre_goal = state_check(pre, self.step, self.task)
        # The snapshot carries no wrist orientation, so read one raw
        # observation here: without it the retreat cannot restore the pose.
        if self._last_obs is None and _quat_of_snapshot(pre) is None:
            try:
                self._last_obs = self.env.raw_obs()
            except Exception:  # noqa: BLE001 - orientation stays best-effort
                self._last_obs = None
        block["pre"] = {
            "observation_id": pre.get("observation_id"),
            "sim_action_count": pre.get("sim_action_count"),
            "scene_epoch": pre.get("scene_epoch"),
            "goal": pre_goal,
            "native_goal": bool(pre.get("native_goal")),
            "eef_pos": self._eef(pre),
            "gripper_opening_m": (pre.get("sim") or {}).get("gripper_opening_m"),
            "eef_quat": _quat_of_snapshot(pre) or _quat_from_obs(self._last_obs),
        }
        if not pre_goal.get("ok"):
            block.update(
                {
                    "status": "step_1_unverified",
                    "reason_code": "step_1_goal_not_verified_at_handoff",
                    "reason": "the step-1 predicate does not hold before the handoff; refusing to start step 2",
                    "elapsed_s": round(time.perf_counter() - started, 4),
                }
            )
            return block, self._last_obs or {}, pre, ""

        # ---- release: only for a step whose own kind requires it ----------
        if block["requires_release"]:
            block["release"] = {
                "required": True,
                "executed": False,
                "reason": "release for a held object is owned by the step's finalizer path",
            }
        else:
            block["release"] = {
                "required": False,
                "executed": False,
                "reason": f"{self.step_task.get('kind')} step has no held object to release",
            }

        # ---- bounded retreat towards the episode's initial EEF pose ------
        # Both position and orientation are restored: the policy continues
        # from a wrist pose close to the one the episode started in, so step 2
        # is not asked to grasp from an arbitrary orientation left by step 1.
        retreat = {
            "applied": False,
            "actions": 0,
            "start_eef": self._eef(pre),
            "target_eef": self.home_eef,
            "target_quat": self.home_quat,
            "reason": None,
        }
        start_quat = _quat_of_snapshot(pre) or _quat_from_obs(self._last_obs)
        if start_quat is not None and self.home_quat is not None:
            retreat["start_angle_deg"] = round(
                float(np.degrees(np.linalg.norm(_axis_angle(start_quat, self.home_quat)))), 2
            )
        current = retreat["start_eef"]
        if self.home_eef is None or current is None:
            retreat["reason"] = "home_unknown"
        else:
            distance = max(abs(float(self.home_eef[i]) - float(current[i])) for i in range(3))
            retreat["start_distance_m"] = round(distance, 4)
            pos_tol = float(self.opts["tolerance_m"])
            rot_tol = float(self.opts["rot_tolerance_rad"])
            quat = start_quat
            if distance < pos_tol and (quat is None or self.home_quat is None
                                       or np.linalg.norm(_axis_angle(quat, self.home_quat)) < rot_tol):
                retreat["reason"] = "already_home"
            else:
                limit = int(self.opts["retreat_max_actions"])
                step_m = float(self.opts["retreat_step_m"])
                rot_step = float(self.opts["retreat_rot_step_rad"])
                for _ in range(limit):
                    self._check_stop()
                    eef = self._eef(self._last_snapshot or pre)
                    if eef is None:
                        retreat["reason"] = "eef_unknown"
                        break
                    delta = [float(self.home_eef[i]) - eef[i] for i in range(3)]
                    pos_done = max(abs(value) for value in delta) < pos_tol
                    rot_vector = np.zeros(3)
                    if quat is not None and self.home_quat is not None:
                        rot_vector = _axis_angle(quat, self.home_quat)
                    rot_done = self.home_quat is None or quat is None or float(
                        np.linalg.norm(rot_vector)
                    ) < rot_tol
                    if pos_done and rot_done:
                        retreat["reason"] = "home_reached"
                        break
                    bounded = [max(-step_m, min(step_m, value)) for value in delta]
                    command = [value / RETREAT_SCALE_M for value in bounded]
                    rot_norm = float(np.linalg.norm(rot_vector))
                    rot_command = [0.0, 0.0, 0.0]
                    if rot_norm > rot_tol:
                        clipped = rot_vector * min(1.0, rot_step / max(rot_norm, 1e-9))
                        rot_command = [float(v) for v in clipped]
                    action = np.array(
                        [
                            command[0],
                            command[1],
                            command[2],
                            rot_command[0],
                            rot_command[1],
                            rot_command[2],
                            float(self.opts["gripper_open_command"]),
                        ],
                        dtype=np.float64,
                    )
                    obs, _reward, _term, _trunc, _info = self.env.step(action)
                    snap = self.env.state_snapshot()
                    self._last_obs = obs
                    self._last_snapshot = snap
                    quat = _quat_of_snapshot(snap) or _quat_from_obs(obs) or quat
                    if self.frame_sink is not None:
                        self.frame_sink(obs, snap)
                    retreat["actions"] += 1
                    retreat["applied"] = True
                    if rot_norm > rot_tol:
                        retreat["rotation_applied"] = True
                else:
                    retreat["reason"] = "bounded_actions_exhausted"
        post_retreat = self._last_snapshot or pre
        retreat["end_eef"] = self._eef(post_retreat)
        end_quat = _quat_of_snapshot(post_retreat) or _quat_from_obs(self._last_obs) or start_quat
        if end_quat is not None and self.home_quat is not None:
            retreat["end_angle_deg"] = round(
                float(np.degrees(np.linalg.norm(_axis_angle(end_quat, self.home_quat)))), 2
            )
        if retreat.get("start_eef") and retreat.get("end_eef"):
            retreat["moved_m"] = round(
                max(abs(float(retreat["end_eef"][i]) - float(retreat["start_eef"][i])) for i in range(3)),
                4,
            )
        block["retreat"] = retreat

        # ---- state checks after the retreat ------------------------------
        goal_after = state_check(post_retreat, self.step, self.task)
        epoch_same = post_retreat.get("scene_epoch") == pre.get("scene_epoch")
        block["post_retreat"] = {
            "observation_id": post_retreat.get("observation_id"),
            "sim_action_count": post_retreat.get("sim_action_count"),
            "scene_epoch": post_retreat.get("scene_epoch"),
            "goal": goal_after,
            "retreat_preserved_goal": bool(goal_after.get("ok")),
        }

        # ---- clear the policy buffer, then take a fresh observation ------
        token = ""
        reset_block: dict[str, Any] = {"requested": bool(self.opts["reset_policy_buffer"])}
        if self.opts["reset_policy_buffer"]:
            reset_started = time.perf_counter()
            token = self.inference.reset(f"handoff:{self.episode_id}:{self.step.get('step_id')}")
            reset_block["token"] = token
            reset_block["reset_s"] = round(time.perf_counter() - reset_started, 4)
            reset_block["reason"] = "no step-1 action chunk may be consumed by step 2"
            self.journal("policy_buffer_reset", {"episode_id": self.episode_id, **reset_block})
        self._check_stop()
        obs = self.env.raw_obs()
        fresh = self.env.state_snapshot()
        self._last_obs = obs
        self._last_snapshot = fresh
        if self.frame_sink is not None:
            self.frame_sink(obs, fresh)
        fresh_goal = state_check(fresh, self.step, self.task)
        block["post_reset"] = {
            "observation_id": fresh.get("observation_id"),
            "sim_action_count": fresh.get("sim_action_count"),
            "scene_epoch": fresh.get("scene_epoch"),
            "goal": fresh_goal,
            "fresh_observation": fresh.get("observation_id") != pre.get("observation_id"),
        }
        block["buffer_reset"] = reset_block
        block["scene_epoch_same"] = bool(epoch_same and fresh.get("scene_epoch") == pre.get("scene_epoch"))
        block["sim_actions"] = int(fresh.get("sim_action_count") or 0) - int(pre.get("sim_action_count") or 0)
        block["elapsed_s"] = round(time.perf_counter() - started, 4)
        ready = bool(
            goal_after.get("ok")
            and fresh_goal.get("ok")
            and block["scene_epoch_same"]
            and block["post_reset"]["fresh_observation"]
            and (token or not self.opts["reset_policy_buffer"])
        )
        block["status"] = "ready" if ready else "not_ready"
        if not ready:
            block["reason_code"] = (
                "retreat_lost_step_1_goal" if not goal_after.get("ok") else "handoff_not_ready"
            )
            block["reason"] = "step 2 was not started because the handoff checks did not all pass"
        self.journal("handoff", {"episode_id": self.episode_id, **block})
        return block, obs, fresh, token


def step_summary(
    *,
    step: dict[str, Any],
    record: dict[str, Any],
    snapshot: dict[str, Any],
    verification: dict[str, Any],
    release_state: str,
    release_source: str,
    final_verification: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """One row of the composite evidence table (never a success claim).

    ``ok`` is the step's own completion at the time it stopped; ``ok_final``
    is the fresh re-check after the whole episode. They are kept apart so a
    step that was met and later disturbed is not reported as never met.
    """
    final = dict(final_verification or {})
    return {
        "index": step.get("step_index"),
        "step_id": step.get("step_id"),
        "task_id": step.get("step_task_id"),
        "task_key": step.get("step_task_key"),
        "prompt": step.get("prompt"),
        "goal_kind": step.get("goal_kind"),
        "ok": bool(verification.get("ok")),
        "ok_final": (bool(final.get("ok")) if final else None),
        "evidence": verification,
        "final_evidence": final or None,
        "native_current": bool(snapshot.get("native_goal")),
        "native_ever": bool(record.get("native_events")),
        "sim_actions": record.get("sim_actions"),
        "chunks_used": record.get("chunks_used"),
        "duration_s": (record.get("timing") or {}).get("total_s"),
        "stop_reason": record.get("stop_reason"),
        "failure_code": None if verification.get("ok") else verification.get("reason_code"),
        "release_state": release_state,
        "release_source": release_source,
    }


__all__ = [
    "DEFAULT_HANDOFF",
    "DEFAULT_PAIRS",
    "Handoff",
    "SCHEMA",
    "build_plan",
    "handoff_options",
    "options",
    "release_required",
    "state_check",
    "step_budget",
    "step_goal",
    "step_summary",
    "support",
    "validate_plan",
]
