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
    "mode": "action_retreat",
    "retreat_m": 0.18,
    "retreat_step_m": 0.04,
    "retreat_max_actions": 30,
    "tolerance_m": 0.02,
    "retreat_rot_step_rad": 1.0,
    "rot_tolerance_rad": 0.139626,
    "min_clear_m": 0.02,
    "reset_policy_buffer": True,
    "gripper_open_command": -1.0,
    "settle_actions": 2,
    "settle_tol_m": 0.01,
    "settle_tol_rot_rad": 0.05,
    "restore_penetration_tol_m": 0.002,
    "inflight_wait_s": 10.0,
    "detach_from_contact": True,
    "detach_max_actions": 8,
    "detach_step_m": 0.04,
    "settle_scene_tol_m": 0.01,
    "settle_object_tol_m": 0.005,
}
HANDOFF_MODES = ("action_retreat", "robot_state_restore")
"""``action_retreat`` moves the arm home with closed-loop actions.

``robot_state_restore`` writes the captured robot joints directly, keeping the
scene. It is a teleport: ``physical_continuity`` is false by construction.
"""
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
    opts = {**DEFAULT_HANDOFF, **{k: v for k, v in (block.get("handoff") or {}).items()}}
    mode = str(opts.get("mode") or "action_retreat")
    opts["mode"] = mode if mode in HANDOFF_MODES else "action_retreat"
    opts["mode_requested"] = mode
    opts["mode_fallback"] = bool(mode not in HANDOFF_MODES)
    return opts


def handoff_mode(cfg: dict[str, Any] | None = None) -> str:
    return str(handoff_options(cfg)["mode"])


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
    """The transition between step 1 and step 2 of one episode.

    Two modes, same contract: the scene is never reset, reloaded, or
    teleported, step 1's goal must still hold when the handoff ends, the
    policy buffer is cleared, and step 2 starts from a fresh observation.

    * ``action_retreat``: the arm walks home with bounded OSC actions.
    * ``robot_state_restore``: the arm's own joints are written back to the
      captured episode-initial value inside the env worker. That is a robot
      teleport, so ``physical_continuity`` is false and the mode says so.
    """

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
        robot_state: dict[str, Any] | None = None,
        prior_step: dict[str, Any] | None = None,
    ):
        self.env = env
        self.inference = inference
        self.stop_event = stop_event
        self.opts = handoff_options(cfg)
        self.mode = str(self.opts["mode"])
        self.episode_id = episode_id
        self.step = step
        self.step_task = step_task
        self.task = task
        self.home_eef = home_eef
        self.home_quat = [float(v) for v in home_quat] if home_quat else None
        self.journal = journal or (lambda _stage, _row: None)
        self.frame_sink = frame_sink
        # The stamped capture row the env service returned at episode init.
        self.robot_state = robot_state or {}
        # The finished step's own result row: a placement step must have
        # passed its release verification before the arm may be moved.
        self.prior_step = prior_step or {}
        self._last_snapshot: dict[str, Any] | None = None
        self._last_obs: dict[str, Any] | None = None

    # ---- helpers ------------------------------------------------------

    def _check_stop(self) -> None:
        from engine.skills import Stopped

        if self.stop_event.is_set():
            raise Stopped("stop requested during step handoff")

    def _eef(self, snapshot: dict[str, Any]) -> list[float] | None:
        value = (snapshot.get("sim") or {}).get("eef_pos")
        if isinstance(value, list) and len(value) >= 3:
            return [float(v) for v in value[:3]]
        return None

    def _base_block(self) -> dict[str, Any]:
        restore_mode = self.mode == "robot_state_restore"
        return {
            "schema": f"{SCHEMA}.handoff",
            "step_id": self.step.get("step_id"),
            "step_task_key": self.step_task.get("key"),
            "requires_release": release_required(self.step_task),
            "status": "started",
            "mode": self.mode,
            "mode_requested": self.opts.get("mode_requested"),
            "mode_fallback": bool(self.opts.get("mode_fallback")),
            # Teleport bookkeeping is explicit: the scene stays, the arm does
            # not travel there, so a later reader cannot call this continuous.
            "robot_state_restore": restore_mode,
            "robot_teleport": restore_mode,
            "physical_continuity": False,
            "scene_objects_preserved": None,
            "options": {
                key: self.opts[key]
                for key in (
                    "retreat_m",
                    "retreat_step_m",
                    "retreat_max_actions",
                    "tolerance_m",
                    "rot_tolerance_rad",
                    "settle_actions",
                    "settle_tol_m",
                    "settle_tol_rot_rad",
                    "restore_penetration_tol_m",
                    "reset_policy_buffer",
                    "gripper_open_command",
                )
            },
        }

    def _goal_joint(self, snapshot: dict[str, Any]) -> str | None:
        """The scene joint this handoff's step goal is about, resolved live.

        Evidence rows name the joint the previous goal owns instead of a
        literal, so the same code reports a different cabinet level or a
        different scene without edits.
        """
        try:
            goal = state_check(snapshot, self.step, self.task)
        except Exception:  # noqa: BLE001 - diagnostics only
            return None
        name = goal.get("joint") if isinstance(goal, dict) else None
        return str(name) if name else None

    def _read_orientation(self, pre: dict[str, Any]) -> None:
        # The snapshot carries no wrist orientation, so read one raw
        # observation here: without it the retreat cannot restore the pose.
        if self._last_obs is None and _quat_of_snapshot(pre) is None:
            try:
                self._last_obs = self.env.raw_obs()
            except Exception:  # noqa: BLE001 - orientation stays best-effort
                self._last_obs = None

    def _reset_policy_buffer(self, reason: str) -> tuple[str, dict[str, Any]]:
        block: dict[str, Any] = {"requested": bool(self.opts["reset_policy_buffer"])}
        token = ""
        if not self.opts["reset_policy_buffer"]:
            return token, block
        started = time.perf_counter()
        token = self.inference.reset(reason)
        block["token"] = token
        block["reset_s"] = round(time.perf_counter() - started, 4)
        block["reason"] = "no step-1 action chunk may be consumed by step 2"
        self.journal("policy_buffer_reset", {"episode_id": self.episode_id, **block})
        return token, block

    def _inflight_wait(self) -> dict[str, Any]:
        """Wait (bounded) until the action model has no request in flight.

        ``reset`` already invalidates the queue and the model-side buffer, so
        this is an extra barrier for the *write*, not the correctness
        mechanism. A model that never reports idle is recorded, not trusted.
        """
        budget = float(self.opts["inflight_wait_s"])
        started = time.perf_counter()
        row: dict[str, Any] = {"budget_s": budget, "waited": False, "checks": 0}
        deadline = started + budget
        while True:
            try:
                metrics = self.inference.metrics() or {}
            except Exception as exc:  # noqa: BLE001
                row.update({"status": "metrics_unavailable", "error": f"{type(exc).__name__}: {exc}"})
                break
            row["checks"] = int(row["checks"]) + 1
            row["in_flight"] = int(metrics.get("in_flight") or 0)
            row["pending_prefetch"] = int(metrics.get("pending_prefetch") or 0)
            if not row["in_flight"] and not row["pending_prefetch"]:
                row["waited"] = True
                break
            if time.perf_counter() >= deadline:
                row["status"] = "inflight_timeout"
                break
            time.sleep(0.05)
        row["wait_s"] = round(time.perf_counter() - started, 4)
        return row

    def _hold_action(self):
        return np.array(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, float(self.opts["gripper_open_command"])],
            dtype=np.float64,
        )

    def _not_ready(
        self,
        block: dict[str, Any],
        started: float,
        pre: dict[str, Any],
        reason_code: str,
        reason: str,
        *,
        obs: dict[str, Any] | None = None,
        snapshot: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], str]:
        block["status"] = "not_ready"
        block["reason_code"] = reason_code
        block["reason"] = reason
        block["elapsed_s"] = round(time.perf_counter() - started, 4)
        self.journal("handoff", {"episode_id": self.episode_id, **block})
        post_snapshot = snapshot or self._last_snapshot or pre
        return block, obs or self._last_obs or {}, post_snapshot, ""

    def _fresh_read(self) -> tuple[dict[str, Any], dict[str, Any]]:
        obs = self.env.raw_obs()
        fresh = self.env.state_snapshot()
        self._last_obs = obs
        self._last_snapshot = fresh
        if self.frame_sink is not None:
            self.frame_sink(obs, fresh)
        return obs, fresh

    @staticmethod
    def _non_robot_diff(
        pre: dict[str, Any],
        post: dict[str, Any],
        robot_joints: set[str],
        *,
        joint_tol: float = 1e-6,
        object_tol: float = 1e-4,
    ) -> dict[str, Any]:
        """What moved outside the robot between the two snapshots.

        ``joint_tol`` is passed in by the caller: the direct write is checked
        by the env service's exact non-robot digest, while the settle actions
        are real physics, so a hinge that keeps swinging a few millimetres is
        recorded with its delta instead of being called a scene change.
        """
        joints: dict[str, Any] = {}
        worst = 0.0
        pre_joints = pre.get("joints") or {}
        post_joints = post.get("joints") or {}
        for name, value in pre_joints.items():
            if name in robot_joints:
                continue
            before = value
            after = post_joints.get(name)
            if not isinstance(before, (int, float)) or not isinstance(after, (int, float)):
                continue
            delta = abs(float(after) - float(before))
            worst = max(worst, delta)
            joints[name] = {
                "before": round(float(before), 6),
                "after": round(float(after), 6),
                # ``delta`` stays a magnitude so existing rows keep their
                # meaning; ``delta_signed`` carries the direction a slide joint
                # needs (a drawer that opened and one that closed by the same
                # amount both report delta=0.005).
                "delta": round(delta, 6),
                "delta_signed": round(float(after) - float(before), 6),
            }
        objects: dict[str, Any] = {}
        pre_objects = (pre.get("sim") or {}).get("objects") or {}
        post_objects = (post.get("sim") or {}).get("objects") or {}
        object_worst = 0.0
        for name, row in pre_objects.items():
            before = (row or {}).get("pos")
            after = (post_objects.get(name) or {}).get("pos")
            if not isinstance(before, list) or not isinstance(after, list):
                continue
            if len(before) < 3 or len(after) < 3:
                continue
            delta = max(abs(float(after[i]) - float(before[i])) for i in range(3))
            object_worst = max(object_worst, delta)
            objects[name] = {
                "before": [round(float(v), 6) for v in before[:3]],
                "after": [round(float(v), 6) for v in after[:3]],
                "delta": round(delta, 6),
            }
        return {
            "joints": joints,
            "objects": objects,
            "joint_max_delta": round(worst, 6),
            "object_max_delta": round(object_worst, 6),
            "joint_tol": float(joint_tol),
            "object_tol": float(object_tol),
            "preserved": bool(worst <= float(joint_tol) and object_worst <= float(object_tol)),
        }

    # ---- entry point --------------------------------------------------

    def run(self, snapshot: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], str]:
        started = time.perf_counter()
        block = self._base_block()
        pre = snapshot
        pre_goal = state_check(pre, self.step, self.task)
        self._read_orientation(pre)
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
            self.journal("handoff", {"episode_id": self.episode_id, **block})
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

        if self.mode == "robot_state_restore":
            return self._run_restore(pre, block, started)
        return self._run_retreat(pre, block, started)

    # ---- mode: bounded action retreat ---------------------------------

    def _run_retreat(
        self, pre: dict[str, Any], block: dict[str, Any], started: float
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], str]:
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
        # Convergence is measured, not assumed: "I ran out of actions but the
        # goal still holds" is not a restored pose.
        pos_ok = None
        rot_ok = None
        if retreat.get("end_eef") is not None and self.home_eef is not None:
            end_distance = max(
                abs(float(self.home_eef[i]) - float(retreat["end_eef"][i])) for i in range(3)
            )
            retreat["end_distance_m"] = round(end_distance, 4)
            pos_ok = bool(end_distance < float(self.opts["tolerance_m"]))
        if self.home_quat is not None and end_quat is not None:
            end_angle = float(np.linalg.norm(_axis_angle(end_quat, self.home_quat)))
            rot_ok = bool(end_angle < float(self.opts["rot_tolerance_rad"]))
        retreat["converged"] = bool(
            (pos_ok is not False and rot_ok is not False)
            and (pos_ok or rot_ok)
            and retreat["reason"] in {"home_reached", "already_home"}
        )
        block["retreat"] = retreat
        block["counts"] = {
            "retreat_actions": int(retreat["actions"]),
            "settle_actions": 0,
            "policy_actions": 0,
            "sim_actions": int(retreat["actions"]),
        }

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
        token, reset_block = self._reset_policy_buffer(
            f"handoff:{self.episode_id}:{self.step.get('step_id')}"
        )
        self._check_stop()
        obs, fresh = self._fresh_read()
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
        block["scene_objects_preserved"] = bool(block["scene_epoch_same"])
        block["sim_actions"] = int(fresh.get("sim_action_count") or 0) - int(pre.get("sim_action_count") or 0)
        block["elapsed_s"] = round(time.perf_counter() - started, 4)
        ready = bool(
            goal_after.get("ok")
            and fresh_goal.get("ok")
            and block["scene_epoch_same"]
            and block["post_reset"]["fresh_observation"]
            and retreat.get("converged")
            and (token or not self.opts["reset_policy_buffer"])
        )
        block["status"] = "ready" if ready else "not_ready"
        if not ready:
            if not goal_after.get("ok"):
                block["reason_code"] = "retreat_lost_step_1_goal"
                block["reason"] = "the retreat moved the scene enough to break step 1's goal"
            elif not retreat.get("converged"):
                block["reason_code"] = "retreat_not_converged"
                block["reason"] = (
                    f"the bounded retreat ended {retreat.get('end_distance_m')} m / "
                    f"{retreat.get('end_angle_deg')} deg from home ({retreat.get('reason')}); "
                    "step 2 would start from an unverified pose"
                )
            else:
                block["reason_code"] = "handoff_not_ready"
                block["reason"] = "step 2 was not started because the handoff checks did not all pass"
        self.journal("handoff", {"episode_id": self.episode_id, **block})
        return block, obs, fresh, token

    # ---- mode: robot state restore ------------------------------------

    def _gripper_contacts(self) -> list[dict[str, Any]] | None:
        """Finger-to-non-robot contacts, or ``None`` when unreadable."""
        reader = getattr(self.env, "robot_state_inspect", None)
        if not callable(reader):
            return None
        try:
            row = reader()
        except Exception:  # noqa: BLE001 - the write itself will refuse
            return None
        if not isinstance(row, dict) or not row.get("ok"):
            return None
        robot = row.get("robot") or {}
        state = robot.get("state") or {}
        contacts = (state.get("contacts") or {}).get("gripper_vs_nonrobot")
        return list(contacts or [])

    def _detach(
        self,
        started: float,
        pre: dict[str, Any],
        block: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], str] | None:
        """Open the fingers and back off until the sim reports no contact.

        Step 1 of 0->8 ends with the fingers still on the drawer handle, so a
        direct write of the arm's initial joints would drag the drawer. The
        contact is *measured* here (never assumed from the step's task type),
        and the detach is a small, counted, policy-free motion. If the fingers
        cannot be freed inside the budget, the handoff is refused.
        """
        detach: dict[str, Any] = {
            "enabled": bool(self.opts["detach_from_contact"]),
            "required": False,
            "actions": 0,
            "contacts_before": None,
            "contacts_after": None,
            "reason": None,
            "steps": [],
        }
        block["detach"] = detach
        if not detach["enabled"]:
            detach["reason"] = "disabled_by_config"
            return None
        contacts = self._gripper_contacts()
        if contacts is None:
            # No read: fall through, the write's own contact check decides.
            detach["reason"] = "contact_read_unavailable"
            return None
        detach["contacts_before"] = contacts
        if not contacts:
            detach["reason"] = "already_clear"
            return None
        detach["required"] = True
        budget = int(self.opts["detach_max_actions"])
        step_m = float(self.opts["detach_step_m"])
        joint_name = self._goal_joint(pre)
        detach["goal_joint"] = joint_name
        for _ in range(budget):
            self._check_stop()
            snap = self._last_snapshot or pre
            eef = self._eef(snap)
            if eef is not None and self.home_eef is not None:
                delta = [float(self.home_eef[i]) - float(eef[i]) for i in range(3)]
            else:
                delta = [-1.0, 0.0, 0.0]
            if max(abs(value) for value in delta) < 1e-6:
                delta = [-step_m, 0.0, 0.0]
            bounded = [max(-step_m, min(step_m, value)) for value in delta]
            action = np.array(
                [value / RETREAT_SCALE_M for value in bounded] + [0.0, 0.0, 0.0,
                                                                 float(self.opts["gripper_open_command"])],
                dtype=np.float64,
            )
            obs, _reward, _term, _trunc, _info = self.env.step(action)
            snap = self.env.state_snapshot()
            self._last_obs = obs
            self._last_snapshot = snap
            if self.frame_sink is not None:
                self.frame_sink(obs, snap)
            detach["actions"] += 1
            detach["steps"].append(
                {
                    "action": [round(float(v), 4) for v in action],
                    "eef_pos": self._eef(snap),
                    "observation_id": snap.get("observation_id"),
                    "sim_action_count": snap.get("sim_action_count"),
                    "drawer_qpos": (snap.get("joints") or {}).get(joint_name) if joint_name else None,
                }
            )
            contacts = self._gripper_contacts()
            if contacts is not None and not contacts:
                break
        detach["contacts_after"] = contacts
        if contacts:
            return self._not_ready(
                block,
                started,
                pre,
                "gripper_contact_after_detach",
                f"the fingers are still in contact after {detach['actions']} detach actions; "
                "restoring the arm would drag what they touch",
            )
        return None

    def _run_restore(
        self, pre: dict[str, Any], block: dict[str, Any], started: float
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], str]:
        capture_row = self.robot_state or {}
        capture = capture_row.get("robot") or {}
        captured_state = capture.get("state") or {}
        robot_names = set((captured_state.get("robot") or {}).get("joints") or [])
        restore: dict[str, Any] = {
            "requested": True,
            "available": bool(capture.get("ok")),
            "capture_id": capture_row.get("capture_id"),
            "capture_observation_id": capture_row.get("observation_id"),
            "capture_sim_action_count": capture_row.get("sim_action_count"),
            "capture_scene_epoch": capture_row.get("scene_epoch"),
            "applied": False,
            "verified": False,
            "reason": None,
            "settle_actions": 0,
        }
        block["restore"] = restore

        # A placement step must have measured its release first: moving the
        # arm while the fingers still hold the object would drag it.
        if block["requires_release"]:
            placement = (self.prior_step or {}).get("placement") or {}
            release_ok = bool(placement.get("placement_verified"))
            block["release"].update(
                {
                    "required": True,
                    "executed": release_ok,
                    "verified": release_ok,
                    "evidence_source": placement.get("evidence_source"),
                }
            )
            if not release_ok:
                return self._not_ready(
                    block,
                    started,
                    pre,
                    "release_not_verified",
                    "the finished step's release was not verified; the gripper may still hold the object",
                )
        else:
            block["release"]["verified"] = None

        if not capture.get("ok"):
            return self._not_ready(
                block,
                started,
                pre,
                "robot_state_capture_missing",
                "no verified robot-state capture exists for this episode: "
                + str(capture.get("reason_code") or capture.get("reason") or "unknown"),
            )

        # ---- barrier: no action may be in flight or buffered --------------
        token, reset_block = self._reset_policy_buffer(
            f"handoff:{self.episode_id}:{self.step.get('step_id')}"
        )
        block["buffer_reset"] = reset_block
        barrier = {"policy_buffer": reset_block, "inflight": self._inflight_wait()}
        block["barrier"] = barrier
        self._check_stop()

        # ---- measured detach: the fingers must be off whatever they touched -
        # A drawer step ends with the fingers still on the handle, so restoring
        # the arm with them closed would drag the drawer. Whether they are in
        # contact is read from the sim, never inferred from the task type.
        detach = self._detach(started, pre, block)
        if detach is not None:
            return detach
        pre_write = self._last_snapshot or pre

        # ---- the write itself, inside the env service ---------------------
        request = {
            "capture_id": capture_row.get("capture_id"),
            "mode": self.mode,
            "reason": f"handoff:{self.episode_id}:{self.step.get('step_id')}",
            "penetration_tol_m": float(self.opts["restore_penetration_tol_m"]),
            "reject_gripper_contact": True,
            "require_collision_check": True,
        }
        write_started = time.perf_counter()
        try:
            reply = self.env.robot_state_restore(request)
        except Exception as exc:  # noqa: BLE001 - a refused write is not a crash
            restore["write_error"] = f"{type(exc).__name__}: {exc}"
            return self._not_ready(
                block,
                started,
                pre,
                "robot_state_restore_rpc_failed",
                f"the env service refused the restore call: {type(exc).__name__}: {exc}",
            )
        restore["write_s"] = round(time.perf_counter() - write_started, 4)
        worker = reply.get("robot") if isinstance(reply, dict) else {}
        worker = worker if isinstance(worker, dict) else {}
        restore["applied"] = bool(reply.get("ok"))
        restore["reason_code"] = reply.get("reason_code") or worker.get("reason_code")
        restore["reason"] = reply.get("reason") or worker.get("reason")
        restore["service_instance_id"] = reply.get("service_instance_id")
        restore["sim_action_count_during_write"] = reply.get("sim_action_count")
        restore["write"] = {
            "verification": (worker.get("block") or {}).get("verification"),
            "candidate": (worker.get("block") or {}).get("candidate"),
            "controller_sync": (worker.get("block") or {}).get("controller_sync"),
            "gripper_contacts": (worker.get("block") or {}).get("gripper_contacts"),
            "before": (worker.get("block") or {}).get("before"),
            "after_write": (worker.get("block") or {}).get("after_write"),
        }
        restore["robot"] = worker.get("capture")
        if not restore["applied"]:
            return self._not_ready(
                block,
                started,
                pre,
                str(restore["reason_code"] or "robot_state_restore_refused"),
                str(restore["reason"] or "the env service refused to restore the robot state"),
            )
        restore["verified"] = bool(
            (restore["write"]["verification"] or {}).get("non_robot_preserved")
            and float((restore["write"]["verification"] or {}).get("robot_qpos_error") or 0.0) <= 1e-6
            and float((restore["write"]["verification"] or {}).get("robot_qvel_abs_max") or 0.0) <= 1e-9
        )
        if not restore["verified"]:
            # An unverified write is an unknown robot state: applying actions on
            # top of it could drag whatever the fingers are near.
            return self._not_ready(
                block,
                started,
                pre,
                "restore_write_unverified",
                "the robot write or the untouched scene state failed its own verification",
            )

        # ---- a stop that arrived during/after the write is honoured now ---
        if self.stop_event.is_set():
            restore["stop_seen_after_restore"] = True
            self._check_stop()

        # ---- counted settle actions: a real controller hold ----------------
        settle_target = int(self.opts["settle_actions"])
        settle_rows: list[dict[str, Any]] = []
        settle_started = time.perf_counter()
        joint_name = self._goal_joint(pre)
        previous_eef = self._eef(pre)
        for index in range(settle_target):
            self._check_stop()
            obs, _reward, _term, _trunc, _info = self.env.step(self._hold_action())
            snap = self.env.state_snapshot()
            self._last_obs = obs
            self._last_snapshot = snap
            if self.frame_sink is not None:
                self.frame_sink(obs, snap)
            eef = self._eef(snap)
            delta = None
            if eef is not None and previous_eef is not None:
                delta = round(max(abs(float(eef[i]) - float(previous_eef[i])) for i in range(3)), 6)
            settle_rows.append(
                {
                    "action": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, float(self.opts["gripper_open_command"])],
                    "observation_id": snap.get("observation_id"),
                    "sim_action_count": snap.get("sim_action_count"),
                    "eef_pos": eef,
                    "eef_step_delta_m": delta,
                    "goal_joint": joint_name,
                    "goal_joint_qpos": (snap.get("joints") or {}).get(joint_name) if joint_name else None,
                }
            )
            previous_eef = eef
            restore["settle_actions"] = index + 1
        restore["settle_s"] = round(time.perf_counter() - settle_started, 4)
        restore["settle"] = settle_rows
        post_settle = self._last_snapshot or pre

        # ---- did the hold move the robot or the scene? --------------------
        target_eef = (restore.get("robot") or {}).get("eef_pos")
        drift_m = None
        if target_eef and self._eef(post_settle):
            drift_m = round(
                max(abs(float(target_eef[i]) - float(self._eef(post_settle)[i])) for i in range(3)), 6
            )
        drift_rot = None
        if self.home_quat is not None:
            quat_now = _quat_of_snapshot(post_settle) or _quat_from_obs(self._last_obs)
            target_quat = (restore.get("robot") or {}).get("eef_quat")
            if quat_now is not None and target_quat:
                drift_rot = round(
                    float(np.linalg.norm(_axis_angle([float(v) for v in target_quat], quat_now))), 6
                )
        # Two separate questions: what the detach itself moved since the
        # handoff began, and what the settle actions moved after the write.
        # The write's own exactness is the env service's non-robot digest; the
        # settle is real physics and gets a millimetre-scale tolerance.
        block["detach_check"] = self._non_robot_diff(
            pre,
            pre_write,
            robot_names,
            joint_tol=float(self.opts["settle_scene_tol_m"]),
            object_tol=float(self.opts["settle_object_tol_m"]),
        )
        scene_diff = self._non_robot_diff(
            pre_write,
            post_settle,
            robot_names,
            joint_tol=float(self.opts["settle_scene_tol_m"]),
            object_tol=float(self.opts["settle_object_tol_m"]),
        )
        goal_after = state_check(post_settle, self.step, self.task)
        block["post_restore"] = {
            "observation_id": post_settle.get("observation_id"),
            "sim_action_count": post_settle.get("sim_action_count"),
            "scene_epoch": post_settle.get("scene_epoch"),
            "goal": goal_after,
            "eef_pos": self._eef(post_settle),
            "target_eef": target_eef,
            "eef_drift_m": drift_m,
            "eef_drift_rot_rad": drift_rot,
            "robot_joints": {
                name: (post_settle.get("joints") or {}).get(name) for name in sorted(robot_names)
            },
        }
        block["scene_check"] = scene_diff

        # ---- fresh observation, then step 2 -------------------------------
        obs, fresh = self._fresh_read()
        fresh_goal = state_check(fresh, self.step, self.task)
        block["post_reset"] = {
            "observation_id": fresh.get("observation_id"),
            "sim_action_count": fresh.get("sim_action_count"),
            "scene_epoch": fresh.get("scene_epoch"),
            "goal": fresh_goal,
            "fresh_observation": fresh.get("observation_id") != pre.get("observation_id"),
        }
        block["scene_epoch_same"] = bool(
            pre.get("scene_epoch")
            == post_settle.get("scene_epoch")
            == fresh.get("scene_epoch")
        )
        block["scene_objects_preserved"] = bool(
            block["scene_epoch_same"] and scene_diff["preserved"]
        )
        block["counts"] = {
            "retreat_actions": 0,
            "detach_actions": int((block.get("detach") or {}).get("actions") or 0),
            "settle_actions": int(restore["settle_actions"]),
            "policy_actions": 0,
            "sim_actions": int(fresh.get("sim_action_count") or 0) - int(pre.get("sim_action_count") or 0),
            "restore_writes": 1 if restore["applied"] else 0,
        }
        block["sim_actions"] = int(block["counts"]["sim_actions"])
        block["elapsed_s"] = round(time.perf_counter() - started, 4)
        drift_ok = drift_m is not None and float(drift_m) <= float(self.opts["settle_tol_m"])
        drift_rot_ok = drift_rot is None or float(drift_rot) <= float(self.opts["settle_tol_rot_rad"])
        ready = bool(
            restore["applied"]
            and restore["verified"]
            and scene_diff["preserved"]
            and goal_after.get("ok")
            and fresh_goal.get("ok")
            and block["scene_epoch_same"]
            and block["post_reset"]["fresh_observation"]
            and drift_ok
            and drift_rot_ok
            and (token or not self.opts["reset_policy_buffer"])
        )
        block["status"] = "ready" if ready else "not_ready"
        if not ready:
            if not restore["verified"]:
                block["reason_code"] = "restore_write_unverified"
                block["reason"] = "the robot write or the untouched scene state failed its own verification"
            elif not scene_diff["preserved"]:
                block["reason_code"] = "settle_moved_the_scene"
                block["reason"] = (
                    "the counted settle actions moved a scene joint or object: "
                    f"joint {scene_diff['joint_max_delta']} m, object {scene_diff['object_max_delta']} m"
                )
            elif not goal_after.get("ok"):
                block["reason_code"] = "restore_lost_step_1_goal"
                block["reason"] = "the restored scene no longer satisfies step 1's goal"
            elif not drift_ok or not drift_rot_ok:
                block["reason_code"] = "restore_drift"
                block["reason"] = (
                    f"after {restore['settle_actions']} settle actions the arm sits {drift_m} m / "
                    f"{drift_rot} rad from the restored pose"
                )
            else:
                block["reason_code"] = "handoff_not_ready"
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
    "HANDOFF_MODES",
    "Handoff",
    "SCHEMA",
    "build_plan",
    "handoff_mode",
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
