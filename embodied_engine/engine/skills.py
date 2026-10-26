"""Skill executors: closed-loop VLA chunks plus explicit release/verification.

The policy produces a 5-action chunk. The runner observes after every action
and checks task-specific stop conditions at action granularity. A single
bounded prefetch slot may compute the next chunk from a mid-chunk observation
while the rest of the current chunk executes; the result is used only after
token/scene/observation-age validation. Grasp and release boundaries cancel
the prefetch and use a fresh observation.
"""

from __future__ import annotations

import math
import threading
import time
from typing import Any, Callable

import numpy as np

from engine import completion
from engine.inference_client import InferenceClient, model_view, request_row

RELEASE_ACTION = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
RETREAT_STEP_M = 0.025
RETREAT_SCALE_M = 0.05


class Stopped(Exception):
    pass


class SkillFailure(Exception):
    def __init__(self, reason_code: str, message: str, evidence: dict[str, Any] | None = None):
        super().__init__(message)
        self.reason_code = reason_code
        self.message = message
        self.evidence = evidence or {}


def _vec(row: Any) -> list[float] | None:
    if row is None:
        return None
    try:
        values = [float(v) for v in list(row)[:3]]
    except (TypeError, ValueError):
        return None
    return values if len(values) == 3 else None


EXTENSION_WINDOW_ACTIONS = 60
EXTENSION_MIN_QPOS_DELTA = 0.01
EXTENSION_STEP_CHUNKS = 10
EXTENSION_MIN_OBJECT_MOVE_M = 0.02

# A retry that starts where the failed attempt stopped inherits the failed
# pose: the drawer stays at qpos 0.0 for the whole second attempt because the
# arm is already pressed against the cabinet face. Successful first attempts
# always start from the episode's initial end-effector pose, so retries are
# brought back to that pose first (bounded Cartesian move, same OSC contract
# as the retreat). Nothing is relaxed: the completion evidence is unchanged.
REHOME_TOLERANCE_M = 0.02
REHOME_STEP_M = 0.06
REHOME_MAX_ACTIONS = 26
REHOME_MIN_OPENING_M = 0.05


class SkillRunner:
    def __init__(
        self,
        *,
        env,
        inference: InferenceClient,
        token: str,
        episode_id: str,
        task: dict[str, Any],
        step: dict[str, Any],
        obs: dict[str, Any],
        snapshot: dict[str, Any],
        baseline: dict[str, Any],
        stop_event: threading.Event,
        journal: Callable[[str, dict[str, Any]], None] | None = None,
        max_chunk_age_actions: int = 5,
        allow_prefetch: bool = True,
        chunk_budget: int = 30,
        chunk_budget_hard: int | None = None,
        frame_sink: Callable[[dict[str, Any], dict[str, Any]], None] | None = None,
        home_eef: Any = None,
    ):
        self.env = env
        self.inference = inference
        self.token = token
        self.episode_id = episode_id
        self.task = task
        self.step = step
        self.obs = obs
        self.snapshot = snapshot
        self.baseline = baseline
        self.stop_event = stop_event
        self.journal = journal or (lambda _stage, _row: None)
        self.max_chunk_age_actions = max_chunk_age_actions
        self.allow_prefetch = allow_prefetch and bool(inference.health().get("identity", {}).get("server", {}).get("reachable"))
        self.chunk_budget = chunk_budget
        self.chunk_budget_hard = max(int(chunk_budget_hard or chunk_budget), int(chunk_budget))
        self.budget_extensions = 0
        self.frame_sink = frame_sink
        self.home_eef = _vec(home_eef)
        self.rehome: dict[str, Any] | None = None
        self.attempt = int(step.get("attempt") or 1)
        self.object_name = step.get("object_n")
        self.target_name = step.get("target_n")
        self.prompt = self._prompt()
        self.held = False
        self.held_streak = 0
        self.chunk_index = 0
        self.sim_actions = 0
        self.inference_s = 0.0
        self.env_step_s = 0.0
        self.snapshot_s = 0.0
        self.chunk_gap_s = 0.0
        self.prefetch_wait_s = 0.0
        self.prefetch_saved_s = 0.0
        self.decision_wait_s = 0.0
        self.last_release_s = 0.0
        self.last_verify_s = 0.0
        self.prefetch_used = 0
        self.prefetch_discarded = 0
        self.prefetch_reinfer = 0
        self.chunks: list[dict[str, Any]] = []
        self.observations: list[dict[str, Any]] = []
        self.action_trace: list[dict[str, Any]] = []
        self.native_events: list[dict[str, Any]] = []
        self._native_latched = False
        self._pending: dict[str, Any] | None = None
        self._chunk_record: dict[str, Any] | None = None

    def _prompt(self) -> str:
        prompt = str(self.step.get("prompt") or "").strip()
        if prompt:
            return prompt
        skill = self.step.get("skill")
        if skill == "open_drawer":
            level = completion.drawer_level(self.target_name) or self.task.get("drawer_level") or "middle"
            return f"open the {level} drawer of the cabinet"
        return str(self.task.get("instruction") or "")

    def _check_stop(self) -> None:
        if self.stop_event.is_set():
            raise Stopped("stop requested")

    def _observe(self, obs: dict[str, Any] | None = None) -> dict[str, Any]:
        started = time.perf_counter()
        snapshot = self.env.state_snapshot()
        self.snapshot_s += time.perf_counter() - started
        if obs is not None:
            self.obs = obs
        if snapshot.get("scene_epoch") != self.snapshot.get("scene_epoch"):
            raise SkillFailure("scene_changed", "scene epoch changed during skill", {"snapshot": snapshot})
        self.snapshot = snapshot
        state = completion.goal_state(snapshot, self.step, self.task, self.baseline)
        self.observations.append(
            {
                "sim_action_count": snapshot.get("sim_action_count"),
                "observation_id": snapshot.get("observation_id"),
                "ok": state.get("ok"),
                "ready": state.get("ready"),
                "native_goal": snapshot.get("native_goal"),
                "predicate": state.get("predicate"),
                "joint": state.get("joint"),
                "qpos": state.get("qpos"),
                "relation": state.get("relation"),
                "eef_pos": (snapshot.get("sim") or {}).get("eef_pos"),
                "gripper_opening_m": (snapshot.get("sim") or {}).get("gripper_opening_m"),
            }
        )
        if snapshot.get("native_goal") and not self._native_latched:
            self._native_latched = True
            self.native_events.append(
                {
                    "sim_action_count": snapshot.get("sim_action_count"),
                    "observation_id": snapshot.get("observation_id"),
                }
            )
        return state

    def _step(self, action: Any, phase: str, index: int) -> dict[str, Any]:
        self._check_stop()
        started = time.perf_counter()
        pair = [round(float(value), 4) for value in np.asarray(action, dtype=np.float64).reshape(-1)[:7]]
        obs, _reward, _term, _trunc, _info = self.env.step(action)
        self.env_step_s += time.perf_counter() - started
        self.sim_actions += 1
        state = self._observe(obs)
        self.action_trace.append(
            {"n": self.sim_actions, "a": pair, "eef": (self.snapshot.get("sim") or {}).get("eef_pos"),
             "grip": (self.snapshot.get("sim") or {}).get("gripper_opening_m")}
        )
        if self.frame_sink is not None and (self.sim_actions % 2 == 0 or state.get("ready")):
            self.frame_sink(self.obs, self.snapshot)
        if self.step.get("skill") == "pick_place" and self.object_name:
            self._track_held(state)
        if state.get("ready"):
            self._discard_prefetch("skill_ready_boundary")
        return state

    def _track_held(self, state: dict[str, Any]) -> None:
        opening = ((self.snapshot.get("sim") or {}).get("gripper_opening_m"))
        object_pos = ((self.snapshot.get("sim") or {}).get("objects") or {}).get(self.object_name or "", {}).get("pos")
        baseline_pos = (
            ((self.baseline.get("sim") or {}).get("objects") or {})
            .get(self.object_name or "", {})
            .get("pos")
        )
        try:
            lifted = float(object_pos[2]) - float(baseline_pos[2])
        except (TypeError, ValueError, IndexError):
            lifted = None
        closed = isinstance(opening, (int, float)) and float(opening) < 0.05
        sufficient = lifted is not None and lifted >= 0.02
        if closed and sufficient:
            self.held_streak += 1
        else:
            self.held_streak = 0
        if not self.held and self.held_streak >= 2:
            self.held = True
            self._discard_prefetch("grasp_boundary")
            self.journal(
                "grasp",
                {
                    "sim_action_count": self.snapshot.get("sim_action_count"),
                    "observation_id": self.snapshot.get("observation_id"),
                    "gripper_opening_m": opening,
                    "object_lift_m": round(float(lifted), 4) if lifted is not None else None,
                    "relation_ok": state.get("ok"),
                },
            )

    def _infer(self, obs: dict[str, Any], snapshot: dict[str, Any], step_id: int) -> dict[str, Any]:
        self._check_stop()
        request = request_row(
            token=self.token,
            episode_id=self.episode_id,
            scene_epoch=snapshot.get("scene_epoch"),
            snapshot=snapshot,
            skill=str(self.step.get("skill")),
            step_id=step_id,
            prompt=self.prompt,
        )
        started = time.perf_counter()
        out = self.inference.infer(request, model_view(obs), timeout_s=180)
        waited = time.perf_counter() - started
        self.decision_wait_s += waited
        if out.get("status") != "ok":
            raise SkillFailure("inference_failed", str(out.get("error") or out.get("status")), {"response": out})
        self.inference_s += float(out.get("inference_s") or 0.0)
        self._check_stop()
        return {"actions": out["actions"], "inference_s": out.get("inference_s"), "wait_s": waited, "prefetched": False}

    def _submit_prefetch(self, obs: dict[str, Any], snapshot: dict[str, Any], step_id: int) -> None:
        if not self.allow_prefetch or self._pending is not None:
            return
        request = request_row(
            token=self.token,
            episode_id=self.episode_id,
            scene_epoch=snapshot.get("scene_epoch"),
            snapshot=snapshot,
            skill=str(self.step.get("skill")),
            step_id=step_id,
            prompt=self.prompt,
        )
        out = self.inference.prefetch(request, model_view(obs))
        if out.get("status") != "queued":
            self.prefetch_discarded += 1
            return
        self._pending = {
            "request_id": out["request_id"],
            "submitted_at": time.monotonic(),
            "sim_action_count": snapshot.get("sim_action_count"),
            "observation_id": snapshot.get("observation_id"),
            "scene_epoch": snapshot.get("scene_epoch"),
            "step_id": step_id,
        }

    def _take_prefetch(self) -> dict[str, Any] | None:
        if self._pending is None:
            return None
        pending = self._pending
        self._pending = None
        self._check_stop()
        wait_started = time.perf_counter()
        out = self.inference.result(pending["request_id"], wait_s=3.0)
        self.prefetch_wait_s += time.perf_counter() - wait_started
        current_count = self.snapshot.get("sim_action_count")
        age_ok = (
            isinstance(current_count, int)
            and isinstance(pending.get("sim_action_count"), int)
            and current_count - int(pending["sim_action_count"]) <= self.max_chunk_age_actions
        )
        if out.get("status") != "ready" or not age_ok:
            self.prefetch_discarded += 1
            self.prefetch_reinfer += 1
            self.inference.cancel("prefetch_stale_or_failed")
            self.journal(
                "prefetch_discard",
                {
                    "status": out.get("status"),
                    "reason": "age" if out.get("status") == "ready" and not age_ok else out.get("reason"),
                    "age_actions": None if not isinstance(current_count, int) else current_count - int(pending["sim_action_count"]),
                    "observation_id": pending.get("observation_id"),
                },
            )
            return None
        self.prefetch_used += 1
        self.prefetch_saved_s += float(out.get("inference_s") or 0.0)
        self.inference_s += float(out.get("inference_s") or 0.0)
        self.decision_wait_s += time.perf_counter() - wait_started
        return {
            "actions": out["actions"],
            "inference_s": out.get("inference_s"),
            "wait_s": time.perf_counter() - wait_started,
            "prefetched": True,
            "age_actions": current_count - int(pending["sim_action_count"]),
            "observation_id": pending.get("observation_id"),
        }

    def _extend_budget(self, state: dict[str, Any]) -> bool:
        """Extend a drawer budget only while the joint keeps moving.

        A fixed budget either aborts a policy that is still making progress or
        wastes actions on a stuck one. The joint trace is the cheapest honest
        progress signal: at exhaustion, extend by a fixed step only if the
        owner joint moved by at least ``EXTENSION_MIN_QPOS_DELTA`` inside the
        recent window.
        """
        if self.chunk_index >= self.chunk_budget_hard:
            return False
        skill = self.step.get("skill")
        if skill == "open_drawer":
            window = [
                float(row["qpos"])
                for row in self.observations[-EXTENSION_WINDOW_ACTIONS:]
                if isinstance(row.get("qpos"), (int, float))
            ]
            if len(window) < 2:
                return False
            delta = abs(max(window) - min(window))
            if delta < EXTENSION_MIN_QPOS_DELTA:
                return False
        elif skill == "pick_place":
            # A carried object that is still moving means the policy is
            # working; freezing it at the budget would abort mid-carry.
            poses = [
                [float(v) for v in row["object_pose"][:3]]
                for row in self.observations[-EXTENSION_WINDOW_ACTIONS:]
                if isinstance(row.get("object_pose"), list) and len(row["object_pose"]) >= 3
            ]
            if len(poses) < 2:
                return False
            span = max(math.dist(a, b) for a in poses for b in poses[:: max(1, len(poses) // 6)])
            if span < EXTENSION_MIN_OBJECT_MOVE_M:
                return False
            delta = span
        else:
            return False
        self.chunk_budget = min(self.chunk_budget + EXTENSION_STEP_CHUNKS, self.chunk_budget_hard)
        self.budget_extensions += 1
        self.journal(
            "chunk_budget_extended",
            {
                "chunk": self.chunk_index,
                "new_budget": self.chunk_budget,
                "qpos_window_delta": round(delta, 4),
                "qpos": state.get("qpos"),
            },
        )
        return True

    def _discard_prefetch(self, reason: str) -> None:
        if self._pending is None:
            return
        pending = self._pending
        self._pending = None
        self.prefetch_discarded += 1
        self.inference.cancel(reason)
        self.journal(
            "prefetch_discard",
            {
                "reason": reason,
                "request_id": pending.get("request_id"),
                "observation_id": pending.get("observation_id"),
                "sim_action_count": pending.get("sim_action_count"),
            },
        )

    def run(self) -> dict[str, Any]:
        started = time.perf_counter()
        if self.attempt > 1:
            self.rehome = self._goto_home()
        state = completion.goal_state(self.snapshot, self.step, self.task, self.baseline)
        last_action_end = None
        while not state.get("ready"):
            if self.chunk_index >= self.chunk_budget and not self._extend_budget(state):
                raise SkillFailure(
                    "chunk_budget_exhausted",
                    f"{self.step.get('skill')} used {self.chunk_index} chunks without reaching its goal",
                    {"last_state": state, "prompt": self.prompt},
                )
            if last_action_end is not None:
                self.chunk_gap_s += time.perf_counter() - last_action_end
            row = self._take_prefetch()
            if row is None:
                row = self._infer(self.obs, self.snapshot, self.chunk_index)
            actions = np.asarray(row["actions"])
            if actions.ndim != 2 or actions.shape[1] < 7:
                raise SkillFailure("bad_action_shape", f"expected [chunk,7], got {actions.shape}", {})
            record = {
                "chunk": self.chunk_index,
                "prefetched": bool(row.get("prefetched")),
                "inference_s": row.get("inference_s"),
                "wait_s": round(float(row.get("wait_s") or 0.0), 4),
                "age_actions": row.get("age_actions"),
                "actions": int(actions.shape[0]),
                "env_step_start_s": None,
                "env_step_end_s": None,
                "prompt": self.prompt,
            }
            self._chunk_record = record
            self.chunks.append(record)
            action_start = time.perf_counter()
            prefetch_at = 2 if self.allow_prefetch else None
            for index in range(actions.shape[0]):
                action = actions[index]
                state = self._step(action, "follow", index)
                if prefetch_at is not None and index == prefetch_at and not state.get("ready") and self._pending is None:
                    self._submit_prefetch(self.obs, self.snapshot, self.chunk_index + 1)
                if state.get("ready"):
                    break
            last_action_end = time.perf_counter()
            record["env_step_start_s"] = round(action_start - started, 4)
            record["env_step_end_s"] = round(last_action_end - started, 4)
            record["duration_s"] = round(last_action_end - action_start, 4)
            self.chunk_index += 1
        self._discard_prefetch("skill_done")
        state = completion.goal_state(self.snapshot, self.step, self.task, self.baseline)
        phase = "verify"
        release = None
        release_started = time.perf_counter()
        if self.step.get("skill") == "pick_place":
            release = self._release_and_retreat(state)
            phase = "release"
        else:
            self._retreat_only()
        self.last_release_s = time.perf_counter() - release_started
        verification = self._verify(state if release is None else None, release)
        self.last_verify_s = float(verification.get("verify_s") or 0.0)
        total = time.perf_counter() - started
        return {
            "skill": self.step.get("skill"),
            "object_n": self.object_name,
            "target_n": self.target_name,
            "relation": self.step.get("relation"),
            "prompt": self.prompt,
            "phase_reached": phase,
            "ok": bool(verification.get("ok")),
            "verification": verification,
            "release": release,
            "grasp": {
                "held": self.held,
                "held_streak": self.held_streak,
                "final_opening_m": (self.snapshot.get("sim") or {}).get("gripper_opening_m"),
            },
            "native_events": self.native_events,
            "chunks_used": self.chunk_index,
            "attempt": self.attempt,
            "rehome": self.rehome,
            "chunk_budget": self.chunk_budget,
            "chunk_budget_hard": self.chunk_budget_hard,
            "budget_extensions": self.budget_extensions,
            "sim_actions": self.sim_actions,
            "timing": {
                "total_s": round(total, 3),
                "inference_s_sum": round(self.inference_s, 3),
                "env_step_s_sum": round(self.env_step_s, 3),
                "snapshot_s_sum": round(self.snapshot_s, 3),
                "chunk_gap_s_sum": round(self.chunk_gap_s, 3),
                "chunk_decision_wait_s_sum": round(self.decision_wait_s, 3),
                "prefetch_wait_s_sum": round(self.prefetch_wait_s, 3),
                "prefetch_inference_overlap_s_est": round(self.prefetch_saved_s, 3),
                "release_s": round(self.last_release_s, 3),
                "verification_s": round(self.last_verify_s, 3),
            },
            "prefetch": {
                "enabled": self.allow_prefetch,
                "used": self.prefetch_used,
                "discarded": self.prefetch_discarded,
                "reinfer": self.prefetch_reinfer,
                "max_age_actions": self.max_chunk_age_actions,
            },
            "chunks": self.chunks,
            "observations": self.observations,
            "action_trace": self.action_trace,
        }

    def _goto_home(self) -> dict[str, Any]:
        """Return the arm to the episode's initial EEF pose before a retry.

        Only the arm is moved (bounded Cartesian OSC steps, gripper stays
        open); the scene, the objects and the completion evidence are not
        touched. Skipped when the gripper is closed, because the previous
        attempt may still hold an object and dragging it home would change
        the task state.
        """
        record: dict[str, Any] = {"attempt": self.attempt, "applied": False, "actions": 0}
        current = _vec((self.snapshot.get("sim") or {}).get("eef_pos"))
        opening = (self.snapshot.get("sim") or {}).get("gripper_opening_m")
        record["start_eef"] = current
        record["start_opening_m"] = opening
        if self.home_eef is None or current is None:
            record["reason"] = "home_unknown"
            return record
        if isinstance(opening, (int, float)) and float(opening) < REHOME_MIN_OPENING_M:
            record["reason"] = "gripper_closed_may_hold_object"
            return record
        for _ in range(REHOME_MAX_ACTIONS):
            self._check_stop()
            eef = _vec((self.snapshot.get("sim") or {}).get("eef_pos"))
            if eef is None:
                record["reason"] = "eef_unknown"
                break
            delta = [self.home_eef[i] - eef[i] for i in range(3)]
            if max(abs(value) for value in delta) < REHOME_TOLERANCE_M:
                record["reason"] = "home_reached"
                break
            action = [max(-REHOME_STEP_M, min(REHOME_STEP_M, value)) / RETREAT_SCALE_M for value in delta]
            self._step(action + [0.0, 0.0, 0.0, -1.0], "rehome", 0)
            record["actions"] += 1
            record["applied"] = True
        record["end_eef"] = _vec((self.snapshot.get("sim") or {}).get("eef_pos"))
        record.setdefault("reason", "bounded_actions_exhausted")
        self.journal("rehome", record)
        return record

    def _lower_into_box(self, state: dict[str, Any]) -> dict[str, Any]:
        """Descend the held object into the container opening before release.

        The policy often stops 2-5 cm above an open drawer. Releasing there
        makes the object bounce off the cabinet face. This is a bounded,
        closed-loop Cartesian descent (same OSC contract as the retreat), and
        it stops as soon as the official relation and height both hold.
        """
        drop = state.get("drop_release") or {}
        check = state.get("relation") or {}
        lower = check.get("lower")
        upper = check.get("upper")
        if not drop.get("trigger") or not isinstance(lower, list) or not isinstance(upper, list):
            return {"applied": False, "reason": "no_drop_trigger"}
        target_z = float(lower[2]) + 0.4 * (float(upper[2]) - float(lower[2]))
        center_x = 0.5 * (float(lower[0]) + float(upper[0]))
        center_y = 0.5 * (float(lower[1]) + float(upper[1]))
        record = {"applied": True, "target_z": round(target_z, 4), "actions": 0, "align_actions": 0,
                  "start_object_pose": (self.snapshot.get("sim") or {}).get("objects", {}).get(self.object_name, {}).get("pos") if self.object_name else None}
        # Align over the opening first; descending against the drawer face
        # stalls the arm and only lowers a few millimetres per action.
        for _ in range(8):
            self._check_stop()
            current = completion.goal_state(self.snapshot, self.step, self.task, self.baseline)
            obj = current.get("object_pose")
            if current.get("ok") or not isinstance(obj, list):
                break
            dx = center_x - float(obj[0])
            dy = center_y - float(obj[1])
            if abs(dx) < 0.005 and abs(dy) < 0.005:
                break
            ax = max(-RETREAT_STEP_M, min(RETREAT_STEP_M, dx)) / RETREAT_SCALE_M
            ay = max(-RETREAT_STEP_M, min(RETREAT_STEP_M, dy)) / RETREAT_SCALE_M
            self._step([ax, ay, 0.0, 0.0, 0.0, 0.0, 1.0], "align_over_box", 0)
            record["align_actions"] += 1
        for _ in range(16):
            self._check_stop()
            current = completion.goal_state(self.snapshot, self.step, self.task, self.baseline)
            obj = current.get("object_pose")
            box = current.get("relation") or {}
            if current.get("ok"):
                record["reason"] = "relation_reached"
                break
            if not isinstance(obj, list) or not isinstance(box.get("lower"), list):
                record["reason"] = "state_missing"
                break
            if not (float(box["lower"][0]) < float(obj[0]) < float(box["upper"][0])
                    and float(box["lower"][1]) < float(obj[1]) < float(box["upper"][1])):
                record["reason"] = "outside_box_xy"
                break
            dz = target_z - float(obj[2])
            if dz > -0.005:
                record["reason"] = "target_reached"
                break
            step_z = max(-RETREAT_STEP_M, min(RETREAT_STEP_M, dz)) / RETREAT_SCALE_M
            # Gripper stays commanded closed while carrying.
            self._step([0.0, 0.0, step_z, 0.0, 0.0, 0.0, 1.0], "lower_into_box", 0)
            record["actions"] += 1
        record["end_object_pose"] = ((self.snapshot.get("sim") or {}).get("objects", {}).get(self.object_name, {}) or {}).get("pos") if self.object_name else None
        self.journal("lower_into_box", record)
        return record

    def _release_and_retreat(self, state: dict[str, Any] | None = None) -> dict[str, Any]:
        self._discard_prefetch("release_boundary")
        self._check_stop()
        descent = self._lower_into_box(state or {})
        release_snapshot = self.snapshot
        release_record = {
            "commanded": True,
            "gripper_command": -1.0,
            "command_sim_action_count": release_snapshot.get("sim_action_count"),
            "command_observation_id": release_snapshot.get("observation_id"),
            "opening_before_m": (release_snapshot.get("sim") or {}).get("gripper_opening_m"),
            "relation_before": completion.goal_state(release_snapshot, self.step, self.task, self.baseline).get("relation"),
            "descent": descent,
            "actions": 0,
        }
        release_started = time.perf_counter()
        for _ in range(12):
            self._step(list(RELEASE_ACTION), "release", 0)
            release_record["actions"] += 1
            opening = (self.snapshot.get("sim") or {}).get("gripper_opening_m")
            if isinstance(opening, (int, float)) and float(opening) >= 0.06:
                break
        # Release evidence is the first observed state after the open command,
        # not the closed gripper before it.
        release_record["sim_action_count"] = self.snapshot.get("sim_action_count")
        release_record["observation_id"] = self.snapshot.get("observation_id")
        release_record["gripper_opening_m"] = (self.snapshot.get("sim") or {}).get("gripper_opening_m")
        self._retreat(0.12)
        release_record["retreat_s"] = round(time.perf_counter() - release_started, 3)
        release_record["after_retreat"] = {
            "sim_action_count": self.snapshot.get("sim_action_count"),
            "observation_id": self.snapshot.get("observation_id"),
        }
        self.journal("release", release_record)
        return release_record

    def _retreat(self, distance_m: float) -> None:
        start_eef = _vec((self.snapshot.get("sim") or {}).get("eef_pos"))
        if start_eef is None:
            return
        target = [start_eef[0], start_eef[1], start_eef[2] + distance_m]
        for _ in range(16):
            self._check_stop()
            eef = _vec((self.snapshot.get("sim") or {}).get("eef_pos"))
            if eef is None or eef[2] >= target[2] - 0.005:
                break
            delta = [target[i] - eef[i] for i in range(3)]
            action = [max(-RETREAT_STEP_M, min(RETREAT_STEP_M, v)) / RETREAT_SCALE_M for v in delta] + [0.0, 0.0, 0.0, -1.0]
            self._step(action, "retreat", 0)

    def _retreat_only(self) -> None:
        self._discard_prefetch("skill_done")
        self._retreat(0.08)

    def _verify(self, _extra_state: dict[str, Any] | None, release: dict[str, Any] | None) -> dict[str, Any]:
        self._check_stop()
        started = time.perf_counter()
        samples = []
        for _ in range(3):
            state = self._step(list(RELEASE_ACTION), "verify", 0)
            samples.append(self.snapshot)
            if not state.get("ok"):
                # A single transient frame is not enough to call success, but
                # keep sampling; final verification requires all three.
                pass
        out = completion.verification(samples, self.step, self.task, self.baseline, release=release)
        out["verify_s"] = round(time.perf_counter() - started, 3)
        out["release_matches_step"] = bool(release) if self.step.get("skill") == "pick_place" else None
        return out
