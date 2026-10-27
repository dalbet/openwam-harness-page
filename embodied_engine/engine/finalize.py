"""Bounded execution-layer placement finalizer.

Why this module exists
----------------------
LIBERO's native goal is a set of BDDL predicates (``On bowl stove_cook_region``).
Those predicates can hold *while the gripper is still closed on the object*,
and OpenWAM's whole-task rollout stops at the first sample where the goal
latches. The measured consequence (2026-10-27, 20-round loop) was 11/13
pick-place episodes ending with the fingers closed and no release evidence.

The rollout now gives the policy a bounded window to finish its own release
(``openwam_rollout`` post-goal phase). When that window ends without a
verified release, *this* module finishes the placement with an explicit,
bounded, execution-layer sequence:

    verify release candidate → clear the policy buffer → open the gripper in
    place → retreat a bounded distance → take fresh stability samples

Every action here is tagged ``evidence_source="execution_finalizer"``. It is
never reported as OpenWAM policy output: the record keeps
``policy_owned: false`` and the action counts of both layers separate.

Nothing in this module hard-codes scene coordinates, object names, or a
task-specific trajectory. The release action is "open the fingers, keep the
pose"; the retreat is "move up along the current tool axis".
"""

from __future__ import annotations

import math
import time
from typing import Any, Callable

import numpy as np

from engine import grasp_geometry
from engine.skills import Stopped

OPEN_GRIPPER = -1.0
HOLD_POSE = np.zeros(7, dtype=np.float64)
HOLD_POSE[6] = OPEN_GRIPPER

SCHEMA = "embodied_engine.placement.v1"


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class PlacementFinalizer:
    """One bounded release/retreat/verify sequence inside the live episode."""

    def __init__(
        self,
        *,
        env,
        task: dict[str, Any],
        relation_step: dict[str, Any],
        execution_cfg: dict[str, Any],
        post_goal_cfg: dict[str, Any],
        stop_event,
        episode_id: str = "",
        journal: Callable[[str, dict[str, Any]], None] | None = None,
        frame_sink: Callable[[dict[str, Any], dict[str, Any]], None] | None = None,
        reset_policy: Callable[[str], str] | None = None,
    ):
        self.env = env
        self.task = task
        self.relation_step = relation_step
        self.stop_event = stop_event
        self.episode_id = episode_id
        self.journal = journal or (lambda _stage, _row: None)
        self.frame_sink = frame_sink
        self.reset_policy = reset_policy

        finalizer_cfg = dict((post_goal_cfg or {}).get("finalizer") or {})
        self.release_steps = int(finalizer_cfg.get("release_steps") or execution_cfg.get("release_steps") or 8)
        self.retreat_m = float(execution_cfg.get("retreat_m") or 0.12)
        self.move_action_scale_m = float(execution_cfg.get("move_action_scale_m") or 0.05)
        self.move_step_clip_m = float(execution_cfg.get("move_step_clip_m") or 0.025)
        self.retreat_step_m = float(
            execution_cfg.get("retreat_step_m") or min(self.move_step_clip_m, 0.02)
        )
        self.samples_required = max(3, int(execution_cfg.get("stable_samples") or 3))
        self.open_min_m = float(execution_cfg.get("release_open_min_m") or 0.05)
        self.detach_probe_m = float(post_goal_cfg.get("detach_probe_m") or 0.03)
        self.object_motion_tol_m = float(post_goal_cfg.get("object_motion_tol_m") or 0.005)
        self.object_name = (task.get("objects") or [None])[0]

        steps_needed = (
            self.release_steps
            + int(math.ceil(self.retreat_m / max(self.retreat_step_m, 1e-6)))
            + self.samples_required
        )
        self.max_actions = int(post_goal_cfg.get("finalizer_max_actions") or (steps_needed + 4))

        self.rows: list[dict[str, Any]] = []
        self.actions = 0
        self.started = 0.0
        self._sim_start: int | None = None

    # ---- primitives ----

    def _check_stop(self) -> None:
        if self.stop_event is not None and self.stop_event.is_set():
            raise Stopped("stop requested during placement finalizer")

    def _step(self, action: np.ndarray, phase: str) -> dict[str, Any]:
        if self.actions >= self.max_actions:
            raise RuntimeError(f"placement finalizer exceeded its {self.max_actions}-action budget")
        self._check_stop()
        obs, _reward, term, trunc, info = self.env.step(action)
        self.actions += 1
        self._last_obs = obs
        snap = self.env.state_snapshot()
        if self.frame_sink is not None:
            try:
                self.frame_sink(obs, snap)
            except Exception:  # noqa: BLE001 - frame push must not break the contract
                pass
        row = {
            "phase": phase,
            "n": self.actions,
            "action7": [round(float(v), 4) for v in action],
            "sim_action_count": snap.get("sim_action_count"),
            "observation_id": snap.get("observation_id"),
            "native_goal": bool(snap.get("native_goal")),
            "gripper_opening_m": grasp_geometry.opening_m(snap),
            "eef_pos": grasp_geometry.eef_pos(snap),
            "object_pos": grasp_geometry.object_pos(snap, self.object_name),
            "terminated": bool(term),
            "truncated": bool(trunc),
        }
        frame = grasp_geometry.finger_frame(snap, self.object_name)
        row["separation_m"] = frame.get("separation_m")
        row["object_gap_m"] = frame.get("object_gap_m")
        row["object_along_m"] = frame.get("object_along_m")
        self.rows.append(row)
        self._last_snapshot = snap
        self.journal("finalizer_step", {"episode_id": self.episode_id, "row": row})
        return row

    @staticmethod
    def _relation_ok(snapshot: dict[str, Any], step: dict[str, Any], task: dict[str, Any]) -> dict[str, Any]:
        state = grasp_geometry.relation_facts(snapshot, step, task)
        return {
            "ok": bool(state.get("ok")),
            "predicate": state.get("predicate"),
            "relation": state.get("relation"),
            "height": state.get("height"),
            "evidence_source": state.get("evidence_source"),
        }

    # ---- verdicts ----

    def _base_block(self, *, required: bool = True) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "required": required,
            "attempted": False,
            "reason_code": None,
            "native_goal": False,
            "release_executed": False,
            "release_verified": False,
            "placement_verified": False,
            "retreat_status": "skipped",
            "execution_status": "not_attempted",
            "evidence_source": "none",
            "policy_owned": False,
            "release": None,
            "detach": None,
            "retreat": None,
            "samples": [],
            "budget": {
                "release_steps": self.release_steps,
                "retreat_steps": int(math.ceil(self.retreat_m / max(self.retreat_step_m, 1e-6))),
                "sample_steps": self.samples_required,
                "max_actions": self.max_actions,
            },
            "extra_actions": 0,
            "extra_s": 0.0,
            "token": None,
        }

    def _sample_rows(self, snapshots: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out = []
        for snap in snapshots:
            relation = self._relation_ok(snap, self.relation_step, self.task)
            frame = grasp_geometry.finger_frame(snap, self.object_name)
            out.append(
                {
                    "observation_id": snap.get("observation_id"),
                    "sim_action_count": snap.get("sim_action_count"),
                    "native_goal": bool(snap.get("native_goal")),
                    "relation_ok": relation["ok"],
                    "relation": relation,
                    "object_pos": grasp_geometry.object_pos(snap, self.object_name),
                    "eef_pos": grasp_geometry.eef_pos(snap),
                    "gripper_opening_m": grasp_geometry.opening_m(snap),
                    "separation_m": frame.get("separation_m"),
                    "object_gap_m": frame.get("object_gap_m"),
                }
            )
        return out

    # ---- main entry ----

    def run(self, snapshot: dict[str, Any]) -> dict[str, Any]:
        block = self._base_block()
        block["native_goal"] = bool(snapshot.get("native_goal"))
        start_relation = self._relation_ok(snapshot, self.relation_step, self.task)
        block["precondition"] = {
            "native_goal": bool(snapshot.get("native_goal")),
            "relation_ok": start_relation["ok"],
            "relation": start_relation,
            "continuable": snapshot.get("continuable"),
            "gripper_opening_m": grasp_geometry.opening_m(snapshot),
            "object_eef_m": grasp_geometry.eef_object_m(snapshot, self.object_name),
            "finger_frame": grasp_geometry.finger_frame(snapshot, self.object_name),
        }
        if snapshot.get("continuable") is False:
            # Case C: the environment refuses further steps. Issuing any action
            # here would be an error, not a fix.
            block["reason_code"] = "episode_not_continuable"
            block["execution_status"] = "unknown"
            return block
        if not block["native_goal"] or not start_relation["ok"]:
            block["reason_code"] = "not_at_release_candidate"
            block["execution_status"] = "unknown"
            return block

        self.started = time.perf_counter()
        self._sim_start = snapshot.get("sim_action_count")
        block["attempted"] = True
        try:
            # 1. The policy's remaining chunk must never interleave with the
            #    finalizer: dropping it here also resets the model-side buffer.
            if self.reset_policy is not None:
                block["token"] = self.reset_policy(f"finalizer:{self.episode_id}")
                block["buffer_cleared"] = True

            # 2. Open the fingers in place, bounded by execution.release_steps.
            opening_before = grasp_geometry.opening_m(snapshot)
            for _ in range(self.release_steps):
                self._step(HOLD_POSE, "release")
            release_rows = [r for r in self.rows if r["phase"] == "release"]
            openings = [r["gripper_opening_m"] for r in release_rows if isinstance(r["gripper_opening_m"], (int, float))]
            opening_after = openings[-1] if openings else None
            opening_max = max(openings) if openings else None
            release_verified = bool(
                isinstance(opening_after, (int, float)) and float(opening_after) >= self.open_min_m
            )
            block["release_executed"] = True
            block["release"] = {
                "command7": [round(float(v), 4) for v in HOLD_POSE],
                "steps": len(release_rows),
                "opening_before_m": opening_before,
                "opening_after_m": opening_after,
                "opening_max_m": opening_max,
                "min_opening_m": self.open_min_m,
                "sim_action_count": [release_rows[0]["sim_action_count"], release_rows[-1]["sim_action_count"]] if release_rows else None,
            }
            if not release_verified:
                block["reason_code"] = "release_not_measured"
                block["execution_status"] = "unknown"
                return self._finish(block)

            # 3. Retreat along +z of the current pose. This is the detachment
            #    probe: a held object follows the arm, a released one does not.
            before = self._last_snapshot
            retreat_steps = int(math.ceil(self.retreat_m / max(self.retreat_step_m, 1e-6)))
            z_cmd = _clip(self.retreat_step_m / max(self.move_action_scale_m, 1e-6), 0.0, 1.0)
            retreat_action = np.array([0.0, 0.0, z_cmd, 0.0, 0.0, 0.0, OPEN_GRIPPER], dtype=np.float64)
            for _ in range(retreat_steps):
                self._step(retreat_action, "retreat")
            after = self._last_snapshot
            eef_before = grasp_geometry.eef_pos(before)
            eef_after = grasp_geometry.eef_pos(after)
            obj_before = grasp_geometry.object_pos(before, self.object_name)
            obj_after = grasp_geometry.object_pos(after, self.object_name)
            eef_dz = None if eef_before is None or eef_after is None else round(eef_after[2] - eef_before[2], 5)
            obj_dz = None if obj_before is None or obj_after is None else round(obj_after[2] - obj_before[2], 5)
            retreat_ok = bool(eef_dz is not None and eef_dz >= min(self.detach_probe_m, self.retreat_m) * 0.5)
            coupled = bool(
                eef_dz is not None
                and obj_dz is not None
                and abs(eef_dz) >= 1e-6
                and obj_dz >= 0.3 * eef_dz
                and abs(obj_dz) > self.object_motion_tol_m
            )
            detached = None
            if retreat_ok and obj_dz is not None and eef_dz is not None:
                # A held object follows the arm; a released one does not.
                detached = bool(abs(obj_dz) <= max(self.object_motion_tol_m, 0.3 * abs(eef_dz)))
            block["retreat"] = {
                "status": "completed" if retreat_ok else "failed",
                "command7": [round(float(v), 4) for v in retreat_action],
                "steps": retreat_steps,
                "eef_dz_m": eef_dz,
                "object_dz_m": obj_dz,
                "object_follows_gripper": coupled,
            }
            block["retreat_status"] = block["retreat"]["status"]

            # 4. Stability samples: each one follows a real hold action, so the
            #    sample spans a new simulator state instead of re-reading one.
            snapshots: list[dict[str, Any]] = []
            for _ in range(self.samples_required):
                self._step(HOLD_POSE, "sample")
                snapshots.append(self._last_snapshot)
            samples = self._sample_rows(snapshots)
            block["samples"] = samples
            ids = {row.get("observation_id") for row in samples}
            counts = [row.get("sim_action_count") for row in samples]
            fresh = (
                len(samples) >= 3
                and None not in ids
                and len(ids) == len(samples)
                and all(isinstance(c, int) for c in counts)
                and all(counts[i] < counts[i + 1] for i in range(len(counts) - 1))
            )
            stable = fresh and all(row["relation_ok"] for row in samples)
            final_gap = [row.get("object_gap_m") for row in samples if isinstance(row.get("object_gap_m"), (int, float))]
            block["detach"] = {
                "detached_by_motion": detached,
                "object_follows_gripper": coupled,
                "eef_dz_m": eef_dz,
                "object_dz_m": obj_dz,
                "object_gap_m": final_gap[-1] if final_gap else None,
                "object_motion_tol_m": self.object_motion_tol_m,
                "basis": "retreat probe: object z-motion vs eef z-motion",
            }
            block["release_verified"] = bool(release_verified)
            block["fresh_samples"] = fresh
            block["samples_stable"] = bool(stable)
            # Retreat failure is recorded on its own: a verified placement in
            # which the arm could not retreat is still a verified placement.
            block["placement_verified"] = bool(release_verified and stable and detached is not False)
            block["evidence_source"] = "execution_finalizer"
            block["execution_status"] = "completed" if block["placement_verified"] else "unknown"
            if not block["placement_verified"]:
                if not release_verified:
                    block["reason_code"] = "release_not_measured"
                elif detached is False:
                    block["reason_code"] = "object_follows_gripper"
                elif not fresh:
                    block["reason_code"] = "stale_samples"
                else:
                    block["reason_code"] = "relation_lost_after_release"
            return self._finish(block)
        except Stopped:
            block["execution_status"] = "stopped"
            block["reason_code"] = "user_stop"
            block["evidence_source"] = "execution_finalizer" if block["release_executed"] else "none"
            return self._finish(block)
        except Exception as exc:  # noqa: BLE001 - report, never fake success
            block["execution_status"] = "failed"
            block["reason_code"] = "finalizer_error"
            block["error"] = f"{type(exc).__name__}: {exc}"
            return self._finish(block)

    def _finish(self, block: dict[str, Any]) -> dict[str, Any]:
        block["extra_actions"] = self.actions
        block["extra_s"] = round(time.perf_counter() - self.started, 3) if self.started else 0.0
        block["action_rows"] = self.rows
        block["sim_action_count"] = [
            self._sim_start,
            self._last_snapshot.get("sim_action_count") if getattr(self, "_last_snapshot", None) else None,
        ]
        block["policy_owned"] = False
        block["note"] = (
            "execution-layer release/retreat/verify; not OpenWAM policy output. "
            "The policy's remaining chunk was cleared before the first finalizer action."
        )
        return block


def policy_release_block(
    *,
    required: bool,
    native_goal: bool,
    post_goal: dict[str, Any],
    snapshot: dict[str, Any],
    object_name: str | None,
    open_min_m: float = 0.05,
) -> dict[str, Any]:
    """Placement block for an episode the policy finished on its own.

    ``release_verified`` means the *policy* was measured opening its fingers to
    at least ``open_min_m`` for two consecutive post-goal observations. The
    fingers being closed at the very end does not by itself mean the object is
    still held (a policy may open, release, then close again), so the final
    opening is reported but is not the criterion. Whether the object really
    left the fingers is reported separately in ``detach`` and is finally
    settled by relation stability across fresh post-release samples.
    """
    release = (post_goal or {}).get("release") or {}
    frame = grasp_geometry.finger_frame(snapshot, object_name)
    final_opening = grasp_geometry.opening_m(snapshot)
    gap = frame.get("object_gap_m")
    separation = frame.get("separation_m")
    outside_span = (
        isinstance(gap, (int, float))
        and isinstance(separation, (int, float))
        and float(gap) >= float(separation) / 2.0 + 0.01
    )
    block = {
        "schema": SCHEMA,
        "required": required,
        "attempted": False,
        "reason_code": None,
        "native_goal": bool(native_goal),
        "release_executed": bool(release.get("verified")),
        "release_verified": bool(release.get("verified")),
        "placement_verified": False,
        "retreat_status": "skipped",
        "execution_status": "unknown",
        "evidence_source": "policy_chunk" if release.get("verified") else "none",
        "policy_owned": True,
        "release": release,
        "detach": {
            "detached_by_motion": None,
            "final_gripper_opening_m": final_opening,
            "object_gap_m": gap,
            "separation_m": separation,
            "outside_finger_span": outside_span,
            "basis": "finger span at the final observation (no execution-layer retreat)",
        },
        "retreat": {"status": "skipped", "reason": "policy released on its own"},
        "samples": [],
        "budget": None,
        "extra_actions": 0,
        "extra_s": 0.0,
        "token": None,
        "open_min_m": open_min_m,
    }
    block["execution_status"] = "completed" if not required else "unknown"
    if required and not block["release_verified"]:
        block["reason_code"] = release.get("reason_code") or "release_not_verified"
    return block


__all__ = ["PlacementFinalizer", "policy_release_block", "SCHEMA"]
