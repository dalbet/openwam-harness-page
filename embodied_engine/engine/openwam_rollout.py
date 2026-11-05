"""Closed-loop whole-task rollout for the OpenWAM backend.

OpenWAM-Alpha-Sim-LIBERO is a single whole-task policy: it was fine-tuned on
LIBERO episodes conditioned on the **full official task instruction** and it
emits a 32-step native EEF10 action horizon. It was not trained on the
engine's pick/place/open-drawer sub-skill prompts, so feeding it those would
be an out-of-distribution instruction. This runner therefore executes one
planned ``whole_task`` step:

    observe (raw LIBERO) → OpenWAM contract obs → one action → env.step
    → repeat until the native LIBERO goal latches, the stop flag is set, or
      the step budget is exhausted

The server owns the receding-horizon buffer (one generated chunk of
``generate_horizon`` actions, ``execute_horizon`` of which are executed before
re-planning). One call to the inference service returns exactly one action,
which keeps stop/reset isolation per action: a cancelled token or a raised
stop flag invalidates the very next action.

Completion evidence is measured, never impersonated: the native LIBERO goal,
the measured finger separation and the actual open commands are recorded as
separate facts. No Pi0.5-style release receipt is fabricated for an action
sequence this policy produced by itself.
"""

from __future__ import annotations

import math
import subprocess
import threading
import time
from typing import Any, Callable

import numpy as np

from adapters.openwam_contract import build_openwam_obs, native_eef10_to_libero7d
from engine import completion, grasp_geometry
from engine.inference_client import InferenceClient, request_row
from engine.skills import Stopped

OPEN_GRIPPER_COMMAND = -1.0
OPENING_DELTA_M = 0.05


class GpuSampler:
    """Sample one GPU's used memory while a rollout runs (peak = measured)."""

    def __init__(self, index: int | None, period_s: float = 0.5):
        self.index = index
        self.period_s = period_s
        self.peak_mib: int | None = None
        self.samples = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _read(self) -> int | None:
        if self.index is None:
            return None
        try:
            out = subprocess.run(
                [
                    "nvidia-smi",
                    "--id",
                    str(self.index),
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            ).stdout.strip()
            return int(out.splitlines()[0]) if out else None
        except (OSError, ValueError, subprocess.SubprocessError, IndexError):
            return None

    def _loop(self) -> None:
        while not self._stop.is_set():
            value = self._read()
            if value is not None:
                self.samples += 1
                self.peak_mib = value if self.peak_mib is None else max(self.peak_mib, value)
            self._stop.wait(self.period_s)

    def __enter__(self) -> "GpuSampler":
        if self.index is not None:
            self._thread = threading.Thread(target=self._loop, name="gpu-sampler", daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *_exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)


class WholeTaskRunner:
    """One bounded whole-task OpenWAM rollout inside an already-reset scene."""

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
        max_steps: int = 600,
        timeout_s: float = 300.0,
        gpu_index: int | None = None,
        frame_sink: Callable[[dict[str, Any], dict[str, Any]], None] | None = None,
        post_goal: dict[str, Any] | None = None,
        goal: dict[str, Any] | None = None,
        goal_confirm_samples: int = 2,
        step_ref: str | None = None,
        watch_joints: list[str] | None = None,
        watch_goals: list[dict[str, Any]] | None = None,
        carry_objects: list[str] | None = None,
        watch_every: int = 5,
        policy_seed: int | None = None,
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
        self.max_steps = int(max_steps)
        self.timeout_s = float(timeout_s)
        self.gpu_index = gpu_index
        self.frame_sink = frame_sink
        # Recorded by both runners; only the Pi0.5 runner can honour it (its
        # server takes explicit initial noise). The OpenWAM path reports it as
        # unsupported rather than pretending the sample was pinned.
        self.policy_seed = None if policy_seed is None else int(policy_seed)

        # ---- step goal (composite step 1) ----------------------------------
        # ``goal`` is an engine predicate (drawer_open / stove_button / relation)
        # evaluated on every snapshot of this rollout. When it is set the
        # native LIBERO flag of the *loaded* scene is recorded but never ends
        # the step: a composite loads the last step's BDDL, so its native goal
        # has nothing to do with an earlier step.
        self.goal = dict(goal) if isinstance(goal, dict) else None
        self.goal_confirm_samples = max(1, int(goal_confirm_samples))
        self.step_ref = step_ref or str(step.get("step_id") or "step")
        self.step_goal_events: list[dict[str, Any]] = []
        self.step_goal_streak = 0
        self.step_goal_latched = False
        self._goal_facts: dict[str, Any] | None = None
        self._last_window_facts: dict[str, Any] = {}

        # ---- earlier steps' goal joints (diagnostics only) ------------------
        # Step N traces the joints a previous step had to achieve, so a report
        # can say *when* that earlier goal started to regress. Reads the
        # snapshot that was already taken for this action; it never issues an
        # extra environment call and never changes the action stream.
        self.watch_joints: list[str] = [str(name) for name in (watch_joints or [])]
        self.watch_baseline: dict[str, float] = {}
        self.watch_events: list[dict[str, Any]] = []
        self._last_watched: dict[str, Any] | None = None

        # ---- earlier steps' *goals*, carried along with a moved object ------
        # A later step can move the very object an earlier step placed into
        # (cheese in a bowl, then the bowl is carried). A stored end-of-step
        # pose would answer the wrong question, so each watched goal is
        # re-evaluated against the *live* snapshot, i.e. against the moved
        # object's current frame. Rows are written every ``watch_every``
        # actions from a snapshot the loop already takes: no extra RPC, and
        # one contact probe only at the first loss.
        self.watch_goals: list[dict[str, Any]] = [dict(row) for row in (watch_goals or [])]
        self.watch_objects: list[str] = []
        for name in [
            *(carry_objects or []),
            *[row.get("object_n") for row in self.watch_goals],
            *[row.get("target_n") for row in self.watch_goals],
        ]:
            if isinstance(name, str) and name and name not in self.watch_objects:
                self.watch_objects.append(name)
        self.watch_every = max(1, int(watch_every))
        self.goal_watch: dict[str, Any] = {
            "watch_goals": [
                {
                    "step_id": row.get("step_id"),
                    "key": row.get("key"),
                    "goal_kind": row.get("goal_kind"),
                    "object_n": row.get("object_n"),
                    "target_n": row.get("target_n"),
                    "relation": row.get("relation"),
                }
                for row in self.watch_goals
            ],
            "objects": list(self.watch_objects),
            "timeline_every_actions": self.watch_every,
            "samples": 0,
            "timeline": [],
            "per_goal": {},
            "events": [],
            "first_slide_out": None,
            "carry": {
                "object_n": (carry_objects[0] if carry_objects else None),
                "target_n": (self.watch_goals[0].get("target_n") if self.watch_goals else None),
                "held_through": None,
            },
            "note": (
                "each watched goal is evaluated on the live snapshot, so a carried object's goal is checked "
                "against that object's current pose, not its pose at the end of the earlier step"
            ),
        }
        self._goal_watch_last_ok: dict[str, bool] = {}

        self.prompt = str(step.get("prompt") or task.get("instruction") or "")
        self.sim_actions = 0
        self.inference_s = 0.0
        self.env_step_s = 0.0
        self.snapshot_s = 0.0
        self.observe_s = 0.0
        self.convert_s = 0.0
        self.frame_s = 0.0
        self.first_action_s: float | None = None
        self.action_intervals: list[float] = []
        self.action_trace: list[dict[str, Any]] = []
        self.observations: list[dict[str, Any]] = []
        self.inference_rows: list[dict[str, Any]] = []
        self.release_events: list[dict[str, Any]] = []
        self.native_events: list[dict[str, Any]] = []
        self.native_latched = False
        self._last_opening: float | None = None
        self.peak_gpu_mib: int | None = None

        # ---- post-native-goal release window -------------------------------
        # LIBERO's native goal is a predicate about object placement; it can
        # hold while the fingers are still closed on the object. Breaking the
        # loop at that instant cuts the policy off in the middle of its own
        # release. ``window_actions`` lets the already-generated chunk finish,
        # bounded, and records what the gripper actually did.
        cfg = dict(post_goal or {})
        self.post_goal_cfg = cfg
        self.window_actions = max(0, int(cfg.get("window_actions") or 0))
        self.release_open_m = float(cfg.get("release_open_m") or 0.05)
        self.release_confirm_samples = max(1, int(cfg.get("release_confirm_samples") or 2))
        self.detach_probe_m = float(cfg.get("detach_probe_m") or 0.03)
        self.object_motion_tol_m = float(cfg.get("object_motion_tol_m") or 0.005)
        objects = task.get("objects") or []
        targets = task.get("targets") or []
        # Only a placement task holds an object that must be released. A
        # drawer/stove task also lists objects+targets, but its relation
        # ("open"/"turn_on") is not a placement: treating it as one made the
        # window report ``relation_unconfirmed`` and refuse the action chunk
        # even though the native goal was already true.
        self.placement_task = str(task.get("kind") or "") in {"pick_place", "composite"}
        self.object_name = (objects[0] if objects else None) if self.placement_task else None
        self.relation_step = (
            {
                "skill": "pick_place",
                "object_n": objects[0],
                "target_n": targets[0],
                "relation": (task.get("semantics") or {}).get("relation") or "on",
            }
            if self.placement_task and objects and targets
            else None
        )
        self.window_relation_lost = False
        self.window_started_at: int | None = None
        self.window_used = 0
        self.window_rows: list[dict[str, Any]] = []
        self.window_open_streak = 0
        self.window_open_max: float | None = None
        self.window_release: dict[str, Any] = {"verified": False, "samples": 0}
        self.window_goal_facts: dict[str, Any] | None = None
        self.stop_reason: str | None = None
        self.buffer_remaining: int | None = None
        self._first_action_after_goal = True

    # ---- helpers ----

    def _check_stop(self) -> None:
        if self.stop_event.is_set():
            raise Stopped("stop requested")

    def _snapshot(self) -> dict[str, Any]:
        started = time.perf_counter()
        snap = self.env.state_snapshot()
        self.snapshot_s += time.perf_counter() - started
        if snap.get("scene_epoch") != self.baseline.get("scene_epoch"):
            raise RuntimeError(
                f"scene epoch changed during the OpenWAM rollout: "
                f"{self.baseline.get('scene_epoch')} -> {snap.get('scene_epoch')}"
            )
        self.snapshot = snap
        state = self._record_observation(snap)
        self._last_state = state
        self._last_window_facts = self._window_observe(snap)
        if self.goal is not None:
            self._goal_facts = self._observe_step_goal(snap)
        self._last_watched = self._watch(snap)
        self._observe_watch_goals(snap)
        return snap

    # ---- earlier steps' goals under a carried object ---------------------

    def _observe_watch_goals(self, snap: dict[str, Any]) -> None:
        if not self.watch_goals:
            return
        block = self.goal_watch
        block["samples"] = int(block["samples"]) + 1
        positions = {name: grasp_geometry.object_pos(snap, name) for name in self.watch_objects}
        row: dict[str, Any] = {
            "n": self.sim_actions,
            "sim_action_count": snap.get("sim_action_count"),
            "gripper_opening_m": grasp_geometry.opening_m(snap),
            "eef": grasp_geometry.eef_pos(snap),
            "positions": positions,
        }
        if len(self.watch_objects) >= 2:
            relatives: dict[str, Any] = {}
            for item in self.watch_goals:
                obj_name = item.get("object_n")
                target_name = item.get("target_n")
                a = positions.get(obj_name) if isinstance(obj_name, str) else None
                b = positions.get(target_name) if isinstance(target_name, str) else None
                if not isinstance(a, list) or not isinstance(b, list):
                    continue
                delta = [round(float(a[i]) - float(b[i]), 5) for i in range(3)]
                relatives[str(item.get("step_id") or "")] = {
                    "of": obj_name,
                    "to": target_name,
                    "delta_m": delta,
                    "distance_m": round(sum(value * value for value in delta) ** 0.5, 5),
                    "xy_m": round((delta[0] ** 2 + delta[1] ** 2) ** 0.5, 5),
                }
            if relatives:
                row["relative"] = relatives
        dynamics = self._watch_dynamics(snap)
        if dynamics:
            row["dynamics"] = dynamics
        states: dict[str, Any] = {}
        for item in self.watch_goals:
            step_id = str(item.get("step_id") or "")
            goal = item.get("goal") or {}
            try:
                facts = completion.goal_state(snap, goal, self.task)
            except Exception as exc:  # noqa: BLE001 - evidence must not break the rollout
                facts = {"ok": False, "reason_code": "goal_evaluation_failed",
                         "error": f"{type(exc).__name__}: {exc}"}
            ok = bool(facts.get("ok"))
            summary = {
                "ok": ok,
                "predicate": facts.get("predicate"),
                "relation": facts.get("relation"),
                "height": facts.get("height"),
                "object_pose": facts.get("object_pose"),
            }
            states[step_id] = summary
            per = block["per_goal"].setdefault(
                step_id,
                {
                    "step_id": step_id,
                    "key": item.get("key"),
                    "goal_kind": item.get("goal_kind"),
                    "ok_first": ok,
                    "ok_last": ok,
                    "samples": 0,
                    "samples_ok": 0,
                    "longest_ok_streak": 0,
                    "first_lost": None,
                    "last_transition": None,
                },
            )
            per["samples"] = int(per["samples"]) + 1
            per["samples_ok"] = int(per["samples_ok"]) + (1 if ok else 0)
            per["ok_last"] = ok
            streak = int(per.get("_streak") or 0)
            per["_streak"] = (streak + 1) if ok else 0
            per["longest_ok_streak"] = max(int(per.get("longest_ok_streak") or 0), int(per["_streak"]))
            before = self._goal_watch_last_ok.get(step_id)
            if before is not None and before != ok:
                transition = {
                    "sim_action_count": snap.get("sim_action_count"),
                    "skill_sim_actions": self.sim_actions,
                    "observation_id": snap.get("observation_id"),
                    "step_id": step_id,
                    "key": item.get("key"),
                    "from": "held" if before else "lost",
                    "to": "held" if ok else "lost",
                    "state": summary,
                    "gripper_opening_m": row["gripper_opening_m"],
                    "eef": row["eef"],
                    "positions": positions,
                    "relative": row.get("relative"),
                }
                if not ok:
                    # One contact read at the first loss only: it says whether
                    # the arm itself touched the object, which a position-only
                    # record cannot answer.
                    transition["contacts"] = self._watch_contact_probe()
                per["last_transition"] = transition
                if not ok and per["first_lost"] is None:
                    per["first_lost"] = transition
                    if block["first_slide_out"] is None:
                        block["first_slide_out"] = dict(transition)
                if len(block["events"]) < 200:
                    block["events"].append(transition)
            self._goal_watch_last_ok[step_id] = ok
            if not ok and per["first_lost"] is None and before is None:
                # A goal that never held during this step is not a slide-out;
                # it is recorded once as the starting state instead.
                per["never_held"] = True
        row["goals"] = states
        at_boundary = self.sim_actions % self.watch_every == 0
        if at_boundary:
            block["timeline"].append(row)
        elif block["events"] and block["events"][-1].get("skill_sim_actions") == self.sim_actions:
            # The transition row and the periodic row are the same state.
            block["timeline"].append(row)
        block["current"] = states

    # ---- earlier steps' goals (diagnostics only) -------------------------

    def _watch_dynamics(self, snap: dict[str, Any]) -> dict[str, Any] | None:
        """Orientation/velocity/contact evidence for the watched bodies.

        Reads the snapshot the loop already took: no extra round trip, no
        extra simulator action. ``tilt_deg`` is the angle between the body's
        own +z axis and world +z, which is what a bowl's "tilt" means visually.

        The composite names objects the way the catalog does
        (``akita_black_bowl_1``) while the simulator names the same body with a
        ``_main`` suffix. Both spellings are accepted and the row is keyed by
        the *catalog* name so a trace stays comparable across environment
        versions; ``body`` records which simulator body answered.
        """
        if not self.watch_objects:
            return None
        named = snap.get("named_poses") or {}
        body_dyn = named.get("dynamics") or {}
        if not isinstance(body_dyn, dict) or not body_dyn:
            return None

        def resolve(name: str) -> str | None:
            if not isinstance(name, str) or not name:
                return None
            if name in body_dyn:
                return name
            for candidate in (f"{name}_main", name[: -len("_main")] if name.endswith("_main") else name):
                if candidate in body_dyn:
                    return candidate
            for key in body_dyn:
                if key.startswith(f"{name}_"):
                    return key
            return None

        resolved: dict[str, str] = {}
        rows: dict[str, Any] = {}
        for name in self.watch_objects:
            body = resolve(name)
            if body is None:
                continue
            entry = body_dyn.get(body)
            if not isinstance(entry, dict):
                continue
            resolved[name] = body
            quat = entry.get("quat")
            tilt = None
            if isinstance(quat, (list, tuple)) and len(quat) == 4:
                w, x, y, z = (float(v) for v in quat)
                norm = (w * w + x * x + y * y + z * z) ** 0.5
                if norm > 1e-9:
                    w, x, y, z = w / norm, x / norm, y / norm, z / norm
                    cos_tilt = max(-1.0, min(1.0, 1.0 - 2.0 * (x * x + y * y)))
                    tilt = round(math.degrees(math.acos(cos_tilt)), 3)
            rows[name] = {
                "body": body,
                "pos": entry.get("pos"),
                "quat": None if quat is None else [round(float(v), 5) for v in quat],
                "tilt_deg": tilt,
                "vel": entry.get("vel"),
                "angvel": entry.get("angvel"),
            }
        # Relative velocity is the quantity a "was it thrown?" question needs:
        # a cheese moving with the bowl is not sliding in it.
        relatives: dict[str, Any] = {}
        for item in self.watch_goals:
            a = rows.get(item.get("object_n"))
            b = rows.get(item.get("target_n"))
            if not a or not b or a.get("vel") is None or b.get("vel") is None:
                continue
            relative = [round(float(a["vel"][i]) - float(b["vel"][i]), 5) for i in range(3)]
            relatives[str(item.get("step_id") or "")] = {
                "of": item.get("object_n"),
                "to": item.get("target_n"),
                "vel": relative,
                "speed": round(sum(v * v for v in relative) ** 0.5, 5),
            }
        # Only the pairs a transport question asks about: carried object
        # against its container, and the fingers against the carried object.
        pairs: list[dict[str, Any]] = []
        contacts = named.get("contacts")
        if isinstance(contacts, list):
            watched = set(resolved.values()) | set(self.watch_objects)
            for contact in contacts:
                if not isinstance(contact, dict):
                    continue
                one, two = contact.get("body1"), contact.get("body2")
                if one in watched or two in watched:
                    pairs.append(
                        {
                            "a": one,
                            "b": two,
                            "geom_a": contact.get("geom1"),
                            "geom_b": contact.get("geom2"),
                            "dist": contact.get("dist"),
                            "robot": bool(contact.get("robot1") or contact.get("robot2")),
                        }
                    )
        block: dict[str, Any] = {"bodies": rows}
        if relatives:
            block["relative_vel"] = relatives
        # Robot-vs-object pairs first: they are the ones that answer "did the
        # fingers touch the cheese while closing on the bowl". ``total`` keeps a
        # truncated list from being read as "no other contact existed".
        pairs.sort(
            key=lambda row: (
                0 if row.get("robot") else 1,
                row.get("dist") if isinstance(row.get("dist"), (int, float)) else 0.0,
            )
        )
        block["contacts"] = pairs[:24]
        block["contacts_total"] = len(pairs)
        block["contacts_available"] = contacts is not None
        inference = self.inference_rows[-1] if self.inference_rows else {}
        if isinstance(inference, dict):
            block["chunk_id"] = inference.get("chunk_id")
            block["buffer_index"] = inference.get("buffer_index")
            block["cache_hit"] = inference.get("cache_hit")
        return block

    def _watch(self, snap: dict[str, Any]) -> dict[str, Any] | None:
        """Record the first action after which an earlier goal joint moved.

        The baseline is the joint value at this step's first action, i.e. the
        state the handoff handed over. One contact read is taken at detection
        so the report can name the geom pair that touched the drawer.
        """
        if not self.watch_joints:
            return None
        joints = snap.get("joints") or {}
        now = {
            name: joints.get(name)
            for name in self.watch_joints
            if isinstance(joints.get(name), (int, float))
        }
        if not now:
            return None
        if not self.watch_baseline:
            self.watch_baseline = {name: float(value) for name, value in now.items()}
        moved = [
            name
            for name, value in now.items()
            if abs(float(value) - float(self.watch_baseline.get(name, value))) > 1e-4
        ]
        if moved and not self.watch_events:
            name = moved[0]
            event = {
                "sim_action_count": snap.get("sim_action_count"),
                "skill_sim_actions": self.sim_actions,
                "observation_id": snap.get("observation_id"),
                "joint": name,
                "before": round(float(self.watch_baseline.get(name, 0.0)), 6),
                "now": round(float(now[name]), 6),
                "delta": round(float(now[name]) - float(self.watch_baseline.get(name, 0.0)), 6),
                "eef_pos": (snap.get("sim") or {}).get("eef_pos"),
                "gripper_opening_m": (snap.get("sim") or {}).get("gripper_opening_m"),
                "contacts": self._watch_contact_probe(),
            }
            self.watch_events.append(event)
            self.journal("watched_joint_moved", {**event, "step_id": self.step_ref})
        return {name: round(float(value), 6) for name, value in now.items()}

    def _watch_contact_probe(self) -> dict[str, Any] | None:
        reader = getattr(self.env, "robot_state_inspect", None)
        if not callable(reader):
            return None
        try:
            row = reader()
        except Exception as exc:  # noqa: BLE001 - diagnostics only
            return {"error": f"{type(exc).__name__}: {exc}"}
        if not isinstance(row, dict) or not row.get("ok"):
            return {"error": row.get("reason_code") if isinstance(row, dict) else "no_reply"}
        state = (row.get("robot") or {}).get("state") or {}
        contacts = state.get("contacts") or {}
        return {
            "count": contacts.get("count"),
            "robot_vs_nonrobot": (contacts.get("robot_vs_nonrobot") or [])[:8],
            "gripper_vs_nonrobot": (contacts.get("gripper_vs_nonrobot") or [])[:8],
            "penetration_max_m": contacts.get("penetration_max_m"),
        }

    # ---- step goal (composite early step) --------------------------------

    def _observe_step_goal(self, snap: dict[str, Any]) -> dict[str, Any]:
        try:
            facts = completion.goal_state(snap, self.goal, self.task)
        except Exception as exc:  # noqa: BLE001 - evidence must not break the rollout
            facts = {"ok": False, "reason_code": "goal_evaluation_failed", "error": f"{type(exc).__name__}: {exc}"}
        facts = dict(facts)
        facts["native_goal"] = bool(snap.get("native_goal"))
        facts["sim_action_count"] = snap.get("sim_action_count")
        facts["observation_id"] = snap.get("observation_id")
        if facts.get("ok"):
            self.step_goal_streak += 1
            if not self.step_goal_latched and self.step_goal_streak >= self.goal_confirm_samples:
                self.step_goal_latched = True
                self.step_goal_events.append(
                    {
                        "sim_action_count": facts.get("sim_action_count"),
                        "observation_id": facts.get("observation_id"),
                        "predicate": facts.get("predicate"),
                        "confirm_samples": self.step_goal_streak,
                        "source": "engine_predicate",
                        "native_goal_at_latch": facts.get("native_goal"),
                    }
                )
                self.journal(
                    "step_goal_latched",
                    {
                        "step_ref": self.step_ref,
                        "sim_actions": self.sim_actions,
                        "predicate": facts.get("predicate"),
                        "native_goal_at_latch": facts.get("native_goal"),
                    },
                )
        else:
            self.step_goal_streak = 0
        return facts

    # ---- post-goal window ------------------------------------------------

    def _window_facts(self, snap: dict[str, Any]) -> dict[str, Any]:
        frame = grasp_geometry.finger_frame(snap, self.object_name)
        opening = grasp_geometry.opening_m(snap)
        separation = frame.get("separation_m")
        gap = frame.get("object_gap_m")
        outside = (
            isinstance(gap, (int, float))
            and isinstance(separation, (int, float))
            and float(gap) >= float(separation) / 2.0 + 0.01
        )
        open_now = isinstance(opening, (int, float)) and float(opening) >= self.release_open_m
        relation_ok = None
        if self.relation_step is not None:
            try:
                relation_ok = bool(
                    completion.goal_state(snap, self.relation_step, self.task).get("ok")
                )
            except Exception:  # noqa: BLE001 - geometry must not break the rollout
                relation_ok = None
        return {
            "sim_action_count": snap.get("sim_action_count"),
            "observation_id": snap.get("observation_id"),
            "native_goal": bool(snap.get("native_goal")),
            "gripper_opening_m": opening,
            "separation_m": separation,
            "object_gap_m": gap,
            "object_along_m": frame.get("object_along_m"),
            "object_eef_m": grasp_geometry.eef_object_m(snap, self.object_name),
            "object_pos": grasp_geometry.object_pos(snap, self.object_name),
            "eef_pos": grasp_geometry.eef_pos(snap),
            "outside_finger_span": bool(outside),
            "open_now": bool(open_now),
            "relation_ok": relation_ok,
        }

    def _window_observe(self, snap: dict[str, Any]) -> dict[str, Any] | None:
        if self.window_started_at is None:
            return None
        facts = self._window_facts(snap)
        if facts["open_now"]:
            self.window_open_streak += 1
            self.window_open_max = max(self.window_open_max or 0.0, float(facts["gripper_opening_m"]))
        else:
            self.window_open_streak = 0
        facts["open_streak"] = self.window_open_streak
        if facts.get("relation_ok") is False:
            self.window_relation_lost = True
        if (
            not self.window_release.get("verified")
            and self.window_open_streak >= self.release_confirm_samples
        ):
            self.window_release = {
                "verified": True,
                "samples": self.window_open_streak,
                "opening_m": facts["gripper_opening_m"],
                "opening_max_m": self.window_open_max,
                "separation_m": facts["separation_m"],
                "object_gap_m": facts["object_gap_m"],
                "object_eef_m": facts["object_eef_m"],
                "sim_action_count": facts["sim_action_count"],
                "observation_id": facts["observation_id"],
                "evidence_source": "policy_chunk",
            }
        return facts

    def _goal_decision(self) -> str:
        """What to do once the native goal holds: keep the policy running or stop.

        The window is bounded by config *and* by the already-generated chunk:
        no new plan is requested here, so every window action comes from the
        chunk the policy produced from a pre-goal observation.
        """
        if self.window_started_at is None:
            if self.relation_step is None:
                # No held object to release (drawer/stove task): the native
                # goal *is* the whole goal, and post-goal policy actions could
                # undo it (e.g. push a drawer closed again).
                self.stop_reason = "native_goal"
                return "break"
            if self.window_actions <= 0:
                self.stop_reason = "native_goal"
                return "break"
            if self.sim_actions == 0:
                self.stop_reason = "native_goal_before_actions"
                return "break"
            facts = self._window_facts(self.snapshot)
            if facts.get("relation_ok") is False:
                # Our own relation check disagrees with the native goal; do not
                # spend policy actions here. The finalizer re-checks the
                # candidate before it releases anything.
                self.stop_reason = "relation_unconfirmed"
                return "break"
            self.window_started_at = self.sim_actions
            self.window_goal_facts = facts
            self.window_open_streak = 1 if facts.get("open_now") else 0
            self.window_open_max = facts.get("gripper_opening_m") if facts.get("open_now") else None
            self.journal(
                "openwam_post_goal_window",
                {
                    "sim_actions": self.sim_actions,
                    "window_actions": self.window_actions,
                    "gripper_opening_m": facts.get("gripper_opening_m"),
                    "object_eef_m": facts.get("object_eef_m"),
                    "buffer_remaining": self.buffer_remaining,
                },
            )
            return "continue"
        if self.window_release.get("verified"):
            self.stop_reason = "policy_release"
            return "break"
        if self.window_relation_lost:
            # The policy kept moving and broke the placement relation before
            # releasing it: stop the window instead of letting it drift away.
            self.stop_reason = "relation_lost_in_window"
            return "break"
        if self.window_used >= self.window_actions:
            self.stop_reason = "window_exhausted"
            return "break"
        if self.buffer_remaining is not None and self.buffer_remaining <= 0:
            self.stop_reason = "chunk_exhausted"
            return "break"
        return "continue"

    def _record_observation(self, snap: dict[str, Any]) -> dict[str, Any]:
        sim = snap.get("sim") or {}
        opening = sim.get("gripper_opening_m")
        row = {
            "sim_action_count": snap.get("sim_action_count"),
            "observation_id": snap.get("observation_id"),
            "native_goal": bool(snap.get("native_goal")),
            "eef_pos": sim.get("eef_pos"),
            "gripper_opening_m": opening,
        }
        object_name = (self.task.get("objects") or [None])[0]
        if object_name:
            row["object_pose"] = ((sim.get("objects") or {}).get(object_name) or {}).get("pos")
        self.observations.append(row)
        if row["native_goal"] and not self.native_latched:
            self.native_latched = True
            self.native_events.append(
                {
                    "sim_action_count": row["sim_action_count"],
                    "observation_id": row["observation_id"],
                    "source": "native_libero_goal",
                }
            )
        if isinstance(opening, (int, float)):
            if self._last_opening is None:
                self._last_opening = float(opening)
            elif float(opening) - self._last_opening >= OPENING_DELTA_M:
                self.release_events.append(
                    {
                        "sim_action_count": row["sim_action_count"],
                        "observation_id": row["observation_id"],
                        "opening_before_m": round(self._last_opening, 4),
                        "opening_after_m": round(float(opening), 4),
                        "source": "measured_gripper_separation",
                    }
                )
                self._last_opening = float(opening)
            else:
                self._last_opening = max(self._last_opening, float(opening))
        return row

    def _raw_gripper(self, raw_action) -> tuple[str, float | None]:
        """Where the policy's *native* gripper command lives, per model.

        OpenWAM emits a 10-D EEF action whose gripper is slot 9; Pi0.5 emits the
        7-D OSC command whose gripper is slot 6.  Reading slot 9 unconditionally
        is exactly the kind of silent OpenWAM assumption that must not leak into
        the Pi0.5 path, so the slot is a seam instead of a literal.
        """
        value = np.asarray(raw_action, dtype=np.float64).reshape(-1)
        if value.shape[0] <= 9:
            return "grip_cmd_raw", None
        return "grip_cmd10", float(value[9])

    def _to_action7(self, raw_action) -> np.ndarray:
        """Model-specific action adapter.

        OpenWAM emits native EEF10 and needs the 6-D/rotation conversion; a
        backend whose native action already is the 7-D LIBERO command may
        override this and change nothing else in the rollout.
        """
        return native_eef10_to_libero7d(raw_action)

    def _observe_model_input(self) -> dict[str, Any]:
        started = time.perf_counter()
        raw = self.env.raw_obs()
        obs = build_openwam_obs(raw)
        self.observe_s += time.perf_counter() - started
        return obs

    def _infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        self._check_stop()
        request = request_row(
            token=self.token,
            episode_id=self.episode_id,
            scene_epoch=self.snapshot.get("scene_epoch"),
            snapshot=self.snapshot,
            skill="whole_task",
            step_id=self.sim_actions,
            prompt=self.prompt,
            step_ref=self.step_ref,
        )
        out = self.inference.infer(request, obs, timeout_s=self.timeout_s)
        if out.get("status") != "ok":
            raise RuntimeError(
                f"OpenWAM inference failed ({out.get('status')}): {out.get('error') or out.get('reason')}"
            )
        actions = np.asarray(out["actions"], dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != 10:
            raise RuntimeError(f"OpenWAM returned actions with shape {actions.shape}, expected [n,10] native EEF10")
        self.inference_s += float(out.get("inference_s") or 0.0)
        return {
            "actions": actions,
            "inference_s": out.get("inference_s"),
            "inference_total_s": out.get("total_s"),
            "queued_s": out.get("queued_s"),
            "encode_s": out.get("encode_s"),
            "send_s": out.get("send_s"),
            "recv_s": out.get("recv_s"),
            "server_latency_ms": out.get("server_latency_ms"),
            "server_step": out.get("server_step"),
            "chunk": out.get("chunk"),
            "action_raw": out.get("action_raw"),
        }

    # ---- timing summaries ----

    @staticmethod
    def _summary(values: list[Any]) -> dict[str, Any] | None:
        vals = sorted(float(v) for v in values if isinstance(v, (int, float)))
        if not vals:
            return None

        def at(p: float) -> float:
            return round(vals[min(len(vals) - 1, int(round(p * (len(vals) - 1))))], 4)

        return {
            "n": len(vals),
            "mean": round(sum(vals) / len(vals), 4),
            "min": round(vals[0], 4),
            "median": at(0.5),
            "p95": at(0.95),
            "max": round(vals[-1], 4),
        }

    def _chunk_stats(self) -> dict[str, Any]:
        """Separate real chunk generations from cached dequeues.

        The server's buffer means only 1 in ``execute_horizon`` requests runs the
        model; the rest are ~tens of ms of preprocessing + dequeue. Reporting a
        single average would mislabel the cached path as model latency.
        """
        rows = self.inference_rows
        generated = [r for r in rows if r.get("chunk_generation") is True]
        cached = [r for r in rows if r.get("cache_hit") is True]
        first = generated[0] if generated else None
        return {
            "requests": len(rows),
            "chunk_generations": len(generated),
            "cache_hits": len(cached),
            "chunk_ids": sorted({int(r["chunk_id"]) for r in rows if r.get("chunk_id") is not None}),
            "generated_actions_total": sum(int(r.get("generated_actions") or 0) for r in rows),
            "consumed_actions_total": sum(int(r.get("consumed_actions") or 0) for r in rows),
            "first_generation": None
            if first is None
            else {
                "server_latency_ms": first.get("server_latency_ms"),
                "client_inference_s": first.get("inference_s"),
                "engine_total_s": first.get("inference_total_s"),
                "encode_s": first.get("encode_s"),
            },
            "generation": {
                "server_latency_ms": self._summary([r.get("server_latency_ms") for r in generated]),
                "client_inference_s": self._summary([r.get("inference_s") for r in generated]),
                "engine_total_s": self._summary([r.get("inference_total_s") for r in generated]),
            },
            "cache": {
                "server_latency_ms": self._summary([r.get("server_latency_ms") for r in cached]),
                "client_inference_s": self._summary([r.get("inference_s") for r in cached]),
                "engine_total_s": self._summary([r.get("inference_total_s") for r in cached]),
            },
        }

    def _wall_timing(self, total: float) -> dict[str, Any]:
        rows = self.inference_rows

        def phase(key: str) -> float:
            return sum(float(r[key]) for r in rows if isinstance(r.get(key), (int, float)))

        sim_seconds = self.sim_actions * 0.05
        measured = sum(
            phase(key)
            for key in ("observe_s", "infer_wall_s", "convert_s", "env_step_s", "snapshot_s", "frame_s")
        )
        return {
            "total_s": round(total, 4),
            "sim_actions": self.sim_actions,
            "sim_time_s": round(sim_seconds, 3),
            "control_hz_wall": round(self.sim_actions / total, 2) if total > 0 else None,
            "sim_real_time_factor": round(sim_seconds / total, 3) if total > 0 else None,
            "first_action_s": None if self.first_action_s is None else round(self.first_action_s, 4),
            "per_action_total_s": self._summary([r.get("action_total_s") for r in rows]),
            "action_interval_s": self._summary(self.action_intervals),
            "phases_sum_s": {
                "observe": round(phase("observe_s"), 4),
                "inference_wall": round(phase("infer_wall_s"), 4),
                "convert": round(phase("convert_s"), 4),
                "env_step": round(phase("env_step_s"), 4),
                "snapshot": round(phase("snapshot_s"), 4),
                "frame": round(phase("frame_s"), 4),
            },
            "unattributed_s": round(max(0.0, total - measured), 4),
        }

    # ---- main loop ----

    def run(self) -> dict[str, Any]:
        started = time.perf_counter()
        self._snapshot()
        deadline = started + max(self.timeout_s, 60.0) * 20  # hard wall-clock guard
        previous_loop_start: float | None = None
        with GpuSampler(self.gpu_index) as sampler:
            while True:
                loop_start = time.perf_counter()
                self._check_stop()
                if self.sim_actions >= self.max_steps:
                    break
                if loop_start > deadline:
                    raise RuntimeError(f"OpenWAM rollout exceeded its wall-clock guard after {self.sim_actions} actions")
                if self.goal is not None:
                    if self.step_goal_latched:
                        self.stop_reason = self.stop_reason or "step_goal"
                        break
                elif self._last_state.get("native_goal") and self._goal_decision() == "break":
                    break
                model_obs = self._observe_model_input()
                observe_done = time.perf_counter()
                row = self._infer(model_obs)
                infer_done = time.perf_counter()

                raw_action = row["actions"][0]
                convert_started = time.perf_counter()
                action7 = self._to_action7(raw_action)
                convert_s = time.perf_counter() - convert_started
                self.convert_s += convert_s

                self._check_stop()
                step_started = time.perf_counter()
                if self.first_action_s is None:
                    self.first_action_s = step_started - started
                obs, reward, term, trunc, info = self.env.step(action7)
                env_step_s = time.perf_counter() - step_started
                self.env_step_s += env_step_s
                self.sim_actions += 1
                in_window = self.window_started_at is not None
                if in_window:
                    self.window_used += 1
                if self.stop_event.is_set():
                    # The action was already handed to the simulator when the
                    # stop flag rose; nothing new is issued from here on.
                    self.journal(
                        "openwam_stop_in_flight",
                        {"sim_actions": self.sim_actions, "note": "stop raised during env.step"},
                    )
                    raise Stopped("stop requested during env.step")
                snapshot_started = time.perf_counter()
                snap = self._snapshot()
                snapshot_done = time.perf_counter()
                frame_started = time.perf_counter()
                if self.frame_sink is not None:
                    self.frame_sink(obs, snap)
                frame_s = time.perf_counter() - frame_started
                self.frame_s += frame_s
                loop_end = time.perf_counter()

                interval_s = None if previous_loop_start is None else loop_start - previous_loop_start
                if interval_s is not None:
                    self.action_intervals.append(interval_s)
                previous_loop_start = loop_start
                chunk = row.get("chunk") or {}
                horizon = chunk.get("execute_horizon")
                buffer_index = chunk.get("buffer_index")
                if isinstance(horizon, int) and isinstance(buffer_index, int):
                    self.buffer_remaining = max(0, int(horizon) - 1 - int(buffer_index))
                else:
                    self.buffer_remaining = None
                if in_window:
                    facts = self._last_window_facts or {}
                    grip_key, grip_value = self._raw_gripper(raw_action)
                    self.window_rows.append(
                        {
                            "n": self.sim_actions,
                            "window_n": self.window_used,
                            "grip_cmd7": round(float(action7[6]), 4),
                            grip_key: None if grip_value is None else round(grip_value, 4),
                            "gripper_opening_m": facts.get("gripper_opening_m"),
                            "separation_m": facts.get("separation_m"),
                            "object_gap_m": facts.get("object_gap_m"),
                            "object_eef_m": facts.get("object_eef_m"),
                            "object_pos": facts.get("object_pos"),
                            "native_goal": facts.get("native_goal"),
                            "outside_finger_span": facts.get("outside_finger_span"),
                            "relation_ok": facts.get("relation_ok"),
                            "open_streak": facts.get("open_streak"),
                            "cache_hit": chunk.get("cache_hit"),
                            "buffer_index": buffer_index,
                            "terminated": bool(term),
                            "truncated": bool(trunc),
                        }
                    )

                self.inference_rows.append(
                    {
                        "n": self.sim_actions,
                        "inference_s": row["inference_s"],
                        "inference_total_s": row.get("inference_total_s"),
                        "queued_s": row.get("queued_s"),
                        "encode_s": row.get("encode_s"),
                        "send_s": row.get("send_s"),
                        "recv_s": row.get("recv_s"),
                        "server_latency_ms": row.get("server_latency_ms"),
                        "server_step": row.get("server_step"),
                        "chunk_generation": chunk.get("chunk_generation"),
                        "cache_hit": chunk.get("cache_hit"),
                        "chunk_id": chunk.get("chunk_id"),
                        "buffer_index": chunk.get("buffer_index"),
                        "generated_actions": chunk.get("generated_actions"),
                        "consumed_actions": chunk.get("consumed_actions"),
                        "action_raw": row.get("action_raw"),
                        "action7": [round(float(v), 4) for v in action7],
                        "native_goal": bool(snap.get("native_goal")),
                        "reward": float(reward),
                        "terminated": bool(term),
                        "truncated": bool(trunc),
                        "observe_s": round(observe_done - loop_start, 4),
                        "infer_wall_s": round(infer_done - observe_done, 4),
                        "convert_s": round(convert_s, 4),
                        "env_step_s": round(env_step_s, 4),
                        "snapshot_s": round(snapshot_done - snapshot_started, 4),
                        "frame_s": round(frame_s, 4),
                        "action_total_s": round(loop_end - loop_start, 4),
                        "interval_s": None if interval_s is None else round(interval_s, 4),
                        "action_offset_s": round(step_started - started, 4),
                    }
                )
                self.action_trace.append(
                    {
                        "n": self.sim_actions,
                        "a7": [round(float(v), 4) for v in action7],
                        "a10": [round(float(v), 4) for v in raw_action],
                        "eef": (snap.get("sim") or {}).get("eef_pos"),
                        "grip": (snap.get("sim") or {}).get("gripper_opening_m"),
                        "watched": self._last_watched,
                    }
                )
                self.obs = obs
                if trunc:
                    self.stop_reason = self.stop_reason or "truncated"
                    self.journal("openwam_truncated", {"sim_action_count": snap.get("sim_action_count")})
                    break
            self.peak_gpu_mib = sampler.peak_mib

        total = time.perf_counter() - started
        latencies = [row["inference_s"] for row in self.inference_rows if isinstance(row.get("inference_s"), (int, float))]
        first = latencies[0] if latencies else None
        warm = sorted(latencies[1:]) if len(latencies) > 1 else []
        warm_median = warm[len(warm) // 2] if warm else None
        state = self._last_state
        native = bool(self._last_state.get("native_goal"))
        if self.goal is not None:
            ok = bool(self.step_goal_latched)
        else:
            ok = bool(native)
        if self.stop_reason is None:
            self.stop_reason = (
                ("step_goal" if ok else "budget")
                if self.goal is not None
                else ("native_goal" if ok else ("budget" if self.sim_actions >= self.max_steps else "unknown"))
            )
        chunk_stats = self._chunk_stats()
        wall = self._wall_timing(total)
        if self.goal is not None:
            facts = dict(self._goal_facts or {})
            verification = {
                "ok": ok,
                "method": f"engine_predicate:{facts.get('predicate')}",
                "predicate": facts.get("predicate"),
                "target": self.goal.get("target_n") or self.goal.get("relation"),
                "qpos": facts.get("qpos"),
                "threshold": facts.get("threshold"),
                "relation": facts.get("relation"),
                "height": facts.get("height"),
                "object_pose": facts.get("object_pose"),
                "native_goal": native,
                "native_goal_note": "the loaded scene's flag; not this step's goal",
                "samples": len(self.observations),
                "confirm_samples": self.goal_confirm_samples,
                "sim_actions": self.sim_actions,
                "limit": "max_steps" if (not ok and self.sim_actions >= self.max_steps) else None,
                "reason_code": None if ok else "step_goal_not_reached",
                "prompt": self.prompt,
            }
        else:
            verification = {
                "ok": ok,
                "method": "native_libero_goal",
                "native_goal": native,
                "samples": len(self.observations),
                "sim_actions": self.sim_actions,
                "limit": "max_steps" if (not ok and self.sim_actions >= self.max_steps) else None,
                "reason_code": None if ok else "native_goal_not_reached",
                "prompt": self.prompt,
            }
        return {
            "skill": "whole_task",
            "object_n": None,
            "target_n": None,
            "relation": "task",
            "prompt": self.prompt,
            "step_id": self.step_ref,
            "goal_kind": (self.goal or {}).get("skill") or "native_goal",
            "phase_reached": "step_goal" if (self.goal is not None and ok) else ("native_goal" if ok else "budget"),
            "ok": ok,
            "verification": verification,
            "step_goal_events": self.step_goal_events,
            "release": {
                "events": self.release_events,
                "count": len(self.release_events),
                "source": "measured_gripper_separation",
                "threshold_m": self.release_open_m,
                "max_opening_m": max(
                    [float(r["gripper_opening_m"]) for r in self.observations if isinstance(r.get("gripper_opening_m"), (int, float))],
                    default=None,
                ),
                "threshold_crossed": any(
                    float(r["gripper_opening_m"]) >= self.release_open_m
                    for r in self.observations
                    if isinstance(r.get("gripper_opening_m"), (int, float))
                ),
                "policy_window": self.window_release,
                "note": "policy-authored release measured from finger separation; not a Pi0.5 executor receipt",
            },
            "grasp": {
                "final_opening_m": (self.snapshot.get("sim") or {}).get("gripper_opening_m"),
                "min_opening_m": min(
                    [float(r["gripper_opening_m"]) for r in self.observations if isinstance(r.get("gripper_opening_m"), (int, float))],
                    default=None,
                ),
            },
            "native_events": self.native_events,
            "stop_reason": self.stop_reason,
            "post_goal": {
                "window_actions": self.window_actions,
                "actions_used": self.window_used,
                "stop_reason": self.stop_reason,
                "first_goal": self.window_goal_facts,
                "release": self.window_release,
                "rows": self.window_rows,
                "buffer_remaining_at_stop": self.buffer_remaining,
                "release_open_m": self.release_open_m,
                "release_confirm_samples": self.release_confirm_samples,
                "note": (
                    "actions the policy ran after the first native goal, taken only from the "
                    "chunk it had already generated; no new plan was requested"
                ),
            },
            "chunks_used": chunk_stats["chunk_generations"],
            "sim_actions": self.sim_actions,
            "max_steps": self.max_steps,
            "timing": {
                "total_s": round(total, 3),
                "inference_s_sum": round(self.inference_s, 3),
                "env_step_s_sum": round(self.env_step_s, 3),
                "snapshot_s_sum": round(self.snapshot_s, 3),
                "observe_s_sum": round(self.observe_s, 3),
                "convert_s_sum": round(self.convert_s, 3),
                "frame_s_sum": round(self.frame_s, 3),
                "first_inference_s": first,
                "warm_inference_median_s": warm_median,
                "first_action_s": None if self.first_action_s is None else round(self.first_action_s, 3),
                "peak_gpu_mib": self.peak_gpu_mib,
            },
            "chunk_stats": chunk_stats,
            "wall": wall,
            "inference_rows": self.inference_rows,
            "observations": self.observations,
            "action_trace": self.action_trace,
            "watch_joints": self.watch_joints,
            "watch_baseline": self.watch_baseline,
            "watch_events": self.watch_events,
            "goal_watch": self._goal_watch_record(),
            "final_state": state,
        }

    def _goal_watch_record(self) -> dict[str, Any]:
        """The carried-object watch, without the internal streak counters."""
        block = dict(self.goal_watch)
        per: dict[str, Any] = {}
        for key, row in (block.get("per_goal") or {}).items():
            row = dict(row)
            row.pop("_streak", None)
            if str(row.get("step_id")) == "":
                continue
            per[str(key)] = row
        block["per_goal"] = per
        block["carry"]["held_through"] = (
            all(bool(row.get("ok_last")) for row in per.values()) if per else None
        )
        return block


__all__ = ["GpuSampler", "WholeTaskRunner", "completion"]
