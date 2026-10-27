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

import subprocess
import threading
import time
from typing import Any, Callable

import numpy as np

from adapters.openwam_contract import build_openwam_obs, native_eef10_to_libero7d
from engine import completion
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

        self.prompt = str(step.get("prompt") or task.get("instruction") or "")
        self.sim_actions = 0
        self.inference_s = 0.0
        self.env_step_s = 0.0
        self.snapshot_s = 0.0
        self.observe_s = 0.0
        self.action_trace: list[dict[str, Any]] = []
        self.observations: list[dict[str, Any]] = []
        self.inference_rows: list[dict[str, Any]] = []
        self.release_events: list[dict[str, Any]] = []
        self.native_events: list[dict[str, Any]] = []
        self.native_latched = False
        self._last_opening: float | None = None
        self.peak_gpu_mib: int | None = None

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
        return snap

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
            "server_latency_ms": out.get("server_latency_ms"),
            "action_raw": out.get("action_raw"),
        }

    # ---- main loop ----

    def run(self) -> dict[str, Any]:
        started = time.perf_counter()
        self._snapshot()
        deadline = started + max(self.timeout_s, 60.0) * 20  # hard wall-clock guard
        with GpuSampler(self.gpu_index) as sampler:
            while True:
                self._check_stop()
                if self.sim_actions >= self.max_steps:
                    break
                if time.perf_counter() > deadline:
                    raise RuntimeError(f"OpenWAM rollout exceeded its wall-clock guard after {self.sim_actions} actions")
                if self._last_state.get("native_goal"):
                    break

                model_obs = self._observe_model_input()
                row = self._infer(model_obs)
                raw_action = row["actions"][0]
                action7 = native_eef10_to_libero7d(raw_action)

                self._check_stop()
                step_started = time.perf_counter()
                obs, reward, term, trunc, info = self.env.step(action7)
                self.env_step_s += time.perf_counter() - step_started
                self.sim_actions += 1
                if self.stop_event.is_set():
                    # The action was already handed to the simulator when the
                    # stop flag rose; nothing new is issued from here on.
                    self.journal(
                        "openwam_stop_in_flight",
                        {"sim_actions": self.sim_actions, "note": "stop raised during env.step"},
                    )
                    raise Stopped("stop requested during env.step")
                snap = self._snapshot()
                if self.frame_sink is not None:
                    self.frame_sink(obs, snap)

                self.inference_rows.append(
                    {
                        "n": self.sim_actions,
                        "inference_s": row["inference_s"],
                        "server_latency_ms": row.get("server_latency_ms"),
                        "action_raw": row.get("action_raw"),
                        "action7": [round(float(v), 4) for v in action7],
                        "native_goal": bool(snap.get("native_goal")),
                        "reward": float(reward),
                        "terminated": bool(term),
                        "truncated": bool(trunc),
                    }
                )
                self.action_trace.append(
                    {
                        "n": self.sim_actions,
                        "a7": [round(float(v), 4) for v in action7],
                        "a10": [round(float(v), 4) for v in raw_action],
                        "eef": (snap.get("sim") or {}).get("eef_pos"),
                        "grip": (snap.get("sim") or {}).get("gripper_opening_m"),
                    }
                )
                self.obs = obs
                if trunc:
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
        ok = bool(native)
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
            "phase_reached": "native_goal" if ok else "budget",
            "ok": ok,
            "verification": verification,
            "release": {
                "events": self.release_events,
                "count": len(self.release_events),
                "source": "measured_gripper_separation",
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
            "chunks_used": None,
            "sim_actions": self.sim_actions,
            "max_steps": self.max_steps,
            "timing": {
                "total_s": round(total, 3),
                "inference_s_sum": round(self.inference_s, 3),
                "env_step_s_sum": round(self.env_step_s, 3),
                "snapshot_s_sum": round(self.snapshot_s, 3),
                "observe_s_sum": round(self.observe_s, 3),
                "first_inference_s": first,
                "warm_inference_median_s": warm_median,
                "peak_gpu_mib": self.peak_gpu_mib,
            },
            "inference_rows": self.inference_rows,
            "observations": self.observations,
            "action_trace": self.action_trace,
            "final_state": state,
        }


__all__ = ["GpuSampler", "WholeTaskRunner", "completion"]
