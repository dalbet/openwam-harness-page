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
                if self._last_state.get("native_goal"):
                    break

                model_obs = self._observe_model_input()
                observe_done = time.perf_counter()
                row = self._infer(model_obs)
                infer_done = time.perf_counter()

                raw_action = row["actions"][0]
                convert_started = time.perf_counter()
                action7 = native_eef10_to_libero7d(raw_action)
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
        chunk_stats = self._chunk_stats()
        wall = self._wall_timing(total)
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
            "final_state": state,
        }


__all__ = ["GpuSampler", "WholeTaskRunner", "completion"]
