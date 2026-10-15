"""VLA scheduling. Display frames on another thread are not VLA async.

The live Pi0.5 path is ``predict`` and then ``chunk_step`` on the environment
thread. A prefetch that is discarded and followed by another synchronous
predict is not issued. The page reports synchronous execution.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any

# Reviewed order in robots/libero/tools.py LiberoPrimitives._vlm_chunk.
CALL_ORDER = (
    {"name": "model.predict", "kind": "vla_infer", "thread": "env", "overlaps_env_step": False},
    {"name": "vla.predict RPC", "kind": "rpc", "thread": "env", "inside": "model.predict"},
    {"name": "env.chunk_step", "kind": "env_step", "thread": "env", "after_predict": True},
    {"name": "render_camera 512", "kind": "render_512", "thread": "env", "counts_as_vla_async": False},
    {"name": "next tool", "kind": "tool_switch", "thread": "env"},
)

RTC_ASSESSMENT = {
    "backend": "RLinf OpenPI Pi05VLAFacade.predict",
    "call": "predict_action_batch(obs, mode)",
    "accepts_action_prefix": False,
    "accepts_inference_delay": False,
    "rtc": False,
    "thread_prefetch_is_rtc": False,
    "unaligned_policy": "sync",
    "reason": "The server predict method has no action prefix, delay, or RTC argument. A background prefetch is not RTC.",
}

_CONTROLLER = None
_LOCK = threading.Lock()


def execution_mode() -> str:
    value = os.environ.get("VLA_ASYNC", "sync").strip().lower()
    return "async" if value == "async" else "sync"


def get_controller() -> "VLAController":
    global _CONTROLLER
    with _LOCK:
        if _CONTROLLER is None:
            _CONTROLLER = VLAController(mode=execution_mode())
        return _CONTROLLER


def classify_span(name: str) -> str:
    text = name.lower()
    if "display" in text or "frame_bus" in text:
        return "display"
    if "render" in text or "512" in text:
        return "render_512"
    if "rpc" in text or "vla.predict" in text:
        return "rpc"
    if "predict" in text or text.startswith("vla"):
        return "vla_infer"
    if "chunk" in text or "env.step" in text or "env_step" in text:
        return "env_step"
    if "tool" in text or "switch" in text:
        return "tool_switch"
    return "other"


def control_stalls(spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Longest blocking spans on the env thread. Display work is excluded."""
    blocking = []
    for span in spans:
        kind = span.get("kind") or classify_span(str(span.get("name") or ""))
        if kind == "display":
            continue
        blocking.append({**span, "kind": kind, "vla_async": False})
    blocking.sort(key=lambda item: float(item.get("ms") or 0), reverse=True)
    return blocking[:2]


class VLAController:
    """One in-flight prediction and a short action queue with an explicit owner."""

    def __init__(self, mode: str = "sync", max_depth: int = 8) -> None:
        self.mode = "async" if mode == "async" else "sync"
        self.max_depth = max(1, int(max_depth))
        self.queue: list[dict[str, Any]] = []
        self.inflight: dict[str, Any] | None = None
        self.token: tuple | None = None
        self.closed = False
        self.events: list[dict[str, Any]] = []
        self.underrun_waits: list[dict[str, Any]] = []
        self.env_version = 0
        self._lock = threading.Lock()

    def arm(self, episode_id: str, scene_epoch: int, plan_version: int, skill: str) -> None:
        token = (str(episode_id), int(scene_epoch), int(plan_version), str(skill))
        with self._lock:
            if self.token != token:
                self._clear_locked("skill_or_version")
            self.token = token
            self.closed = False

    def clear(self, reason: str) -> None:
        with self._lock:
            self._clear_locked(reason)
            if reason in {"terminated", "cancel", "reset", "replan"}:
                self.closed = True

    def begin_predict(self, obs_version: int, token: tuple) -> bool:
        """Allow one predict. A second caller is refused instead of being spliced in."""
        with self._lock:
            if self.closed or token != self.token:
                self.events.append({"kind": "discard", "reason": "stale_or_closed", "obs_version": obs_version})
                return False
            if self.inflight is not None:
                self.events.append({"kind": "reject_inflight", "obs_version": obs_version})
                return False
            self.inflight = {"obs_version": int(obs_version), "token": token, "started": time.perf_counter()}
            return True

    def finish_predict(self, actions: list, obs_version: int, token: tuple, *, apply: bool = True) -> bool:
        """Queue actions only when they still match the observation they were computed from."""
        with self._lock:
            self.inflight = None
            if not apply:
                self.events.append(
                    {
                        "kind": "discard",
                        "reason": "unaligned",
                        "obs_version": int(obs_version),
                        "env_version": int(self.env_version),
                        "rtc": False,
                        "fallback": "sync",
                    }
                )
                return False
            if self.closed or token != self.token:
                self.events.append({"kind": "discard", "reason": "stale_token", "obs_version": obs_version})
                return False
            if self.mode == "async" and int(obs_version) != int(self.env_version):
                self.events.append(
                    {
                        "kind": "discard",
                        "reason": "unaligned",
                        "obs_version": int(obs_version),
                        "env_version": int(self.env_version),
                        "rtc": False,
                        "fallback": "sync",
                    }
                )
                return False
            room = self.max_depth - len(self.queue)
            if room <= 0:
                self.events.append({"kind": "discard", "reason": "queue_full", "obs_version": obs_version})
                return False
            accepted = list(actions)[:room]
            for offset, action in enumerate(accepted):
                self.queue.append(
                    {
                        "action": action,
                        "action_index": offset,
                        "obs_version": int(obs_version),
                        "token": token,
                    }
                )
            self.events.append(
                {
                    "kind": "enqueue",
                    "count": len(accepted),
                    "obs_version": int(obs_version),
                    "queue_depth": len(self.queue),
                    "mode": self.mode,
                    "rtc": False,
                }
            )
            return True

    def pop_action(self, token: tuple) -> dict[str, Any] | None:
        with self._lock:
            if self.closed or token != self.token:
                return None
            if not self.queue:
                self.underrun_waits.append({"obs_version": self.env_version, "queue_depth": 0, "token": token})
                return None
            item = self.queue.pop(0)
            item["queue_depth_after"] = len(self.queue)
            return item

    def mark_env(self, version: int) -> None:
        with self._lock:
            self.env_version = int(version)

    def owner_step(self, chunk_step, actions: list, token: tuple, obs_version: int):
        """The env thread is the only stepper. A closed or stale token does not step."""
        with self._lock:
            allowed = (not self.closed) and token == self.token
        if not allowed:
            self.events.append({"kind": "skip_step", "reason": "closed_or_stale"})
            return None
        self.mark_env(int(obs_version) + len(list(actions)))
        return chunk_step(actions)

    def _clear_locked(self, reason: str) -> None:
        dropped = len(self.queue)
        self.queue.clear()
        self.inflight = None
        self.events.append({"kind": "clear", "reason": reason, "dropped": dropped})


def replay_queue(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize a recorded controller log. This does not execute actions."""
    return {
        "enqueued": sum(int(item.get("count") or 0) for item in events if item.get("kind") == "enqueue"),
        "discarded_unaligned": sum(1 for item in events if item.get("reason") == "unaligned"),
        "clears": [item.get("reason") for item in events if item.get("kind") == "clear"],
        "rtc_labels": [bool(item.get("rtc")) for item in events if "rtc" in item],
    }
