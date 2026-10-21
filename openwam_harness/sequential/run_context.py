"""One run owns its phase. Service health is not this object."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from queue import Empty, Queue
from typing import Any

ACTIVE = frozenset({"preparing", "planning", "executing", "waiting_confirmation", "cancelling"})
TERMINAL = frozenset({"succeeded", "failed", "cancelled"})
CONFIRM_TIMEOUT_S = 90.0


class RunContext:
    def __init__(self) -> None:
        self.phase = "idle"
        self.run_id = 0
        self.episode_id = ""
        self.scene_epoch = None
        self.user_text = ""
        self.root_error = ""
        self.root_code = ""
        self.plan_summary = ""
        self.events: list[dict[str, Any]] = []
        self.execution = None
        self.native_task_success = None
        self.user_goal_success = None
        self.user_goal_status = None
        self.execution_status = None
        self.retreat_error = None
        self.evidence_source = None
        self.subgoal_results = None
        self.first_error = ""
        self.verification_error = ""
        self.planner_stage = ""
        self.execution_started = False
        self.confirm_started_at = None
        self.confirm_q: Queue = Queue()
        self.thread: threading.Thread | None = None
        self.generation = 0
        self.finished = False
        self.execution_mode = "benchmark"

    def active(self) -> bool:
        return self.phase in ACTIVE

    def producer_alive(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def begin(self, run_id: int, episode_id: str, scene_epoch: int | None, user_text: str, execution_mode: str) -> None:
        self.generation += 1
        self.phase = "preparing"
        self.run_id = run_id
        self.episode_id = episode_id or ""
        self.scene_epoch = scene_epoch
        self.user_text = user_text
        self.root_error = ""
        self.root_code = ""
        self.plan_summary = ""
        self.events = []
        self.execution = None
        self.native_task_success = None
        self.user_goal_success = None
        self.user_goal_status = None
        self.execution_status = None
        self.retreat_error = None
        self.evidence_source = None
        self.subgoal_results = None
        self.first_error = ""
        self.verification_error = ""
        self.planner_stage = ""
        self.execution_started = False
        self.confirm_started_at = None
        self.confirm_q = Queue()
        self.finished = False
        self.execution_mode = execution_mode
        self.thread = None

    def set_phase(self, phase: str) -> None:
        if self.finished or phase not in ACTIVE:
            return
        if self.phase == "cancelling" and phase != "cancelling":
            return
        self.phase = phase

    def note_error(self, reason: str, code: str = "") -> None:
        if self.root_error or not reason:
            return
        self.root_error = str(reason)[:500]
        self.root_code = code

    def add_event(self, stage: str, status: str, detail: str = "", code: str = "") -> None:
        self.events.append(
            {
                "t": round(time.time(), 3),
                "stage": stage,
                "status": status,
                "code": code,
                "detail": str(detail)[:500],
            }
        )
        self.events = self.events[-80:]

    def finish(self, phase: str, reason: str = "", code: str = "") -> bool:
        """Idempotent. Does not claim the action producer has stopped."""
        if phase not in TERMINAL:
            raise ValueError(phase)
        self._wake()
        if self.finished:
            return False
        if reason:
            self.note_error(reason, code)
        self.finished = True
        self.phase = phase
        return True

    def attach(self, thread: threading.Thread) -> None:
        self.thread = thread

    def release(self, thread: threading.Thread) -> bool:
        """Drop ownership only after this producer has stopped."""
        if thread is not self.thread:
            return False
        if thread.is_alive():
            return False
        self.thread = None
        if not self.finished:
            self.finish("failed", "producer exited without a terminal state", "producer_exit")
        return True

    def complete_from_producer(self, callback) -> None:
        """Join this producer from another thread, then release and notify.

        Calling ``release`` on the producer itself always sees ``is_alive()``.
        The callback runs only after that check can succeed.
        """
        thread = threading.current_thread()
        if self.thread is not None and thread is not self.thread:
            callback(False)
            return

        def watch() -> None:
            thread.join()
            callback(self.release(thread))

        threading.Thread(target=watch, name="run-release", daemon=True).start()

    def wait_confirm(self, timeout: float = CONFIRM_TIMEOUT_S):
        if self.phase == "cancelling" or self.finished:
            return None
        self.phase = "waiting_confirmation"
        self.confirm_started_at = time.time()
        generation = self.generation
        try:
            verdict = self.confirm_q.get(timeout=timeout)
            timed_out = False
        except Empty:
            verdict = None
            timed_out = True
        if generation != self.generation or self.finished:
            return None
        if self.phase == "cancelling":
            return None
        if timed_out:
            self.note_error("确认超时，已结束等待", "confirm_timeout")
            return "timeout"
        if self.phase == "waiting_confirmation":
            self.phase = "executing"
        return verdict

    def submit_confirm(self, verdict: str) -> bool:
        if self.phase != "waiting_confirmation":
            return False
        self.confirm_q.put(verdict)
        return True

    def cancel(self) -> None:
        if self.active():
            self.phase = "cancelling"
        self._wake()

    def _wake(self) -> None:
        try:
            self.confirm_q.put_nowait(None)
        except Exception:
            return

    def public(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "run_id": self.run_id,
            "episode_id": self.episode_id,
            "scene_epoch": self.scene_epoch,
            "user_text": self.user_text,
            "root_error": self.root_error,
            "root_code": self.root_code,
            "plan_summary": self.plan_summary,
            "execution": self.execution,
            "native_task_success": self.native_task_success,
            "user_goal_success": self.user_goal_success,
            "user_goal_status": self.user_goal_status,
            "execution_status": self.execution_status,
            "retreat_error": self.retreat_error,
            "evidence_source": self.evidence_source,
            "subgoal_results": self.subgoal_results,
            "first_error": self.first_error,
            "verification_error": self.verification_error,
            "planner_stage": self.planner_stage,
            "execution_started": self.execution_started,
            "execution_mode": self.execution_mode,
            "awaiting_confirm": self.phase == "waiting_confirmation",
            "confirm_started_at": self.confirm_started_at,
            "confirm_timeout_s": CONFIRM_TIMEOUT_S,
            "producer_alive": self.producer_alive(),
        }

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        payload = self.public()
        payload["events"] = list(self.events)
        path = directory / f"{self.episode_id or 'none'}-{self.run_id}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
