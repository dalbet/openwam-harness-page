"""Schema-constrained sequential executor.

Tool execution, subtask verdict, and native environment success stay separate.
The executor does not call the planner after an action.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from schema import PlanError, resolve_args, split_move, validate_plan


class StaleResult(RuntimeError):
    pass


@dataclass
class ToolOutcome:
    executed: bool
    error: str | None = None
    grasp_signal: bool | None = None
    native_terminated: bool = False
    xyz: list[float] | None = None
    eef_pos: list[float] | None = None
    raw: dict[str, Any] = field(default_factory=dict)


class Session:
    def __init__(self, scene_id: str) -> None:
        self.scene_id = scene_id
        self.episode_id = uuid.uuid4().hex[:12]
        self.scene_epoch = 0
        self.plan_version = 1
        self.cancelled = False

    def begin_episode(self) -> str:
        self.scene_epoch += 1
        self.episode_id = uuid.uuid4().hex[:12]
        self.cancelled = False
        return self.episode_id

    def cancel(self) -> None:
        self.cancelled = True


def _discard(reason: str, session: Session, steps: list[dict]) -> dict[str, Any]:
    return {
        "status": "discarded",
        "reason": reason,
        "episode_id": session.episode_id,
        "scene_epoch": session.scene_epoch,
        "plan_version": session.plan_version,
        "steps": steps,
        "native_success": None,
        "llm_calls_during_execution": 0,
    }


def run_plan(
    plan: dict[str, Any],
    backend: Any,
    session: Session,
    confirm: Callable[[str], str | None],
    *,
    epoch: int | None = None,
    plan_version: int | None = None,
    deadline_s: float | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    validate_plan(plan)
    started_epoch = session.scene_epoch if epoch is None else epoch
    started_version = session.plan_version if plan_version is None else plan_version
    bindings: dict[str, list[float]] = {}
    steps: list[dict] = []
    eef: list[float] | None = None

    def stale() -> bool:
        return (
            session.cancelled
            or session.scene_epoch != started_epoch
            or session.plan_version != started_version
        )

    for step in plan["steps"]:
        if stale():
            return _discard("cancelled or epoch/plan_version changed", session, steps)
        if deadline_s is not None and clock() >= deadline_s:
            return {
                "status": "diagnostic_saved",
                "reason": "deadline reached; scene saved and execution stopped",
                "episode_id": session.episode_id,
                "scene_epoch": started_epoch,
                "plan_version": started_version,
                "steps": steps,
                "native_success": backend.native_success(),
                "llm_calls_during_execution": 0,
            }
        action = step["action"]
        if action == "pause_for_confirm":
            subtask = step["subtask"]
            verdict = confirm(subtask)
            if stale():
                return _discard("stale confirmation discarded", session, steps)
            if verdict is None:
                return _discard("confirmation cancelled", session, steps)
            if verdict not in {"success", "failure", "unknown"}:
                raise PlanError(f"verdict must be success, failure, or unknown: {verdict}")
            steps.append(
                {
                    "id": step["id"],
                    "action": action,
                    "subtask": subtask,
                    "tool_executed": False,
                    "subtask_status": verdict,
                    "grasp_signal": None,
                    "native_terminated": bool(backend.native_success()),
                }
            )
            continue

        try:
            args = resolve_args(step.get("args") or {}, bindings)
        except PlanError as first_error:
            try:
                args = resolve_args(step.get("args") or {}, bindings)
            except PlanError as exc:
                outcome = ToolOutcome(executed=False, error=str(exc))
                steps.append(_step_record(step, outcome))
                return _finish(
                    "no_progress",
                    str(first_error),
                    session,
                    started_epoch,
                    started_version,
                    steps,
                    backend,
                )
        calls = _calls_for(action, args, eef)
        outcome = _execute_with_one_retry(backend, calls)
        if outcome.executed and action == "move_to":
            eef = list(args["xyz"])
        elif outcome.eef_pos is not None:
            eef = outcome.eef_pos
        if action == "segment" and outcome.xyz is not None:
            prompt = str((step.get("args") or {}).get("prompt", step["id"]))
            bindings[f"segment:{prompt}"] = outcome.xyz
        if not outcome.executed:
            steps.append(_step_record(step, outcome))
            return _finish(
                "no_progress",
                outcome.error or "tool failed twice",
                session,
                started_epoch,
                started_version,
                steps,
                backend,
            )
        # Observation is refreshed by the backend. That does not call the planner.
        backend.observe()
        steps.append(_step_record(step, outcome))

    return _finish("completed", "", session, started_epoch, started_version, steps, backend)


def _calls_for(action: str, args: dict, eef: list[float] | None) -> list[tuple[str, dict]]:
    if action != "move_to":
        return [(action, args)]
    points = split_move(eef, list(args["xyz"]))
    calls = []
    for point in points:
        call = dict(args)
        call["xyz"] = point
        calls.append(("move_to", call))
    return calls


def _execute_with_one_retry(backend: Any, calls: list[tuple[str, dict]]) -> ToolOutcome:
    last = ToolOutcome(executed=False, error="no call")
    for attempt in range(2):
        failed = False
        for action, args in calls:
            last = backend.execute(action, args)
            if not last.executed:
                failed = True
                break
        if not failed:
            return last
        if attempt == 1:
            return last
    return last


def _step_record(step: dict, outcome: ToolOutcome) -> dict[str, Any]:
    grasp = outcome.grasp_signal if step["action"] == "pi0_pick" else None
    # A grasp signal or a tool return is never the subtask verdict.
    subtask_status = "unknown" if outcome.executed else "failure"
    return {
        "id": step["id"],
        "action": step["action"],
        "subtask": step.get("subtask"),
        "tool_executed": outcome.executed,
        "subtask_status": subtask_status,
        "grasp_signal": grasp,
        "native_terminated": outcome.native_terminated,
        "error": outcome.error,
    }


def _finish(status, reason, session, epoch, version, steps, backend) -> dict[str, Any]:
    return {
        "status": status,
        "reason": reason,
        "episode_id": session.episode_id,
        "scene_epoch": epoch,
        "plan_version": version,
        "steps": steps,
        "native_success": backend.native_success(),
        "llm_calls_during_execution": 0,
    }
