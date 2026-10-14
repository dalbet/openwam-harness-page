"""Schema-constrained sequential executor.

Tool execution, subtask verdict, and native environment success stay separate.
The executor does not call the planner after an action unless a local replan
was explicitly requested by recovery exhaustion, target ambiguity, or a new goal.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from recovery import RecoveryBudget, event_for, propose, signature
from schema import PlanError, resolve_args, split_move, validate_plan
from store import LocalizationCache, new_observation
from verify import THRESHOLDS, receipt, verify_step

WORLD_SKILLS = frozenset({"pi0_pick", "move_to", "set_gripper"})


class StaleResult(RuntimeError):
    pass


@dataclass
class ToolOutcome:
    executed: bool
    error: str | None = None
    grasp_signal: bool | None = None
    native_terminated: bool = False
    native_truncated: bool = False
    xyz: list[float] | None = None
    eef_pos: list[float] | None = None
    raw: dict[str, Any] = field(default_factory=dict)
    real: bool = True
    error_class: str | None = None
    env_step: int | None = None


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
        "tool_results": [step.get("tool_result") for step in steps],
        "subtask_results": [step.get("verification") for step in steps],
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
    on_event: Callable[[dict[str, Any]], None] | None = None,
    task: str | None = None,
    replan_fn: Callable[[dict[str, Any]], dict[str, Any] | None] | None = None,
) -> dict[str, Any]:
    plan = validate_plan(plan, task=task if task is not None else plan.get("goal"))
    started_epoch = session.scene_epoch if epoch is None else epoch
    started_version = session.plan_version if plan_version is None else plan_version
    bindings: dict[str, list[float]] = {}
    steps: list[dict] = []
    eef: list[float] | None = None
    seen_real_world = False
    llm_calls = 0
    cache = LocalizationCache()
    recovery = RecoveryBudget()
    support_z: dict[str, float] = {}
    replanned = False
    last_obs = _capture(backend, started_epoch)
    _remember_support(last_obs, support_z)

    def emit(phase: str, **detail: Any) -> None:
        if on_event is None:
            return
        on_event(
            {
                "phase": phase,
                "episode_id": session.episode_id,
                "scene_epoch": started_epoch,
                "plan_version": started_version,
                "plan_source": plan.get("plan_source"),
                **detail,
            }
        )

    def stale() -> bool:
        return (
            session.cancelled
            or session.scene_epoch != started_epoch
            or session.plan_version != started_version
        )

    emit("executing", detail="plan accepted")
    queue = list(plan["steps"])
    index = 0
    while index < len(queue):
        step = queue[index]
        if stale():
            return _discard("cancelled or epoch/plan_version changed", session, steps)
        if deadline_s is not None and clock() >= deadline_s:
            return _finish(
                "diagnostic_saved",
                "deadline reached; scene saved and execution stopped",
                session,
                started_epoch,
                started_version,
                steps,
                backend,
                llm_calls,
            )
        skill = step["skill"]
        emit("executing", step_id=step["step_id"], skill=skill, detail=step.get("subtask") or skill)
        if skill == "pause_for_confirm":
            if not seen_real_world:
                emit("failed", reason="confirmation_before_real_action", step_id=step["step_id"])
                return _finish(
                    "failed",
                    "confirmation_before_real_action",
                    session,
                    started_epoch,
                    started_version,
                    steps,
                    backend,
                    llm_calls,
                )
            subtask = step["subtask"]
            emit(
                "waiting_confirmation",
                step_id=step["step_id"],
                subtask=subtask,
                detail=subtask,
                reason="subtask is waiting for manual confirmation",
            )
            verdict = confirm(subtask)
            if stale():
                return _discard("stale confirmation discarded", session, steps)
            if verdict is None:
                return _discard("confirmation cancelled", session, steps)
            if verdict not in {"success", "failure", "unknown"}:
                raise PlanError(f"verdict must be success, failure, or unknown: {verdict}")
            steps.append(
                {
                    "id": step["step_id"],
                    "step_id": step["step_id"],
                    "action": skill,
                    "skill": skill,
                    "subtask": subtask,
                    "tool_executed": False,
                    "tool_result": None,
                    "subtask_status": verdict,
                    "verification": None,
                    "confirmation_source": "manual",
                    "manual_confirmation": {
                        "verdict": verdict,
                        "step_id": step["step_id"],
                        "episode_id": session.episode_id,
                        "scene_epoch": started_epoch,
                        "plan_version": started_version,
                    },
                    "grasp_signal": None,
                    "native_terminated": bool(backend.native_success()),
                }
            )
            index += 1
            continue

        try:
            args = resolve_args(step.get("args") or {}, bindings)
        except PlanError as exc:
            outcome = ToolOutcome(executed=False, error=str(exc), real=False, error_class="plan")
            steps.append(_step_record(step, outcome, None))
            emit("failed", step_id=step["step_id"], reason=str(exc))
            return _finish(
                "no_progress",
                str(exc),
                session,
                started_epoch,
                started_version,
                steps,
                backend,
                llm_calls,
            )
        before_step = _env_step(backend)
        calls = _calls_for(skill, args, eef)
        outcome = _execute_with_one_retry(backend, calls)
        if outcome.executed and skill == "move_to":
            eef = list(args["xyz"])
        elif outcome.eef_pos is not None:
            eef = outcome.eef_pos
        if skill == "segment" and outcome.xyz is not None:
            prompt = str((step.get("args") or {}).get("prompt", step["step_id"]))
            bindings[f"segment:{prompt}"] = outcome.xyz
            cache.put(
                step.get("object_ref") or prompt,
                scene_epoch=started_epoch,
                camera=str((step.get("args") or {}).get("camera") or "agentview"),
                observation_id=str((last_obs or {}).get("observation_id") or prompt),
                xyz=outcome.xyz,
                env_step=outcome.env_step,
            )
        if outcome.real and skill in WORLD_SKILLS and outcome.executed:
            seen_real_world = True
            if step.get("object_ref"):
                cache.invalidate(skill, object_ref=step.get("object_ref"), scene_epoch=started_epoch)
        post, verification = _judge(
            step, outcome, backend, last_obs, started_epoch, before_step, plan, support_z,
        )
        if post:
            last_obs = post[-1]
            _remember_support(last_obs, support_z)
        verification, outcome, post = _recover_or_confirm(
            step=step,
            outcome=outcome,
            verification=verification,
            post=post,
            backend=backend,
            bindings=bindings,
            cache=cache,
            recovery=recovery,
            support_z=support_z,
            plan=plan,
            started_epoch=started_epoch,
            before_step=before_step,
            confirm=confirm,
            emit=emit,
            stale=stale,
            session=session,
            eef=eef,
        )
        if isinstance(verification, dict) and verification.get("status") == "discarded":
            return _discard(verification.get("reason") or "discarded", session, steps)
        if post:
            last_obs = post[-1]
        if verification.get("reason_code") == "RECOVERY_EXHAUSTED" and replan_fn is not None and not replanned:
            replanned = True
            llm_calls += 1
            replacement = replan_fn(
                {
                    "goal": plan.get("goal"),
                    "completed": [item["step_id"] for item in steps if item.get("verification", {}) and item["verification"].get("status") == "success"],
                    "failure": verification,
                    "observation": _public_observation(last_obs),
                    "remaining_budget_s": step.get("budget", {}).get("max_s"),
                }
            )
            if replacement:
                replacement = validate_plan(replacement, task=plan.get("goal"))
                session.plan_version += 1
                started_version = session.plan_version
                _invalidate_broken(steps, last_obs, plan, support_z, started_epoch)
                queue[index + 1 :] = replacement["steps"]
                emit("executing", detail="local replan", plan_version=started_version, recovery_attempts=recovery.attempts.get(step["step_id"], 0))
        confirmed = (verification.get("manual_confirmation") or {}).get("verdict") == "success"
        if (
            verification.get("status") != "success"
            and not confirmed
            and verification.get("reason_code") not in {"NOT_REAL"}
        ):
            if not (verification.get("reason_code") == "RECOVERY_EXHAUSTED" and replanned):
                steps.append(_step_record(step, outcome, verification, last_obs))
                emit(
                    "failed",
                    step_id=step["step_id"],
                    skill=skill,
                    reason=verification.get("reason_code") or outcome.error,
                    detail=outcome.error or verification.get("reason_code"),
                    evidence_source=verification.get("evidence_source"),
                    recovery_attempts=recovery.attempts.get(step["step_id"], 0),
                )
                status = "RECOVERY_EXHAUSTED" if verification.get("reason_code") == "RECOVERY_EXHAUSTED" else "no_progress"
                return _finish(status, verification.get("reason_code") or outcome.error or "failed", session, started_epoch, started_version, steps, backend, llm_calls)
        steps.append(_step_record(step, outcome, verification, last_obs))
        emit(
            "executing",
            step_id=step["step_id"],
            skill=skill,
            detail=verification.get("reason_code"),
            verification_status=verification.get("status"),
            reason_code=verification.get("reason_code"),
            evidence_source=verification.get("evidence_source"),
            grasp_signal=outcome.grasp_signal,
            env_step=outcome.env_step,
            recovery_attempts=recovery.attempts.get(step["step_id"], 0),
        )
        index += 1

    return _finish("completed", "", session, started_epoch, started_version, steps, backend, llm_calls)


def _capture(backend: Any, scene_epoch: int) -> dict[str, Any]:
    raw = backend.observe() if hasattr(backend, "observe") else {}
    return new_observation(raw if isinstance(raw, dict) else {}, scene_epoch=scene_epoch)


def _remember_support(obs: dict | None, support_z: dict[str, float]) -> None:
    objects = ((obs or {}).get("sim") or {}).get("objects") or {}
    for name, item in objects.items():
        if name in support_z or not isinstance(item, dict) or not item.get("identified"):
            continue
        pos = item.get("pos")
        if isinstance(pos, list) and len(pos) >= 3:
            support_z[name] = float(pos[2])


def _judge(step, outcome, backend, before, scene_epoch, min_step, plan, support_z):
    if not outcome.executed:
        return [], receipt("failure", "tool_failed", "tool_return", [], {"error": outcome.error})
    post = [_capture(backend, scene_epoch)]
    verdict = verify_step(
        step,
        outcome,
        post,
        scene_epoch=scene_epoch,
        min_step=min_step,
        objects=plan.get("objects") or {},
        support_z=support_z,
        goal_relation=plan.get("goal_relation"),
        before=before,
    )
    if verdict["reason_code"] == "NEED_STABILITY_SAMPLE":
        post.extend(_hold_window(backend, scene_epoch))
        verdict = verify_step(
            step,
            outcome,
            post,
            scene_epoch=scene_epoch,
            min_step=min_step,
            objects=plan.get("objects") or {},
            support_z=support_z,
            goal_relation=plan.get("goal_relation"),
            before=before,
        )
    return post, verdict


def _hold_window(backend: Any, scene_epoch: int) -> list[dict[str, Any]]:
    """Advance the sim with the existing close-gripper primitive, then observe."""
    grasp = THRESHOLDS["grasp"]
    calls = int(grasp.get("window_calls", 2))
    steps = int(grasp.get("window_steps", 4))
    gripper = float(grasp.get("window_gripper", 1.0))
    samples = []
    for _ in range(calls):
        try:
            outcome = backend.execute("set_gripper", {"gripper": gripper, "steps": steps})
        except Exception:
            break
        samples.append(_capture(backend, scene_epoch))
        if not getattr(outcome, "executed", False):
            break
    return samples


def _recover_or_confirm(
    *,
    step,
    outcome,
    verification,
    post,
    backend,
    bindings,
    cache,
    recovery,
    support_z,
    plan,
    started_epoch,
    before_step,
    confirm,
    emit,
    stale,
    session,
    eef,
):
    if outcome.executed and not outcome.real:
        failed = receipt("failure", "NOT_REAL", "tool_return", [], {"real": False})
        return failed, outcome, post
    if verification["status"] == "unknown":
        return _confirm_unknown(step, verification, confirm, emit, stale, session, outcome, post)
    event = event_for(verification, outcome)
    if event is None or step.get("on_failure") != "recover":
        return verification, outcome, post
    for attempt_index in range(recovery.limit):
        if verification["status"] != "failure":
            break
        holding = _holding(post[-1] if post else None, step, plan, support_z)
        opening = _opening(post[-1] if post else None)
        proposal = propose(
            event,
            step,
            budget=recovery,
            holding=holding,
            released=opening is not None and opening >= 0.05,
            stalled=event == "NO_PROGRESS",
            env_step_changed=_env_step(backend) is not None and before_step is not None and _env_step(backend) != before_step,
            attempt_index=attempt_index,
        )
        if proposal["kind"] == "service_retry":
            if not recovery.allow_service(step["step_id"]):
                verification = receipt("failure", "SERVICE_ERROR", "tool_return", [], {})
                break
            recovery.mark_service(step["step_id"])
        if proposal["kind"] != "actions" and proposal["kind"] != "service_retry":
            verification = receipt("failure", "RECOVERY_EXHAUSTED", "sim_state_assisted", [], {"event": event})
            break
        if proposal["kind"] == "actions" and not recovery.consume(step["step_id"]):
            verification = receipt("failure", "RECOVERY_EXHAUSTED", "sim_state_assisted", [], {"event": event})
            break
        moved = False
        for action in proposal["actions"]:
            try:
                args = resolve_args(action["args"], bindings)
            except PlanError:
                continue
            sig = signature(action["skill"], args)
            if action["skill"] in WORLD_SKILLS and not recovery.allow(step["step_id"], sig):
                continue
            follow = _execute_with_one_retry(backend, _calls_for(action["skill"], args, eef))
            if action["skill"] in WORLD_SKILLS:
                recovery.note(step["step_id"], sig)
            moved = True
            outcome = follow
            if action["skill"] == "segment" and follow.xyz is not None:
                prompt = str(action["args"].get("prompt") or "")
                bindings[f"segment:{prompt}"] = follow.xyz
            if follow.eef_pos is not None:
                eef = follow.eef_pos
            post.append(_capture(backend, started_epoch))
        if not moved:
            verification = receipt("failure", "RECOVERY_EXHAUSTED", "sim_state_assisted", [], {"event": event})
            break
        verification = verify_step(
            step,
            outcome,
            post,
            scene_epoch=started_epoch,
            min_step=before_step,
            objects=plan.get("objects") or {},
            support_z=support_z,
            goal_relation=plan.get("goal_relation"),
            before=post[0] if post else None,
        )
        event = event_for(verification, outcome) or event
    if verification.get("status") == "unknown":
        return _confirm_unknown(step, verification, confirm, emit, stale, session, outcome, post)
    if verification.get("status") == "failure":
        verification = receipt("failure", "RECOVERY_EXHAUSTED", verification.get("evidence_source") or "sim_state_assisted", verification.get("observation_ids") or [], {"event": event})
    return verification, outcome, post


def _confirm_unknown(step, verification, confirm, emit, stale, session, outcome, post):
    emit(
        "waiting_confirmation",
        step_id=step["step_id"],
        subtask=step.get("subtask"),
        detail=verification["reason_code"],
        reason=verification["reason_code"],
    )
    user = confirm(step.get("subtask") or step["step_id"])
    if stale() or user is None:
        return {"status": "discarded", "reason": "stale confirmation discarded"}, outcome, post
    verification = dict(verification)
    verification["manual_confirmation"] = {
        "verdict": user,
        "step_id": step["step_id"],
        "observation_ids": verification.get("observation_ids"),
        "episode_id": session.episode_id,
        "scene_epoch": session.scene_epoch,
        "plan_version": session.plan_version,
    }
    if user != "success":
        verification["status"] = "failure"
        verification["reason_code"] = f"manual_{user}"
    return verification, outcome, post


def _holding(obs, step, plan, support_z) -> bool:
    spec = (plan.get("objects") or {}).get(step.get("object_ref") or "")
    sim = (obs or {}).get("sim") or {}
    if not isinstance(spec, dict):
        return False
    item = (sim.get("objects") or {}).get(spec.get("sim_name"))
    eef = sim.get("eef_pos")
    if not isinstance(item, dict) or not isinstance(eef, list):
        return False
    pos = item.get("pos")
    base = support_z.get(spec.get("sim_name"))
    if not isinstance(pos, list) or base is None:
        return False
    near = sum((float(pos[i]) - float(eef[i])) ** 2 for i in range(3)) ** 0.5 < 0.06
    return near and float(pos[2]) >= base + 0.03


def _opening(obs) -> float | None:
    value = ((obs or {}).get("sim") or {}).get("gripper_opening_m")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _invalidate_broken(steps, obs, plan, support_z, scene_epoch) -> None:
    for record in steps:
        verification = record.get("verification") or {}
        if verification.get("status") != "success":
            continue
        if verification.get("reason_code") != "GRASP_HELD":
            continue
        if not _holding(obs, record, plan, support_z):
            record["invalidated"] = True
            record["verification"] = dict(verification)
            record["verification"]["status"] = "failure"
            record["verification"]["reason_code"] = "INVALIDATED_BY_RECOVERY"


def _calls_for(skill: str, args: dict, eef: list[float] | None) -> list[tuple[str, dict]]:
    if skill != "move_to":
        return [(skill, args)]
    points = split_move(eef, list(args["xyz"]))
    calls = []
    for point in points:
        call = dict(args)
        call["xyz"] = point
        calls.append(("move_to", call))
    return calls


def _execute_with_one_retry(backend: Any, calls: list[tuple[str, dict]]) -> ToolOutcome:
    last = ToolOutcome(executed=False, error="no call", real=False)
    for attempt in range(2):
        failed = False
        for skill, args in calls:
            before = _env_step(backend)
            last = backend.execute(skill, args)
            after = _env_step(backend)
            if last.error_class == "service" and after is not None and before is not None and after > before:
                last.executed = False
                last.error = last.error or "service error after the action already changed the env step"
                return last
            if not last.executed:
                failed = True
                break
        if not failed:
            return last
        if attempt == 1 or last.error_class == "service":
            return last
    return last


def _env_step(backend: Any) -> int | None:
    reader = getattr(backend, "env_step", None)
    if reader is None:
        return None
    try:
        return reader()
    except Exception:
        return None


def _step_record(step: dict, outcome: ToolOutcome, verification: dict | None, observation: Any = None) -> dict[str, Any]:
    grasp = outcome.grasp_signal if step["skill"] == "pi0_pick" else None
    subtask_status = (verification or {}).get("status") or ("unknown" if outcome.executed else "failure")
    return {
        "id": step["step_id"],
        "step_id": step["step_id"],
        "action": step["skill"],
        "skill": step["skill"],
        "subtask": step.get("subtask"),
        "tool_executed": outcome.executed,
        "tool_result": {
            "executed": outcome.executed,
            "error": outcome.error,
            "error_class": outcome.error_class,
            "real": outcome.real,
            "grasp_signal": grasp,
            "native_terminated": outcome.native_terminated,
            "native_truncated": outcome.native_truncated,
            "env_step": outcome.env_step,
        },
        "subtask_status": subtask_status,
        "verification": verification,
        "confirmation_source": None,
        "grasp_signal": grasp,
        "native_terminated": outcome.native_terminated,
        "error": outcome.error,
        "observation": _public_observation(observation),
        "real": outcome.real,
    }


def _public_observation(observation: Any) -> dict | None:
    if not isinstance(observation, dict):
        return None
    return {
        "observation_id": observation.get("observation_id"),
        "scene_epoch": observation.get("scene_epoch"),
        "env_step": observation.get("env_step"),
        "timestamp": observation.get("timestamp"),
        "cameras": observation.get("cameras"),
        "llm": False,
    }


def _finish(status, reason, session, epoch, version, steps, backend, llm_calls) -> dict[str, Any]:
    # Official success is only the backend's LIBERO terminated flag.
    # Truncation, an unfinished plan, and planner finish do not set it.
    native = bool(backend.native_success())
    if status != "completed":
        native = False
    return {
        "status": status,
        "reason": reason,
        "episode_id": session.episode_id,
        "scene_epoch": epoch,
        "plan_version": version,
        "steps": steps,
        "native_success": native,
        "llm_calls_during_execution": llm_calls,
        "tool_results": [step.get("tool_result") for step in steps],
        "subtask_results": [step.get("verification") for step in steps],
    }
