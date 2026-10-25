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
from verify import THRESHOLDS, call_failure_receipt, receipt, verify_step

WORLD_SKILLS = frozenset({"pi0_pick", "move_to", "set_gripper"})
NATIVE_FINISH_PHASES = frozenset({"descending", "releasing", "retreat"})


def _vla_arm(episode_id: str, scene_epoch: int, plan_version: int, skill: str) -> None:
    from vla_async import get_controller

    get_controller().arm(episode_id, scene_epoch, plan_version, skill)


def _vla_clear(reason: str) -> None:
    from vla_async import get_controller

    get_controller().clear(reason)


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
    category: str | None = None
    state_after: dict[str, Any] | None = None


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
    stop_after: str | None = None,
) -> dict[str, Any]:
    carried = {
        key: plan.pop(key)
        for key in ("request_id", "planner_stage", "episode_id", "replay_note")
        if key in plan
    }
    plan = validate_plan(plan, task=task if task is not None else plan.get("goal"))
    plan.update(carried)
    if plan.get("plan_source") == "model" and plan.get("execution_mode") == "free" and not plan.get("bindings_checked"):
        return _finish(
            "failed",
            "绑定契约未检查，不会开始动作",
            session,
            session.scene_epoch if epoch is None else epoch,
            session.plan_version if plan_version is None else plan_version,
            [],
            backend,
            0,
            plan=plan,
        )
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
    native_finish_scope = None
    native_finish_deadline = None
    native_finish_budget_s = None
    native_finish_start_count = None
    while index < len(queue):
        step = queue[index]
        if stale():
            _vla_clear("cancel")
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
        _vla_arm(session.episode_id, started_epoch, started_version, step.get("skill") or "")
        gate = native_execution_gate(plan, backend, step=step, steps=steps)
        if gate == "finish_native":
            budget = _finish_window_budget(queue[index:], step.get("subtask"))
            if budget <= 0:
                gate = "stop_native"
            else:
                if native_finish_scope is None:
                    native_finish_scope = step.get("subtask")
                    native_finish_budget_s = budget
                    native_finish_start_count = _env_step(backend)
                    native_finish_deadline = clock() + budget
                    emit(
                        "executing",
                        step_id=step["step_id"],
                        skill=step.get("skill"),
                        detail="NATIVE_GOAL_FINISH",
                        reason_code="NATIVE_GOAL",
                        evidence_source="libero_terminated",
                        native_finish_scope=native_finish_scope,
                        native_finish_budget_s=native_finish_budget_s,
                    )
        if gate == "stop_uncontinuable":
            return _finish(
                "failed",
                "episode_not_continuable",
                session,
                started_epoch,
                started_version,
                steps,
                backend,
                llm_calls,
                plan=plan,
            )
        if gate == "stop_native":
            _vla_clear("terminated")
            emit(
                "executing",
                step_id=step["step_id"],
                skill=step.get("skill"),
                detail="NATIVE_SUCCESS",
                reason_code="NATIVE_SUCCESS",
                evidence_source="libero_terminated",
            )
            if plan.get("execution_mode") == "free":
                return _finish(
                    "native_goal_stopped",
                    "NATIVE_GOAL_STOPPED",
                    session,
                    started_epoch,
                    started_version,
                    steps,
                    backend,
                    llm_calls,
                    plan=plan,
                )
            return _finish(
                "completed",
                "NATIVE_SUCCESS",
                session,
                started_epoch,
                started_version,
                steps,
                backend,
                llm_calls,
                plan=plan,
            )
        if gate == "record_native":
            emit(
                "executing",
                step_id=step["step_id"],
                skill=step.get("skill"),
                detail="native_task_success",
                reason_code="NATIVE_GOAL",
                evidence_source="libero_terminated",
            )
        if gate == "finish_native":
            if native_finish_deadline is not None and clock() >= native_finish_deadline:
                return _finish(
                    "failed",
                    "native_finish_budget_exhausted",
                    session,
                    started_epoch,
                    started_version,
                    steps,
                    backend,
                    llm_calls,
                    plan=plan,
                )
            step["_native_finish"] = True
            step["_native_finish_scope"] = native_finish_scope
            step["_native_finish_budget_s"] = native_finish_budget_s
            step["_native_finish_start_sim_action"] = native_finish_start_count
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

        _ensure_carry_offset(bindings, last_obs, step, plan)
        if _needs_preplace(step, plan):
            from placement import preplace_gate

            gate_result = preplace_gate(step, last_obs, plan, bindings)
            if not gate_result.get("ok"):
                outcome = ToolOutcome(
                    executed=False,
                    error=gate_result.get("summary") or gate_result.get("reason") or "preplace blocked",
                    real=False,
                    error_class="plan",
                )
                verification = receipt(
                    "failure",
                    "preplace_blocked",
                    "per_item",
                    [],
                    {
                        "summary": gate_result.get("summary"),
                        "error": gate_result.get("summary"),
                        "unmet": gate_result.get("unmet"),
                        "missing": gate_result.get("unmet"),
                        "evidence_sources": gate_result.get("evidence_sources"),
                        "released_required": False,
                        "placed_required": False,
                    },
                )
                steps.append(_step_record(step, outcome, verification, last_obs))
                emit(
                    "failed",
                    step_id=step["step_id"],
                    skill=skill,
                    reason=outcome.error,
                    reason_code="preplace_blocked",
                    execution=_diagnosis(step, outcome, verification, last_obs, plan),
                )
                return _finish(
                    "no_progress",
                    outcome.error,
                    session,
                    started_epoch,
                    started_version,
                    steps,
                    backend,
                    llm_calls,
                    plan=plan,
                    execution=_diagnosis(step, outcome, verification, last_obs, plan),
                )
        if _release_blocked(step, last_obs, plan, bindings):
            outcome = ToolOutcome(
                executed=False,
                error="释放前相对位姿已偏离抓取时的对象-末端关系",
                real=True,
                error_class="plan",
            )
            verification = receipt("failure", "GRASP_SLIP", "sim_state", [], {"error": outcome.error})
            _write_trace(backend, step, verification)
            steps.append(_step_record(step, outcome, verification, last_obs))
            emit(
                "failed",
                step_id=step["step_id"],
                skill=skill,
                reason=outcome.error,
                reason_code="GRASP_SLIP",
                detail=outcome.error,
                execution=_diagnosis(step, outcome, verification, last_obs, plan),
            )
            return _finish("no_progress", outcome.error, session, started_epoch, started_version, steps, backend, llm_calls)
        from placement import region_pose_ready, resolve_motion_step

        if skill == "move_to" and not region_pose_ready(step, last_obs, plan):
            last_obs = _capture(backend, started_epoch)
        prepared = resolve_motion_step(step, bindings, last_obs, plan) if skill == "move_to" else {"ok": True, "args": None}
        if skill != "move_to":
            try:
                prepared = {"ok": True, "args": resolve_args(step.get("args") or {}, bindings), "stage": skill}
            except PlanError as exc:
                prepared = {"ok": False, "code": "plan", "reason": str(exc), "stage": skill}
        if not prepared["ok"]:
            outcome = ToolOutcome(executed=False, error=prepared.get("reason") or "motion rejected", real=False, error_class="plan")
            record = _step_record(step, outcome, None)
            record["stage"] = prepared.get("stage") or skill
            record["motion"] = {key: value for key, value in prepared.items() if key != "args"}
            steps.append(record)
            verification = receipt(
                "failure",
                prepared.get("code") or "pose_not_provided",
                prepared.get("evidence_source") or "per_item",
                [],
                {
                    "summary": outcome.error,
                    "error": outcome.error,
                    "missing": ["target_region_pose"],
                    "unmet": [outcome.error],
                    "actual": {"target_region_pose": None, "region_key": prepared.get("region_key")},
                    "condition": "区域运动前必须读到本次观测中的放置 site，不能用分割点或设施原点代替",
                    "evidence_sources": {"target_region": prepared.get("evidence_source") or "pose_not_provided"},
                },
            )
            emit(
                "failed",
                step_id=step["step_id"],
                skill=skill,
                reason=outcome.error,
                reason_code=prepared.get("code"),
                execution=_diagnosis(step, outcome, verification, last_obs, plan),
            )
            if (step.get("args") or {}).get("motion") == "retreat":
                return _finish(
                    "retreat_error",
                    outcome.error,
                    session,
                    started_epoch,
                    started_version,
                    steps,
                    backend,
                    llm_calls,
                    plan=plan,
                    retreat_error=outcome.error,
                )
            return _finish(
                "no_progress",
                outcome.error,
                session,
                started_epoch,
                started_version,
                steps,
                backend,
                llm_calls,
                plan=plan,
            )
        if skill == "pi0_pick":
            from placement import grasp_placement_block

            blocked = grasp_placement_block(plan, step, last_obs)
            if blocked:
                outcome = ToolOutcome(executed=False, error=blocked.get("reason") or "placement not commandable", real=False, error_class="plan")
                verification = receipt(
                    "failure",
                    blocked.get("code") or "orientation_not_commandable",
                    "mujoco_site",
                    [],
                    {"summary": outcome.error, "error": outcome.error, "placement": blocked.get("placement")},
                )
                steps.append(_step_record(step, outcome, verification, last_obs))
                emit(
                    "failed",
                    step_id=step["step_id"],
                    skill=skill,
                    reason=outcome.error,
                    reason_code=blocked.get("code") or "orientation_not_commandable",
                    execution=_diagnosis(step, outcome, verification, last_obs, plan),
                )
                return _finish(
                    "no_progress",
                    outcome.error,
                    session,
                    started_epoch,
                    started_version,
                    steps,
                    backend,
                    llm_calls,
                    plan=plan,
                    execution=_diagnosis(step, outcome, verification, last_obs, plan),
                )
        args = prepared["args"]
        before_step = _env_step(backend)
        calls = _calls_for(skill, args, eef)
        outcome = _execute_with_one_retry(backend, calls)
        if outcome.eef_pos is not None and skill != "move_to":
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
        if outcome.executed and skill == "move_to":
            from placement import arrival
            from verify import _eef as _observed_eef

            observed = _observed_eef(post[-1]) if post else None
            check = arrival(args.get("xyz"), observed)
            metrics = dict((verification or {}).get("metrics") or {})
            metrics["arrival"] = check
            if isinstance(verification, dict):
                verification["metrics"] = metrics
            if not check["ok"]:
                verification = receipt(
                    "failure",
                    check["code"] or "MOTION_NOT_ARRIVED",
                    "sim_state",
                    (verification or {}).get("observation_ids") or [],
                    {
                        "summary": "动作后暂停：末端没有到达要求位置",
                        "error": check["reason"],
                        "unmet": [check["reason"]],
                        "actual": {"eef_pos": check.get("actual"), "requested_eef": check.get("requested")},
                        "condition": f"gap_m <= {check.get('tolerance_m')}",
                        "arrival": check,
                        "evidence_sources": {"eef": "sim_state"},
                    },
                )
            elif observed is not None:
                eef = observed
        if outcome.executed and _native_stop(plan, backend):
            verification = _as_native_success(verification)
        _write_trace(backend, step, verification)
        if verification.get("reason_code") == "NATIVE_SUCCESS" and _native_stop(plan, backend):
            _vla_clear("terminated")
            if post:
                last_obs = post[-1]
            steps.append(_step_record(step, outcome, verification, last_obs))
            emit(
                "executing",
                step_id=step["step_id"],
                skill=skill,
                detail="NATIVE_SUCCESS",
                verification_status="success",
                reason_code="NATIVE_SUCCESS",
                evidence_source="libero_terminated",
            )
            return _finish(
                "completed",
                "NATIVE_SUCCESS",
                session,
                started_epoch,
                started_version,
                steps,
                backend,
                llm_calls,
                plan=plan,
            )
        if post:
            last_obs = post[-1]
            _remember_support(last_obs, support_z)
        if (
            plan.get("execution_mode") == "free"
            and skill == "segment"
            and verification.get("reason_code") == "TARGET_NOT_IDENTIFIED"
            and not step.get("_point_retried")
        ):
            step["_point_retried"] = True
            projector = getattr(backend, "project_instance", None)
            point = projector(step.get("object_ref")) if projector else None
            if isinstance(point, dict) and point.get("ok"):
                emit(
                    "executing",
                    step_id=step["step_id"],
                    skill="segment",
                    detail="point retry",
                    evidence_source="sim_state",
                    reason_code="POINT_RETRY",
                )
                outcome = _execute_with_one_retry(
                    backend,
                    [(
                        "segment",
                        {
                            "point": [int(point["row"]), int(point["col"])],
                            "camera": str((step.get("args") or {}).get("camera") or "agentview"),
                            "object_ref": step.get("object_ref"),
                        },
                    )],
                )
                if outcome.xyz is not None:
                    prompt = str((step.get("args") or {}).get("prompt") or step.get("grounding_query") or "")
                    if prompt:
                        bindings[f"segment:{prompt}"] = outcome.xyz
                post, verification = _judge(
                    step, outcome, backend, last_obs, started_epoch, before_step, plan, support_z,
                )
                if post:
                    last_obs = post[-1]
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
        if isinstance(verification, dict) and verification.get("status") == "success" and verification.get("reason_code") == "GRASP_HELD":
            samples = (verification.get("metrics") or {}).get("samples") or []
            relative = samples[-1].get("relative_pos") if samples and isinstance(samples[-1], dict) else None
            ref = step.get("object_ref")
            if isinstance(ref, str) and isinstance(relative, list) and len(relative) >= 3:
                bindings[f"hold:{ref}"] = [float(value) for value in relative[:3]]
        if isinstance(verification, dict) and skill == "segment":
            place_xyz = (verification.get("metrics") or {}).get("place_xyz")
            prompt = str((step.get("args") or {}).get("prompt") or "")
            if isinstance(place_xyz, list) and prompt:
                bindings[f"segment:{prompt}"] = place_xyz
        if isinstance(verification, dict) and verification.get("status") == "clarify":
            steps.append(_step_record(step, outcome, verification, last_obs))
            reason = (verification.get("metrics") or {}).get("error") or verification.get("reason_code")
            emit(
                "awaiting_clarification",
                step_id=step["step_id"],
                skill=skill,
                detail=reason,
                reason=reason,
                reason_code=verification.get("reason_code"),
                candidates=(verification.get("metrics") or {}).get("candidates"),
            )
            return _finish(
                "needs_clarification",
                reason,
                session,
                started_epoch,
                started_version,
                steps,
                backend,
                llm_calls,
                plan=plan,
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
                _vla_clear("replan")
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
                diag = _diagnosis(step, outcome, verification, last_obs, plan)
                emit(
                    "failed",
                    step_id=step["step_id"],
                    skill=skill,
                    reason=diag["reason"],
                    reason_code=diag["reason_code"],
                    category=diag["category"],
                    detail=diag["reason"],
                    evidence_source=verification.get("evidence_source"),
                    recovery_attempts=recovery.attempts.get(step["step_id"], 0),
                    execution=diag,
                )
                if (step.get("args") or {}).get("motion") == "retreat":
                    status = "retreat_error"
                elif verification.get("reason_code") == "RECOVERY_EXHAUSTED":
                    status = "RECOVERY_EXHAUSTED"
                else:
                    status = "no_progress"
                return _finish(
                    status,
                    diag["reason"] or "failed",
                    session,
                    started_epoch,
                    started_version,
                    steps,
                    backend,
                    llm_calls,
                    execution=diag,
                    plan=plan,
                    retreat_error=diag["reason"] if status == "retreat_error" else None,
                )
        record = _step_record(step, outcome, verification, last_obs)
        if skill == "set_gripper" and step.get("completion_check") == "object_placed":
            record["stage"] = "place"
        elif (step.get("args") or {}).get("motion") == "retreat":
            record["stage"] = "retreat"
            record["retreat"] = prepared.get("retreat")
        elif prepared.get("stage"):
            record["stage"] = prepared.get("stage")
            record["motion"] = {key: prepared[key] for key in ("object_origin", "eef_origin", "hold_offset", "support_origin", "applied_hold_offset") if key in prepared}
        steps.append(record)
        if (step.get("args") or {}).get("motion") == "retreat" and verification.get("status") != "success":
            return _finish(
                "retreat_error",
                verification.get("reason_code") or "retreat failed",
                session,
                started_epoch,
                started_version,
                steps,
                backend,
                llm_calls,
                plan=plan,
                retreat_error=verification.get("reason_code") or "retreat failed",
            )
        if stop_after and step.get("completion_check") == stop_after:
            diag = _diagnosis(step, outcome, verification, last_obs, plan)
            diag["stopped"] = stop_after
            emit(
                "executing",
                step_id=step["step_id"],
                skill=skill,
                detail=f"stop_after {stop_after}",
                verification_status=verification.get("status"),
                reason_code=verification.get("reason_code"),
                category=diag["category"],
                execution=diag,
            )
            return _finish(
                "stopped_at_check",
                verification.get("reason_code") or stop_after,
                session,
                started_epoch,
                started_version,
                steps,
                backend,
                llm_calls,
                execution=diag,
            )
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

    return _finish("completed", "", session, started_epoch, started_version, steps, backend, llm_calls, plan=plan)


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
        return [], call_failure_receipt(outcome)
    fresh = _capture(backend, scene_epoch)
    returned = getattr(outcome, "state_after", None)
    if isinstance(returned, dict):
        fresh = new_observation(returned, scene_epoch=scene_epoch)
        fresh["state_source"] = returned.get("state_source") or "action_return"
    stop_native = _native_stop(plan, backend)
    if stop_native:
        # Benchmark keeps the geometric read of the action return. Do not
        # sample or retreat after the official goal. Free mode verifies the
        # release instead, and the executor gate bounds what may follow.
        verdict = verify_step(
            step,
            outcome,
            [fresh],
            scene_epoch=scene_epoch,
            min_step=min_step,
            objects=plan.get("objects") or {},
            support_z=support_z,
            goal_relation=_relation_for(plan, step),
            before=before,
        )
        return [fresh], _as_native_success(verdict)
    sampled = _observations_from_samples((outcome.raw or {}).get("samples"), scene_epoch)
    poses = fresh.get("named_poses") if isinstance(fresh, dict) else None
    if isinstance(poses, dict):
        for row in sampled:
            row.setdefault("named_poses", poses)
            row.setdefault("site_observation_id", fresh.get("observation_id"))
    post = (sampled + [fresh]) if sampled else [fresh]
    verdict = verify_step(
        step,
        outcome,
        post,
        scene_epoch=scene_epoch,
        min_step=min_step,
        objects=plan.get("objects") or {},
        support_z=support_z,
        goal_relation=_relation_for(plan, step),
        before=before,
    )
    if (
        verdict["reason_code"] == "NEED_STABILITY_SAMPLE"
        and step.get("completion_check") == "grasp_held"
        and (verdict.get("metrics") or {}).get("lifted")
    ):
        holder = getattr(backend, "hold_for_grasp", None)
        if holder is not None:
            post.extend(_observations_from_samples(holder(4), scene_epoch))
            verdict = verify_step(
                step,
                outcome,
                post,
                scene_epoch=scene_epoch,
                min_step=min_step,
                objects=plan.get("objects") or {},
                support_z=support_z,
                goal_relation=_relation_for(plan, step),
                before=before,
            )
    if (
        verdict["reason_code"] == "NEED_STABILITY_SAMPLE"
        and step.get("completion_check") != "grasp_held"
        and _sampling_allowed(plan, backend)
    ):
        # Keep the gripper where this step left it. A placement window must
        # not close the gripper, and a finished episode must not be stepped.
        if not step.get("_stability_window"):
            step["_stability_window"] = True
            gripper = float((step.get("args") or {}).get("gripper", -1.0))
            post.extend(_hold_window(backend, scene_epoch, gripper))
            verdict = verify_step(
                step,
                outcome,
                post,
                scene_epoch=scene_epoch,
                min_step=min_step,
                objects=plan.get("objects") or {},
                support_z=support_z,
                goal_relation=_relation_for(plan, step),
                before=before,
            )
        if verdict["reason_code"] == "NEED_STABILITY_SAMPLE":
            metrics = dict(verdict.get("metrics") or {})
            metrics["summary"] = "放置后失败：稳定样本来自同一环境状态"
            metrics["error"] = metrics["summary"]
            metrics["unmet"] = (metrics.get("unmet") or []) + ["distinct env_step samples < stable_samples"]
            verdict = receipt("failure", "stable_sample_unchanged", verdict.get("evidence_source") or "per_item", verdict.get("observation_ids") or [], metrics)
    elif (
        verdict["reason_code"] == "NEED_STABILITY_SAMPLE"
        and step.get("completion_check") != "grasp_held"
        and plan.get("execution_mode") != "free"
        and native_execution_gate(plan, backend) == "stop_native"
    ):
        verdict = _as_native_success(verdict)
    elif (
        verdict["reason_code"] == "NEED_STABILITY_SAMPLE"
        and step.get("completion_check") != "grasp_held"
        and native_execution_gate(plan, backend) == "stop_uncontinuable"
    ):
        metrics = dict(verdict.get("metrics") or {})
        metrics["summary"] = "放置后没有再采样：回合已经结束，缺失样本不交给人工确认"
        metrics["error"] = metrics["summary"]
        verdict = receipt("failure", "episode_not_continuable", verdict.get("evidence_source") or "per_item", verdict.get("observation_ids") or [], metrics)
    return post, verdict


def _relation_for(plan: dict[str, Any], step: dict[str, Any]) -> str | None:
    if plan.get("execution_mode") == "free":
        return step.get("relation") or plan.get("goal_relation")
    return plan.get("goal_relation")


def native_execution_gate(
    plan: dict[str, Any],
    backend: Any,
    *,
    step: dict[str, Any] | None = None,
    steps: list[dict[str, Any]] | None = None,
) -> str:
    """Decide what a native goal means for this plan.

    Benchmark stops at the goal. Free mode records it, and only the current
    transfer's descend/release/retreat may still run. Any other step stops.
    """
    checker = getattr(backend, "episode_continuable", None)
    if checker is not None and not bool(checker()):
        return "stop_uncontinuable"
    if not _native(backend):
        return "continue"
    if plan.get("execution_mode") == "free":
        if plan.get("task_scope") == "combo":
            return "record_native"
        if _native_finish_step(step, steps):
            return "finish_native"
        return "stop_native"
    if plan.get("task_scope") == "combo":
        return "record_native"
    return "stop_native"


def _native_stop(plan: dict[str, Any], backend: Any) -> bool:
    """True only when this plan's native goal must stop execution immediately."""
    return plan.get("execution_mode") != "free" and _native(backend)


def _native_finish_step(step: dict[str, Any] | None, steps: list[dict[str, Any]] | None) -> bool:
    if not isinstance(step, dict) or not steps:
        return False
    trigger = steps[-1].get("subtask") if isinstance(steps[-1], dict) else None
    if not trigger or step.get("subtask") != trigger:
        return False
    return step.get("transfer_phase") in NATIVE_FINISH_PHASES


def _finish_window_budget(remaining: list[dict[str, Any]], scope: str | None) -> float:
    total = 0.0
    for item in remaining:
        if item.get("subtask") != scope:
            break
        if item.get("transfer_phase") not in NATIVE_FINISH_PHASES:
            continue
        total += float((item.get("budget") or {}).get("max_s") or 0.0)
    return round(total, 3)


def _sampling_allowed(plan: dict[str, Any], backend: Any) -> bool:
    checker = getattr(backend, "episode_continuable", None)
    if checker is not None and not bool(checker()):
        return False
    return plan.get("execution_mode") == "free" or not _native(backend)


def _native(backend: Any) -> bool:
    checker = getattr(backend, "native_success", None)
    return bool(checker()) if checker is not None else False


def _native_stop_receipt() -> dict:
    return receipt(
        "success",
        "NATIVE_SUCCESS",
        "libero_terminated",
        [],
        {"stopped_before_extra_motion": True, "truncated_is_success": False},
    )


def _as_native_success(verification: dict) -> dict:
    """The LIBERO terminated flag is the task result. Keep the geometric reason beside it."""
    metrics = dict(verification.get("metrics") or {})
    metrics["geometric_reason_code"] = verification.get("reason_code")
    upgraded = receipt(
        "success",
        "NATIVE_SUCCESS",
        "libero_terminated",
        verification.get("observation_ids") or [],
        metrics,
    )
    return upgraded


def _hold_window(backend: Any, scene_epoch: int, gripper: float = -1.0) -> list[dict[str, Any]]:
    """Advance a non-grasp check without changing the gripper command."""
    checker = getattr(backend, "episode_continuable", None)
    if checker is not None and not bool(checker()):
        return []
    grasp = THRESHOLDS["grasp"]
    calls = int(grasp.get("window_calls", 2))
    steps = int(grasp.get("window_steps", 4))
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


def _observations_from_samples(samples, scene_epoch: int) -> list[dict[str, Any]]:
    """Turn chunk snapshots into observations. Identical versions are not repeated."""
    rows = []
    seen = set()
    for sample in samples or []:
        if not isinstance(sample, dict) or not isinstance(sample.get("sim"), dict):
            continue
        version = sample.get("env_action_count")
        if not isinstance(version, int) or version in seen:
            continue
        seen.add(version)
        rows.append({
            "observation_id": sample.get("observation_id") or f"sample-{version}",
            "scene_epoch": scene_epoch,
            "sim_action_count": sample.get("sim_action_count") if isinstance(sample.get("sim_action_count"), int) else version,
            "env_step": version,
            "step_clock": "sim_action",
            "state_source": sample.get("state_source") or "action_sample",
            "service_instance_id": sample.get("service_instance_id"),
            "scene_generation": sample.get("scene_generation"),
            "timestamp": sample.get("timestamp"),
            "hold_step": sample.get("hold_step"),
            "cameras": {
                "agentview": {"id": f"agentview:{version}"},
                "wrist": {"id": f"wrist:{version}"},
            },
            "sim": sample["sim"],
        })
    return rows


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
    if (
        plan.get("execution_mode") == "free"
        and verification.get("reason_code") == "TARGET_NOT_IDENTIFIED"
        and verification.get("status") != "success"
    ):
        failed = dict(verification)
        failed["status"] = "failure"
        return failed, outcome, post
    reread_codes = {
        "pose_not_provided",
        "state_service_error",
        "stale_snapshot",
        "pre_action_state",
        "sim_action_count_missing",
        "cross_scene",
        "cross_reset",
        "cross_service_instance",
    }
    if verification.get("reason_code") in reread_codes and not step.get("_state_reread"):
        step["_state_reread"] = True
        post.append(_capture(backend, started_epoch))
        verification = verify_step(
            step,
            outcome,
            post,
            scene_epoch=started_epoch,
            min_step=before_step,
            objects=plan.get("objects") or {},
            support_z=support_z,
            goal_relation=_relation_for(plan, step),
            before=post[0] if post else None,
        )
    if verification.get("status") == "clarify":
        return verification, outcome, post
    if verification.get("reason_code") in {"pose_not_provided", "state_service_error", "instance_not_found", "stale_observation", "stale_snapshot", "STALE_OBSERVATION"}:
        failed = dict(verification)
        failed["status"] = "failure"
        return failed, outcome, post
    if verification["status"] == "unknown":
        if step.get("completion_check") == "object_placed":
            failed = dict(verification)
            failed["status"] = "failure"
            return failed, outcome, post
        return _confirm_unknown(step, verification, confirm, emit, stale, session, outcome, post)
    event = event_for(verification, outcome)
    _remember_attempt(step, verification)
    handle = (verification.get("metrics") or {}).get("handle_xyz")
    if event == "DRAWER_STALLED" and isinstance(handle, list) and len(handle) >= 3:
        step["_entry_xyz"] = [float(handle[0]), float(handle[1]), float(handle[2]) + 0.05]
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
            if proposal.get("reason"):
                metrics = dict(verification.get("metrics") or {})
                metrics["recovery"] = proposal["reason"]
                verification = dict(verification)
                verification["metrics"] = metrics
            verification = _finish_recovery(step, verification, event)
            break
        if proposal["kind"] == "actions" and not recovery.consume(step["step_id"]):
            verification = _finish_recovery(step, verification, event)
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
            verification = _finish_recovery(step, verification, event)
            break
        verification = verify_step(
            step,
            outcome,
            post,
            scene_epoch=started_epoch,
            min_step=before_step,
            objects=plan.get("objects") or {},
            support_z=support_z,
            goal_relation=_relation_for(plan, step),
            before=post[0] if post else None,
        )
        _remember_attempt(step, verification)
        mapped = event_for(verification, outcome)
        if verification.get("status") == "failure" and mapped is None:
            break
        event = mapped or event
    if verification.get("status") == "unknown":
        if step.get("completion_check") == "object_placed":
            failed = dict(verification)
            failed["status"] = "failure"
            metrics = dict(failed.get("metrics") or {})
            metrics["checker_reason_code"] = verification.get("reason_code")
            metrics["summary"] = metrics.get("summary") or metrics.get("error") or verification.get("reason_code")
            failed["metrics"] = metrics
            return failed, outcome, post
        return _confirm_unknown(step, verification, confirm, emit, stale, session, outcome, post)
    if verification.get("status") == "failure" and event_for(verification, outcome):
        verification = _finish_recovery(step, verification, event)
    return verification, outcome, post


def _remember_attempt(step, verification) -> None:
    attempt = (verification.get("metrics") or {}).get("attempt") if isinstance(verification, dict) else None
    if not isinstance(attempt, dict):
        return
    step.setdefault("_attempt_log", [])
    if attempt not in step["_attempt_log"]:
        step["_attempt_log"].append(attempt)


def _finish_recovery(step, verification, event) -> dict:
    failed = _exhausted(verification, event)
    log = step.get("_attempt_log")
    if log:
        failed.setdefault("metrics", {})
        failed["metrics"]["attempts"] = list(log)
        failed["metrics"]["last_grasp_stage"] = (log[-1] or {}).get("grasp_stage")
    return failed


def _exhausted(verification: dict, event: str | None) -> dict:
    """Keep the last checker reason. RECOVERY_EXHAUSTED is the ending, not the cause."""
    metrics = dict(verification.get("metrics") or {})
    last_code = verification.get("reason_code")
    last_reason = metrics.get("error") or metrics.get("summary") or last_code
    kept = dict(metrics)
    kept["event"] = event
    kept["last_reason_code"] = last_code
    kept["last_reason"] = last_reason
    kept["last_unmet"] = metrics.get("unmet")
    kept["error"] = f"RECOVERY_EXHAUSTED: {last_reason}"
    kept["summary"] = kept["error"]
    return receipt(
        "failure",
        "RECOVERY_EXHAUSTED",
        verification.get("evidence_source") or "sim_state_assisted",
        verification.get("observation_ids") or [],
        kept,
    )


def _confirm_unknown(step, verification, confirm, emit, stale, session, outcome, post):
    metrics = verification.get("metrics") or {}
    reason = outcome.error or metrics.get("error") or verification["reason_code"]
    emit(
        "waiting_confirmation",
        step_id=step["step_id"],
        skill=step.get("skill"),
        subtask=step.get("subtask"),
        detail=reason,
        reason=reason,
        reason_code=verification["reason_code"],
        category=getattr(outcome, "category", None) or metrics.get("category") or "subtask_incomplete",
        execution={
            "step_id": step.get("step_id"),
            "skill": step.get("skill"),
            "category": getattr(outcome, "category", None) or metrics.get("category") or "subtask_incomplete",
            "reason": reason,
            "reason_code": verification["reason_code"],
            "args": {key: value for key, value in (step.get("args") or {}).items() if key != "code"},
            "object_ref": step.get("object_ref"),
            "grounding_query": step.get("grounding_query"),
            "vla_prompt": step.get("vla_prompt"),
            "return_summary": {
                "executed": outcome.executed,
                "error": outcome.error,
                "score": metrics.get("score"),
                "found": metrics.get("found"),
                "grasp_signal": outcome.grasp_signal,
                "native_terminated": outcome.native_terminated,
            },
            "observation_id": (verification.get("observation_ids") or [None])[-1],
            "record_index_note": "states.json 的序号是工具记录，不是仿真步数",
            "transport_started": step.get("completion_check") in {"object_transported", "object_placed"},
            "missing": metrics.get("missing"),
            "unmet": metrics.get("unmet"),
        },
    )
    user = confirm(step.get("subtask") or step["step_id"])
    if stale() or user is None:
        metrics = dict(verification.get("metrics") or {})
        metrics["checker_reason_code"] = verification.get("reason_code")
        metrics["error"] = metrics.get("error") or metrics.get("summary") or verification.get("reason_code")
        metrics["summary"] = "确认超时，已结束等待。活动步骤记为失败，不保持 running。"
        failed = dict(verification)
        failed["status"] = "failure"
        failed["reason_code"] = "confirm_timeout"
        failed["metrics"] = metrics
        return failed, outcome, post
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
        metrics = dict(verification.get("metrics") or {})
        metrics["checker_reason_code"] = verification.get("reason_code")
        verification["metrics"] = metrics
        verification["status"] = "failure"
        verification["reason_code"] = f"manual_{user}"
    return verification, outcome, post


def _ensure_carry_offset(bindings, obs, step, plan) -> None:
    """Record object-minus-eef so later targets move the object, not the eef origin."""
    ref = (step.get("args") or {}).get("carry_offset")
    if not isinstance(ref, str) or not ref:
        return
    spec = (plan.get("objects") or {}).get(ref) or {}
    sim_name = spec.get("sim_name") or ref
    item = (((obs or {}).get("sim") or {}).get("objects") or {}).get(sim_name) or {}
    pos = item.get("pos") if isinstance(item, dict) else None
    eef = ((obs or {}).get("sim") or {}).get("eef_pos")
    if not isinstance(pos, list) or not isinstance(eef, list) or len(pos) < 3 or len(eef) < 3:
        return
    bindings[f"carry:{ref}"] = [float(pos[i]) - float(eef[i]) for i in range(3)]


def _needs_preplace(step, plan) -> bool:
    """Fixture regions use the new gate. Older movable targets keep their own checks."""
    args = step.get("args") or {}
    target = (plan.get("objects") or {}).get(step.get("target_ref") or "")
    regional = isinstance(target, dict) and bool(target.get("place_site"))
    if not regional:
        return False
    if step.get("skill") == "move_to" and args.get("motion") == "vertical":
        return True
    return step.get("skill") == "set_gripper" and step.get("completion_check") == "object_placed" and float(args.get("gripper") or 0) < 0


def _release_blocked(step, obs, plan, bindings) -> bool:
    if step.get("skill") != "set_gripper" or step.get("completion_check") != "object_placed":
        return False
    if float((step.get("args") or {}).get("gripper") or 0) >= 0:
        return False
    ref = step.get("object_ref")
    offset = bindings.get(f"hold:{ref}") or bindings.get(f"carry:{ref}")
    if not isinstance(offset, list) or len(offset) < 3:
        return False
    spec = (plan.get("objects") or {}).get(ref) or {}
    sim_name = spec.get("sim_name") or ref
    item = (((obs or {}).get("sim") or {}).get("objects") or {}).get(sim_name) or {}
    pos = item.get("pos") if isinstance(item, dict) else None
    eef = ((obs or {}).get("sim") or {}).get("eef_pos")
    if not isinstance(pos, list) or not isinstance(eef, list):
        return False
    delta = sum((float(pos[i]) - float(eef[i]) - float(offset[i])) ** 2 for i in range(3)) ** 0.5
    return delta > float(THRESHOLDS["transport"]["hold_relative_m"])


def _write_trace(backend, step, verification) -> None:
    out = getattr(backend, "output_dir", None)
    if out is None or not isinstance(verification, dict):
        return
    if step.get("completion_check") not in {"grasp_held", "object_transported", "object_placed"}:
        return
    import json
    from pathlib import Path

    path = Path(out) / "verification_trace.jsonl"
    metrics = verification.get("metrics") or {}
    row = {
        "step_id": step.get("step_id"),
        "skill": step.get("skill"),
        "status": verification.get("status"),
        "reason_code": verification.get("reason_code"),
        "evidence_source": verification.get("evidence_source"),
        "relative_span_m": metrics.get("relative_span_m"),
        "relative_delta_m": metrics.get("relative_delta_m"),
        "support_object": metrics.get("support_object"),
        "support_baseline": metrics.get("support_baseline"),
        "possible_external_support": metrics.get("possible_external_support"),
        "contact_status": metrics.get("contact_status"),
        "samples": metrics.get("samples"),
        "missing": metrics.get("missing"),
        "unmet": metrics.get("unmet"),
        "actual": metrics.get("actual"),
        "condition": metrics.get("condition"),
        "summary": metrics.get("summary"),
        "evidence_sources": metrics.get("evidence_sources"),
        "observation_ids": verification.get("observation_ids"),
        "validator_version": verification.get("validator_version"),
        "parent_step_id": step.get("subtask"),
        "check_name": step.get("completion_check"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


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
            unclear = _action_unclear(last)
            if unclear and (after is None or before is None or after == before):
                last.executed = False
                last.error = (last.error or "动作超时或取消") + "；动作是否完成不明确，不重发、不 reset"
                last.category = "action_unclear"
                return last
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


def _action_unclear(outcome: ToolOutcome) -> bool:
    text = f"{outcome.error or ''} {outcome.error_class or ''}".lower()
    return any(marker in text for marker in ("timeout", "timed out", "cancel"))


def _env_step(backend: Any) -> int | None:
    reader = getattr(backend, "env_step", None)
    if reader is None:
        return None
    try:
        return reader()
    except Exception:
        return None


def _terminal_outcomes(steps: list[dict]) -> dict[str, Any]:
    place_rows = [
        item
        for item in steps
        if isinstance(item, dict)
        and (
            item.get("stage") == "place"
            or (item.get("action") == "set_gripper" and item.get("completion_check") == "object_placed")
        )
    ]
    released = []
    for item in place_rows:
        verification = item.get("verification") or {}
        metrics = verification.get("metrics") or {}
        if verification.get("reason_code") == "PLACED" and metrics.get("released_required"):
            released.append(item)
    first = place_rows[-1] if place_rows else None
    metrics = ((first or {}).get("verification") or {}).get("metrics") if first else {}
    observation = (first or {}).get("observation") or {}
    retreat_rows = [
        item
        for item in steps
        if isinstance(item, dict)
        and (
            item.get("stage") == "retreat"
            or (item.get("action") == "move_to" and item.get("completion_check") == "retreated")
        )
    ]
    return {
        "release": {
            "command_executed": any(bool(item.get("tool_executed")) for item in place_rows),
            "verified": bool(released),
            "reason_code": ((first or {}).get("verification") or {}).get("reason_code"),
            "observation_id": observation.get("observation_id"),
            "sim_action_count": observation.get("sim_action_count"),
            "gripper_opening_m": (metrics or {}).get("gripper_opening_m"),
        },
        "retreat_complete": any(
            ((item.get("verification") or {}).get("reason_code") == "RETREATED") for item in retreat_rows
        ),
    }


def _native_finish_window(steps: list[dict]) -> dict[str, Any] | None:
    rows = [item for item in steps if isinstance(item, dict) and item.get("native_finish")]
    if not rows:
        return None
    return {
        "scope": rows[0].get("native_finish_scope"),
        "budget_s": rows[0].get("native_finish_budget_s"),
        "start_sim_action": rows[0].get("native_finish_start_sim_action"),
        "steps": [item.get("step_id") for item in rows],
    }


def _step_record(step: dict, outcome: ToolOutcome, verification: dict | None, observation: Any = None) -> dict[str, Any]:
    grasp = outcome.grasp_signal if step["skill"] == "pi0_pick" else None
    subtask_status = (verification or {}).get("status") or ("unknown" if outcome.executed else "failure")
    return {
        "id": step["step_id"],
        "step_id": step["step_id"],
        "action": step["skill"],
        "skill": step["skill"],
        "subtask": step.get("subtask"),
        "completion_check": step.get("completion_check"),
        "transfer_phase": step.get("transfer_phase"),
        "object_ref": step.get("object_ref"),
        "target_ref": step.get("target_ref"),
        "relation": step.get("relation"),
        "native_finish": bool(step.get("_native_finish")),
        "native_finish_scope": step.get("_native_finish_scope"),
        "native_finish_budget_s": step.get("_native_finish_budget_s"),
        "native_finish_start_sim_action": step.get("_native_finish_start_sim_action"),
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
        "scene_generation": observation.get("scene_generation"),
        "sim_action_count": observation.get("sim_action_count"),
        "env_step": observation.get("env_step"),
        "timestamp": observation.get("timestamp"),
        "cameras": observation.get("cameras"),
        "llm": False,
    }


def _diagnosis(step, outcome, verification, observation, plan) -> dict[str, Any]:
    metrics = (verification or {}).get("metrics") or {}
    category = outcome.category or metrics.get("category")
    code = (verification or {}).get("reason_code") or ""
    if not category:
        if code in {"TOOL_UNAVAILABLE"}:
            category = "service"
        elif code in {"PARAMETER"}:
            category = "parameter"
        elif code in {"TOOL_EXCEPTION"}:
            category = "tool_exception"
        elif not outcome.executed:
            category = "tool_exception"
        else:
            category = "subtask_incomplete"
    checker_reason = metrics.get("checker_reason_code")
    reason = outcome.error or metrics.get("error") or code
    raw = outcome.raw or {}
    obs = _public_observation(observation) or {}
    args = {key: value for key, value in (step.get("args") or {}).items() if key != "code"}
    return {
        "parent_step_id": step.get("subtask"),
        "step_id": step.get("step_id"),
        "primitive": step.get("skill"),
        "skill": step.get("skill"),
        "check_name": step.get("completion_check"),
        "transfer_phase": step.get("transfer_phase"),
        "category": category,
        "reason": metrics.get("summary") or reason,
        "summary": metrics.get("summary") or reason,
        "reason_code": code,
        "error_code": code,
        "checker_reason_code": checker_reason,
        "args": args,
        "object_ref": step.get("object_ref") or args.get("object_ref"),
        "grounding_query": step.get("grounding_query") if step.get("skill") == "segment" else None,
        "vla_prompt": step.get("vla_prompt") if step.get("skill") == "pi0_pick" else None,
        "return_summary": {
            "executed": outcome.executed,
            "error": outcome.error,
            "error_class": outcome.error_class,
            "grasp_signal": outcome.grasp_signal,
            "native_terminated": outcome.native_terminated,
            "native_truncated": outcome.native_truncated,
            "found": raw.get("found"),
            "score": raw.get("score"),
            "world_xyz": outcome.xyz,
        },
        "request_id": plan.get("request_id"),
        "episode_id": plan.get("episode_id"),
        "env_action_count": raw.get("env_action_count") if isinstance(raw.get("env_action_count"), int) else None,
        "observation_id": obs.get("observation_id"),
        "record_index": outcome.env_step,
        "record_index_note": "states.json 的序号是工具记录，不是仿真步数",
        "planner_stage": plan.get("planner_stage"),
        "transport_started": step.get("completion_check") in {"object_transported", "object_placed"},
        "missing": metrics.get("missing"),
        "unmet": metrics.get("unmet"),
        "unmet_conditions": metrics.get("unmet"),
        "evidence_sources": metrics.get("evidence_sources"),
        "actual": metrics.get("actual"),
        "condition": metrics.get("condition"),
    }


def _failed_step(steps: list[dict]) -> str | None:
    for step in reversed(steps or []):
        if (step.get("verification") or {}).get("status") == "failure":
            return step.get("step_id")
    return None


def _last_completed(steps: list[dict]) -> str | None:
    for step in reversed(steps or []):
        if (step.get("verification") or {}).get("status") == "success":
            return step.get("step_id")
    return None


def _stamp_steps(execution, steps: list[dict]) -> dict:
    body = dict(execution) if isinstance(execution, dict) else {}
    body["failed_step"] = _failed_step(steps)
    body["last_completed_step"] = _last_completed(steps)
    return body


def _finish(status, reason, session, epoch, version, steps, backend, llm_calls, execution=None, plan=None, retreat_error=None) -> dict[str, Any]:
    # Official success is only the backend's LIBERO terminated flag.
    # Truncation, an unfinished plan, and planner finish do not set it.
    # A retreat error does not assign the user goal.
    native = bool(backend.native_success()) if hasattr(backend, "native_success") else False
    payload = {
        "status": status,
        "execution_status": "retreat_error" if retreat_error else status,
        "reason": reason,
        "episode_id": session.episode_id,
        "scene_epoch": epoch,
        "plan_version": version,
        "steps": steps,
        "native_success": native,
        "native_task_success": native,
        "user_goal_success": None,
        "user_goal_status": None,
        "subgoal_results": None,
        "retreat_error": retreat_error,
        "evidence_source": None,
        "place_records": [step for step in steps if step.get("stage") in {"place", "place_motion"}],
        "retreat_records": [step for step in steps if step.get("stage") == "retreat"],
        "llm_calls_during_execution": llm_calls,
        "tool_results": [step.get("tool_result") for step in steps],
        "subtask_results": [step.get("verification") for step in steps],
        "execution": _stamp_steps(execution, steps),
        "failed_step": _failed_step(steps),
        "last_completed_step": _last_completed(steps),
        "native_finish_window": _native_finish_window(steps),
    }
    payload.update(_terminal_outcomes(steps))
    if plan and plan.get("execution_mode") == "free":
        from finalize import _read_user_goal

        _read_user_goal(payload, backend, plan)
        payload["final_check"] = {
            "user_goal_status": payload["user_goal_status"],
            "native_task_success": native,
            "retreat_error": retreat_error,
            "retreat_complete": payload.get("retreat_complete"),
            "release": payload.get("release"),
            "native_finish_window": payload.get("native_finish_window"),
            "evidence_source": payload["evidence_source"],
            "extra_step": False,
        }
        payload["final_check"]["unknown_reason"] = payload.get("unknown_reason")
    return payload
