"""One idempotent terminal report. A failed read still returns the five fields."""

from __future__ import annotations

from typing import Any


def finalize_run(
    result: dict[str, Any] | None,
    *,
    backend: Any = None,
    plan: dict[str, Any] | None = None,
    exit_kind: str,
    first_error: str = "",
    verification_error: str = "",
) -> dict[str, Any]:
    """Fill the terminal report. Does not call env.step and does not raise."""
    try:
        return _finalize(
            result,
            backend=backend,
            plan=plan,
            exit_kind=exit_kind,
            first_error=first_error,
            verification_error=verification_error,
        )
    except Exception as exc:
        payload = dict(result or {})
        payload["finalized"] = True
        payload["execution_status"] = payload.get("execution_status") or exit_kind or "exception"
        payload["user_goal_status"] = payload.get("user_goal_status") or "unknown"
        payload["native_task_success"] = payload.get("native_task_success")
        payload["retreat_error"] = payload.get("retreat_error")
        payload["evidence_source"] = payload.get("evidence_source") or "unavailable"
        payload["unknown_reason"] = f"finalize failed: {exc}"
        payload["first_error"] = first_error or payload.get("first_error") or ""
        payload["verification_error"] = verification_error or payload.get("verification_error") or ""
        payload["extra_step"] = False
        return payload


def _finalize(result, *, backend, plan, exit_kind, first_error, verification_error) -> dict[str, Any]:
    payload = dict(result or {})
    if payload.get("finalized"):
        return payload
    payload["finalized"] = True
    payload["extra_step"] = False
    payload["execution_status"] = payload.get("execution_status") or exit_kind or "failed"
    if "retreat_error" not in payload:
        payload["retreat_error"] = None
    if "native_task_success" not in payload or payload.get("native_task_success") is None:
        if backend is not None and hasattr(backend, "native_success"):
            try:
                payload["native_task_success"] = bool(backend.native_success())
            except Exception as exc:
                payload["native_task_success"] = None
                payload["native_read_error"] = str(exc)
        else:
            payload["native_task_success"] = None
    steps = payload.get("steps") or []
    payload["completed_subgoals"] = [
        item.get("step_id")
        for item in steps
        if isinstance(item, dict) and ((item.get("verification") or {}).get("reason_code") == "PLACED" or item.get("stage") == "place")
    ]
    if plan and plan.get("execution_mode") == "free" and payload.get("user_goal_status") is None:
        _read_user_goal(payload, backend, plan)
    elif payload.get("user_goal_status") is None:
        payload["user_goal_status"] = "unknown"
        payload["evidence_source"] = payload.get("evidence_source") or "unavailable"
        payload["unknown_reason"] = payload.get("unknown_reason") or "这次退出没有用户目标复查"
    if not payload.get("evidence_source"):
        payload["evidence_source"] = "unavailable" if payload.get("user_goal_status") == "unknown" else payload.get("evidence_source")
    payload["first_error"] = first_error or payload.get("reason") or ""
    payload["verification_error"] = verification_error or _first_verification(steps)
    payload["final_check"] = {
        "user_goal_status": payload.get("user_goal_status"),
        "native_task_success": payload.get("native_task_success"),
        "retreat_error": payload.get("retreat_error"),
        "evidence_source": payload.get("evidence_source"),
        "completed_subgoals": payload.get("completed_subgoals"),
        "extra_step": False,
        "unknown_reason": payload.get("unknown_reason"),
    }
    return payload


def _read_user_goal(payload: dict[str, Any], backend: Any, plan: dict[str, Any]) -> None:
    from free_task import judge_user_goal
    from object_state import goal_evidence

    observation = None
    if backend is not None and hasattr(backend, "observe"):
        try:
            observation = backend.observe()
        except Exception as exc:
            payload["state_read_error"] = str(exc)
    if not isinstance(observation, dict) and backend is not None and hasattr(backend, "sim_state"):
        try:
            observation = {"sim": backend.sim_state(), "scene_epoch": plan.get("scene_epoch")}
        except Exception as exc:
            payload["state_read_error"] = str(exc)
            observation = None
    if not isinstance(observation, dict):
        payload["user_goal_status"] = "unknown"
        payload["user_goal_success"] = None
        payload["evidence_source"] = "unavailable"
        payload["unknown_reason"] = payload.get("state_read_error") or "状态服务没有返回可复查的仿真状态"
        return
    evidence = goal_evidence(
        plan.get("predicates") or [],
        observation,
        plan.get("objects") or {},
        scene_epoch=plan.get("scene_epoch") if plan.get("scene_epoch") is not None else observation.get("scene_epoch"),
    )
    payload["goal_queries"] = evidence.get("queries")
    if evidence["status"] != "ready":
        payload["user_goal_status"] = "unknown"
        payload["user_goal_success"] = None
        payload["evidence_source"] = evidence.get("evidence_source") or "unavailable"
        payload["unknown_reason"] = evidence.get("reason") or evidence.get("reason_code")
        return
    judged = judge_user_goal(plan.get("predicates") or [], evidence["sim"])
    payload["subgoal_results"] = judged["subgoal_results"]
    payload["user_goal_success"] = judged["user_goal_success"]
    payload["user_goal_status"] = "met" if judged["user_goal_success"] else "unmet"
    payload["evidence_source"] = evidence.get("evidence_source") or judged.get("evidence_source") or "sim_state"


def _first_verification(steps: list) -> str:
    for item in steps:
        if not isinstance(item, dict):
            continue
        verification = item.get("verification") or {}
        if verification.get("status") not in (None, "success"):
            return str(verification.get("reason_code") or verification.get("status") or "")
    return ""
