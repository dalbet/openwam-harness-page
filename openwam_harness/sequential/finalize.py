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
        "retreat_complete": payload.get("retreat_complete"),
        "release": payload.get("release"),
        "native_finish_window": payload.get("native_finish_window"),
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
    judged = judge_user_goal(
        plan.get("predicates") or [],
        evidence["sim"],
        observation,
        plan.get("objects") or {},
    )
    releases = _release_evidence((payload.get("steps") or []), observation)
    rows, status, success, unknown_reason, missing = _release_gate(judged["subgoal_results"], releases, observation)
    payload["subgoal_results"] = rows
    payload["user_goal_success"] = success
    payload["user_goal_status"] = status
    payload["release_evidence"] = {
        key: value
        for key, value in releases.items()
        if isinstance(key, tuple)
    }
    payload["evidence_source"] = evidence.get("evidence_source") or judged.get("evidence_source") or "sim_state"
    if unknown_reason:
        payload["unknown_reason"] = unknown_reason
    if missing:
        payload["missing"] = missing
    if payload.get("retreat_error") and status == "met":
        payload["finalization"] = "user_goal_met_retreat_error"


PLACEMENT_RELATIONS = frozenset({"on", "in", "inside_drawer"})


def _release_evidence(steps: list, observation: dict[str, Any]) -> dict[tuple, dict[str, Any]]:
    """A PLACED receipt and its post-release read. A grasp pose is not release."""
    from verify import THRESHOLDS

    minimum = float(THRESHOLDS["place"]["gripper_open_min_m"])
    found: dict[tuple, dict[str, Any]] = {}
    for item in steps or []:
        if not isinstance(item, dict):
            continue
        verification = item.get("verification") or {}
        metrics = verification.get("metrics") or {}
        if verification.get("reason_code") != "PLACED" or not metrics.get("released_required"):
            continue
        key = (item.get("object_ref"), item.get("target_ref"), item.get("relation"))
        if not all(isinstance(value, str) and value for value in key):
            continue
        opening = metrics.get("gripper_opening_m")
        released = isinstance(opening, (int, float)) and float(opening) >= minimum
        pos = item.get("observation") or {}
        found[key] = {
            "status": "verified" if released else "unverified",
            "reason_code": None if released else "GRIPPER_NOT_OPEN",
            "observation_id": pos.get("observation_id"),
            "sim_action_count": pos.get("sim_action_count"),
            "gripper_opening_m": opening,
            "gripper_open_min_m": minimum,
            "final_observation_id": observation.get("observation_id") if isinstance(observation, dict) else None,
            "final_sim_action_count": observation.get("sim_action_count") if isinstance(observation, dict) else None,
        }
    return found


def _release_gate(rows: list, releases: dict[tuple, dict[str, Any]], observation: dict[str, Any]) -> tuple[list, str, bool | None, str, list]:
    """Reject a held On pose as user-goal completion.

    Geometry false is unmet. Geometry true without release evidence is unknown.
    Geometry true after a verified release is met.
    """
    from verify import THRESHOLDS

    minimum = float(THRESHOLDS["place"]["gripper_open_min_m"])
    sim = (observation or {}).get("sim") if isinstance(observation, dict) else None
    final_opening = (sim or {}).get("gripper_opening_m") if isinstance(sim, dict) else None
    final_count = (observation or {}).get("sim_action_count") if isinstance(observation, dict) else None
    missing: list[str] = []
    unknown = False
    geometry_ok = True
    release_ok = True
    for row in rows or []:
        if not isinstance(row, dict) or row.get("hold") != "end":
            continue
        relation = row.get("relation")
        geometry = bool(row.get("held_at_end"))
        row["geometry_held_at_end"] = geometry
        if relation not in PLACEMENT_RELATIONS:
            if not geometry:
                geometry_ok = False
            continue
        row["release_required"] = True
        key = (row.get("object_ref"), row.get("target_ref"), relation)
        entry = releases.get(key)
        check = dict(entry or {"status": "missing", "reason_code": "release_not_executed"})
        if check.get("status") == "verified":
            post_count = check.get("sim_action_count")
            post_id = check.get("observation_id")
            if not isinstance(post_count, int) or not isinstance(final_count, int):
                check["status"] = "unverified"
                check["reason_code"] = "post_release_observation_missing"
            elif final_count < post_count:
                check["status"] = "unverified"
                check["reason_code"] = "post_release_observation_stale"
            elif not post_id or post_id == check.get("final_observation_id"):
                check["status"] = "unverified"
                check["reason_code"] = "post_release_observation_missing"
            elif not isinstance(final_opening, (int, float)):
                check["status"] = "unverified"
                check["reason_code"] = "gripper_opening_missing"
            elif float(final_opening) < minimum:
                check["status"] = "unverified"
                check["reason_code"] = "gripper_closed_after_release"
        check["final_gripper_opening_m"] = final_opening
        row["release_evidence"] = check
        if not geometry:
            geometry_ok = False
        if check.get("status") != "verified":
            release_ok = False
            if geometry:
                unknown = True
                row["held_at_end"] = False
                code = check.get("reason_code") or "release_evidence_missing"
                missing.append(f"{row.get('object_ref')}:{code}")
    if not rows:
        return rows, "unknown", None, "没有用户目标谓词", ["predicates"]
    if not geometry_ok:
        reason = "释放后对象与目标区域关系不成立"
        return rows, "unmet", False, "", missing
    if unknown or not release_ok:
        reason = "释放后证据不足：" + ", ".join(missing or ["release_evidence"])
        return rows, "unknown", None, reason, missing
    return rows, "met", True, "", []


def _first_verification(steps: list) -> str:
    for item in steps:
        if not isinstance(item, dict):
            continue
        verification = item.get("verification") or {}
        if verification.get("status") not in (None, "success"):
            return str(verification.get("reason_code") or verification.get("status") or "")
    return ""
