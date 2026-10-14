"""Schema for the local sequential plan. The model cannot emit arbitrary code."""

from __future__ import annotations

import json
import math
from typing import Any

ALLOWED_ACTIONS = frozenset(
    {
        "view_env_state",
        "segment",
        "pi0_pick",
        "move_to",
        "set_gripper",
        "pause_for_confirm",
    }
)
REJECTED_ACTIONS = frozenset(
    {"pi0_doubled", "execute_action", "python", "code", "reset", "shell"}
)
VERDICTS = frozenset({"success", "failure", "unknown"})
TOOL_SCHEMA = json.dumps(
    {
        "actions": sorted(ALLOWED_ACTIONS),
        "rejected": sorted(REJECTED_ACTIONS),
        "move_to": {"xyz": "[x,y,z] or xyz_from + z_offset", "gripper": "+1 hold, -1 open"},
        "set_gripper": {"gripper": "-1 open, +1 close"},
        "pi0_pick": {"prompt": "grasp only"},
        "pause_for_confirm": "required after each place; verdict is success|failure|unknown",
    },
    ensure_ascii=False,
)


class PlanError(ValueError):
    pass


def validate_plan(plan: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(plan, dict):
        raise PlanError("plan must be an object")
    steps = plan.get("steps")
    if not isinstance(steps, list) or not steps:
        raise PlanError("plan.steps must be a non-empty list")
    seen = set()
    for step in steps:
        if not isinstance(step, dict):
            raise PlanError("step must be an object")
        action = step.get("action")
        if action in REJECTED_ACTIONS or action not in ALLOWED_ACTIONS:
            raise PlanError(f"action not allowed: {action}")
        step_id = step.get("id")
        if not isinstance(step_id, str) or not step_id or step_id in seen:
            raise PlanError(f"step id missing or duplicated: {step_id}")
        seen.add(step_id)
        args = step.get("args", {})
        if not isinstance(args, dict):
            raise PlanError(f"{step_id} args must be an object")
        for key in args:
            if key in {"code", "python", "script", "exec"}:
                raise PlanError(f"{step_id} contains executable {key}")
        if action == "pause_for_confirm" and not step.get("subtask"):
            raise PlanError(f"{step_id} pause requires subtask")
    return plan


def parse_plan_text(text: str) -> dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise PlanError("planner output has no JSON object")
    try:
        value = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise PlanError(f"planner JSON invalid: {exc}") from exc
    return validate_plan(value)


def resolve_args(args: dict[str, Any], bindings: dict[str, list[float]]) -> dict[str, Any]:
    out = dict(args)
    if "xyz_from" in out:
        key = str(out.pop("xyz_from"))
        z_offset = float(out.pop("z_offset", 0.0))
        if key not in bindings:
            raise PlanError(f"missing perception binding {key}")
        xyz = [float(v) for v in bindings[key]]
        if len(xyz) != 3 or not all(math.isfinite(v) for v in xyz):
            raise PlanError(f"binding {key} is not a finite xyz")
        xyz[2] += z_offset
        out["xyz"] = [round(v, 4) for v in xyz]
    return out


def split_move(current: list[float] | None, target: list[float], span: float = 0.25) -> list[list[float]]:
    """Split a long XY move locally so one OSC command stays inside 0.30 m."""
    if current is None:
        return [target]
    dx = target[0] - current[0]
    dy = target[1] - current[1]
    distance = math.hypot(dx, dy)
    if distance <= 0.30:
        return [target]
    pieces = max(2, math.ceil(distance / span))
    points = []
    for index in range(1, pieces + 1):
        scale = index / pieces
        points.append(
            [
                round(current[0] + dx * scale, 4),
                round(current[1] + dy * scale, 4),
                round(current[2] + (target[2] - current[2]) * scale, 4),
            ]
        )
    return points
