"""Bounded local recovery. Unknown evidence never opens the gripper or moves the arm."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
LIMIT = int(json.loads((ROOT / "thresholds.json").read_text(encoding="utf-8"))["recovery_attempts"])
SERVICE_RETRIES = int(json.loads((ROOT / "thresholds.json").read_text(encoding="utf-8"))["service_retries"])

EVENTS = frozenset({
    "EMPTY_GRASP",
    "GRASP_LOST",
    "NO_PROGRESS",
    "PLACEMENT_UNVERIFIED",
    "TARGET_AMBIGUOUS",
    "SERVICE_ERROR",
    "DRAWER_STALLED",
})
WORLD = frozenset({"pi0_pick", "move_to", "set_gripper"})


def signature(skill: str, args: dict[str, Any]) -> str:
    xyz = args.get("xyz")
    rounded = [round(float(value), 2) for value in xyz[:3]] if isinstance(xyz, list) and len(xyz) >= 3 else None
    return json.dumps(
        {
            "skill": skill,
            "xyz": rounded,
            "xyz_from": args.get("xyz_from"),
            "prompt": args.get("prompt"),
            "camera": args.get("camera"),
            "gripper": args.get("gripper"),
            "z_offset": None if args.get("z_offset") is None else round(float(args["z_offset"]), 2),
        },
        sort_keys=True,
    )


class RecoveryBudget:
    def __init__(self, limit: int = LIMIT) -> None:
        self.limit = limit
        self.attempts: dict[str, int] = {}
        self.seen: dict[str, set[str]] = {}
        self.service_retries: dict[str, int] = {}

    def remaining(self, step_id: str) -> int:
        return self.limit - self.attempts.get(step_id, 0)

    def allow(self, step_id: str, sig: str) -> bool:
        return sig not in self.seen.get(step_id, set())

    def consume(self, step_id: str) -> bool:
        if self.attempts.get(step_id, 0) >= self.limit:
            return False
        self.attempts[step_id] = self.attempts.get(step_id, 0) + 1
        return True

    def note(self, step_id: str, sig: str) -> None:
        self.seen.setdefault(step_id, set()).add(sig)

    def allow_service(self, step_id: str) -> bool:
        return self.service_retries.get(step_id, 0) < SERVICE_RETRIES

    def mark_service(self, step_id: str) -> None:
        self.service_retries[step_id] = self.service_retries.get(step_id, 0) + 1


def event_for(verification: dict[str, Any] | None, outcome: Any) -> str | None:
    if getattr(outcome, "error_class", None) == "service":
        return "SERVICE_ERROR"
    if not verification:
        return None
    if verification.get("status") == "unknown":
        return None
    code = verification.get("reason_code")
    if code in EVENTS:
        return code
    if code == "WRONG_PLACEMENT":
        return "PLACEMENT_UNVERIFIED"
    if code == "TARGET_NOT_IDENTIFIED":
        return "TARGET_AMBIGUOUS"
    if code == "zero_progress":
        return "DRAWER_STALLED"
    return None


def propose(
    event: str,
    step: dict[str, Any],
    *,
    budget: RecoveryBudget,
    holding: bool,
    released: bool,
    stalled: bool,
    env_step_changed: bool,
    attempt_index: int,
) -> dict[str, Any]:
    """Return recovery actions that reuse registered primitives."""
    if event not in EVENTS:
        return {"kind": "none", "actions": [], "event": event}
    if event == "SERVICE_ERROR":
        if env_step_changed or not budget.allow_service(step["step_id"]):
            return {"kind": "service_no_retry", "actions": [], "event": event}
        return {"kind": "service_retry", "actions": [{"skill": step["skill"], "args": dict(step.get("args") or {})}], "event": event}
    if not stalled and event == "NO_PROGRESS":
        return {"kind": "continue", "actions": [], "event": event}
    if event == "DRAWER_STALLED":
        entry = step.get("_entry_xyz")
        if not isinstance(entry, list) or len(entry) < 3:
            return {
                "kind": "entry_unmeasured",
                "actions": [],
                "event": event,
                "changed_variables": [],
                "reason": "没有把手位姿，不重复同一开抽屉指令",
            }
        return {
            "kind": "actions",
            "actions": _actions(event, step, holding=holding, released=released, attempt_index=attempt_index),
            "event": event,
            "changed_variables": ["entry_pose"],
        }
    if budget.remaining(step["step_id"]) <= 0:
        return {"kind": "exhausted", "actions": [], "event": event}
    actions = _actions(event, step, holding=holding, released=released, attempt_index=attempt_index)
    if holding and event == "EMPTY_GRASP":
        actions = [action for action in actions if action["skill"] != "set_gripper"]
    if not actions:
        return {"kind": "exhausted", "actions": [], "event": event}
    return {
        "kind": "actions",
        "actions": actions,
        "event": event,
        "changed_variables": _changed_variables(event, attempt_index),
    }


def _actions(event: str, step: dict, *, holding: bool, released: bool, attempt_index: int) -> list[dict]:
    ref = step.get("object_ref")
    args = step.get("args") or {}
    prompt = args.get("prompt") or ""
    phrase = _phrase(step)
    camera = "wrist" if attempt_index else "agentview"
    approach_z = 0.08 if attempt_index else 0.15
    locate = {
        "skill": "segment",
        "args": {"prompt": phrase, "camera": camera, "object_ref": ref},
    }
    if event == "EMPTY_GRASP":
        actions = []
        if not holding:
            actions.append({"skill": "set_gripper", "args": {"gripper": -1.0, "steps": 8, "object_ref": ref}})
        actions.append(locate)
        actions.append({
            "skill": "move_to",
            "args": {"xyz_from": f"segment:{phrase}", "z_offset": approach_z, "gripper": 1.0, "object_ref": ref},
        })
        actions.append({
            "skill": "pi0_pick",
            "args": {"prompt": prompt or f"pick up the {phrase}", "max_chunks": int(args.get("max_chunks") or 12), "object_ref": ref},
        })
        return actions
    if event == "GRASP_LOST":
        return [
            locate,
            {"skill": "pi0_pick", "args": {"prompt": prompt or f"pick up the {phrase}", "max_chunks": 12, "object_ref": ref}},
        ]
    if event == "NO_PROGRESS":
        return [
            locate,
            {"skill": "move_to", "args": {"xyz_from": f"segment:{phrase}", "z_offset": approach_z, "gripper": args.get("gripper", 1.0), "object_ref": ref}},
        ]
    if event == "PLACEMENT_UNVERIFIED":
        target = step.get("target_ref") or ""
        carried = ref if ref and ref != target else ""
        if holding and not released and carried and target:
            return [
                {"skill": "move_to", "args": {"xyz_from": f"segment:{target}", "z_offset": 0.15, "gripper": 1.0, "object_ref": carried}},
                {"skill": "set_gripper", "args": {"gripper": -1.0, "steps": 8, "object_ref": carried}},
            ]
        if released and carried:
            # The VLA and the segmenter need the natural phrase. An instance
            # id such as "cream_cheese_1" is not in the policy's language
            # distribution and degrades into grasping the wrong object.
            phrase = _phrase(step) or carried
            return [
                {"skill": "segment", "args": {"prompt": phrase, "camera": camera, "object_ref": carried}},
                {"skill": "pi0_pick", "args": {"prompt": f"pick up the {phrase}", "max_chunks": 12, "object_ref": carried}},
            ]
        return [locate] if carried else []
    if event == "TARGET_AMBIGUOUS":
        return [locate]
    if event == "DRAWER_STALLED":
        entry = step.get("_entry_xyz")
        args = dict(step.get("args") or {})
        return [
            {"skill": "move_to", "args": {"xyz": [float(entry[0]), float(entry[1]), float(entry[2])], "gripper": 1.0}},
            {"skill": "open_drawer", "args": args},
        ]
    return []


def _changed_variables(event: str, attempt_index: int) -> list[str]:
    if event in {"EMPTY_GRASP", "GRASP_LOST", "NO_PROGRESS"}:
        changed = ["camera", "approach_z"]
        if attempt_index:
            changed.append("attempt_index")
        return changed
    return []


def _phrase(step: dict) -> str:
    query = step.get("grounding_query")
    if isinstance(query, str) and query.strip():
        return query.strip()
    args = step.get("args") or {}
    prompt = args.get("prompt")
    if isinstance(prompt, str) and prompt.strip():
        return prompt.strip()
    source = args.get("xyz_from")
    if isinstance(source, str) and source.startswith("segment:"):
        return source.split(":", 1)[1]
    return str(step.get("object_ref") or "")
