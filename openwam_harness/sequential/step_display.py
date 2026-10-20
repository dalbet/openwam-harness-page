"""Chinese action rows mapped from compiled steps. The planner does not write these."""

from __future__ import annotations

from typing import Any

_NAMES = (
    ("cream cheese", "奶油奶酪"),
    ("alphabet soup", "字母汤"),
    ("tomato sauce", "番茄酱"),
    ("wine bottle", "酒瓶"),
    ("black bowl", "黑碗"),
    ("wine rack", "酒架"),
    ("basket", "篮子"),
    ("stove", "炉灶"),
    ("plate", "盘子"),
    ("bowl", "碗"),
)

_SUCCESS = {
    "OBSERVE_READY",
    "OBJECT_LOCALIZED",
    "GRASP_HELD",
    "TRANSPORTED",
    "PLACED",
    "RETREATED",
}


def display_steps(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per real compiled action. A missing retreat is not invented."""
    rows: list[dict[str, Any]] = []
    segments = [step for step in steps if step.get("skill") == "segment"]
    if segments:
        obj = _phrase(segments[0])
        target = _phrase(segments[1]) if len(segments) > 1 else "目标"
        rows.append(_row(
            "locate",
            [step["step_id"] for step in segments],
            f"定位{obj}与{target}放置区域",
            f"Locate { _english(segments[0]) } and the placement region",
        ))
    for step in steps:
        skill = step.get("skill")
        motion = (step.get("args") or {}).get("motion")
        phase = step.get("transfer_phase")
        obj = _phrase(step)
        target = _target_phrase(step, steps)
        if skill == "pi0_pick":
            prompt = step.get("vla_prompt") or (step.get("args") or {}).get("prompt") or ""
            rows.append(_row("pick", [step["step_id"]], f"抓取{obj}", str(prompt)))
        elif skill == "move_to" and (motion == "travel" or phase == "transporting"):
            rows.append(_row("carry", [step["step_id"]], f"搬运至{target}上方", f"Move above the { _english_target(step, steps) }"))
        elif skill == "move_to" and (motion == "vertical" or phase == "descending"):
            rows.append(_row("lower", [step["step_id"]], "下降至放置位置", "Lower to the placement pose"))
        elif skill == "set_gripper" and step.get("completion_check") == "object_placed":
            rows.append(_row("release", [step["step_id"]], "松开夹爪", "Release the object"))
            rows.append(_row("verify", [step["step_id"]], "验证放置结果", "Verify placement"))
        elif skill == "move_to" and (motion == "retreat" or phase == "retreat"):
            rows.append(_row("retreat", [step["step_id"]], "安全退离", "Retreat"))
    return rows


def apply_display_event(rows: list[dict[str, Any]], event: dict[str, Any]) -> list[dict[str, Any]]:
    """Advance only the compiled row that owns this step. Planning success is not grasp success."""
    step_id = event.get("step_id")
    if not isinstance(step_id, str):
        return rows
    phase = event.get("phase")
    code = event.get("reason_code") or event.get("verification_status")
    copied = [dict(row) for row in rows]
    for row in copied:
        if step_id not in (row.get("step_ids") or []):
            continue
        if phase == "failed":
            if row["kind"] == "verify" and code not in {None, "", "PLACED"}:
                row["status"] = "failed"
            elif row["kind"] != "verify":
                row["status"] = "failed"
            continue
        if code in _SUCCESS or event.get("verification_status") == "success":
            if row["kind"] == "verify" and code not in {"PLACED", "OBJECT_PLACED"} and event.get("verification_status") != "success":
                continue
            if row["kind"] == "verify" and code in {"GRASP_HELD", "TRANSPORTED", "OBJECT_LOCALIZED"}:
                continue
            row["status"] = "success"
        elif phase == "executing" and row["status"] != "success":
            if row["kind"] == "verify":
                continue
            row["status"] = "running"
    return copied


def _row(kind: str, step_ids: list[str], zh: str, en: str) -> dict[str, Any]:
    return {
        "kind": kind,
        "step_ids": step_ids,
        "text_zh": zh,
        "text_en": en,
        "status": "pending",
    }


def _phrase(step: dict[str, Any]) -> str:
    return _zh(_english(step)) or "物体"


def _english(step: dict[str, Any]) -> str:
    query = step.get("grounding_query") or (step.get("args") or {}).get("grounding_query") or ""
    return str(query).strip()


def _target_phrase(step: dict[str, Any], steps: list[dict[str, Any]]) -> str:
    ref = step.get("target_ref")
    for item in steps:
        if item.get("skill") == "segment" and item.get("object_ref") == ref:
            return _phrase(item)
    return "目标"


def _english_target(step: dict[str, Any], steps: list[dict[str, Any]]) -> str:
    ref = step.get("target_ref")
    for item in steps:
        if item.get("skill") == "segment" and item.get("object_ref") == ref:
            return _english(item) or "target"
    return "target"


def _zh(text: str) -> str:
    lowered = text.lower()
    for english, chinese in _NAMES:
        if english in lowered:
            return chinese
    return text
