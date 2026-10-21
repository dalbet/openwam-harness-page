"""Skills the sequential executor can run.

``transfer`` is the only composite skill that has been executed and checked.
Contact tools in RPent (``move_to``, ``rotate_wrist``, ``rotate_pitch``,
``move_pose``, ``release``) are not wrapped as drawer, push, or knob skills:
none of those motions has a completion check or a local acceptance record.
Passing a schema does not make a raw tool available to the planner.
"""

from __future__ import annotations

from typing import Any


IMPLEMENTED = (
    "view_env_state",
    "segment",
    "pi0_pick",
    "move_to",
    "set_gripper",
    "pause_for_confirm",
)


def _transfer() -> dict[str, Any]:
    return {
        "name": "transfer",
        "implemented": True,
        "planner_visible": True,
        "params": ["object_n", "target_n", "object_ref", "target_ref", "relation", "prompt"],
        "preconditions": ["object_in_scene", "object_movable", "target_accepts_relation"],
        "executor": "expand_semantic_plan",
        "completion": "object_placed",
        "cancel": "stop",
        "recovery": "recover",
        "expands_to": list(IMPLEMENTED),
        "note": "一次 transfer 是完整的抓取、搬运、放置和检查。",
    }


def _unimplemented(name: str, reason: str) -> dict[str, Any]:
    return {
        "name": name,
        "implemented": False,
        "planner_visible": False,
        "params": [],
        "preconditions": [],
        "executor": None,
        "completion": None,
        "cancel": "stop",
        "recovery": None,
        "reason": reason,
    }


REGISTRY: dict[str, dict[str, Any]] = {
    "transfer": _transfer(),
    "open_drawer": {
        "name": "open_drawer",
        "implemented": True,
        "planner_visible": True,
        "params": ["target_n", "target_ref", "target_region"],
        "preconditions": ["cabinet_in_scene", "middle_region_resolved"],
        "executor": "open_drawer.run_open_drawer",
        "completion": "drawer_open",
        "cancel": "LiberoPrimitives._check_cancelled",
        "recovery": None,
        "reason": "只包装中层抽屉。完成条件是 Open 谓词，不是 pi0_doubled 的 success。",
    },
    "close": _unimplemented(
        "close",
        "RPent 没有已验证的关抽屉技能。release 只张开夹爪。",
    ),
    "push": _unimplemented(
        "push",
        "RPent 没有已验证的推物体技能。move_to 是抓取后的携带，不是推动。",
    ),
    "turn_on_stove": _unimplemented(
        "turn_on_stove",
        "RPent 没有已验证的旋钮技能。rotate_wrist 没有炉灶完成条件。",
    ),
}


def implemented_skills() -> list[str]:
    return [name for name, row in REGISTRY.items() if row.get("implemented") and row.get("planner_visible")]


def unimplemented_skills() -> list[dict[str, str]]:
    return [
        {"name": name, "reason": row.get("reason") or "未实现"}
        for name, row in REGISTRY.items()
        if not row.get("implemented")
    ]


def skill_rejection(name: str) -> str | None:
    row = REGISTRY.get(name)
    if row is None:
        return f"没有技能 {name}。不会执行模型生成的工具或代码。"
    if not row.get("implemented"):
        return f"技能 {name} 未实现。{row.get('reason') or ''}"
    return None
