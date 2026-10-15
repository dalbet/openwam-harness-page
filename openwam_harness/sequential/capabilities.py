"""What the sequential executor can actually run.

A task that only parses is not marked successful. local_acceptance stays
"not_run" until a committed native-success record exists for that task.
"""

from __future__ import annotations

from task_zh import SUITE_LABELS, TASK_ZH
from schema import transfer_slots

IMPLEMENTED = (
    "view_env_state",
    "segment",
    "pi0_pick",
    "move_to",
    "set_gripper",
    "pause_for_confirm",
)
CONTACT_SKILLS = {
    "open": "open_drawer",
    "close": "close",
    "turn_on": "turn_on_stove",
    "push": "push",
}

SPATIAL = (
    "pick up the black bowl between the plate and the ramekin and place it on the plate",
    "pick up the black bowl next to the ramekin and place it on the plate",
    "pick up the black bowl from table center and place it on the plate",
    "pick up the black bowl on the cookie box and place it on the plate",
    "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate",
    "pick up the black bowl on the ramekin and place it on the plate",
    "pick up the black bowl next to the cookie box and place it on the plate",
    "pick up the black bowl on the stove and place it on the plate",
    "pick up the black bowl next to the plate and place it on the plate",
    "pick up the black bowl on the wooden cabinet and place it on the plate",
)
OBJECT = (
    "pick up the alphabet soup and place it in the basket",
    "pick up the cream cheese and place it in the basket",
    "pick up the salad dressing and place it in the basket",
    "pick up the bbq sauce and place it in the basket",
    "pick up the ketchup and place it in the basket",
    "pick up the tomato sauce and place it in the basket",
    "pick up the butter and place it in the basket",
    "pick up the milk and place it in the basket",
    "pick up the chocolate pudding and place it in the basket",
    "pick up the orange juice and place it in the basket",
)
GOAL = (
    "open the middle drawer of the cabinet",
    "put the bowl on the stove",
    "put the wine bottle on top of the cabinet",
    "open the top drawer and put the bowl inside",
    "put the bowl on top of the cabinet",
    "push the plate to the front of the stove",
    "put the cream cheese in the bowl",
    "turn on the stove",
    "put the bowl on the plate",
    "put the wine bottle on the rack",
)
LONG = (
    "put both the alphabet soup and the tomato sauce in the basket",
    "put both the cream cheese box and the butter in the basket",
    "turn on the stove and put the moka pot on it",
    "put the black bowl in the bottom drawer of the cabinet and close it",
    "put the white mug on the left plate and put the yellow and white mug on the right plate",
    "pick up the book and place it in the back compartment of the caddy",
    "put the white mug on the plate and put the chocolate pudding to the right of the plate",
    "put both the alphabet soup and the cream cheese box in the basket",
    "put both moka pots on the stove",
    "put the yellow and white mug in the microwave and close it",
)
GROUPS = (
    ("libero_spatial", SPATIAL),
    ("libero_object", OBJECT),
    ("libero_goal", GOAL),
    ("libero_10", LONG),
)


def contact_kinds(language: str) -> list[str]:
    text = language.lower()
    found = []
    if text.startswith("open ") or "open the" in text:
        found.append("open")
    if "close it" in text or "and close" in text:
        found.append("close")
    if "turn on" in text:
        found.append("turn_on")
    if text.startswith("push ") or " push " in text:
        found.append("push")
    return found


def classify(language: str, suite: str) -> dict:
    """Return the capability row. demo_open means the executor may attempt it."""
    text = language.strip()
    lowered = text.lower()
    contacts = contact_kinds(text)
    slots = transfer_slots(text)
    unsupported = []
    if "to the right" in lowered:
        unsupported.append("to_the_right")
    if "back compartment" in lowered:
        unsupported.append("back_compartment")
    if "left plate" in lowered or "right plate" in lowered:
        unsupported.append("left_right_plate")
    transport = bool(slots) and not contacts and not unsupported
    if suite == "libero_object" and transport:
        kind = "object_transport"
        demo_open = True
    elif suite == "libero_spatial" and transport:
        kind = "spatial_pick_place"
        demo_open = True
    elif suite == "libero_goal" and transport:
        kind = "goal_transport"
        demo_open = True
    elif suite == "libero_10" and lowered.startswith("put both the") and "in the basket" in lowered and not contacts:
        kind = "long_two_foods_basket"
        demo_open = True
    elif contacts:
        kind = "contact"
        demo_open = False
    else:
        kind = "unsupported"
        demo_open = False
    required = ["view_env_state", "segment", "pi0_pick", "move_to", "set_gripper"] if slots else []
    required.extend(CONTACT_SKILLS[item] for item in contacts)
    missing = [skill for skill in required if skill not in IMPLEMENTED]
    limits = []
    if "drawer" in lowered and suite == "libero_spatial":
        limits.append("不打开抽屉。只在碗已经位于抽屉中时，才按抓放尝试。")
    if contacts:
        limits.append("接触技能（开合、旋钮、推动）执行器未实现，动作开始前拒绝。")
    if unsupported:
        limits.append("目标关系不在 in/on 抓放范围内：" + ", ".join(unsupported))
    if not demo_open and not limits:
        limits.append("不在当前开放的搬运范围内。")
    limits.append("可编译或可展开不等于环境原生成功。")
    return {
        "suite": suite,
        "suite_label": SUITE_LABELS[suite],
        "language": text,
        "summary_zh": TASK_ZH.get(text, text),
        "kind": kind,
        "demo_open": demo_open,
        "required_skills": required,
        "executor_implemented": not missing and demo_open,
        "missing_skills": missing,
        "local_acceptance": "not_run",
        "compilable_is_success": False,
        "limitations": limits,
        "source": ["task_zh.py", "schema.transfer_slots"],
        "reject_before_action": not demo_open,
    }


def builtin_matrix() -> list[dict]:
    rows = []
    for suite, languages in GROUPS:
        for index, language in enumerate(languages):
            row = classify(language, suite)
            row["task_id"] = index
            row["task_key"] = f"{suite}:t{index}"
            rows.append(row)
    return rows


def merge_registry(rows: list[dict], suites: list[dict]) -> list[dict]:
    """Attach BDDL instance lists. This still does not mark a task successful."""
    index = {(item["suite"], item["task_id"]): item for item in rows}
    for suite in suites:
        for task in suite.get("tasks") or []:
            row = index.get((suite["id"], int(task["id"])))
            if row is None:
                continue
            if task.get("language") != row["language"]:
                row["registry_language"] = task.get("language")
                row["registry_match"] = False
            else:
                row["registry_match"] = True
            row["instances"] = task.get("instances") or []
            row["fixtures"] = task.get("fixtures") or []
            sources = list(row["source"])
            if "bddl" not in sources:
                sources.append("bddl")
            row["source"] = sources
    return rows


def reject_reason(row: dict) -> str | None:
    if row.get("demo_open"):
        return None
    missing = "、".join(row.get("missing_skills") or [])
    limit = " ".join(row.get("limitations") or [])
    if missing:
        return f"动作前拒绝：执行器没有 {missing}。{limit}"
    return f"动作前拒绝：{limit}"
