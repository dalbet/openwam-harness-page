"""Free-task understanding on a fixed scene.

The task catalog is not a gate. An unknown sentence is not rewritten into
the nearest official task, and it does not change ``task_id``.
"""

from __future__ import annotations

import re
from typing import Any

from schema import THRESHOLDS, _err
from skills import skill_rejection

# Longer phrases first. These names are scene vocabulary, not a task whitelist.
_ZH = (
    ("奶油奶酪", ("cream cheese",)),
    ("黑碗", ("black bowl", "akita black bowl")),
    ("酒瓶", ("wine bottle",)),
    ("炉灶", ("stove",)),
    ("盘子", ("plate",)),
    ("酒架", ("wine rack", "rack")),
    ("木柜", ("wooden cabinet", "cabinet")),
    ("柜子", ("cabinet", "wooden cabinet")),
    ("碗", ("bowl", "black bowl")),
)

_SKILL_INTENTS = (
    (re.compile(r"打开[^。；]*抽屉|拉开[^。；]*抽屉"), "open_drawer"),
    (re.compile(r"关[上闭][^。；]*抽屉|合上[^。；]*抽屉"), "close"),
    (re.compile(r"推动|推到|把[^。；]{0,12}推"), "push"),
    (re.compile(r"打开炉灶|拧[^。；]*旋钮|转动旋钮"), "turn_on_stove"),
)

_PLACE = re.compile(r"把(?P<obj>.+?)放(?:到|在|进)(?P<dst>.+?)(?:上|里|内|中)?$")


class EpisodeFlags:
    """Separate a native goal hit from an episode that can still be stepped."""

    def __init__(self, mode: str = "benchmark") -> None:
        if mode not in {"benchmark", "free"}:
            raise ValueError(mode)
        self.mode = mode
        self.native_goal_met = False
        self.truncated = False

    def absorb(self, terminated: bool, truncated: bool) -> dict[str, Any]:
        if truncated:
            self.truncated = True
            return {"kind": "time_limit", "clear_terminated": False, "continuable": False}
        if terminated:
            self.native_goal_met = True
            if self.mode == "free":
                return {"kind": "native_goal", "clear_terminated": True, "continuable": True}
            return {"kind": "native_goal", "clear_terminated": False, "continuable": False}
        return {"kind": "running", "clear_terminated": False, "continuable": self.continuable()}

    def continuable(self) -> bool:
        if self.truncated:
            return False
        if self.mode != "free" and self.native_goal_met:
            return False
        return True


def prepare_free_request(
    text: str,
    objects: dict[str, Any],
    poses: dict[str, list[float]] | None = None,
) -> dict[str, Any]:
    """Compile a user sentence into predicates. Never selects another task."""
    typed = text.strip()
    blocked = _unsupported_skill(typed)
    if blocked:
        return {"status": "rejected", "changed": False, "reason": blocked, "code": "skill_unimplemented"}
    clauses = _clauses(typed)
    if not clauses:
        return {
            "status": "rejected",
            "changed": False,
            "code": "no_transfer",
            "reason": "这句话没有当前场景可执行的搬运目标，没有改写成相近的预设任务。",
        }
    predicates = []
    for index, clause in enumerate(clauses):
        parsed = _PLACE.search(_compact(clause))
        if parsed is None:
            return {
                "status": "rejected",
                "changed": False,
                "code": "no_transfer",
                "reason": f"无法从“{clause.strip()}”确定搬运对象和目标，没有改写成相近的预设任务。",
            }
        relation = "in" if any(mark in clause for mark in ("放进", "里", "内")) else "on"
        obj = _resolve_mention(parsed.group("obj"), objects, poses, role="对象")
        dst = _resolve_mention(parsed.group("dst"), objects, poses, role="目标")
        for item in (obj, dst):
            if item["status"] != "resolved":
                item["changed"] = False
                return item
        if not objects[obj["object_ref"]].get("movable"):
            return {
                "status": "rejected",
                "changed": False,
                "code": "not_movable",
                "reason": f"{obj['object_ref']} 不能被搬运。",
            }
        relations = objects[dst["object_ref"]].get("target_relations") or []
        if relation not in relations:
            return {
                "status": "rejected",
                "changed": False,
                "code": "relation_unsupported",
                "reason": f"{dst['object_ref']} 不支持 {relation}。",
            }
        predicates.append(
            {
                "object_ref": obj["object_ref"],
                "target_ref": dst["object_ref"],
                "relation": relation,
                "hold": "end",
                "order": index,
                "source": "user_text",
            }
        )
    ordered = len(clauses) > 1 and any(mark in typed for mark in ("先", "再", "然后"))
    return {
        "status": "ready",
        "changed": False,
        "user_text": typed,
        "predicates": predicates,
        "ordered": ordered,
        "evidence_source": "sim_state",
        "vision": False,
    }


def candidate_number_errors(plan: dict[str, Any]) -> list[dict[str, Any]]:
    errors = []
    steps = plan.get("steps") if isinstance(plan, dict) else None
    if not isinstance(steps, list):
        return errors
    for index, step in enumerate(steps):
        if not isinstance(step, dict) or step.get("skill") != "transfer":
            continue
        if step.get("object_n") is None:
            errors.append(_err(f"steps[{index}].object_n", "candidate_number", None, "当前场景候选编号"))
        if step.get("target_n") is None:
            errors.append(_err(f"steps[{index}].target_n", "candidate_number", None, "当前场景候选编号"))
    return errors


def coverage_errors(plan: dict[str, Any], predicates: list[dict[str, Any]], *, ordered: bool) -> list[dict[str, Any]]:
    """The plan must place every user predicate. Official task text is not the target."""
    errors = []
    steps = [step for step in plan.get("steps") or [] if isinstance(step, dict)]
    open_steps = list(steps)
    chosen = []
    for predicate in predicates:
        found = None
        for step in open_steps:
            if (
                step.get("object_ref") == predicate["object_ref"]
                and step.get("target_ref") == predicate["target_ref"]
                and step.get("relation") == predicate["relation"]
                and step.get("skill") == "transfer"
            ):
                found = step
                break
        if found is None:
            errors.append(
                _err(
                    "plan.steps",
                    "user_goal_uncovered",
                    predicate,
                    "一个覆盖该用户要求的 transfer",
                )
            )
            continue
        open_steps.remove(found)
        chosen.append(found)
    if ordered and len(chosen) > 1:
        earlier = chosen[0].get("step_id")
        for step in chosen[1:]:
            depends = step.get("depends_on") or []
            if earlier not in depends:
                errors.append(
                    _err(
                        f"steps.{step.get('step_id')}.depends_on",
                        "sequence_dependency",
                        depends,
                        f"依赖已完成的 {earlier}。先后关系不是重复的 transfer",
                    )
                )
            earlier = step.get("step_id")
    return errors


def judge_user_goal(predicates: list[dict[str, Any]], sim: dict[str, Any] | None) -> dict[str, Any]:
    """Backend check. A planner success flag is not read."""
    state = sim or {}
    objects = state.get("objects") or {}
    cfg = THRESHOLDS["place"]
    rows = []
    for predicate in predicates:
        obj = _pos(objects, predicate["object_ref"])
        dst = _pos(objects, predicate["target_ref"])
        held = False
        metrics: dict[str, Any] = {"relation": predicate["relation"], "hold": predicate.get("hold") or "end"}
        if obj is not None and dst is not None:
            xy = ((obj[0] - dst[0]) ** 2 + (obj[1] - dst[1]) ** 2) ** 0.5
            radius = float(cfg["in_radius_m"] if predicate["relation"] == "in" else cfg["on_radius_m"])
            z_ok = abs(obj[2] - dst[2]) <= float(cfg["support_z_tolerance_m"])
            held = xy <= radius and z_ok
            metrics["xy_m"] = round(xy, 4)
            metrics["z_delta_m"] = round(obj[2] - dst[2], 4)
        rows.append(
            {
                "object_ref": predicate["object_ref"],
                "target_ref": predicate["target_ref"],
                "relation": predicate["relation"],
                "hold": predicate.get("hold") or "end",
                "held_at_end": held,
                "instant_only": False,
                "evidence_source": "sim_state",
                "metrics": metrics,
            }
        )
    end_rows = [row for row in rows if row["hold"] == "end"]
    return {
        "subgoal_results": rows,
        "user_goal_success": bool(end_rows) and all(row["held_at_end"] for row in end_rows),
        "evidence_source": "sim_state",
        "vision": False,
    }


def _unsupported_skill(text: str) -> str | None:
    for pattern, name in _SKILL_INTENTS:
        if pattern.search(text):
            return skill_rejection(name)
    return None


def _clauses(text: str) -> list[str]:
    match = re.search(r"先(?P<a>.+?)(?:，|,)?再(?P<b>.+)", text)
    if match:
        return [match.group("a"), match.group("b")]
    parts = re.split(r"然后|，|。|；", text)
    return [part.strip() for part in parts if "把" in part and "放" in part]


def _compact(text: str) -> str:
    return re.sub(r"[\s。！？，、,.]+", "", text.strip())


def _resolve_mention(
    text: str,
    objects: dict[str, Any],
    poses: dict[str, list[float]] | None,
    *,
    role: str,
) -> dict[str, Any]:
    phrase = text.strip()
    keys = _alias_keys(phrase)
    matches = []
    for ref, spec in objects.items():
        hay = " ".join(
            [str(spec.get("semantic_type") or ""), *(str(item) for item in spec.get("phrases") or [])]
        ).lower()
        if keys and any(key in hay for key in keys):
            matches.append(ref)
        elif phrase and phrase.lower() in hay:
            matches.append(ref)
    if not matches:
        return {
            "status": "rejected",
            "code": "object_missing",
            "reason": f"当前场景没有“{phrase}”，不能执行。",
        }
    if len(matches) > 1:
        side = "left" if "左" in phrase else "right" if "右" in phrase else None
        poses = poses or {}
        if side and all(ref in poses and len(poses[ref]) >= 1 for ref in matches):
            ordered = sorted(matches, key=lambda ref: float(poses[ref][0]))
            chosen = ordered[0] if side == "left" else ordered[-1]
            return {
                "status": "resolved",
                "object_ref": chosen,
                "note": "左右参照系是仿真世界坐标 x，较小为左。evidence_source=sim_state，不是相机画面。",
            }
        shown = "、".join(matches)
        return {
            "status": "clarify",
            "code": "ambiguous_reference",
            "question": f"哪一个{role}是“{phrase}”？候选 {shown}。不会默认第一个。",
            "candidates": matches,
        }
    return {"status": "resolved", "object_ref": matches[0]}


def _alias_keys(phrase: str) -> list[str]:
    keys = []
    for zh, english in _ZH:
        if zh in phrase:
            keys.extend(english)
            break
    return keys


def _pos(objects: dict[str, Any], ref: str) -> list[float] | None:
    item = objects.get(ref)
    if isinstance(item, dict):
        pos = item.get("pos")
        if isinstance(pos, list) and len(pos) >= 3:
            return [float(pos[0]), float(pos[1]), float(pos[2])]
    return None
