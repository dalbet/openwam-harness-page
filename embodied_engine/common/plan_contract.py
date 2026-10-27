"""Structured plan contract for the agent layer.

The language model proposes a plan; this module validates it against the
loaded scene and the skill catalog. It never rewrites a malformed list into a
plan with regular expressions.
"""

from __future__ import annotations

from typing import Any


ALLOWED_RELATIONS = {
    "pick_place": ("on", "in"),
    "push_object": ("on",),
    "open_drawer": ("open",),
    "turn_on": ("turn_on",),
    # Whole-task policies take the full official instruction and produce the
    # whole trajectory themselves; there is no object/site binding to check.
    "whole_task": ("task",),
}



def _error(field: str, code: str, value: Any, expected: str) -> dict[str, Any]:
    return {"field": field, "code": code, "value": value, "expected": expected}


def validate_plan(
    plan: Any,
    *,
    instruction: str,
    task: dict[str, Any],
    snapshot: dict[str, Any],
    skills: dict[str, Any],
    prior_steps: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    errors: list[dict[str, Any]] = []
    if not isinstance(plan, dict):
        return None, [_error("plan", "not_object", type(plan).__name__, "JSON object")]
    steps = plan.get("steps")
    if not isinstance(steps, list) or not steps:
        return None, [_error("steps", "not_nonempty_list", steps, "non-empty list of steps")]
    if len(steps) > 4:
        errors.append(_error("steps", "too_many_steps", len(steps), "at most 4"))
    catalog = {row["name"]: row for row in skills.get("skills", [])}
    sim = snapshot.get("sim") or {}
    objects = sim.get("objects") or {}
    named = snapshot.get("named_poses") or {}
    sites = named.get("sites") or {}
    joints = snapshot.get("joints") or {}
    cleaned = []
    # A replan validates only the new steps. Completed steps are still part of
    # the executed plan, so a placement into a drawer opened by an earlier
    # step must not be rejected as a closed-drawer placement.
    executed = [row for row in (prior_steps or []) if isinstance(row, dict)]
    opened_regions: set[str] = {
        str(row.get("target_n"))
        for row in executed
        if row.get("skill") == "open_drawer" and isinstance(row.get("target_n"), str)
    }
    for index, step in enumerate(steps):
        prefix = f"steps[{index}]"
        if not isinstance(step, dict):
            errors.append(_error(prefix, "not_object", type(step).__name__, "JSON object"))
            continue
        skill = step.get("skill")
        if skill not in catalog:
            errors.append(_error(f"{prefix}.skill", "skill_not_in_catalog", skill, sorted(catalog)))
            continue
        relation = step.get("relation")
        allowed = ALLOWED_RELATIONS.get(skill, ())
        if relation not in allowed:
            errors.append(_error(f"{prefix}.relation", "relation_not_allowed", relation, list(allowed)))
        target = step.get("target_n")
        target_is_site = isinstance(target, str) and target in sites
        target_is_object = isinstance(target, str) and target in objects
        if skill == "whole_task":
            # No binding: the target of the task is the whole scene.
            if target is not None or step.get("object_n") is not None:
                errors.append(
                    _error(f"{prefix}.target_n", "whole_task_takes_no_binding", target, "null object_n and target_n")
                )
            prompt = step.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                errors.append(_error(f"{prefix}.prompt", "empty_prompt", prompt, "full LIBERO task instruction"))
            cleaned.append(
                {
                    "skill": skill,
                    "object_n": None,
                    "target_n": None,
                    "relation": relation,
                    "prompt": (prompt or "").strip() or None,
                }
            )
            continue
        if not (target_is_site or target_is_object):
            errors.append(_error(f"{prefix}.target_n", "target_not_in_scene", target, "a site or object in this scene"))
        obj = step.get("object_n")
        if skill in {"pick_place", "push_object"}:
            if obj not in objects:
                errors.append(_error(f"{prefix}.object_n", "object_not_in_scene", obj, sorted(objects)))
        elif obj is not None and obj not in objects and obj not in {task.get("objects", [None])[0]}:
            errors.append(_error(f"{prefix}.object_n", "object_not_in_scene", obj, "null or a scene object"))
        if skill == "open_drawer":
            if not isinstance(target, str):
                errors.append(_error(f"{prefix}.target_n", "drawer_region_required", target, "drawer region site"))
            else:
                owner, level = _drawer_parts(target)
                if level is None:
                    errors.append(_error(f"{prefix}.target_n", "not_a_drawer_region", target, "*_top/middle/bottom_region"))
                elif owner is None or _drawer_joint(owner, level, joints) is None:
                    # A site named *_top_region is not necessarily a drawer:
                    # wine_rack_1_top_region is a fixed support site and has no
                    # *_top_level joint. Requiring the owner's own joint keeps
                    # object binding tied to the scene, not to the suffix.
                    errors.append(
                        _error(
                            f"{prefix}.target_n",
                            "drawer_joint_missing",
                            target,
                            f"{owner or '<owner>'}_{level}_level joint",
                        )
                    )
                else:
                    opened_regions.add(target)
            action_object = task.get("objects") or []
            if isinstance(obj, str) and action_object and obj not in action_object:
                errors.append(_error(f"{prefix}.object_n", "drawer_owner_mismatch", obj, action_object))
        if skill == "pick_place" and isinstance(target, str) and _is_drawer_region(target, joints):
            owner_region_open = any(
                isinstance(prev.get("target_n"), str)
                and prev.get("skill") == "open_drawer"
                and _same_drawer(prev.get("target_n"), target)
                for prev in [*executed, *steps[:index]]
                if isinstance(prev, dict)
            )
            if not owner_region_open:
                errors.append(
                    _error(
                        f"{prefix}.target_n",
                        "closed_drawer_requires_open_step",
                        target,
                        "an open_drawer step on the same drawer before this placement",
                    )
                )
        prompt = step.get("prompt")
        if prompt is not None and (not isinstance(prompt, str) or not prompt.strip()):
            errors.append(_error(f"{prefix}.prompt", "empty_prompt", prompt, "short English prompt"))
        cleaned.append(
            {
                "skill": skill,
                "object_n": obj if isinstance(obj, str) and obj else None,
                "target_n": target if isinstance(target, str) else None,
                "relation": relation,
                "prompt": (prompt or "").strip() or None,
            }
        )
    if errors:
        return None, errors
    whole_task = any(step.get("skill") == "whole_task" for step in cleaned)
    if whole_task and len(cleaned) != 1:
        return None, [_error("steps", "whole_task_must_be_alone", len(cleaned), "exactly one whole_task step")]
    if not whole_task and task.get("kind") in {"open_drawer", "composite"}:
        has_open = any(step.get("skill") == "open_drawer" for step in [*executed, *cleaned])
        if not has_open:
            errors.append(
                _error("steps", "task_requires_open_drawer", task.get("kind"), "open_drawer before placement")
            )
    if errors:
        return None, errors
    return {
        "interpretation": str(plan.get("interpretation") or "").strip(),
        "success_condition": str(plan.get("success_condition") or "").strip(),
        "steps": cleaned,
        "instruction": instruction,
        "task_id": int(task["task_id"]),
        "kind": task.get("kind"),
        "plan_source": "qwen3.5-9b",
        "planner_vision": False,
        "evidence_source": "scene_metadata+sim_state",
        "snapshot_observation_id": snapshot.get("observation_id"),
        "scene_epoch": snapshot.get("scene_epoch"),
        "sim_action_count": snapshot.get("sim_action_count"),
    }, []


def _drawer_parts(name: str) -> tuple[str | None, str | None]:
    """Split ``owner_level_region`` into the owning body and the level.

    ``wooden_cabinet_1_top_region`` -> ("wooden_cabinet_1", "top").
    A level-only suffix such as ``wine_rack_1_top_region`` still returns a
    level; the caller decides whether the owner actually has a drawer joint.
    """
    text = str(name or "")
    for level in ("top", "middle", "bottom"):
        suffix = f"_{level}_region"
        if text.endswith(suffix):
            owner = text[: -len(suffix)]
            return (owner or None), level
    return None, None


def _drawer_level(name: str) -> str | None:
    return _drawer_parts(name)[1]


def _drawer_joint(owner: str, level: str, joints: dict[str, Any]) -> str | None:
    name = f"{owner}_{level}_level"
    return name if name in joints else None


def _is_drawer_region(name: str, joints: dict[str, Any]) -> bool:
    owner, level = _drawer_parts(name)
    return owner is not None and level is not None and _drawer_joint(owner, level, joints) is not None


def _same_drawer(open_target: str, place_target: str) -> bool:
    return _drawer_parts(open_target)[0] == _drawer_parts(place_target)[0]


def goal_signature(step: dict[str, Any] | None) -> tuple[str, str, str] | None:
    """The goal-defining part of a step: skill, relation and target.

    The object can legitimately change (e.g. replanning to a different bowl),
    but a replan that drops the original target/relation silently changes the
    user's task. Recovery uses this to reject goal drift.
    """
    if not isinstance(step, dict):
        return None
    skill = step.get("skill")
    target = step.get("target_n")
    relation = step.get("relation")
    if not isinstance(skill, str) or not isinstance(target, str) or not isinstance(relation, str):
        return None
    return (skill, relation, target)


def preserves_goal(steps: list[dict[str, Any]], goal: dict[str, Any] | None) -> bool:
    signature = goal_signature(goal)
    if signature is None:
        return True
    for step in steps:
        if goal_signature(step) == signature:
            return True
    return False
