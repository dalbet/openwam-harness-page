"""Catalog-level contract for a same-scene, two-step composite.

Pure data and string logic: no simulator, no numpy, no HTTP. Both the Agent
layer (routing/assembly) and the engine (execution) import this module, so
"which ordered pairs exist" and "what is each step's goal" have exactly one
definition.

The composite is anchored on the **last** step's BDDL: the loaded scene is the
last capability's scene, so the simulator's own ``native_goal`` is exactly the
final step's goal atoms. An earlier step is judged by an engine predicate on
the same snapshots, never by the native flag of a scene it does not belong to.
"""

from __future__ import annotations

from typing import Any

SCHEMA = "embodied_engine.composite.v1"

DEFAULT_PAIRS: tuple[tuple[str, str], ...] = (("open_middle_drawer", "put_bowl_on_plate"),)
DEFAULT_MAX_STEPS = 2
DEFAULT_STEP_MAX_STEPS = {"1": 240, "2": 600}
DEFAULT_GOAL_CONFIRM_SAMPLES = 2


def options(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    block = dict((cfg or {}).get("composite") or {})
    pairs = block.get("supported_pairs") or [list(pair) for pair in DEFAULT_PAIRS]
    return {
        "enabled": bool(block.get("enabled", True)),
        "pairs": tuple(tuple(str(key) for key in pair) for pair in pairs),
        "max_steps": int(block.get("max_steps") or DEFAULT_MAX_STEPS),
        "step_max_steps": {
            **DEFAULT_STEP_MAX_STEPS,
            **{str(k): int(v) for k, v in (block.get("step_max_steps") or {}).items()},
        },
        "goal_confirm_samples": int(block.get("goal_confirm_samples") or DEFAULT_GOAL_CONFIRM_SAMPLES),
    }


def step_budget(step_index: int, cfg: dict[str, Any] | None = None) -> int:
    return int(options(cfg)["step_max_steps"].get(str(step_index), 600))


def completion_for(task: dict[str, Any], *, is_anchor: bool) -> dict[str, Any]:
    """How this step's own goal is judged.

    The anchor step is the loaded scene, so the simulator's native goal is the
    step goal. Any earlier step is judged by an engine predicate computed from
    the same snapshot; its native flag is recorded but never used as its goal.
    """
    if is_anchor:
        return {"kind": "native_goal", "predicate": "native_libero_goal"}
    kind = str(task.get("kind") or "")
    targets = task.get("targets") or []
    objects = task.get("objects") or []
    if kind == "open_drawer":
        return {
            "kind": "drawer_open",
            "predicate": "drawer_open",
            "skill": "open_drawer",
            "target_n": targets[0] if targets else None,
        }
    if kind == "turn_on":
        return {"kind": "stove_button", "predicate": "stove_button", "skill": "turn_on", "target_n": None}
    if kind in {"pick_place", "push_object"} and objects and targets:
        return {
            "kind": "relation",
            "predicate": "relation",
            "skill": "pick_place" if kind == "pick_place" else "push_object",
            "object_n": objects[0],
            "target_n": targets[0],
            "relation": (task.get("semantics") or {}).get("relation") or "on",
        }
    return {"kind": "unsupported", "predicate": None}


def support(
    keys: list[str],
    *,
    loaded_task: dict[str, Any],
    catalog: dict[str, dict[str, Any]],
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Whether this ordered key list is executable in the loaded scene.

    Recognition and execution are deliberately separate: a pair can be a
    correct reading of the instruction and still be ``unsupported`` here.
    """
    opts = options(cfg)
    ordered = [str(key) for key in keys]
    out: dict[str, Any] = {"ok": False, "keys": ordered, "executable": False}
    if not opts["enabled"]:
        out.update({"reason_code": "composite_disabled", "reason": "composite execution is disabled in config"})
        return out
    if len(ordered) < 2:
        out.update({"reason_code": "not_a_composite", "reason": "fewer than two steps"})
        return out
    if len(ordered) > opts["max_steps"]:
        out.update(
            {
                "reason_code": "step_limit_exceeded",
                "reason": f"{len(ordered)} steps requested; at most {opts['max_steps']} are supported",
            }
        )
        return out
    if len(set(ordered)) != len(ordered):
        out.update({"reason_code": "duplicate_step", "reason": "the same capability appears twice"})
        return out
    missing = [key for key in ordered if key not in catalog]
    if missing:
        out.update({"reason_code": "key_not_in_catalog", "reason": f"unknown keys: {missing}"})
        return out
    if tuple(ordered) not in opts["pairs"]:
        out.update(
            {
                "reason_code": "unsupported_combo",
                "reason": (
                    "the request names a valid sequence, but only "
                    + ", ".join("->".join(pair) for pair in opts["pairs"])
                    + " is executable right now"
                ),
                "supported_pairs": [list(pair) for pair in opts["pairs"]],
            }
        )
        return out
    anchor = catalog[ordered[-1]]
    if int(anchor["task_id"]) != int(loaded_task["task_id"]):
        out.update(
            {
                "reason_code": "requires_scene_switch",
                "reason": (
                    f"the composite ends with task {anchor['task_id']} ({anchor['key']}), but task "
                    f"{loaded_task['task_id']} is loaded; load task {anchor['task_id']} to run it"
                ),
            }
        )
        return out
    if any(completion_for(catalog[key], is_anchor=False)["kind"] == "unsupported" for key in ordered[:-1]):
        out.update({"reason_code": "step_goal_unsupported", "reason": "an early step has no engine predicate"})
        return out
    out.update(
        {
            "ok": True,
            "executable": True,
            "anchor": {"task_id": int(anchor["task_id"]), "key": anchor["key"], "bddl": anchor.get("bddl")},
            "steps": [
                {
                    "task_id": int(catalog[key]["task_id"]),
                    "key": key,
                    "goal": completion_for(catalog[key], is_anchor=key == ordered[-1]),
                }
                for key in ordered
            ],
        }
    )
    return out


def build_plan(
    keys: list[str],
    *,
    catalog: dict[str, dict[str, Any]],
    loaded_task: dict[str, Any],
    instruction: str,
    cfg: dict[str, Any] | None = None,
    plan_source: str = "qwen3.5-9b-semantic-router",
) -> dict[str, Any]:
    """The ordered step list for an executable composite (no validation)."""
    ordered = [str(key) for key in keys]
    steps = []
    for index, key in enumerate(ordered, start=1):
        task = catalog[key]
        official = str(task.get("official_instruction") or task.get("instruction") or "").strip()
        is_anchor = index == len(ordered)
        completion = completion_for(task, is_anchor=is_anchor)
        steps.append(
            {
                "skill": "whole_task",
                "object_n": None,
                "target_n": None,
                "relation": "task",
                "prompt": official,
                "step_id": f"c{index}",
                "step_index": index,
                "step_task_id": int(task["task_id"]),
                "step_task_key": key,
                "goal_kind": completion["kind"],
                "completion": completion,
            }
        )
    return {
        "interpretation": (
            f"One scene, two ordered capabilities ({' -> '.join(ordered)}). The episode is loaded from task "
            f"{loaded_task['task_id']} ({loaded_task['key']}) and is never reset between steps; each step "
            "receives its own catalog official instruction."
        ),
        "success_condition": (
            "Step 1's engine predicate and step 2's native LIBERO goal plus a measured release must all hold "
            "on fresh samples of the same episode after step 2."
        ),
        "task_scope": "combo",
        "step_limit": 2,
        "instruction": instruction,
        "composite": {
            "schema": SCHEMA,
            "keys": ordered,
            "anchor_task_id": int(loaded_task["task_id"]),
            "anchor_bddl": loaded_task.get("bddl"),
            "bddl_mutation": False,
            "env_reset_between_steps": False,
            "plan_source": plan_source,
        },
        "steps": steps,
        "plan_source": plan_source,
        "planner_vision": False,
    }


def validate_plan(
    plan: dict[str, Any],
    *,
    catalog: dict[str, dict[str, Any]],
    loaded_task: dict[str, Any],
    cfg: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Structural checks the composite plan must pass before it runs."""
    errors: list[dict[str, Any]] = []

    def error(field: str, code: str, value: Any, expected: str) -> None:
        errors.append({"field": field, "code": code, "value": value, "expected": expected})

    if not isinstance(plan, dict):
        return [{"field": "plan", "code": "not_object", "value": type(plan).__name__, "expected": "object"}]
    if plan.get("task_scope") != "combo":
        error("task_scope", "not_combo", plan.get("task_scope"), "combo")
    block = plan.get("composite") or {}
    keys = [str(key) for key in (block.get("keys") or [])]
    check = support(keys, loaded_task=loaded_task, catalog=catalog, cfg=cfg)
    if not check.get("ok"):
        error("composite.keys", str(check.get("reason_code")), keys, "an executable pair")
    steps = plan.get("steps")
    if not isinstance(steps, list) or len(steps) != len(keys):
        return [*errors, {"field": "steps", "code": "count_mismatch", "value": steps, "expected": len(keys)}]
    for index, (step, key) in enumerate(zip(steps, keys)):
        prefix = f"steps[{index}]"
        if not isinstance(step, dict):
            error(prefix, "not_object", type(step).__name__, "object")
            continue
        if step.get("skill") != "whole_task":
            error(f"{prefix}.skill", "not_whole_task", step.get("skill"), "whole_task")
        if step.get("object_n") is not None or step.get("target_n") is not None:
            error(f"{prefix}.binding", "whole_task_takes_no_binding", True, "null object_n/target_n")
        official = str((catalog.get(key) or {}).get("official_instruction") or "")
        if not official or str(step.get("prompt") or "").strip() != official:
            error(f"{prefix}.prompt", "not_catalog_official", step.get("prompt"), official)
        expected_id = (catalog.get(key) or {}).get("task_id")
        actual_id = step.get("step_task_id")
        if expected_id is None or not isinstance(actual_id, int) or actual_id != int(expected_id):
            error(f"{prefix}.step_task_id", "task_mismatch", actual_id, key)
        if not isinstance(step.get("step_id"), str) or not step.get("step_id"):
            error(f"{prefix}.step_id", "missing", step.get("step_id"), "unique step identity")
    if isinstance(steps, list):
        ids = [str(s.get("step_id")) for s in steps if isinstance(s, dict)]
        if len(set(ids)) != len(ids):
            error("steps.step_id", "not_unique", ids, "distinct step ids")
    if keys and isinstance(steps, list) and steps:
        anchor_step = steps[-1] if isinstance(steps[-1], dict) else {}
        anchor_id = anchor_step.get("step_task_id")
        if not isinstance(anchor_id, int) or anchor_id != int(loaded_task["task_id"]):
            error("steps[-1].step_task_id", "not_loaded_scene", anchor_id, loaded_task["task_id"])
        if str((anchor_step.get("completion") or {}).get("kind")) != "native_goal":
            error("steps[-1].completion", "anchor_goal_must_be_native", anchor_step.get("completion"), "native_goal")
    return errors


def step_goal(step: dict[str, Any]) -> dict[str, Any] | None:
    """The engine-predicate goal for a step, or ``None`` for the native goal."""
    completion_block = step.get("completion") or {}
    kind = str(completion_block.get("kind") or "")
    if kind == "native_goal":
        return None
    if kind == "drawer_open":
        return {"skill": "open_drawer", "target_n": completion_block.get("target_n")}
    if kind == "stove_button":
        return {"skill": "turn_on", "target_n": None}
    if kind == "relation":
        return {
            "skill": completion_block.get("skill") or "pick_place",
            "object_n": completion_block.get("object_n"),
            "target_n": completion_block.get("target_n"),
            "relation": completion_block.get("relation") or "on",
        }
    return None


def release_required(step_task: dict[str, Any]) -> bool:
    """Whether this step's own kind needs a measured release to be complete."""
    return str(step_task.get("kind") or "") in {"pick_place", "composite"}


__all__ = [
    "DEFAULT_PAIRS",
    "SCHEMA",
    "build_plan",
    "completion_for",
    "options",
    "release_required",
    "step_budget",
    "step_goal",
    "support",
    "validate_plan",
]
