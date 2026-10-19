"""Plan contract. Illegal plans never reach the robot."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
THRESHOLDS = json.loads((ROOT / "thresholds.json").read_text(encoding="utf-8"))

REGISTERED_SKILLS = (
    "view_env_state",
    "segment",
    "pi0_pick",
    "move_to",
    "set_gripper",
    "pause_for_confirm",
)
ALLOWED_ACTIONS = frozenset(REGISTERED_SKILLS)
REJECTED_ACTIONS = frozenset(
    {"pi0_doubled", "execute_action", "python", "code", "reset", "shell"}
)
CHECKERS = frozenset(
    {
        "observe_ready",
        "object_localized",
        "grasp_held",
        "object_transported",
        "object_placed",
        "user_confirmed",
    }
)
PRECONDITIONS = frozenset(
    {
        "observe_ready",
        "object_localized",
        "grasp_held",
        "holding_none",
        "gripper_open",
    }
)
ON_FAILURE = frozenset({"stop", "recover", "replan"})
RELATIONS = frozenset({"in", "on", "inside_drawer"})
SKILL_CHECK = {
    "view_env_state": "observe_ready",
    "segment": "object_localized",
    "pi0_pick": "grasp_held",
    "move_to": "object_transported",
    "pause_for_confirm": "user_confirmed",
}
SEMANTIC_PLAN_KEYS = frozenset(
    {
        "plan_id",
        "goal",
        "steps",
        "goal_relation",
        "scene_id",
        "suite",
        "task_id",
        "seed",
        "task",
        "plan_source",
        "semantic_step_ids",
        "execution_mode",
        "interpretation",
        "subgoals",
        "failure_branches",
        "predicates",
        "user_text",
    }
)
SEMANTIC_STEP_KEYS = frozenset(
    {
        "step_id",
        "skill",
        "object_ref",
        "target_ref",
        "object_n",
        "target_n",
        "relation",
        "prompt",
        "depends_on",
        "preconditions",
        "completion_check",
        "budget",
        "on_failure",
        "user_phrase",
        "grounding_query",
        "vla_prompt",
    }
)
PLAN_KEYS = frozenset(
    {
        "plan_id",
        "goal",
        "steps",
        "objects",
        "goal_relation",
        "scene_id",
        "suite",
        "task_id",
        "seed",
        "task",
        "plan_source",
        "semantic_step_ids",
        "execution_mode",
        "predicates",
        "user_text",
    }
)
EXECUTABLE_KEYS = frozenset({"code", "python", "script", "exec", "eval"})


class PlanError(ValueError):
    def __init__(self, errors: str | list[dict[str, Any]]):
        if isinstance(errors, str):
            errors = [
                {
                    "path": "plan",
                    "code": "invalid",
                    "received": None,
                    "expected": errors,
                }
            ]
        self.errors = errors
        super().__init__(format_errors(errors))


def format_errors(errors: list[dict[str, Any]]) -> str:
    parts = []
    for item in errors:
        parts.append(
            f"{item.get('path') or 'plan'}: {item.get('code')} "
            f"received={item.get('received')!r} expected {item.get('expected')}"
        )
    return "; ".join(parts)


def _err(path: str, code: str, received: Any, expected: str) -> dict[str, Any]:
    if isinstance(received, str) and len(received) > 240:
        received = received[:240] + "..."
    return {"path": path, "code": code, "received": received, "expected": expected}


def _workspace() -> dict[str, list[float]]:
    return THRESHOLDS["workspace_m"]


def _in_workspace(xyz: list[float]) -> bool:
    box = _workspace()
    names = ("x", "y", "z")
    return all(box[name][0] <= float(value) <= box[name][1] for name, value in zip(names, xyz))


def _phrases(plan: dict[str, Any]) -> dict[str, str]:
    found = {}
    for ref, spec in (plan.get("objects") or {}).items():
        if not isinstance(spec, dict):
            continue
        for phrase in spec.get("phrases") or []:
            found[str(phrase).strip().lower()] = ref
    return found


def _object_refs(plan: dict[str, Any]) -> set[str]:
    return set((plan.get("objects") or {}).keys())


def example_plan() -> dict[str, Any]:
    """Short example that matches the current schema. Not a runnable scene plan."""
    return {
        "plan_id": "example-alphabet-soup",
        "goal": "pick up the alphabet soup",
        "goal_relation": "in",
        "objects": {
            "alphabet_soup": {
                "sim_name": "alphabet_soup_1",
                "phrases": ["alphabet soup", "alphabet soup can"],
                "role": "movable",
            }
        },
        "steps": [
            {
                "step_id": "see",
                "skill": "view_env_state",
                "args": {"step": -1},
                "depends_on": [],
                "preconditions": [],
                "completion_check": "observe_ready",
                "budget": {"max_s": 30, "max_attempts": 1},
                "on_failure": "stop",
                "subtask": "observe",
            },
            {
                "step_id": "locate",
                "skill": "segment",
                "args": {
                    "prompt": "alphabet soup can",
                    "camera": "agentview",
                    "object_ref": "alphabet_soup",
                },
                "depends_on": ["see"],
                "preconditions": ["observe_ready"],
                "completion_check": "object_localized",
                "budget": {"max_s": 30, "max_attempts": 1},
                "on_failure": "stop",
                "object_ref": "alphabet_soup",
                "subtask": "soup",
            },
            {
                "step_id": "grasp",
                "skill": "pi0_pick",
                "args": {
                    "prompt": "pick up the alphabet soup",
                    "max_chunks": 12,
                    "object_ref": "alphabet_soup",
                },
                "depends_on": ["locate"],
                "preconditions": ["object_localized"],
                "completion_check": "grasp_held",
                "budget": {"max_s": 120, "max_attempts": 2},
                "on_failure": "recover",
                "object_ref": "alphabet_soup",
                "subtask": "soup_pick",
            },
        ],
    }


def plan_parameters_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["plan_id", "goal", "steps"],
        "properties": {
            "plan_id": {"type": "string"},
            "goal": {"type": "string"},
            "goal_relation": {"type": "string", "enum": sorted(RELATIONS)},
            "objects": {"type": "object"},
            "steps": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "step_id",
                        "skill",
                        "args",
                        "depends_on",
                        "preconditions",
                        "completion_check",
                        "budget",
                        "on_failure",
                    ],
                    "properties": {
                        "step_id": {"type": "string"},
                        "skill": {"type": "string", "enum": list(REGISTERED_SKILLS)},
                        "args": {"type": "object"},
                        "depends_on": {"type": "array", "items": {"type": "string"}},
                        "preconditions": {
                            "type": "array",
                            "items": {"type": "string", "enum": sorted(PRECONDITIONS)},
                        },
                        "completion_check": {"type": "string", "enum": sorted(CHECKERS)},
                        "budget": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["max_s", "max_attempts"],
                            "properties": {
                                "max_s": {"type": "number"},
                                "max_attempts": {"type": "integer"},
                            },
                        },
                        "on_failure": {"type": "string", "enum": sorted(ON_FAILURE)},
                        "object_ref": {"type": "string"},
                        "subtask": {"type": "string"},
                        "relation": {"type": "string", "enum": sorted(RELATIONS)},
                    },
                },
            },
        },
    }


SEMANTIC_SKILLS = ("transfer",)


def planner_parameters_schema() -> dict[str, Any]:
    """Flat semantic steps. Object identity is checked against the live scene, not a nested map."""
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["plan_id", "goal", "goal_relation", "steps"],
        "properties": {
            "plan_id": {"type": "string"},
            "goal": {"type": "string"},
            "goal_relation": {"type": "string", "enum": sorted(RELATIONS)},
            "steps": {
                "type": "array",
                "minItems": 1,
                "maxItems": 4,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "step_id",
                        "skill",
                        "object_ref",
                        "target_ref",
                        "relation",
                        "prompt",
                        "depends_on",
                        "completion_check",
                        "budget",
                        "on_failure",
                    ],
                    "properties": {
                        "step_id": {"type": "string"},
                        "skill": {"type": "string", "enum": list(SEMANTIC_SKILLS)},
                        "object_ref": {"type": "string"},
                        "target_ref": {"type": "string"},
                        "object_n": {"type": "integer"},
                        "target_n": {"type": "integer"},
                        "relation": {"type": "string", "enum": sorted(RELATIONS)},
                        "prompt": {"type": "string"},
                        "depends_on": {"type": "array", "items": {"type": "string"}},
                        "preconditions": {"type": "array", "items": {"type": "string"}},
                        "completion_check": {"type": "string", "enum": ["object_placed"]},
                        "budget": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["max_s", "max_attempts"],
                            "properties": {
                                "max_s": {"type": "number"},
                                "max_attempts": {"type": "integer"},
                            },
                        },
                        "on_failure": {"type": "string", "enum": sorted(ON_FAILURE)},
                    },
                },
            },
        },
    }


def free_submit_plan_tools() -> list[dict[str, Any]]:
    """One plan for the user's own goal. Relations may differ across steps."""
    schema = planner_parameters_schema()
    schema["required"] = ["plan_id", "goal", "steps"]
    schema["properties"]["interpretation"] = {"type": "string"}
    schema["properties"]["failure_branches"] = {
        "type": "array",
        "maxItems": 2,
        "items": {
            "type": "object",
            "additionalProperties": False,
            "required": ["step_id", "action"],
            "properties": {
                "step_id": {"type": "string"},
                "action": {"type": "string", "enum": ["stop", "recover"]},
            },
        },
    }
    step = schema["properties"]["steps"]["items"]
    step["properties"]["user_phrase"] = {"type": "string"}
    step["properties"]["grounding_query"] = {"type": "string"}
    step["properties"]["vla_prompt"] = {"type": "string"}
    step["required"] = [
        "step_id",
        "skill",
        "object_n",
        "target_n",
        "object_ref",
        "target_ref",
        "relation",
        "prompt",
        "depends_on",
        "completion_check",
        "budget",
        "on_failure",
    ]
    return [
        {
            "type": "function",
            "function": {
                "name": "submit_plan",
                "description": (
                    "Submit ordered transfer steps for the user goal on the current scene. "
                    "Use candidate numbers. Do not invent tools, do not send code, and do not "
                    "replace the user goal with an official task sentence."
                ),
                "parameters": schema,
            },
        },
        {
            "type": "function",
            "function": {
                "name": "ask_clarification",
                "description": "Ask one concrete question when a mention does not map to one candidate.",
                "parameters": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["question"],
                    "properties": {"question": {"type": "string"}},
                },
            },
        },
    ]


def submit_plan_tools() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "submit_plan",
                "description": (
                    "Submit one flat transfer per placement. A transfer is the whole grasp, "
                    "carry, place, and check. Do not split one placement into a pick transfer "
                    "and a place transfer. Do not nest submit_plan, and do not send code."
                ),
                "parameters": planner_parameters_schema(),
            },
        }
    ]


def semantic_example() -> dict[str, Any]:
    return {
        "plan_id": "example-soup",
        "goal": "put the alphabet soup in the basket",
        "goal_relation": "in",
        "steps": [
            {
                "step_id": "soup",
                "skill": "transfer",
                "object_ref": "alphabet_soup",
                "target_ref": "basket",
                "relation": "in",
                "prompt": "pick up the alphabet soup",
                "depends_on": [],
                "preconditions": [],
                "completion_check": "object_placed",
                "budget": {"max_s": 180, "max_attempts": 2},
                "on_failure": "recover",
            }
        ],
    }


def example_scene_objects() -> dict[str, Any]:
    return {
        "alphabet_soup": {"sim_name": "alphabet_soup_1", "phrases": ["alphabet soup"], "role": "movable"},
        "basket": {"sim_name": "basket_1", "phrases": ["basket"], "role": "container"},
    }


TOOL_SCHEMA = json.dumps(
    {
        "planner_skills": list(SEMANTIC_SKILLS),
        "executor_expands_transfer_to": list(REGISTERED_SKILLS),
        "rejected": sorted(REJECTED_ACTIONS),
        "completion_check": "object_placed",
        "transfer_args": ["object_ref", "target_ref", "relation", "prompt"],
        "pi0_doubled": "not enabled",
    },
    ensure_ascii=False,
)


def validate_plan(plan: dict[str, Any], *, task: str | None = None) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    if not isinstance(plan, dict):
        raise PlanError([_err("plan", "type", type(plan).__name__, "object")])
    for key in plan:
        if key in EXECUTABLE_KEYS or key not in PLAN_KEYS:
            errors.append(_err(f"plan.{key}", "unknown_field", key, "plan contract fields only"))
    if not isinstance(plan.get("plan_id"), str) or not plan.get("plan_id", "").strip():
        errors.append(_err("plan.plan_id", "missing", plan.get("plan_id"), "non-empty string"))
    goal = plan.get("goal")
    if not isinstance(goal, str) or not goal.strip():
        errors.append(_err("plan.goal", "missing", goal, "non-empty string"))
    elif task is not None and plan.get("execution_mode") != "free" and goal.strip() != task.strip():
        errors.append(_err("plan.goal", "task_mismatch", goal, task.strip()))
    if isinstance(plan.get("task"), str) and isinstance(goal, str) and plan["task"].strip() != goal.strip():
        errors.append(_err("plan.task", "goal_mismatch", plan.get("task"), "same text as goal"))
    relation = plan.get("goal_relation")
    if plan.get("execution_mode") != "free" and relation not in RELATIONS:
        errors.append(_err("plan.goal_relation", "relation", relation, "in|on|inside_drawer"))
    objects = plan.get("objects")
    if not isinstance(objects, dict) or not objects:
        errors.append(_err("plan.objects", "type", objects, "non-empty object map"))
    else:
        for ref, spec in objects.items():
            if not isinstance(spec, dict):
                errors.append(_err(f"plan.objects.{ref}", "type", spec, "object"))
                continue
            if not isinstance(spec.get("sim_name"), str) or not spec.get("sim_name"):
                errors.append(_err(f"plan.objects.{ref}.sim_name", "missing", spec.get("sim_name"), "sim object name"))
            phrases = spec.get("phrases")
            if not isinstance(phrases, list) or not phrases or not all(isinstance(item, str) and item.strip() for item in phrases):
                errors.append(_err(f"plan.objects.{ref}.phrases", "type", phrases, "non-empty string list"))
            if "movable" not in spec and spec.get("role") not in {"movable", "container", "support"}:
                errors.append(_err(f"plan.objects.{ref}.role", "role", spec.get("role"), "movable|container|support or explicit movable"))
    steps = plan.get("steps")
    if not isinstance(steps, list) or not steps:
        errors.append(_err("plan.steps", "type", steps, "non-empty list"))
        raise PlanError(errors or [_err("plan.steps", "type", steps, "non-empty list")])
    seen: set[str] = set()
    for index, step in enumerate(steps):
        path = f"steps[{index}]"
        if not isinstance(step, dict):
            errors.append(_err(path, "type", type(step).__name__, "object"))
            continue
        step_id = step.get("step_id", step.get("id"))
        skill = step.get("skill", step.get("action"))
        if not isinstance(step_id, str) or not step_id.strip():
            errors.append(_err(f"{path}.step_id", "missing", step_id, "non-empty string"))
            step_id = f"missing-{index}"
        elif step_id in seen:
            errors.append(_err(f"{path}.step_id", "duplicate_step_id", step_id, "unique step_id"))
        seen.add(step_id)
        if skill in REJECTED_ACTIONS:
            errors.append(_err(f"{path}.skill", "skill_rejected", skill, "registered skills only; pi0_doubled is disabled"))
        elif skill not in ALLOWED_ACTIONS:
            errors.append(_err(f"{path}.skill", "skill_not_registered", skill, ",".join(REGISTERED_SKILLS)))
        args = step.get("args", {})
        if not isinstance(args, dict):
            errors.append(_err(f"{path}.args", "type", type(args).__name__, "object"))
            args = {}
        for key in args:
            if key in EXECUTABLE_KEYS:
                errors.append(_err(f"{path}.args.{key}", "executable_field", key, "no code fields"))
        depends = step.get("depends_on", [])
        if not isinstance(depends, list) or not all(isinstance(item, str) for item in depends):
            errors.append(_err(f"{path}.depends_on", "type", depends, "list of step_id strings"))
            depends = []
        pre = step.get("preconditions", [])
        if not isinstance(pre, list):
            errors.append(_err(f"{path}.preconditions", "type", pre, "list of checker names"))
        else:
            for item in pre:
                if item not in PRECONDITIONS:
                    errors.append(_err(f"{path}.preconditions", "precondition", item, ",".join(sorted(PRECONDITIONS))))
        check = step.get("completion_check")
        expected = SKILL_CHECK.get(skill) if isinstance(skill, str) else None
        if skill == "set_gripper":
            gripper = args.get("gripper") if isinstance(args, dict) else None
            expected = "object_placed" if _gripper_value(gripper) == -1.0 else "grasp_held"
        if skill == "move_to" and isinstance(args, dict) and _gripper_value(args.get("gripper")) == -1.0:
            expected = "object_placed"
        if check not in CHECKERS:
            errors.append(_err(f"{path}.completion_check", "checker", check, ",".join(sorted(CHECKERS))))
        elif expected and check != expected:
            errors.append(_err(f"{path}.completion_check", "checker_skill_mismatch", check, expected))
        budget = step.get("budget")
        errors.extend(_budget_errors(f"{path}.budget", budget))
        failure = step.get("on_failure")
        if failure not in ON_FAILURE:
            errors.append(_err(f"{path}.on_failure", "on_failure", failure, "stop|recover|replan"))
        if isinstance(skill, str) and skill in ALLOWED_ACTIONS:
            errors.extend(_arg_errors(path, skill, args, plan, step))
        if skill == "pause_for_confirm" and not step.get("subtask"):
            errors.append(_err(f"{path}.subtask", "missing", step.get("subtask"), "subtask name"))
    known_ids = [step.get("step_id", step.get("id")) for step in steps if isinstance(step, dict)]
    id_set = {item for item in known_ids if isinstance(item, str)}
    graph: dict[str, list[str]] = {}
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            continue
        step_id = step.get("step_id", step.get("id"))
        depends = step.get("depends_on") or []
        if not isinstance(step_id, str) or not isinstance(depends, list):
            continue
        for dep in depends:
            if dep not in id_set:
                errors.append(_err(f"steps[{index}].depends_on", "missing_step", dep, "an existing step_id"))
            if dep == step_id:
                errors.append(_err(f"steps[{index}].depends_on", "cycle", dep, "no self dependency"))
        graph[step_id] = [dep for dep in depends if isinstance(dep, str)]
    errors.extend(_cycle_errors(graph))
    if errors:
        raise PlanError(errors)
    for step in plan["steps"]:
        step["step_id"] = step.get("step_id", step.get("id"))
        step["skill"] = step.get("skill", step.get("action"))
        step["id"] = step["step_id"]
        step["action"] = step["skill"]
        step.setdefault("depends_on", [])
        step.setdefault("preconditions", [])
    return plan


def _budget_errors(path: str, budget: Any) -> list[dict[str, Any]]:
    if not isinstance(budget, dict):
        return [_err(path, "type", budget, "object with max_s and max_attempts")]
    errors = []
    max_s = budget.get("max_s")
    max_attempts = budget.get("max_attempts")
    cap_s = float(THRESHOLDS["budget_max_s"])
    cap_n = int(THRESHOLDS["budget_max_attempts"])
    if isinstance(max_s, bool) or not isinstance(max_s, (int, float)) or not (0 < float(max_s) <= cap_s):
        errors.append(_err(f"{path}.max_s", "budget_cap", max_s, f"seconds in (0, {cap_s}]"))
    if isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or not (1 <= max_attempts <= cap_n):
        errors.append(_err(f"{path}.max_attempts", "budget_cap", max_attempts, f"integer in [1, {cap_n}]"))
    return errors


def _gripper_value(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if number in (-1.0, 1.0):
        return number
    return None


def _arg_errors(path: str, skill: str, args: dict[str, Any], plan: dict[str, Any], step: dict[str, Any]) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    refs = _object_refs(plan)
    phrases = _phrases(plan)
    object_ref = step.get("object_ref", args.get("object_ref"))

    def need_ref() -> None:
        if skill in {"view_env_state", "pause_for_confirm"}:
            return
        if not isinstance(object_ref, str) or object_ref not in refs:
            errors.append(_err(f"{path}.object_ref", "object_ref", object_ref, "a semantic object in plan.objects"))

    if skill == "view_env_state":
        step_no = args.get("step", -1)
        if isinstance(step_no, bool) or not isinstance(step_no, int) or step_no < -1:
            errors.append(_err(f"{path}.args.step", "type", step_no, "int >= -1"))
        extra = set(args) - {"step"}
        for key in extra:
            errors.append(_err(f"{path}.args.{key}", "unexpected_arg", key, "step"))
    elif skill == "segment":
        need_ref()
        prompt = args.get("prompt")
        owned: set[str] = set()
        if isinstance(object_ref, str):
            spec = (plan.get("objects") or {}).get(object_ref) or {}
            owned = {
                str(item).strip().lower()
                for item in (spec.get("phrases") or [])
                if isinstance(item, str) and item.strip()
            }
        if not isinstance(prompt, str) or prompt.strip().lower() not in phrases:
            errors.append(_err(f"{path}.args.prompt", "phrase", prompt, "a phrase of a scene object"))
        elif prompt.strip().lower() not in owned:
            errors.append(_err(f"{path}.args.prompt", "phrase_object_mismatch", prompt, object_ref))
        if args.get("camera") not in {"agentview", "wrist"}:
            errors.append(_err(f"{path}.args.camera", "camera", args.get("camera"), "agentview|wrist"))
        extra = set(args) - {"prompt", "camera", "object_ref", "step"}
        for key in extra:
            errors.append(_err(f"{path}.args.{key}", "unexpected_arg", key, "prompt, camera, object_ref"))
    elif skill == "pi0_pick":
        need_ref()
        if not isinstance(args.get("prompt"), str) or not args.get("prompt", "").strip():
            errors.append(_err(f"{path}.args.prompt", "missing", args.get("prompt"), "non-empty grasp prompt"))
        chunks = args.get("max_chunks", 12)
        if isinstance(chunks, bool) or not isinstance(chunks, int) or not 1 <= chunks <= 24:
            errors.append(_err(f"{path}.args.max_chunks", "range", chunks, "integer 1..24"))
        extra = set(args) - {"prompt", "max_chunks", "object_ref"}
        for key in extra:
            errors.append(_err(f"{path}.args.{key}", "unexpected_arg", key, "prompt, max_chunks, object_ref"))
    elif skill == "move_to":
        need_ref()
        has_xyz = "xyz" in args
        has_from = "xyz_from" in args
        if has_xyz == has_from:
            errors.append(_err(f"{path}.args", "xyz_source", list(args), "exactly one of xyz or xyz_from"))
        if has_xyz:
            xyz = args.get("xyz")
            if not isinstance(xyz, list) or len(xyz) != 3 or not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v)) for v in xyz):
                errors.append(_err(f"{path}.args.xyz", "type", xyz, "finite [x,y,z] meters"))
            elif not _in_workspace([float(v) for v in xyz]):
                errors.append(_err(f"{path}.args.xyz", "workspace", xyz, str(_workspace())))
        if has_from:
            source = args.get("xyz_from")
            phrase = source.split(":", 1)[1].strip().lower() if isinstance(source, str) and source.startswith("segment:") else None
            if phrase not in phrases:
                errors.append(_err(f"{path}.args.xyz_from", "semantic_ref", source, "segment:<scene phrase>"))
            z_offset = args.get("z_offset", 0.0)
            low, high = THRESHOLDS["approach_z_offset_m"]
            if isinstance(z_offset, bool) or not isinstance(z_offset, (int, float)) or not low <= float(z_offset) <= high:
                errors.append(_err(f"{path}.args.z_offset", "range", z_offset, f"meters in [{low}, {high}]"))
        if _gripper_value(args.get("gripper")) is None:
            errors.append(_err(f"{path}.args.gripper", "gripper", args.get("gripper"), "-1 open or +1 close"))
        motion = args.get("motion", "travel")
        if motion not in {"travel", "vertical"}:
            errors.append(_err(f"{path}.args.motion", "motion", motion, "travel|vertical"))
        carry = args.get("carry_offset")
        if carry is not None and not isinstance(carry, str):
            errors.append(_err(f"{path}.args.carry_offset", "type", carry, "object_ref"))
        xy_offset = args.get("xy_offset")
        if xy_offset is not None:
            if (
                not isinstance(xy_offset, list)
                or len(xy_offset) != 2
                or any(isinstance(v, bool) or not isinstance(v, (int, float)) or abs(float(v)) > 0.08 for v in xy_offset)
            ):
                errors.append(_err(f"{path}.args.xy_offset", "range", xy_offset, "two meters, each within 0.08"))
        extra = set(args) - {"xyz", "xyz_from", "z_offset", "gripper", "object_ref", "max_steps", "motion", "carry_offset", "xy_offset"}
        for key in extra:
            errors.append(_err(f"{path}.args.{key}", "unexpected_arg", key, "xyz source, gripper"))
        if "max_steps" in args and (isinstance(args["max_steps"], bool) or not isinstance(args["max_steps"], int) or not 1 <= args["max_steps"] <= 80):
            errors.append(_err(f"{path}.args.max_steps", "range", args.get("max_steps"), "integer 1..80"))
    elif skill == "set_gripper":
        need_ref()
        if _gripper_value(args.get("gripper")) is None:
            errors.append(_err(f"{path}.args.gripper", "gripper", args.get("gripper"), "-1 open or +1 close"))
        steps = args.get("steps", 8)
        if isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= 40:
            errors.append(_err(f"{path}.args.steps", "range", steps, "integer 1..40"))
        if _gripper_value(args.get("gripper")) == -1.0:
            relation = step.get("relation")
            goal = plan.get("goal_relation")
            if relation not in RELATIONS:
                errors.append(_err(f"{path}.relation", "relation", relation, "in|on|inside_drawer"))
            elif plan.get("execution_mode") != "free" and relation != goal:
                errors.append(_err(f"{path}.relation", "relation_mismatch", relation, f"scene goal_relation {goal}"))
        extra = set(args) - {"gripper", "steps", "object_ref"}
        for key in extra:
            errors.append(_err(f"{path}.args.{key}", "unexpected_arg", key, "gripper, steps"))
    elif skill == "pause_for_confirm":
        extra = set(args)
        for key in extra:
            errors.append(_err(f"{path}.args.{key}", "unexpected_arg", key, "empty args"))
    return errors


def _cycle_errors(graph: dict[str, list[str]]) -> list[dict[str, Any]]:
    errors = []
    state = {key: 0 for key in graph}

    def walk(node: str, stack: list[str]) -> None:
        state[node] = 1
        stack.append(node)
        for dep in graph.get(node, []):
            if dep not in state:
                continue
            if state[dep] == 1:
                errors.append(_err("steps", "cycle", stack[stack.index(dep):] + [dep], "acyclic depends_on"))
            elif state[dep] == 0:
                walk(dep, stack)
        stack.pop()
        state[node] = 2

    for node in list(graph):
        if state[node] == 0:
            walk(node, [])
    return errors


PARSE_CODES = frozenset(
    {"json_decode", "tool_call", "incomplete_stream", "truncated_output", "nested_tool", "type"}
)


def failure_stage(errors: list[dict[str, Any]] | None) -> str:
    if not errors:
        return "ok"
    if any(item.get("code") in PARSE_CODES for item in errors):
        return "parse_failed"
    return "validation_failed"


def transfer_slots(language: str) -> list[dict[str, str]] | None:
    """Placements the instruction actually names. None means this is not a transfer task."""
    text = language.strip().lower()
    text = re.sub(r" and close it$", "", text)
    both = re.fullmatch(r"put both the (.+) and the (.+) in the (.+)", text)
    if both:
        target = _dest_noun(both.group(3))
        return [
            {"object": both.group(1), "target": target, "relation": "in"},
            {"object": both.group(2), "target": target, "relation": "in"},
        ]
    if text == "put both moka pots on the stove":
        return [
            {"object": "moka pot", "target": "stove", "relation": "on"},
            {"object": "moka pot", "target": "stove", "relation": "on"},
        ]
    picked = re.fullmatch(r"pick up the (.+) and place it (on|in) the (.+)", text)
    if picked:
        clause, rel, target = picked.groups()
        noun = re.split(r" between | next to | on the | in the | from ", clause, maxsplit=1)[0].strip()
        return [{"object": noun, "target": _dest_noun(target), "relation": "on" if rel == "on" else "in"}]
    topped = re.fullmatch(r"put the (.+) on top of the (.+)", text)
    if topped:
        return [{"object": topped.group(1), "target": _dest_noun(topped.group(2)), "relation": "on"}]
    placed = re.fullmatch(r"put the (.+) (on|in) the (.+)", text)
    if placed:
        relation = "inside_drawer" if "drawer" in placed.group(3) else ("on" if placed.group(2) == "on" else "in")
        return [{"object": placed.group(1), "target": _dest_noun(placed.group(3)), "relation": relation}]
    opened = re.fullmatch(r"open the top drawer and put the (.+) inside", text)
    if opened:
        return [{"object": opened.group(1), "target": "cabinet", "relation": "inside_drawer"}]
    stove = re.fullmatch(r"turn on the stove and put the (.+) on it", text)
    if stove:
        return [{"object": stove.group(1), "target": "stove", "relation": "on"}]
    parts = re.split(r" and put ", text)
    if len(parts) == 2:
        slots = [_one_put(parts[0]), _one_put("put " + parts[1])]
        if all(slots):
            return slots
    return None


def _one_put(text: str) -> dict[str, str] | None:
    match = re.fullmatch(r"(?:put |pick up )the (.+?) (?:on|in) the (.+)", text.strip())
    if not match:
        return None
    relation = "on" if " on the " in f" {text} " else "in"
    if "drawer" in match.group(2):
        relation = "inside_drawer"
    return {"object": match.group(1), "target": _dest_noun(match.group(2)), "relation": relation}


def _dest_noun(text: str) -> str:
    if "caddy" in text:
        return "caddy"
    if "drawer" in text or "cabinet" in text:
        return "cabinet"
    if "microwave" in text:
        return "microwave"
    if "stove" in text:
        return "stove"
    if "basket" in text:
        return "basket"
    if "plate" in text:
        return "plate"
    if "bowl" in text:
        return "bowl"
    if "rack" in text:
        return "rack"
    return text.strip()


def _phrase_match(spec: dict[str, Any], needle: str) -> bool:
    needle = needle.strip().lower()
    phrases = [str(item).lower() for item in spec.get("phrases") or []]
    semantic = str(spec.get("semantic_type") or "").replace("_", " ")
    return any(phrase and (phrase in needle or needle in phrase) for phrase in phrases) or bool(
        semantic and (semantic in needle or needle in semantic)
    )


def _is_movable(spec: dict[str, Any]) -> bool:
    if "movable" in spec:
        return bool(spec["movable"])
    return spec.get("role") == "movable"


def _accepts_target(spec: dict[str, Any], relation: str) -> bool:
    if "can_be_target" in spec:
        if not spec["can_be_target"]:
            return False
        relations = spec.get("target_relations")
        return True if relations is None else relation in relations
    role = spec.get("role")
    if relation == "in":
        return role == "container"
    return role in {"container", "support"}


def _capability(spec: dict[str, Any]) -> dict[str, Any]:
    return {
        "semantic_type": spec.get("semantic_type"),
        "movable": spec.get("movable", spec.get("role") == "movable"),
        "can_be_target": spec.get("can_be_target"),
        "capability_source": spec.get("capability_source") or spec.get("role"),
    }


def _nested_tool_errors(payload: Any) -> list[dict[str, Any]]:
    """Reject a wrapped plan. Do not read the inner object as another plan."""
    if not isinstance(payload, dict):
        return []
    errors = []
    for key in ("submit_plan", "tool_calls", "function"):
        if key in payload:
            errors.append(_err(f"plan.{key}", "nested_tool", key, "flat submit_plan arguments; inner plans are not extracted"))
    steps = payload.get("steps")
    if isinstance(steps, list):
        for index, step in enumerate(steps):
            if isinstance(step, dict) and (
                step.get("skill") == "submit_plan"
                or any(key in step for key in ("submit_plan", "tool_calls", "function", "arguments"))
            ):
                errors.append(
                    _err(
                        f"steps[{index}]",
                        "nested_tool",
                        step.get("skill") or "wrapper",
                        "a flat transfer step; nested tool calls are not extracted",
                    )
                )
    return errors


def validate_semantic_plan(
    plan: dict[str, Any],
    *,
    task: str | None = None,
    scene_objects: dict[str, Any] | None = None,
    required_object_ref: str | None = None,
    execution_mode: str = "benchmark",
    predicates: list[dict[str, Any]] | None = None,
    ordered: bool = False,
) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    free = execution_mode == "free"
    if not isinstance(plan, dict):
        raise PlanError([_err("plan", "type", type(plan).__name__, "object")])
    for key in plan:
        if key == "objects" or key in EXECUTABLE_KEYS or key not in SEMANTIC_PLAN_KEYS:
            errors.append(_err(f"plan.{key}", "unknown_field", key, "flat semantic fields only; object refs are checked against the scene"))
    goal = plan.get("goal")
    if not isinstance(plan.get("plan_id"), str) or not str(plan.get("plan_id")).strip():
        errors.append(_err("plan.plan_id", "missing", plan.get("plan_id"), "non-empty string"))
    if not isinstance(goal, str) or not goal.strip():
        errors.append(_err("plan.goal", "missing", goal, "non-empty string"))
    elif task is not None and not free and goal.strip() != task.strip():
        errors.append(_err("plan.goal", "task_mismatch", goal, task.strip()))
    if not free and plan.get("goal_relation") not in RELATIONS:
        errors.append(_err("plan.goal_relation", "relation", plan.get("goal_relation"), "in|on|inside_drawer"))
    elif free and "goal_relation" in plan and plan.get("goal_relation") not in RELATIONS:
        errors.append(_err("plan.goal_relation", "relation", plan.get("goal_relation"), "omit it, or in|on|inside_drawer"))
    if free and plan.get("failure_branches") not in (None, []):
        branches = plan.get("failure_branches")
        if not isinstance(branches, list) or len(branches) > 2:
            errors.append(_err("plan.failure_branches", "type", branches, "at most two stop|recover branches"))
        elif isinstance(branches, list):
            for index, branch in enumerate(branches):
                action = branch.get("action") if isinstance(branch, dict) else None
                if action not in {"stop", "recover"}:
                    errors.append(_err(f"plan.failure_branches[{index}].action", "on_failure", action, "stop|recover"))
    objects = scene_objects or {}
    if scene_objects is not None:
        for ref, spec in objects.items():
            if not isinstance(spec, dict):
                errors.append(_err(f"scene.{ref}", "type", spec, "object"))
                continue
            if "movable" not in spec and spec.get("role") not in {"movable", "container", "support"}:
                errors.append(_err(f"scene.{ref}.role", "role", spec.get("role"), "movable|container|support or explicit movable"))
    steps = plan.get("steps")
    if not isinstance(steps, list) or not steps:
        raise PlanError(errors or [_err("plan.steps", "type", steps, "non-empty list")])
    seen: set[str] = set()
    graph: dict[str, list[str]] = {}
    for index, step in enumerate(steps):
        path = f"steps[{index}]"
        if not isinstance(step, dict):
            errors.append(_err(path, "type", type(step).__name__, "object"))
            continue
        step_id = step.get("step_id")
        skill = step.get("skill")
        for key in step:
            if key not in SEMANTIC_STEP_KEYS:
                errors.append(_err(f"{path}.{key}", "unknown_field", key, "flat step fields only"))
        if not isinstance(step_id, str) or not step_id.strip():
            errors.append(_err(f"{path}.step_id", "missing", step_id, "non-empty string"))
            step_id = f"missing-{index}"
        elif step_id in seen:
            errors.append(_err(f"{path}.step_id", "duplicate_step_id", step_id, "unique step_id"))
        seen.add(str(step_id))
        if skill != "transfer":
            if free:
                from skills import skill_rejection

                errors.append(_err(f"{path}.skill", "skill_unimplemented", skill, skill_rejection(str(skill)) or "transfer"))
            else:
                errors.append(_err(f"{path}.skill", "skill_not_registered", skill, "transfer"))
        object_ref = step.get("object_ref")
        target_ref = step.get("target_ref")
        args = {
            "object_ref": object_ref,
            "target_ref": target_ref,
            "relation": step.get("relation"),
            "prompt": step.get("prompt"),
        }
        if object_ref not in objects:
            errors.append(_err(f"{path}.object_ref", "object_ref", object_ref, "an instance id in the current scene"))
        elif not _is_movable(objects[object_ref]):
            errors.append(
                _err(
                    f"{path}.object_ref",
                    "not_movable",
                    _capability(objects[object_ref]),
                    "movable true from the scene definition; container type alone is not movable",
                )
            )
        elif required_object_ref and object_ref != required_object_ref:
            errors.append(_err(f"{path}.object_ref", "binding_mismatch", object_ref, required_object_ref))
        if target_ref not in objects:
            errors.append(_err(f"{path}.target_ref", "object_ref", target_ref, "an instance id in the current scene"))
        elif not _accepts_target(objects[target_ref], str(step.get("relation") or "")):
            errors.append(
                _err(
                    f"{path}.target_ref",
                    "relation_unsupported",
                    _capability(objects[target_ref]),
                    f"can_be_target for relation {step.get('relation')}",
                )
            )
        if not free and args.get("relation") != plan.get("goal_relation"):
            errors.append(_err(f"{path}.args.relation", "relation_mismatch", args.get("relation"), plan.get("goal_relation")))
        if not isinstance(args.get("prompt"), str) or not str(args.get("prompt") or "").strip():
            errors.append(_err(f"{path}.args.prompt", "missing", args.get("prompt"), "non-empty grasp prompt"))
        if step.get("completion_check") != "object_placed":
            errors.append(_err(f"{path}.completion_check", "checker", step.get("completion_check"), "object_placed"))
        errors.extend(_budget_errors(f"{path}.budget", step.get("budget")))
        if step.get("on_failure") not in ON_FAILURE:
            errors.append(_err(f"{path}.on_failure", "on_failure", step.get("on_failure"), "stop|recover|replan"))
        depends = step.get("depends_on", [])
        if not isinstance(depends, list) or not all(isinstance(item, str) for item in depends):
            errors.append(_err(f"{path}.depends_on", "type", depends, "list of step_id strings"))
            depends = []
        graph[str(step_id)] = list(depends)
    id_set = set(graph)
    for step_id, depends in graph.items():
        for dep in depends:
            if dep not in id_set:
                errors.append(_err("steps.depends_on", "missing_step", dep, "an existing step_id"))
            if dep == step_id:
                errors.append(_err("steps.depends_on", "cycle", dep, "no self dependency"))
    errors.extend(_cycle_errors(graph))
    triples = []
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            continue
        triple = (step.get("object_ref"), step.get("target_ref"), step.get("relation"))
        if triple in triples:
            errors.append(
                _err(
                    f"steps[{index}]",
                    "duplicate_transfer",
                    {"object_ref": triple[0], "target_ref": triple[1], "relation": triple[2]},
                    "one transfer for the same object, target, and relation; duplicates are not merged",
                )
            )
        triples.append(triple)
    if free:
        from free_task import coverage_errors

        errors.extend(coverage_errors(plan, predicates or [], ordered=ordered))
    elif task is not None:
        slots = transfer_slots(task)
        if not slots:
            errors.append(_err("plan.goal", "task_form", task, "a registered pick-and-place instruction"))
        else:
            open_slots = list(slots)
            for index, step in enumerate(steps):
                if not isinstance(step, dict):
                    continue
                obj = objects.get(step.get("object_ref")) or {}
                dst = objects.get(step.get("target_ref")) or {}
                found = None
                for slot in open_slots:
                    if (
                        _phrase_match(obj, slot["object"])
                        and _phrase_match(dst, slot["target"])
                        and step.get("relation") == slot["relation"]
                    ):
                        found = slot
                        break
                if found is None:
                    code = "extra_transfer" if not open_slots else "task_target_mismatch"
                    if open_slots and obj and not _phrase_match(obj, open_slots[0]["object"]):
                        code = "task_object_mismatch"
                    errors.append(
                        _err(
                            f"steps[{index}]",
                            code,
                            {"object_ref": step.get("object_ref"), "target_ref": step.get("target_ref"), "relation": step.get("relation")},
                            "a placement named by the task",
                        )
                    )
                else:
                    open_slots.remove(found)
            for slot in open_slots:
                errors.append(_err("plan.steps", "task_incomplete", slot, "a transfer for every placement in the task"))
    if errors:
        raise PlanError(errors)
    used = []
    normalized_steps = []
    for step in steps:
        item = dict(step)
        item["args"] = {
            "object_ref": step.get("object_ref"),
            "target_ref": step.get("target_ref"),
            "relation": step.get("relation"),
            "prompt": step.get("prompt"),
        }
        for key in ("user_phrase", "grounding_query", "vla_prompt"):
            if isinstance(step.get(key), str) and step.get(key).strip():
                item["args"][key] = step[key].strip()
        used.extend([step.get("object_ref"), step.get("target_ref")])
        normalized_steps.append(item)
    normalized = {key: plan[key] for key in plan if key in SEMANTIC_PLAN_KEYS}
    normalized["objects"] = {ref: _with_role(objects[ref]) for ref in used if ref in objects}
    normalized["steps"] = normalized_steps
    if free:
        normalized["execution_mode"] = "free"
        if predicates:
            normalized["predicates"] = list(predicates)
            normalized["predicate_source"] = "request"
        else:
            normalized["predicates"] = [
                {
                    "object_ref": step.get("object_ref"),
                    "target_ref": step.get("target_ref"),
                    "relation": step.get("relation"),
                    "hold": "end",
                    "order": index,
                    "source": "plan_steps",
                }
                for index, step in enumerate(normalized_steps)
            ]
            normalized["predicate_source"] = "plan_steps"
        if not normalized.get("goal_relation") and normalized_steps:
            normalized["goal_relation"] = normalized_steps[0].get("relation")
    return normalized


def _with_role(spec: dict[str, Any]) -> dict[str, Any]:
    copied = dict(spec)
    if copied.get("role") in {"movable", "container", "support"}:
        return copied
    relations = copied.get("target_relations") or []
    if copied.get("can_be_target") and not copied.get("movable"):
        copied["role"] = "container" if ("in" in relations or "inside_drawer" in relations) else "support"
    elif copied.get("semantic_type") in {"plate", "flat_stove"}:
        copied["role"] = "support"
    elif copied.get("movable"):
        copied["role"] = "movable"
    else:
        copied["role"] = "support"
    return copied


def _grounding_query(spec: dict[str, Any]) -> str:
    """Visual phrase for SAM3. Not the instance id and not the longest catalog alias."""
    explicit = spec.get("grounding_query")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    phrases = [item.strip() for item in spec.get("phrases") or [] if isinstance(item, str) and item.strip()]
    sim_name = str(spec.get("sim_name") or "")
    sim_words = sim_name.replace("_", " ")
    visual = [item for item in phrases if item != sim_name and item.lower() != sim_words.lower()]
    pool = visual or phrases
    if not pool:
        return sim_words
    return min(pool, key=len)


def expand_semantic_plan(plan: dict[str, Any]) -> dict[str, Any]:
    """Turn transfer subtasks into the verified primitive sequence."""
    objects = plan["objects"]
    steps: list[dict[str, Any]] = [
        {
            "step_id": "observe",
            "skill": "view_env_state",
            "args": {"step": -1},
            "depends_on": [],
            "preconditions": [],
            "completion_check": "observe_ready",
            "budget": {"max_s": 30, "max_attempts": 1},
            "on_failure": "stop",
            "subtask": "observe",
        }
    ]
    segmented: set[str] = set()
    previous = "observe"
    placed_on: dict[str, int] = {}
    for item in plan["steps"]:
        args = item["args"]
        obj = args["object_ref"]
        target = args["target_ref"]
        obj_query = args.get("grounding_query") or _grounding_query(objects[obj])
        target_query = _grounding_query(objects[target])
        vla_prompt = _short_pick(args.get("vla_prompt") or args.get("prompt") or "", obj_query)
        slot = placed_on.get(target, 0)
        placed_on[target] = slot + 1
        xy_offset = [round(0.04 * slot, 3), 0.0] if slot else None
        if obj not in segmented:
            steps.append(_segment_step(obj, obj_query, previous))
            segmented.add(obj)
        if target not in segmented:
            steps.append(_segment_step(target, target_query, previous))
            segmented.add(target)
        pick = f"pick_{item['step_id']}"
        move = f"move_{item['step_id']}"
        place = f"place_{item['step_id']}"
        pause = f"pause_{item['step_id']}"
        budget = item["budget"]
        failure = item["on_failure"]
        steps.append(
            {
                "step_id": pick,
                "skill": "pi0_pick",
                "args": {"prompt": vla_prompt, "max_chunks": 12, "object_ref": obj},
                "vla_prompt": vla_prompt,
                "user_phrase": args.get("user_phrase") or "",
                "grounding_query": obj_query,
                "depends_on": [f"seg_{obj}", previous] if previous != "observe" else [f"seg_{obj}"],
                "preconditions": ["object_localized"],
                "completion_check": "grasp_held",
                "budget": {"max_s": budget["max_s"], "max_attempts": budget["max_attempts"]},
                "on_failure": failure,
                "object_ref": obj,
                "subtask": item["step_id"],
            }
        )
        steps.append(
            {
                "step_id": move,
                "skill": "move_to",
                "args": {
                    "xyz_from": f"segment:{target_query}",
                    "z_offset": 0.12,
                    "gripper": 1,
                    "object_ref": obj,
                    "carry_offset": obj,
                    "motion": "travel",
                    **({} if xy_offset is None else {"xy_offset": xy_offset}),
                },
                "grounding_query": obj_query,
                "depends_on": [pick, f"seg_{target}"],
                "preconditions": ["grasp_held"],
                "completion_check": "object_transported",
                "target_ref": target,
                "budget": {"max_s": min(180, int(budget["max_s"])), "max_attempts": 1},
                "on_failure": failure,
                "object_ref": obj,
                "subtask": item["step_id"],
            }
        )
        descend = f"descend_{item['step_id']}"
        steps.append(
            {
                "step_id": descend,
                "skill": "move_to",
                "args": {
                    "xyz_from": f"segment:{target_query}",
                    "z_offset": 0.03,
                    "gripper": 1,
                    "object_ref": obj,
                    "carry_offset": obj,
                    "motion": "vertical",
                    **({} if xy_offset is None else {"xy_offset": xy_offset}),
                },
                "grounding_query": obj_query,
                "depends_on": [move],
                "preconditions": ["grasp_held"],
                "completion_check": "object_transported",
                "target_ref": target,
                "budget": {"max_s": min(60, int(budget["max_s"])), "max_attempts": 1},
                "on_failure": failure,
                "object_ref": obj,
                "subtask": item["step_id"],
            }
        )
        steps.append(
            {
                "step_id": place,
                "skill": "set_gripper",
                "args": {"gripper": -1, "steps": 8, "object_ref": obj},
                "grounding_query": obj_query,
                "depends_on": [descend],
                "preconditions": ["grasp_held"],
                "completion_check": "object_placed",
                "target_ref": target,
                "budget": {"max_s": 30, "max_attempts": 1},
                "on_failure": failure,
                "object_ref": obj,
                "relation": args["relation"],
                "subtask": item["step_id"],
            }
        )
        retreat = f"retreat_{item['step_id']}"
        steps.append(
            {
                "step_id": retreat,
                "skill": "move_to",
                "args": {
                    "xyz_from": f"segment:{target_query}",
                    "z_offset": 0.12,
                    "gripper": -1,
                    "object_ref": obj,
                    "carry_offset": obj,
                    "motion": "vertical",
                    **({} if xy_offset is None else {"xy_offset": xy_offset}),
                },
                "depends_on": [place],
                "preconditions": ["grasp_held"],
                "completion_check": "object_placed",
                "target_ref": target,
                "relation": args["relation"],
                "budget": {"max_s": 30, "max_attempts": 1},
                "on_failure": "stop",
                "object_ref": obj,
                "subtask": item["step_id"],
            }
        )
        if plan.get("execution_mode") == "free":
            previous = retreat
        else:
            steps.append(
                {
                    "step_id": pause,
                    "skill": "pause_for_confirm",
                    "args": {},
                    "depends_on": [retreat],
                    "preconditions": [],
                    "completion_check": "user_confirmed",
                    "budget": {"max_s": 180, "max_attempts": 1},
                    "on_failure": "stop",
                    "subtask": item["step_id"],
                }
            )
            previous = pause
    expanded = {
        key: plan[key]
        for key in (
            "plan_id",
            "goal",
            "goal_relation",
            "objects",
            "scene_id",
            "suite",
            "task_id",
            "seed",
            "task",
            "execution_mode",
            "predicates",
            "user_text",
        )
        if key in plan
    }
    expanded["steps"] = steps
    return expanded


def _short_pick(prompt: str, grounding: str) -> str:
    """One English grasp sentence. Chinese user text is not sent to Pi0."""
    text = str(prompt or "").strip()
    english = bool(text) and len(text) <= 80 and "\n" not in text and all(ord(ch) < 128 for ch in text)
    if english and text.lower().startswith(("pick up", "grasp", "lift")):
        return text
    phrase = grounding.strip() or "object"
    return f"pick up the {phrase}"


def _segment_step(ref: str, phrase: str, previous: str) -> dict[str, Any]:
    return {
        "step_id": f"seg_{ref}",
        "skill": "segment",
        "args": {"prompt": phrase, "camera": "agentview", "object_ref": ref},
        "depends_on": ["observe"] if previous == "observe" else [previous],
        "preconditions": ["observe_ready"],
        "completion_check": "object_localized",
        "budget": {"max_s": 40, "max_attempts": 1},
        "on_failure": "stop",
        "object_ref": ref,
        "grounding_query": phrase,
        "user_phrase": phrase,
        "subtask": ref,
    }


def prepare_plan(
    plan: dict[str, Any],
    *,
    task: str | None = None,
    scene_objects: dict[str, Any] | None = None,
    required_object_ref: str | None = None,
    execution_mode: str = "benchmark",
    predicates: list[dict[str, Any]] | None = None,
    ordered: bool = False,
) -> dict[str, Any]:
    """Accept a semantic plan or a primitive plan. Illegal plans raise."""
    if not isinstance(plan, dict):
        return validate_plan(plan, task=task)
    skills = []
    for step in plan.get("steps") or []:
        if isinstance(step, dict):
            skills.append(step.get("skill", step.get("action")))
    if any(skill == "transfer" for skill in skills):
        if any(skill != "transfer" for skill in skills):
            raise PlanError([
                _err("plan.steps", "mixed_plan", skills, "transfer steps only, or primitive steps only")
            ])
        semantic = validate_semantic_plan(
            plan,
            task=task,
            scene_objects=scene_objects,
            required_object_ref=required_object_ref,
            execution_mode=execution_mode,
            predicates=predicates,
            ordered=ordered,
        )
        expanded = expand_semantic_plan(semantic)
        if plan.get("plan_source"):
            expanded["plan_source"] = plan["plan_source"]
        expanded["semantic_step_ids"] = [step["step_id"] for step in semantic["steps"]]
        return validate_plan(expanded, task=task)
    return validate_plan(plan, task=task)


def unwrap_model_text(text: str) -> str:
    """Strip surrounding whitespace and one complete markdown fence only."""
    stripped = text.strip()
    if not stripped.startswith("```") or not stripped.endswith("```"):
        return stripped
    inner = stripped[3:-3]
    if "\n" in inner:
        first, rest = inner.split("\n", 1)
        language = first.strip()
        if language and " " not in language and "{" not in language and "}" not in language:
            return rest.strip()
    return inner.strip()


def parse_model_message(
    message: dict[str, Any],
    *,
    task: str | None = None,
    scene_objects: dict[str, Any] | None = None,
    required_object_ref: str | None = None,
    execution_mode: str = "benchmark",
    predicates: list[dict[str, Any]] | None = None,
    ordered: bool = False,
) -> tuple[str, dict[str, Any]]:
    """Return (response kind, plan). Tool calls are not also parsed as text."""
    finish = message.get("finish_reason")
    if message.get("incomplete"):
        raise PlanError([
            _err(
                "message",
                "incomplete_stream",
                {"finish_reason": finish, "content_chars": len(message.get("content") or "")},
                "a finished completion; incomplete streams are not executed or repaired by splicing",
            )
        ])
    if finish == "length":
        raise PlanError([
            _err(
                "message",
                "truncated_output",
                {"finish_reason": finish, "content_chars": len(message.get("content") or "")},
                "finish_reason other than length; truncated JSON is not completed locally",
            )
        ])
    tool_calls = message.get("tool_calls") or []
    if tool_calls:
        calls = []
        for call in tool_calls:
            fn = call.get("function") or {}
            calls.append(fn.get("name"))
        selected = [
            call for call in tool_calls
            if (call.get("function") or {}).get("name") == "submit_plan"
        ]
        if len(selected) != 1:
            raise PlanError([
                _err("message.tool_calls", "tool_call", calls, "exactly one submit_plan tool call")
            ])
        raw_args = (selected[0].get("function") or {}).get("arguments")
        if isinstance(raw_args, str):
            try:
                payload = json.loads(raw_args)
            except json.JSONDecodeError as exc:
                raise PlanError([
                    _err(
                        "message.tool_calls[0].arguments",
                        "json_decode",
                        {"msg": exc.msg, "lineno": exc.lineno, "colno": exc.colno, "pos": exc.pos},
                        "JSON object in submit_plan.arguments",
                    )
                ]) from exc
        elif isinstance(raw_args, dict):
            payload = raw_args
        else:
            raise PlanError([
                _err("message.tool_calls[0].arguments", "type", type(raw_args).__name__, "JSON object")
            ])
        nested = _nested_tool_errors(payload)
        if nested:
            raise PlanError(nested)
        return "tool_calls", prepare_plan(
            payload,
            task=task,
            scene_objects=scene_objects,
            required_object_ref=required_object_ref,
            execution_mode=execution_mode,
            predicates=predicates,
            ordered=ordered,
        )
    content = message.get("content")
    if not isinstance(content, str):
        raise PlanError([_err("message.content", "type", content, "string JSON or submit_plan tool call")])
    text = unwrap_model_text(content)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        kind = "json" if text[:1] in "{[" else "text"
        raise PlanError([
            _err(
                "message.content",
                "json_decode",
                {"kind": kind, "msg": exc.msg, "lineno": exc.lineno, "colno": exc.colno, "pos": exc.pos},
                "one JSON object after optional whitespace or one complete markdown fence",
            )
        ]) from exc
    if not isinstance(payload, dict):
        raise PlanError([_err("message.content", "type", type(payload).__name__, "JSON object")])
    nested = _nested_tool_errors(payload)
    if nested:
        raise PlanError(nested)
    return "json", prepare_plan(
        payload,
        task=task,
        scene_objects=scene_objects,
        required_object_ref=required_object_ref,
        execution_mode=execution_mode,
        predicates=predicates,
        ordered=ordered,
    )


def parse_plan_text(text: str, *, task: str | None = None) -> dict[str, Any]:
    _kind, plan = parse_model_message({"content": text, "finish_reason": "stop"}, task=task)
    return plan


def resolve_args(args: dict[str, Any], bindings: dict[str, list[float]]) -> dict[str, Any]:
    out = dict(args)
    if "xyz_from" in out:
        key = str(out.pop("xyz_from"))
        z_offset = float(out.pop("z_offset", 0.0))
        low, high = THRESHOLDS["approach_z_offset_m"]
        if not low <= z_offset <= high:
            raise PlanError([_err("args.z_offset", "range", z_offset, f"meters in [{low}, {high}]")])
        if key not in bindings:
            raise PlanError([_err("args.xyz_from", "missing_binding", key, "a segment binding from this episode")])
        xyz = [float(v) for v in bindings[key]]
        if len(xyz) != 3 or not all(math.isfinite(v) for v in xyz):
            raise PlanError([_err("args.xyz_from", "binding", key, "finite xyz")])
        xyz[2] += z_offset
        xy_offset = out.pop("xy_offset", None)
        if isinstance(xy_offset, list) and len(xy_offset) >= 2:
            xyz[0] += float(xy_offset[0])
            xyz[1] += float(xy_offset[1])
        carry_ref = out.pop("carry_offset", None)
        if carry_ref:
            offset = bindings.get(f"carry:{carry_ref}")
            if not isinstance(offset, list) or len(offset) < 3:
                raise PlanError([_err("args.carry_offset", "missing_binding", carry_ref, "object-minus-eef from the current grasp")])
            xyz = [xyz[i] - float(offset[i]) for i in range(3)]
        out.pop("motion", None)
        if not _in_workspace(xyz):
            raise PlanError([_err("args.xyz", "workspace", xyz, str(_workspace()))])
        out["xyz"] = [round(v, 4) for v in xyz]
    elif "xyz" in out:
        xyz = [float(v) for v in out["xyz"]]
        if not _in_workspace(xyz):
            raise PlanError([_err("args.xyz", "workspace", xyz, str(_workspace()))])
        out["xyz"] = [round(v, 4) for v in xyz]
    if "gripper" in out:
        gripper = _gripper_value(out["gripper"])
        if gripper is None:
            raise PlanError([_err("args.gripper", "gripper", out["gripper"], "-1 open or +1 close")])
        out["gripper"] = gripper
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
