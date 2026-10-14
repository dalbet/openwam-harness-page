"""Plan contract. Illegal plans never reach the robot."""

from __future__ import annotations

import json
import math
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


def submit_plan_tools() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "submit_plan",
                "description": "Submit one sequential plan. The arguments are the plan object.",
                "parameters": plan_parameters_schema(),
            },
        }
    ]


TOOL_SCHEMA = json.dumps(
    {
        "skills": list(REGISTERED_SKILLS),
        "rejected": sorted(REJECTED_ACTIONS),
        "completion_check": sorted(CHECKERS),
        "on_failure": sorted(ON_FAILURE),
        "args": {
            "view_env_state": {"step": "int, -1 is latest"},
            "segment": {"prompt": "scene phrase", "camera": "agentview|wrist", "object_ref": "scene object"},
            "pi0_pick": {"prompt": "grasp only", "max_chunks": "1..24", "object_ref": "scene object"},
            "move_to": {
                "xyz_from": "segment:<scene phrase>",
                "z_offset": "meters, configured approach range",
                "gripper": "-1 open or +1 close",
            },
            "set_gripper": {"gripper": "-1 open or +1 close", "steps": "1..40"},
            "pause_for_confirm": {},
        },
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
    elif task is not None and goal.strip() != task.strip():
        errors.append(_err("plan.goal", "task_mismatch", goal, task.strip()))
    if isinstance(plan.get("task"), str) and isinstance(goal, str) and plan["task"].strip() != goal.strip():
        errors.append(_err("plan.task", "goal_mismatch", plan.get("task"), "same text as goal"))
    relation = plan.get("goal_relation")
    if relation not in RELATIONS:
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
            if spec.get("role") not in {"movable", "container", "support"}:
                errors.append(_err(f"plan.objects.{ref}.role", "role", spec.get("role"), "movable|container|support"))
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
        if not isinstance(prompt, str) or prompt.strip().lower() not in phrases:
            errors.append(_err(f"{path}.args.prompt", "phrase", prompt, "a phrase of a scene object"))
        elif isinstance(object_ref, str) and phrases.get(prompt.strip().lower()) not in {None, object_ref}:
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
        extra = set(args) - {"xyz", "xyz_from", "z_offset", "gripper", "object_ref", "max_steps"}
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
            elif relation != goal:
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


def parse_model_message(message: dict[str, Any], *, task: str | None = None) -> tuple[str, dict[str, Any]]:
    """Return (response kind, plan). Tool calls are not also parsed as text."""
    finish = message.get("finish_reason")
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
        return "tool_calls", validate_plan(payload, task=task)
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
    return "json", validate_plan(payload, task=task)


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
