"""Compact Qwen semantic router for the ten supported LIBERO-Goal tasks.

The router does one thing: it reads the user's original instruction together
with the *complete* ten-task capability catalog and returns exactly one
catalog key (or ``clarify`` / ``unsupported``). It never sees the currently
loaded task, its official instruction, or a BDDL goal, so those cannot leak
into the answer.

Everything else is deterministic code:

* look the chosen key up in ``config/tasks.json``;
* check the chosen task's objects/targets against the live scene;
* refuse when the chosen task is not the loaded scene (the user must switch);
* assemble the single ``whole_task`` step from the catalog's official prompt;
* validate the result with ``common.plan_contract``.

There is no keyword rule or synonym dictionary on the decision path.
"""

from __future__ import annotations

import json
import time
from typing import Any, Callable

from agent import planner
from common import config, task_catalog
from common.plan_contract import validate_plan

MAX_TOKENS = 24
"""Default for a choice object; ``agent.semantic_router.max_tokens`` overrides it."""

MAX_REPAIRS = 1
"""Default for the single format-repair turn; overridden by the same config block."""


def router_options() -> dict[str, Any]:
    """Effective routing limits, read from ``config/engine.json``.

    The config block exists so the page and the health response can report the
    limits actually in force instead of a constant copied into three places.
    """
    block = (config.engine_config().get("agent") or {}).get("semantic_router") or {}
    return {
        "max_tokens": int(block.get("max_tokens", MAX_TOKENS)),
        "max_repairs": int(block.get("max_repairs", MAX_REPAIRS)),
        "model_mode": str(block.get("model_mode", "non_thinking")),
    }

SYSTEM_PROMPT = """You are the semantic router for a LIBERO-Goal robot. Your only job is to decide which one supported capability the user's instruction asks for.

Decide in this order:
1. Choose "clarify" when the user's words leave a slot unfilled and the capability list would decide it: an unresolved pronoun or demonstrative ("put it on the stove", "把它放到炉灶上"), no destination at all ("move the bowl"), or a container named without a level while capabilities name levels of that container ("put the bowl inside the cabinet").
2. Otherwise choose exactly one capability key whose object, destination, relation, level and every subgoal match the complete goal. Polite phrasing, word order and ordinary synonyms do not change the goal.
3. Otherwise choose "unsupported" because the request is specific enough to judge and no capability matches: an object, level, relation or place the user did state but no capability has ("open the lower drawer", "put the plate on the cabinet", "put the bowl next to the stove"), one half of a composite capability, or a combination of capabilities.

A stated relation or level is part of the request, not a missing slot: "beside", "near", "behind", "under" are unsupported relations, never reasons to clarify.

Do not substitute a different object or place, and do not split a composite capability into its parts.

Return exactly one JSON object and nothing else:
{"choice":"<key>"} or {"choice":"clarify"} or {"choice":"unsupported"}
"""


def _catalog_lines() -> str:
    rows = []
    for row in task_catalog.capability_rows():
        rows.append(f"- {row['key']} ({row['operation']}): {row['capability']}")
    return "\n".join(rows)


def build_messages(instruction: str) -> list[dict[str, str]]:
    """The whole prompt: ten capabilities, then the raw instruction.

    Deliberately no task_id, official instruction, scene snapshot or BDDL goal
    is present; the model must choose from the capability descriptions.
    """
    system = f"{SYSTEM_PROMPT}\nSupported capabilities:\n{_catalog_lines()}"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": instruction.strip()},
    ]


def parse_choice(content: str) -> tuple[dict[str, str] | None, list[dict[str, Any]]]:
    value, error = planner.parse_json_object(content)
    if value is None:
        return None, [{"field": "response", "code": "invalid_json", "value": error}]
    if "choice" not in value:
        return None, [{"field": "choice", "code": "missing", "value": sorted(value)}]
    choice = value.get("choice")
    if not isinstance(choice, str) or not choice.strip():
        return None, [{"field": "choice", "code": "not_string", "value": choice}]
    cleaned = choice.strip()
    allowed = set(task_catalog.task_by_key()) | {"clarify", "unsupported"}
    if cleaned not in allowed:
        return None, [
            {
                "field": "choice",
                "code": "not_in_catalog",
                "value": cleaned,
                "expected": sorted(allowed),
            }
        ]
    return {"choice": cleaned}, []


def _repair_messages(messages: list[dict[str, str]], content: str) -> list[dict[str, str]]:
    return [
        *messages,
        {"role": "assistant", "content": content},
        {
            "role": "user",
            "content": (
                'Return one corrected JSON object with exactly {"choice":"<key>"} '
                'using a key from the capability list, or "clarify", or "unsupported". '
                "Do not explain."
            ),
        },
    ]


def _attempt(content: str, timing: dict[str, Any], parse_s: float, validate_s: float) -> dict[str, Any]:
    return {
        "timing": {
            **timing,
            "parse_s": round(parse_s, 6),
            "validate_s": round(validate_s, 6),
        },
        "raw_chars": len(content or ""),
    }


def decide(
    instruction: str,
    snapshot: dict[str, Any] | None = None,
    *,
    chat_fn: Callable[..., tuple[str, dict[str, Any]]] | None = None,
    max_repairs: int | None = None,
) -> dict[str, Any]:
    """Ask Qwen for one catalog key and return the parsed, checked decision.

    ``chat_fn`` is injectable so tests can prove the mapping and validation
    logic without a live model; production always uses ``planner.chat``.
    ``snapshot`` is accepted for call-site symmetry but is intentionally not
    placed in the prompt.
    """
    options = router_options()
    if max_repairs is None:
        max_repairs = options["max_repairs"]
    budget = options["max_tokens"]
    started = time.perf_counter()
    messages = build_messages(instruction)
    chat = chat_fn or planner.chat
    attempts: list[dict[str, Any]] = []
    choice: dict[str, str] | None = None
    errors: list[dict[str, Any]] = []
    content = ""
    for index in range(max_repairs + 1):
        content, timing = chat(messages, max_tokens=budget)
        parse_started = time.perf_counter()
        choice, errors = parse_choice(content)
        parse_s = time.perf_counter() - parse_started
        attempts.append(_attempt(content, timing, parse_s, 0.0))
        if choice is not None:
            break
        attempts[-1]["errors"] = errors
        if index < max_repairs:
            messages = _repair_messages(messages, content)

    total_s = round(time.perf_counter() - started, 4)
    result: dict[str, Any] = {
        "status": "ok" if choice is not None else "planning_failed",
        "choice": (choice or {}).get("choice"),
        "errors": errors if choice is None else [],
        "repairs": len(attempts) - 1,
        "attempts": attempts,
        "total_s": total_s,
        "router": {
            "model": planner.client_options()["model"],
            "mode": options["model_mode"],
            "max_tokens": budget,
            "max_repairs": max_repairs,
            "catalog_version": task_catalog.version(),
            "catalog_sha256": task_catalog.fingerprint(),
            "candidate_keys": [row["key"] for row in task_catalog.capability_rows()],
            "prompt_chars": sum(len(row["content"]) for row in messages if row["role"] == "system")
            + len(instruction),
            "snapshot_in_prompt": False,
        },
    }
    if choice is None:
        result["reason_code"] = "router_format_invalid"
        result["reason"] = errors[0].get("value") if errors else "invalid router response"
        return result

    selected = choice["choice"]
    if selected in {"clarify", "unsupported"}:
        result["status"] = "needs_clarification" if selected == "clarify" else "unsupported"
        result["reason_code"] = "ambiguous_or_incomplete" if selected == "clarify" else "outside_catalog"
        result["reason"] = (
            "the instruction is ambiguous or lacks the information needed to choose one capability"
            if selected == "clarify"
            else "the instruction is outside the ten supported capabilities"
        )
        return result

    task = task_catalog.task_by_key().get(selected)
    if task is None:  # parse_choice already checked; kept as a hard invariant.
        result["status"] = "planning_failed"
        result["reason_code"] = "choice_not_in_catalog"
        return result
    support = task_catalog.scene_support(task, snapshot or {})
    result["task"] = task
    result["scene_support"] = support
    if not support["ok"]:
        result["status"] = "scene_not_supported"
        result["reason_code"] = "required_entity_absent"
        missing = ", ".join(f"{row['kind']}:{row['name']}" for row in support["missing"])
        result["reason"] = f"the loaded scene is missing {missing}"
    return result


def assemble_plan(
    decision: dict[str, Any],
    *,
    loaded_task_id: int,
    instruction: str,
    snapshot: dict[str, Any],
    skills: dict[str, Any],
) -> dict[str, Any]:
    """Turn a catalog choice into the single whole_task plan, after checks."""
    task = decision.get("task")
    if decision.get("status") != "ok" or not isinstance(task, dict):
        return {"status": decision.get("status"), "plan": None, "errors": decision.get("errors") or []}

    chosen_id = int(task["task_id"])
    if chosen_id != int(loaded_task_id):
        return {
            "status": "task_mismatch",
            "plan": None,
            "errors": [],
            "reason_code": "requires_scene_switch",
            "reason": (
                f"the instruction means task {chosen_id} ({task['key']}), but task "
                f"{loaded_task_id} is loaded; switch to task {chosen_id} or use a matching instruction"
            ),
        }

    official = str(task.get("official_instruction") or task.get("instruction") or "").strip()
    if not official:
        return {
            "status": "planning_failed",
            "plan": None,
            "errors": [{"field": "official_instruction", "code": "empty"}],
            "reason_code": "catalog_instruction_missing",
        }

    plan = {
        "interpretation": (
            f"User instruction routed to catalog task {chosen_id} ({task['key']}); "
            "the whole-task policy receives the catalog's official instruction."
        ),
        "success_condition": (
            "The native LIBERO goal for the routed task holds on fresh simulation samples. "
            "No success is inferred from the language decision."
        ),
        "steps": [
            {
                "skill": "whole_task",
                "object_n": None,
                "target_n": None,
                "relation": "task",
                "prompt": official,
            }
        ],
    }
    plan, errors = validate_plan(
        plan,
        instruction=instruction,
        task=task,
        snapshot=snapshot,
        skills=skills,
        plan_source="qwen3.5-9b-semantic-router",
    )
    if errors:
        return {
            "status": "planning_failed",
            "plan": None,
            "errors": errors,
            "reason_code": "plan_contract_failed",
        }
    return {"status": "ok", "plan": plan, "errors": []}


__all__ = [
    "MAX_REPAIRS",
    "MAX_TOKENS",
    "SYSTEM_PROMPT",
    "assemble_plan",
    "build_messages",
    "decide",
    "router_options",
    "parse_choice",
]
