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
from common import composite_plan, config, task_catalog
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
        "flow_guard": bool(block.get("flow_guard", True)),
        "flow_guard_max_tokens": int(block.get("flow_guard_max_tokens", 32)),
    }

SYSTEM_PROMPT = """You are the semantic router for a LIBERO-Goal robot. Your only job is to decide which one supported capability, or which ordered sequence of two or three supported capabilities, the user's instruction asks for.

Decide in this order:
1. Choose "clarify" when the user's words leave a slot unfilled and the capability list would decide it: an unresolved pronoun or demonstrative ("put it on the stove", "把它放到炉灶上", "turn it on", "把那个打开"), no destination at all ("move the bowl"), or a container named without a level while capabilities name levels of that container. Missing a level is a missing slot even when another part of the same instruction is complete: "open the drawer, then set the bowl down" and "打开抽屉，然后把碗放下" ask which drawer, so choose "clarify".
   - A pronoun or demonstrative is resolved only by an object the user named in the same instruction. Something the model infers from the scene or from a household routine is not an antecedent, so "get the cream cheese ready and then turn it on" is "clarify", not a flow and not a guessed key.
   - An unfilled slot is still "clarify" when the rest of the instruction is a complete goal, and it is not "unsupported": the request is not outside the catalog, it is missing the words that would fix the slot.
2. Otherwise, if the instruction asks for two or three of the supported capabilities in the same scene, return those keys in the order the user's words put them.
3. Otherwise choose exactly one capability key whose object, destination, relation, level and every subgoal match the complete goal. Polite phrasing, word order and ordinary synonyms do not change the goal.
4. Otherwise choose "unsupported" because the request is specific enough to judge and no capability matches: an object, level, relation or place the user did state but no capability has ("open the lower drawer", "put the plate on the cabinet", "put the bowl next to the stove"), one half of a composite capability, a combination that is not two or three capabilities from the list, or four or more separate capabilities.

Ordered sequences:
- Return {"choice":["<key1>","<key2>"]} or {"choice":["<key1>","<key2>","<key3>"]} only when that many separate capabilities in the list are all asked for. Keep the user's order and do not add, drop, split or reorder a step.
- When the user spells the steps out as separate instructions ("first ... then ... last ...", "先…然后…最后…"), return those keys as a sequence, not a flow id, even when the sequence happens to equal a flow's steps. A named sequence stays a named sequence.
- A named sequence is not incomplete just because a flow has more steps after the ones the user said: a user who names two of the three steps asked for two steps, not for "clarify" and not for the flow.
- One capability can read like two actions ("open the top drawer and place the bowl inside it"). When the list has a capability for the whole goal, choose that single key, not a sequence.
- "clarify" when the order of separate capabilities is neither stated nor implied by which goal depends on the other, when a step's object or place is missing, or when two goals conflict (the same object sent to two different destinations).
- Four or more separate capabilities are outside this contract: choose "unsupported".

A stated relation or level is part of the request, not a missing slot: "beside", "near", "behind", "under" are unsupported relations, never reasons to clarify. 

Household flows:
A flow is a fixed, verified sequence of catalog capabilities for one household goal. The flow's steps are listed after its description, but you never retype them: code expands the flow into its ordered keys and validates the plan.
- Return {"choice":{"flow":"<flow_id>"}} when the instruction asks for the whole household goal that flow describes, in any words.
- Return {"choice":{"flow":"<flow_id>"}} only when every step of the flow is required by the instruction. Never answer with just the flow's last step: a complete household goal is not a request for its final action.
- If the instruction asks for the whole goal *and* for something the flow excludes (stirring, unwrapping, melting detection, measuring temperature, an extra object), choose "unsupported". Do not run the flow minus the extra request.
- If the instruction names only one of the flow's steps and nothing else, return that single capability key instead of the flow.
- If the user named the steps themselves, use the named sequence rule above: the flow id is for a goal described as a whole, not for a goal the user already broke down.
- If a named step is ambiguous on its own (its object or place is only a pronoun), choose "clarify"; a flow does not fill in words the user did not say.

Do not substitute a different object or place.

Tie-break, in this order:
1. An unfilled slot beats everything: a pronoun the user never bound, a missing place, or a missing level means "clarify" — never a flow, never a guessed key. "prepare the cheese for heating, then turn it on" leaves "it" unbound, so it is "clarify".
2. A sequence the user named beats a flow: the flow id is for a household goal the user described as a whole, not for one they already spelled out step by step. Three steps spelled out one after another are that three-key sequence, even when they equal the flow's steps.
3. A flow beats a single key: never reduce a whole household goal to its last step.

Return exactly one JSON object and nothing else:
{"choice":"<key>"} or {"choice":["<key1>","<key2>"]} or {"choice":["<key1>","<key2>","<key3>"]} or {"choice":{"flow":"<flow_id>"}} or {"choice":"clarify"} or {"choice":"unsupported"}
"""


def _catalog_lines() -> str:
    rows = []
    for row in task_catalog.capability_rows():
        rows.append(f"- {row['key']} ({row['operation']}): {row['capability']}")
    return "\n".join(rows)


def _flow_lines() -> str:
    rows = []
    for row in task_catalog.flow_rows():
        excludes = "; ".join(row["excludes"]) or "none"
        rows.append(
            f"- {row['flow_id']} (flow): {row['capability']}"
            f" Steps: {' -> '.join(row['keys'])}. Excludes: {excludes}"
        )
    return "\n".join(rows)


def build_messages(instruction: str) -> list[dict[str, str]]:
    """The whole prompt: ten capabilities and the verified flows, then the raw instruction.

    Deliberately no task_id, official instruction, scene snapshot or BDDL goal
    is present; the model must choose from the capability descriptions.
    """
    system = f"{SYSTEM_PROMPT}\nSupported capabilities:\n{_catalog_lines()}"
    flow_lines = _flow_lines()
    if flow_lines:
        system = f"{system}\nVerified household flows:\n{flow_lines}"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": instruction.strip()},
    ]


def _flow_value(flow_id: str) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Validate a flow id against the catalog; keys are expanded by code."""
    flow = task_catalog.flow_by_id().get(flow_id)
    if flow is None:
        return None, [
            {
                "field": "choice.flow",
                "code": "not_in_flows",
                "value": flow_id,
                "expected": sorted(task_catalog.flow_by_id()),
            }
        ]
    keys = task_catalog.flow_keys(flow_id)
    if not keys:
        return None, [{"field": "flow.keys", "code": "empty", "value": flow_id}]
    return {
        "kind": "flow",
        "choice": {"flow": flow_id},
        "flow_id": flow_id,
        "keys": keys,
    }, []


def parse_choice(content: str) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    value, error = planner.parse_json_object(content)
    if value is None:
        return None, [{"field": "response", "code": "invalid_json", "value": error}]
    if "choice" not in value:
        return None, [{"field": "choice", "code": "missing", "value": sorted(value)}]
    choice = value.get("choice")
    if isinstance(choice, dict):
        # {"choice":{"flow":"<flow_id>"}} — the only structured shape allowed.
        flow_id = choice.get("flow")
        if not isinstance(flow_id, str) or not flow_id.strip():
            return None, [{"field": "choice.flow", "code": "not_string", "value": choice}]
        if len(choice) != 1:
            return None, [{"field": "choice", "code": "unexpected_keys", "value": sorted(choice)}]
        return _flow_value(flow_id.strip())
    if isinstance(choice, str) and choice.strip() == "flow":
        flow_id = value.get("flow_id")
        if not isinstance(flow_id, str) or not flow_id.strip():
            return None, [{"field": "flow_id", "code": "not_string", "value": flow_id}]
        return _flow_value(flow_id.strip())
    allowed = set(task_catalog.task_by_key()) | {"clarify", "unsupported"}
    if isinstance(choice, str):
        cleaned = choice.strip()
        if not cleaned:
            return None, [{"field": "choice", "code": "not_string", "value": choice}]
        if cleaned not in allowed:
            return None, [
                {
                    "field": "choice",
                    "code": "not_in_catalog",
                    "value": cleaned,
                    "expected": sorted(allowed),
                }
            ]
        if cleaned in {"clarify", "unsupported"}:
            return {"kind": "special", "choice": cleaned, "keys": []}, []
        return {"kind": "single", "choice": cleaned, "keys": [cleaned]}, []
    if isinstance(choice, list):
        if not choice:
            return None, [{"field": "choice", "code": "empty_list", "value": choice}]
        keys: list[str] = []
        for item in choice:
            if not isinstance(item, str) or not item.strip():
                return None, [{"field": "choice", "code": "not_string", "value": item}]
            cleaned = item.strip()
            if cleaned not in allowed or cleaned in {"clarify", "unsupported"}:
                return None, [
                    {
                        "field": "choice",
                        "code": "not_in_catalog",
                        "value": cleaned,
                        "expected": sorted(task_catalog.task_by_key()),
                    }
                ]
            keys.append(cleaned)
        if len(keys) == 1:
            # A one-element list is neither shape; ask for a correction instead
            # of quietly accepting a composite contract that was not used.
            return None, [{"field": "choice", "code": "single_key_list", "value": keys, "expected": "a bare key string"}]
        return {"kind": "composite", "choice": keys, "keys": keys}, []
    return None, [{"field": "choice", "code": "not_string_or_list", "value": choice}]


def _repair_messages(messages: list[dict[str, str]], content: str) -> list[dict[str, str]]:
    return [
        *messages,
        {"role": "assistant", "content": content},
        {
            "role": "user",
            "content": (
                'Return one corrected JSON object with exactly {"choice":"<key>"} '
                'using a key from the capability list, or "clarify", or "unsupported", '
                'or {"choice":{"flow":"<flow_id>"}} using a flow id from the verified flow list '
                'when the instruction asks for the whole household goal. '
                'Use {"choice":["<key1>","<key2>"]} or {"choice":["<key1>","<key2>","<key3>"]} only when '
                "that many separate capabilities are asked for. Do not explain."
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


def _composite_choice(keys: list[str], result: dict[str, Any], snapshot: dict[str, Any] | None) -> bool:
    """Shared limits/checks for an explicit ordered sequence or an expanded flow.

    Returns ``True`` when the sequence passes the router's own checks; the
    execution-side contract (``composite_plan.support``) is still applied again
    by :func:`assemble_plan`, so this is not the authority on executability.
    """
    composite_opts = composite_plan.options(config.engine_config())
    if len(keys) > composite_opts["max_steps"]:
        result["status"] = "unsupported"
        result["reason_code"] = "step_limit_exceeded"
        result["reason"] = (
            f"the instruction asks for {len(keys)} separate capabilities; at most "
            f"{composite_opts['max_steps']} can be executed"
        )
        return False
    if len(set(keys)) != len(keys):
        result["status"] = "unsupported"
        result["reason_code"] = "duplicate_step"
        result["reason"] = "the same capability appears twice in the requested sequence"
        return False
    by_key = task_catalog.task_by_key()
    supports = []
    for key in keys:
        row = by_key.get(key)
        if row is None:
            result["status"] = "planning_failed"
            result["reason_code"] = "key_not_in_catalog"
            result["reason"] = f"flow key {key!r} is not a catalog capability"
            return False
        supports.append({"key": key, **task_catalog.scene_support(row, snapshot or {})})
    result["scene_support"] = {
        "ok": all(row.get("ok") for row in supports),
        "steps": supports,
        "missing": [missing for row in supports for missing in (row.get("missing") or [])],
    }
    if not result["scene_support"]["ok"]:
        result["status"] = "scene_not_supported"
        result["reason_code"] = "required_entity_absent"
        missing = ", ".join(
            f"{row['kind']}:{row['name']}" for row in result["scene_support"]["missing"]
        )
        result["reason"] = f"the loaded scene is missing {missing}"
        return False
    return True


FLOW_GUARD_PROMPT = """You check one routing answer for a LIBERO-Goal robot. Do not re-route the whole instruction.

The router answered "{key}", which is the LAST step of this verified household flow:
- {flow_id}: {capability}
  Steps: {steps}

Decide which the instruction actually asks for:
- "flow": the instruction asks for the whole household preparation (every step above).
- "single": the instruction asks only for "{key}" and for nothing else.

Return exactly one JSON object and nothing else: {{"scope":"flow"}} or {{"scope":"single"}}"""


def _flow_guard(
    instruction: str,
    key: str,
    candidates: list[dict[str, Any]],
    chat,
    budget: int,
) -> dict[str, Any]:
    """Ask one bounded question when a single key is a flow's final step.

    Without this, a complete household goal can be answered with its last
    capability (turn the stove on) and the earlier subgoals are silently
    dropped. The question is answered by the same model; no keyword list is
    consulted, and the answer is recorded with its latency.
    """
    flow = candidates[0]
    steps = " -> ".join(str(item) for item in (flow.get("keys") or []))
    system = FLOW_GUARD_PROMPT.format(
        key=key,
        flow_id=flow.get("flow_id"),
        capability=flow.get("capability"),
        steps=steps,
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": instruction.strip()},
    ]
    started = time.perf_counter()
    content, timing = chat(messages, max_tokens=budget)
    elapsed = round(time.perf_counter() - started, 4)
    value, error = planner.parse_json_object(content)
    scope = str((value or {}).get("scope") or "").strip().lower()
    row = {
        "asked": True,
        "key": key,
        "flow_ids": [str(item.get("flow_id")) for item in candidates],
        "raw": content,
        "scope": scope if scope in {"flow", "single"} else None,
        "s": elapsed,
        "timing": timing,
        "note": (
            "a single answer that is a flow's last step is checked once, so a complete "
            "household goal cannot silently degrade to its final capability"
        ),
    }
    if row["scope"] is None:
        row["error"] = error
    return row


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
            "candidate_flows": [row["flow_id"] for row in task_catalog.flow_rows()],
            "flows_version": task_catalog.flows_version(),
            "flow_guard": bool(options["flow_guard"]),
            "prompt_chars": sum(len(row["content"]) for row in messages if row["role"] == "system")
            + len(instruction),
            "snapshot_in_prompt": False,
        },
    }
    if choice is None:
        result["reason_code"] = "router_format_invalid"
        result["reason"] = errors[0].get("value") if errors else "invalid router response"
        return result

    kind = str(choice.get("kind") or "single")
    if kind == "flow":
        # A household flow: the model named the goal, the catalog supplies the
        # ordered keys, and the same composite contract decides executability.
        flow_id = str(choice.get("flow_id") or "")
        keys = list(choice.get("keys") or [])
        result["flow"] = {
            "flow_id": flow_id,
            "keys": keys,
            "source": "catalog flows block",
            "model_choice": "flow",
        }
        result["composite_keys"] = keys
        result["choice"] = {"flow": flow_id}
        check = _composite_choice(keys, result, snapshot)
        if not check:
            return result
        result["flow"]["executable"] = True
        return result
    if kind == "composite":
        keys = list(choice["keys"])
        result["composite_keys"] = keys
        result["choice"] = keys
        _composite_choice(keys, result, snapshot)
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
    # Guard: a single answer that is a verified flow's last step could be a
    # complete household goal reduced to its final action. Ask once; a "flow"
    # answer replaces the single key with the flow's ordered keys.
    terminal = task_catalog.flow_for_terminal_key(selected)
    if terminal and options["flow_guard"]:
        guard = _flow_guard(
            instruction,
            selected,
            terminal,
            chat,
            options["flow_guard_max_tokens"],
        )
        result["flow_guard"] = guard
        if guard.get("scope") == "flow":
            flow_id = str(terminal[0].get("flow_id"))
            keys = task_catalog.flow_keys(flow_id)
            if not keys:
                result["status"] = "unsupported"
                result["reason_code"] = "flow_without_keys"
                result["reason"] = f"flow {flow_id} has no keys"
                return result
            result["flow"] = {
                "flow_id": flow_id,
                "keys": keys,
                "source": "flow_guard",
                "model_choice": "flow",
                "guard_key": selected,
            }
            result.pop("task", None)
            result["composite_keys"] = keys
            result["choice"] = {"flow": flow_id}
            _composite_choice(keys, result, snapshot)
            return result
    return result


def assemble_plan(
    decision: dict[str, Any],
    *,
    loaded_task_id: int,
    instruction: str,
    snapshot: dict[str, Any],
    skills: dict[str, Any],
    loaded_task: dict[str, Any] | None = None,
    plan_source: str = "qwen3.5-9b-semantic-router",
) -> dict[str, Any]:
    """Turn a catalog choice into a plan, after checks.

    A single choice becomes the one ``whole_task`` step. A composite choice is
    only assembled when the ordered sequence is executable in this scene; the model
    recognizes the sequence, the code decides whether it can run.
    """
    task = decision.get("task")
    composite_keys = decision.get("composite_keys")
    if composite_keys:
        if decision.get("status") != "ok":
            return {
                "status": decision.get("status"),
                "plan": None,
                "errors": decision.get("errors") or [],
                "reason_code": decision.get("reason_code"),
                "reason": decision.get("reason"),
            }
        loaded = loaded_task or config.task_for(int(loaded_task_id))
        catalog = task_catalog.task_by_key()
        check = composite_plan.support(
            list(composite_keys), loaded_task=loaded, catalog=catalog, cfg=config.engine_config()
        )
        if not check.get("ok"):
            scene_switch = str(check.get("reason_code")) == "requires_scene_switch"
            return {
                "status": "task_mismatch" if scene_switch else "unsupported",
                "plan": None,
                "errors": [],
                "reason_code": check.get("reason_code"),
                "reason": check.get("reason"),
                "composite": check,
            }
        plan = composite_plan.build_plan(
            list(composite_keys),
            catalog=catalog,
            loaded_task=loaded,
            instruction=instruction,
            cfg=config.engine_config(),
            plan_source=plan_source,
        )
        errors = composite_plan.validate_plan(
            plan, catalog=catalog, loaded_task=loaded, cfg=config.engine_config()
        )
        if errors:
            return {
                "status": "planning_failed",
                "plan": None,
                "errors": errors,
                "reason_code": "composite_plan_contract_failed",
            }
        if decision.get("flow"):
            # The flow id is carried into the plan so a record can always tell
            # an expanded household flow from a user-named sequence.
            plan.setdefault("composite", {})
            plan["composite"]["flow"] = decision["flow"]
            plan["flow_expansion"] = {
                "flow_id": decision["flow"].get("flow_id"),
                "keys": list(decision.get("composite_keys") or []),
                "source": decision["flow"].get("source"),
                "note": (
                    "the language model chose the flow id only; the ordered keys come from "
                    "the catalog and pass the same composite contract as an explicit list"
                ),
            }
        if decision.get("flow_guard"):
            plan["flow_guard"] = decision["flow_guard"]
        return {"status": "ok", "plan": plan, "errors": [], "composite": check}

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
        plan_source=plan_source,
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
