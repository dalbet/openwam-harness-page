"""Qwen planner and recovery decisions for the Agent layer.

The language model proposes structured JSON. This module parses one JSON
object and validates it with ``common.plan_contract``. It does not repair a
prose list into a plan; a failed parse gets at most one format retry that asks
for the same schema again.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any

from common import config
from common.plan_contract import validate_plan


SYSTEM_PROMPT = """You are the Agent layer of a LIBERO-Goal robot system.
You do not see camera images. Your input is the user instruction, task registry metadata, a read-only simulation snapshot, and the skill catalog. The snapshot is simulation truth, not a visual recognition result.

Return exactly one JSON object and nothing else. Use only skills in the catalog and exact instance/site names present in the snapshot. Never invent objects, sites, skills, or controller parameters. Do not output code or a prose list of steps.

Skill rules:
- pick_place: object_n is a movable object; target_n is a site or a support object; relation is "on" or "in". If the target is a drawer region, an open_drawer step on the same drawer must come first.
- open_drawer: target_n is a drawer region site whose joint exists (name contains _top_region, _middle_region, or _bottom_region); relation is "open"; object_n may be null.
- push_object: object_n is movable; target_n is a table region; relation is "on"; no release is needed.
- turn_on: target_n is the appliance; relation is "turn_on"; no grasp is needed.
- At most 4 steps, in execution order.
- prompt is the short English VLA instruction for the whole skill: for pick_place include both the object and the place target (e.g. "put the black bowl on the stove"); for open_drawer use "open the top drawer of the cabinet"; for push_object use "push the plate to the front of the stove"; for turn_on use "turn on the stove". It must not mention candidate numbers or instance IDs.

JSON shape:
{"interpretation": "one sentence",
 "steps": [{"skill": "pick_place", "object_n": "exact_name", "target_n": "exact_name", "relation": "on", "prompt": "short English prompt"}],
 "success_condition": "how the user goal is checked from simulation state"}
"""

RECOVERY_PROMPT = """You are the Agent layer deciding how to recover.
You receive the failed skill, its failure evidence, the remaining user goal, and the current simulation snapshot. Return exactly one JSON object:
{"decision": "retry" | "replan" | "abort",
 "reason": "short reason",
 "steps": [{"skill": "open_drawer", "object_n": null, "target_n": "exact_site", "relation": "open", "prompt": "open the middle drawer of the cabinet"}]}
Use "retry" only for a transient execution problem (timeout, one missed grasp); return an empty steps list. Use "replan" only when you provide a non-empty, complete, validated step list using the same skill rules and exact names as the plan request. Use "abort" when the goal is unreachable with the available skills; return an empty steps list. Do not claim success. Do not output anything outside the JSON object.
"""


def client_options() -> dict[str, Any]:
    agent = config.engine_config()["agent"]
    return {
        "base_url": agent["llm_base_url"].rstrip("/"),
        "model": agent["llm_model"],
        "temperature": float(agent["temperature"]),
        "max_tokens": int(agent["max_tokens"]),
        "timeout_s": 120.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def _post_json(url: str, payload: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:1000]
        raise RuntimeError(f"LLM HTTP {exc.code}: {detail}") from exc


def chat(messages: list[dict[str, Any]], *, max_tokens: int | None = None) -> tuple[str, dict[str, Any]]:
    opts = client_options()
    payload = {
        "model": opts["model"],
        "messages": messages,
        "temperature": opts["temperature"],
        "max_tokens": max_tokens or opts["max_tokens"],
        "response_format": {"type": "json_object"},
        "chat_template_kwargs": opts["chat_template_kwargs"],
    }
    started = time.perf_counter()
    response = _post_json(f"{opts['base_url']}/chat/completions", payload, opts["timeout_s"])
    elapsed = time.perf_counter() - started
    choices = response.get("choices") or []
    content = ""
    if choices:
        content = (choices[0].get("message") or {}).get("content") or ""
    usage = response.get("usage") or {}
    return content, {
        "elapsed_s": round(elapsed, 3),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "model": response.get("model"),
    }


def parse_json_object(text: str) -> tuple[dict[str, Any] | None, str]:
    """Parse one JSON object; a fenced/embedded object is parsed, not edited."""
    text = (text or "").strip()
    if not text:
        return None, "empty response"
    try:
        value = json.loads(text)
        return (value, "") if isinstance(value, dict) else (None, "top-level JSON is not an object")
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    if start < 0:
        return None, "no JSON object"
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                try:
                    value = json.loads(text[start : index + 1])
                    return (value, "") if isinstance(value, dict) else (None, "embedded JSON is not an object")
                except json.JSONDecodeError as exc:
                    return None, f"invalid JSON object: {exc}"
    return None, "unbalanced JSON object"


def scene_summary(task: dict[str, Any], snapshot: dict[str, Any], skills: dict[str, Any]) -> dict[str, Any]:
    sim = snapshot.get("sim") or {}
    objects = sim.get("objects") or {}
    named = snapshot.get("named_poses") or {}
    sites = named.get("sites") or {}
    joints = snapshot.get("joints") or {}
    return {
        "task": {
            "task_id": task.get("task_id"),
            "instruction": task.get("instruction"),
            "kind": task.get("kind"),
            "bddl_goal_atoms": task.get("goal_atoms"),
            "registry_objects": task.get("objects"),
            "registry_targets": task.get("targets"),
            "drawer_level": task.get("drawer_level"),
        },
        "scene": {
            "scene_epoch": snapshot.get("scene_epoch"),
            "sim_action_count": snapshot.get("sim_action_count"),
            "observation_id": snapshot.get("observation_id"),
            "objects": {name: {"pos": row.get("pos")} for name, row in objects.items() if isinstance(row, dict)},
            "sites": sorted(sites.keys()),
            "drawer_joints": {name: value for name, value in joints.items() if "cabinet" in name or "stove" in name},
            "eef_pos": sim.get("eef_pos"),
            "gripper_opening_m": sim.get("gripper_opening_m"),
            "native_goal": snapshot.get("native_goal"),
        },
        "skills": skills.get("skills"),
        "binding": {"evidence_source": "sim_state", "vision_used": False},
    }


def request_plan(
    *,
    instruction: str,
    task: dict[str, Any],
    snapshot: dict[str, Any],
    skills: dict[str, Any],
    attempts: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    summary = scene_summary(task, snapshot, skills)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": json.dumps(
                {
                    "instruction": instruction,
                    "context": summary,
                    "attempts": attempts or [],
                    "request": "Output the JSON plan object now.",
                },
                ensure_ascii=False,
            ),
        },
    ]
    records: list[dict[str, Any]] = []
    content, timing = chat(messages)
    records.append({"attempt": 1, "kind": "initial", "timing": timing, "raw_chars": len(content)})
    plan, parse_error = parse_json_object(content)
    errors: list[dict[str, Any]] = []
    if plan is None:
        errors = [{"field": "response", "code": "invalid_json", "value": parse_error, "expected": "one JSON plan object"}]
    else:
        plan, errors = validate_plan(
            plan, instruction=instruction, task=task, snapshot=snapshot, skills=skills
        )
    if plan is not None and not errors:
        return {"status": "ok", "plan": plan, "attempts": records, "errors": [], "repairs": 0}
    records[0]["errors"] = errors
    repair_messages = [
        *messages,
        {"role": "assistant", "content": content},
        {
            "role": "user",
            "content": json.dumps(
                {
                    "validation_errors": errors,
                    "request": "Return one corrected JSON plan object using the same schema. Do not explain.",
                },
                ensure_ascii=False,
            ),
        },
    ]
    content2, timing2 = chat(repair_messages)
    records.append({"attempt": 2, "kind": "format_repair", "timing": timing2, "raw_chars": len(content2)})
    plan2, parse_error2 = parse_json_object(content2)
    if plan2 is None:
        errors2 = [{"field": "response", "code": "invalid_json", "value": parse_error2, "expected": "one JSON plan object"}]
    else:
        plan2, errors2 = validate_plan(
            plan2, instruction=instruction, task=task, snapshot=snapshot, skills=skills
        )
    if plan2 is not None and not errors2:
        return {"status": "ok", "plan": plan2, "attempts": records, "errors": [], "repairs": 1}
    records[-1]["errors"] = errors2
    return {"status": "planning_failed", "plan": None, "attempts": records, "errors": errors2, "repairs": 1}


def request_recovery(
    *,
    instruction: str,
    task: dict[str, Any],
    snapshot: dict[str, Any],
    skills: dict[str, Any],
    failed_step: dict[str, Any],
    failure: dict[str, Any],
    attempts: list[dict[str, Any]] | None = None,
    completed_steps: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    summary = scene_summary(task, snapshot, skills)
    messages = [
        {"role": "system", "content": RECOVERY_PROMPT},
        {
            "role": "user",
            "content": json.dumps(
                {
                    "instruction": instruction,
                    "context": summary,
                    "failed_step": failed_step,
                    "failure": failure,
                    "completed_steps": completed_steps or [],
                    "previous_attempts": attempts or [],
                    "request": "Output the JSON recovery decision object now.",
                },
                ensure_ascii=False,
            ),
        },
    ]
    attempts_out: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    decision: dict[str, Any] | None = None
    for attempt in range(2):
        content, timing = chat(messages)
        attempts_out.append({"attempt": attempt + 1, "timing": timing, "raw_chars": len(content)})
        decision, parse_error = parse_json_object(content)
        if decision is None:
            errors = [{"field": "response", "code": "invalid_json", "value": parse_error}]
        else:
            kind = decision.get("decision")
            if kind not in {"retry", "replan", "abort"}:
                errors = [{"field": "decision", "code": "unknown_decision", "value": kind, "expected": ["retry", "replan", "abort"]}]
            elif kind == "replan":
                candidate = {"steps": decision.get("steps") or [], "interpretation": decision.get("reason") or ""}
                plan, errors = validate_plan(
                    candidate,
                    instruction=instruction,
                    task=task,
                    snapshot=snapshot,
                    skills=skills,
                    prior_steps=completed_steps or [],
                )
                if plan is not None and not errors:
                    return {
                        "status": "ok",
                        "decision": kind,
                        "reason": str(decision.get("reason") or ""),
                        "steps": plan["steps"],
                        "attempts": attempts_out,
                    }
            else:
                return {
                    "status": "ok",
                    "decision": kind,
                    "reason": str(decision.get("reason") or ""),
                    "steps": [],
                    "attempts": attempts_out,
                }
        attempts_out[-1]["errors"] = errors
        if attempt == 0:
            messages = [
                *messages,
                {"role": "assistant", "content": content},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "validation_errors": errors,
                            "request": "Return one corrected JSON recovery decision object using the same schema. For replan, steps must be a non-empty validated list. Do not explain.",
                        },
                        ensure_ascii=False,
                    ),
                },
            ]
    return {
        "status": "invalid",
        "decision": (decision or {}).get("decision"),
        "errors": errors,
        "attempts": attempts_out,
    }
