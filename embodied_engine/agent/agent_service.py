#!/usr/bin/env python
"""Agent layer service: semantic routing, plan assembly, bounded recovery.

Binds to 127.0.0.1:14700. Qwen3.5-9B is the only language model. It is used
twice, for different jobs:

* ``/plan``: a compact semantic router that chooses one of the ten catalog
  tasks (or clarify/unsupported) from the user's original instruction. Code
  then looks up the official LIBERO instruction, checks the live scene and
  assembles the single ``whole_task`` step for OpenWAM.
* ``/recover``: a bounded retry/abort decision for the whole-task policy.

No camera pixels are sent to the language model.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from aiohttp import web  # noqa: E402

from agent import planner, semantic_router  # noqa: E402
from common import composite_plan, config, task_catalog  # noqa: E402


def _model_identity() -> dict:
    url = f"{config.engine_config()['agent']['llm_base_url'].rstrip('/')}/models"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(url, timeout=3) as response:
            payload = json.loads(response.read().decode("utf-8"))
        rows = payload.get("data") or []
        return {"reachable": True, "models": [row.get("id") for row in rows]}
    except Exception as exc:
        return {"reachable": False, "error": f"{type(exc).__name__}: {exc}"}


def _router_choice(decision: dict) -> dict:
    return {
        "status": decision.get("status"),
        "choice": decision.get("choice"),
        "reason_code": decision.get("reason_code"),
        "reason": decision.get("reason"),
        "model": (decision.get("router") or {}).get("model"),
        "mode": (decision.get("router") or {}).get("mode"),
        "max_tokens": (decision.get("router") or {}).get("max_tokens"),
        "catalog_version": (decision.get("router") or {}).get("catalog_version"),
        "catalog_sha256": (decision.get("router") or {}).get("catalog_sha256"),
        "latency_s": decision.get("total_s"),
        "repairs": decision.get("repairs"),
        "attempts": decision.get("attempts"),
        "scene_support": decision.get("scene_support"),
    }


def _understanding(
    *,
    instruction: str,
    decision: dict,
    task: dict | None,
    loaded_task: dict,
    assembled: dict | None,
) -> dict:
    chosen = task or {}
    composite_keys = list(decision.get("composite_keys") or [])
    by_key = task_catalog.task_by_key() if composite_keys else {}
    step_official = [
        str((by_key.get(key) or {}).get("official_instruction") or "") for key in composite_keys
    ]
    official = str(chosen.get("official_instruction") or "")
    if composite_keys and step_official:
        official = step_official[-1]
    support = decision.get("scene_support") or {}
    composite_check = (assembled or {}).get("composite") or {}
    return {
        "raw_instruction": instruction,
        "semantic_choice": decision.get("choice"),
        "decision_status": decision.get("status"),
        "reason_code": decision.get("reason_code"),
        "reason": decision.get("reason") or (assembled or {}).get("reason"),
        "task_scope": "combo" if composite_keys else "single",
        "composite_keys": composite_keys,
        "composite_executable": bool(composite_check.get("ok")) if composite_keys else None,
        "composite_reason_code": composite_check.get("reason_code") if composite_keys else None,
        "composite_supported_pairs": composite_check.get("supported_pairs") if composite_keys else None,
        "step_official_instructions": step_official,
        "step_goals": [
            {
                "key": row.get("key"),
                "task_id": row.get("task_id"),
                "goal_kind": (row.get("goal") or {}).get("kind"),
            }
            for row in (composite_check.get("steps") or [])
        ],
        "task_key": chosen.get("key"),
        "task_id": chosen.get("task_id"),
        "flow": decision.get("flow"),
        "flow_guard": decision.get("flow_guard"),
        "flow_expanded_keys": (
            list(decision.get("composite_keys") or []) if decision.get("flow") else None
        ),
        "official_instruction": official or None,
        "prompt_source": "config/tasks.json official_instruction" if official else None,
        "model": (decision.get("router") or {}).get("model"),
        "model_mode": (decision.get("router") or {}).get("mode"),
        "catalog_version": task_catalog.version(),
        "catalog_sha256": task_catalog.fingerprint(),
        "router_latency_s": decision.get("total_s"),
        "scene_support": {
            "ok": support.get("ok"),
            "missing": support.get("missing") or [],
        },
        "loaded_task_id": loaded_task.get("task_id"),
        "loaded_task_key": loaded_task.get("key"),
        "required_task_id": chosen.get("task_id")
        if chosen
        else ((by_key.get(composite_keys[-1]) or {}).get("task_id") if composite_keys else None),
        "required_task_key": chosen.get("key")
        if chosen
        else (composite_keys[-1] if composite_keys else None),
        "needs_scene_switch": bool(assembled and assembled.get("status") == "task_mismatch"),
        "target_preserved": (
            all(
                norm_official(step_official[i]) == norm_official((by_key.get(key) or {}).get("instruction"))
                for i, key in enumerate(composite_keys)
            )
            if composite_keys
            else (official == str(chosen.get("instruction") or "") if official else None)
        ),
    }


def norm_official(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


async def health(request: web.Request) -> web.Response:
    agent_cfg = config.engine_config()["agent"]
    identity = _model_identity()
    return web.json_response(
        {
            "service": "agent",
            "status": "ok",
            "mode": "semantic_router",
            "model": agent_cfg["llm_model"],
            "llm_model": agent_cfg["llm_model"],
            "llm_base_url": agent_cfg["llm_base_url"],
            "llm": identity,
            "planner_vision": False,
            "routing": {
                "contract": (
                    "one catalog choice, or an ordered sequence of two or three catalog choices; code "
                    "looks up each official whole_task prompt and checks sequence executability"
                ),
                **semantic_router.router_options(),
                "composite": composite_plan.options(config.engine_config()),
                "catalog": task_catalog.catalog_evidence(),
            },
            "jev": {
                "enabled": False,
                "official_endpoint": agent_cfg["jev"]["endpoint"],
                "reason": "TypeSafe System One is a waitlisted cloud API; no local weights or credentials are present",
            },
            "role": "semantic routing to the task catalog, plan assembly, bounded recovery decisions",
        }
    )


def _structured_plan(keys: list[str], instruction: str, task: dict, snapshot: dict, body: dict) -> dict:
    """Internal screening entry: an explicit ordered key list, no language step.

    It is *not* a way to bypass the catalog: the same ``assemble_plan``
    validation runs, the keys must be catalog keys, and a composite must be an
    executable pair in the loaded scene. The plan is labelled
    ``structured_keys`` so no record can mistake it for a Qwen decision.
    """
    started = time.perf_counter()
    ordered = [str(key) for key in keys]
    catalog = task_catalog.task_by_key()
    if not ordered:
        return {
            "status": "planning_failed",
            "reason_code": "empty_keys",
            "reason": "structured_keys is empty",
            "plan": None,
        }
    missing = [key for key in ordered if key not in catalog]
    if missing:
        return {
            "status": "planning_failed",
            "reason_code": "key_not_in_catalog",
            "reason": f"unknown catalog keys: {missing}",
            "plan": None,
        }
    decision: dict = {"status": "ok", "composite_keys": ordered if len(ordered) > 1 else None}
    if len(ordered) == 1:
        decision["task"] = catalog[ordered[0]]
        decision["choice"] = ordered[0]
    assembled = semantic_router.assemble_plan(
        decision,
        loaded_task_id=int(task["task_id"]),
        instruction=instruction,
        snapshot=snapshot,
        skills=_skills_for(body.get("backend")),
        loaded_task=task,
        plan_source="structured_keys",
    )
    if assembled.get("status") == "ok" and isinstance(assembled.get("plan"), dict):
        assembled["plan"]["structured_entry"] = True
        assembled["plan"]["structured_keys"] = ordered
    result = {
        "agent_mode": "structured",
        "plan_source": "structured_keys",
        "understanding": {
            "status": "ok" if assembled.get("status") == "ok" else assembled.get("status"),
            "instruction": instruction,
            "task_id": int(task["task_id"]),
            "task_key": task.get("key"),
            "mode": "combo" if len(ordered) > 1 else "single",
            "composite_keys": ordered if len(ordered) > 1 else None,
            "routing": {
                "bypassed": True,
                "reason": "structured screening entry: keys were supplied, not inferred",
                "llm_called": False,
            },
            "plan_source": "structured_keys",
        },
        "attempts": [],
        "errors": assembled.get("errors") or [],
        "repairs": 0,
    }
    if assembled.get("status") != "ok" or not assembled.get("plan"):
        result.update(
            {
                "status": assembled.get("status") or "planning_failed",
                "reason_code": assembled.get("reason_code"),
                "reason": assembled.get("reason") or "the structured plan failed its contract",
                "plan": None,
            }
        )
    else:
        result.update({"status": "ok", "plan": assembled["plan"], "errors": [], "composite": assembled.get("composite")})
    result["total_s"] = round(time.perf_counter() - started, 4)
    return result


def _qwen_plan(instruction: str, task: dict, snapshot: dict, body: dict) -> dict:
    """Route the user text, then assemble the official whole_task plan."""
    started = time.perf_counter()
    decision = semantic_router.decide(instruction, snapshot)
    skills = _skills_for(body.get("backend"))
    chosen = decision.get("task") if isinstance(decision.get("task"), dict) else None
    assembled: dict | None = None
    assembly_started = time.perf_counter()
    if decision.get("status") == "ok":
        assembled = semantic_router.assemble_plan(
            decision,
            loaded_task_id=int(task["task_id"]),
            instruction=instruction,
            snapshot=snapshot,
            skills=skills,
            loaded_task=task,
        )
    assembly_s = time.perf_counter() - assembly_started
    understanding = _understanding(
        instruction=instruction,
        decision=decision,
        task=chosen,
        loaded_task=task,
        assembled=assembled,
    )
    understanding["assembly_s"] = round(assembly_s, 6)
    result: dict = {
        "agent_mode": "qwen",
        "plan_source": "qwen3.5-9b-semantic-router",
        "understanding": understanding,
        "attempts": decision.get("attempts") or [],
        "errors": decision.get("errors") or [],
        "repairs": decision.get("repairs") or 0,
    }
    if decision.get("status") != "ok":
        result.update(
            {
                "status": decision["status"],
                "reason_code": decision.get("reason_code"),
                "reason": decision.get("reason"),
                "plan": None,
            }
        )
    elif assembled is None or assembled.get("status") != "ok" or not assembled.get("plan"):
        result.update(
            {
                "status": (assembled or {}).get("status") or "planning_failed",
                "reason_code": (assembled or {}).get("reason_code"),
                "reason": (assembled or {}).get("reason")
                or "the assembled whole_task plan failed its contract",
                "errors": (assembled or {}).get("errors") or result["errors"],
                "plan": None,
            }
        )
    else:
        result.update({"status": "ok", "plan": assembled["plan"], "errors": []})
    result["total_s"] = round(time.perf_counter() - started, 4)
    return result


def _skills_for(backend: str | None) -> dict:
    """Restrict the skill catalog to what the selected backend can execute.

    "whole_task" used to be OpenWAM's alone.  Pi0.5 now has a whole-task runner
    of its own (``engine/pi05_rollout.Pi05WholeTaskRunner``), so the capability
    is declared per backend in ``action_model.<backend>.whole_task`` instead of
    being hardcoded to the OpenWAM name.  A backend that declares it gets the
    whole_task-only catalog, exactly as OpenWAM does, so the plan shape cannot
    silently fall back to short skills when the rollout is whole-task.
    """
    catalog = config.skill_config()
    rows = list(catalog.get("skills") or [])
    action = config.engine_config()["action_model"]
    name = str(backend or action.get("backend") or "pi05").lower()
    section = action.get(name) if isinstance(action.get(name), dict) else {}
    whole = bool((section or {}).get("whole_task", name == "openwam"))
    if whole:
        rows = [row for row in rows if row.get("name") == "whole_task"]
    else:
        rows = [row for row in rows if row.get("name") != "whole_task"]
    return {**catalog, "skills": rows}


TRANSIENT_RETRY_CODES = {"chunk_budget_exhausted"}


REPAIR_SYSTEM_PROMPT = """You are the Agent layer of a LIBERO-Goal robot system repairing a finished composite episode.
The episode ran an ordered two-step plan in one scene. Its final step reached the loaded scene's native goal and released the object. At least one earlier step's goal no longer holds on a fresh state check.

You do not see camera images. Your input is the plan, the fresh per-step state, the lost goals, and the catalog capabilities the engine has already validated as executable in this scene. Return exactly one JSON object and nothing else:
{"decision": "repair" | "abort", "key": "<one catalog key from candidates, or null>", "reason": "one short sentence"}

Rules:
- Choose "repair" only when one listed capability plausibly restores a lost earlier goal without undoing the final step's placement.
- "key" must be one of the candidate keys verbatim. Do not invent keys, objects, sites, or parameters.
- Choose "abort" when no candidate can restore a lost goal, or when any candidate is likely to disturb the final placement.
- You are choosing a task, not writing a trajectory. Do not output code, coordinates, or a step list.
- Do not claim success: the engine verifies every final goal again after the repair.
"""


def _repair_decision(body: dict) -> dict:
    """One compact structured repair choice from Qwen (no control code)."""
    started = time.perf_counter()
    candidates = [row for row in (body.get("candidates") or []) if isinstance(row, dict)]
    keys = {str(row.get("key")) for row in candidates if row.get("key")}
    payload = {
        "instruction": body.get("instruction"),
        "loaded_task_id": body.get("task_id"),
        "plan_keys": (body.get("composite") or {}).get("keys"),
        "steps_on_fresh_state": body.get("steps") or [],
        "lost_goals": body.get("failed") or [],
        "candidates": [
            {
                "key": row.get("key"),
                "official_instruction": row.get("official_instruction"),
                "goal_kind": row.get("goal_kind"),
                "goal_met_now": row.get("goal_met_now"),
            }
            for row in candidates
        ],
    }
    messages = [
        {"role": "system", "content": REPAIR_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]
    attempts: list[dict] = []
    errors: list[dict] = []
    for index in range(2):
        content, timing = planner.chat(messages, max_tokens=192)
        attempts.append({"content": content, "timing": timing})
        decision, key, errors = _parse_repair(content, keys)
        if decision is not None:
            result = {
                "status": "ok",
                "decision": decision,
                "key": key,
                "reason": _repair_reason(content),
                "errors": [],
                "attempts": attempts,
                "policy": "qwen3.5-9b-semantic-router",
                "total_s": round(time.perf_counter() - started, 4),
            }
            return result
        if index == 0:
            messages = messages + [
                {"role": "assistant", "content": content},
                {
                    "role": "user",
                    "content": 'Return exactly one JSON object: {"decision":"repair"|"abort","key":<candidate key or null>,"reason":"..."}',
                },
            ]
    return {
        "status": "planning_failed",
        "decision": None,
        "key": None,
        "reason": errors[0].get("value") if errors else "invalid repair decision",
        "reason_code": "repair_format_invalid",
        "errors": errors,
        "attempts": attempts,
        "policy": "qwen3.5-9b-semantic-router",
        "total_s": round(time.perf_counter() - started, 4),
    }


def _parse_repair(content: str, keys: set[str]) -> tuple[str | None, str | None, list[dict]]:
    text = str(content or "").strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None, None, [{"field": "repair", "code": "not_json", "value": text[:200]}]
    try:
        row = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        return None, None, [{"field": "repair", "code": "invalid_json", "value": str(exc)}]
    if not isinstance(row, dict):
        return None, None, [{"field": "repair", "code": "not_object", "value": type(row).__name__}]
    decision = str(row.get("decision") or "")
    if decision == "abort":
        return "abort", None, []
    if decision != "repair":
        return None, None, [{"field": "decision", "code": "unknown", "value": decision}]
    key = str(row.get("key") or "")
    if key not in keys:
        return None, None, [{"field": "key", "code": "not_a_candidate", "value": key}]
    return "repair", key, []


def _repair_reason(content: str) -> str:
    text = str(content or "").strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return ""
    try:
        row = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return ""
    return str(row.get("reason") or "")[:300] if isinstance(row, dict) else ""


def _bounded_transient_decision(body: dict) -> dict | None:
    """One bounded retry for a transient execution timeout.

    This is a decision-layer rule, not a planner shortcut. It never declares
    success and never relaxes completion evidence.
    """
    failure = body.get("failure") or {}
    code = str(failure.get("reason_code") or "")
    if code not in TRANSIENT_RETRY_CODES:
        return None
    failed_step = body.get("failed_step") or {}
    prior = [
        row
        for row in (body.get("attempts") or [])
        if isinstance(row, dict) and row.get("step") == failed_step
    ]
    if prior:
        return None
    return {
        "status": "ok",
        "decision": "retry",
        "reason": f"bounded transient retry after {code}",
        "steps": [],
        "policy": "bounded_transient_retry",
        "attempts": [],
    }


async def plan(request: web.Request) -> web.Response:
    try:
        body = await request.json()
    except json.JSONDecodeError:
        return web.json_response({"status": "bad_request", "error": "invalid JSON"}, status=400)
    instruction = str(body.get("instruction") or "").strip()
    if not instruction:
        return web.json_response({"status": "bad_request", "error": "instruction is required"}, status=400)
    try:
        task = config.task_for(int(body.get("task_id")))
    except (KeyError, TypeError, ValueError) as exc:
        return web.json_response({"status": "bad_request", "error": str(exc)}, status=400)
    snapshot = body.get("snapshot")
    if not isinstance(snapshot, dict):
        return web.json_response({"status": "bad_request", "error": "snapshot is required"}, status=400)
    started = time.perf_counter()
    try:
        structured = body.get("structured_keys")
        if isinstance(structured, list) and structured:
            result = _structured_plan(
                [str(key) for key in structured], instruction, task, snapshot, body
            )
        else:
            result = _qwen_plan(instruction, task, snapshot, body)
    except Exception as exc:
        return web.json_response({"status": "error", "error": f"{type(exc).__name__}: {exc}"}, status=502)
    result["total_s"] = round(time.perf_counter() - started, 4)
    http_status = {
        "ok": 200,
        "task_mismatch": 409,
        "needs_clarification": 422,
        "unsupported": 422,
        "scene_not_supported": 422,
        "planning_failed": 422,
    }.get(str(result.get("status")), 422)
    return web.json_response(result, status=http_status)


async def recover(request: web.Request) -> web.Response:
    try:
        body = await request.json()
    except json.JSONDecodeError:
        return web.json_response({"status": "bad_request", "error": "invalid JSON"}, status=400)
    try:
        task = config.task_for(int(body.get("task_id")))
    except (KeyError, TypeError, ValueError) as exc:
        return web.json_response({"status": "bad_request", "error": str(exc)}, status=400)
    instruction = str(body.get("instruction") or task.get("instruction"))
    snapshot = body.get("snapshot")
    if not isinstance(snapshot, dict):
        return web.json_response({"status": "bad_request", "error": "snapshot is required"}, status=400)
    started = time.perf_counter()
    fast = _bounded_transient_decision(body)
    if fast is not None:
        fast["total_s"] = round(time.perf_counter() - started, 3)
        return web.json_response(fast)
    try:
        result = planner.request_recovery(
            instruction=instruction,
            task=task,
            snapshot=snapshot,
            skills=_skills_for(body.get("backend")),
            failed_step=body.get("failed_step") or {},
            failure=body.get("failure") or {},
            attempts=body.get("attempts") or [],
            completed_steps=body.get("completed_steps") or [],
        )
    except Exception as exc:
        return web.json_response({"status": "error", "error": f"{type(exc).__name__}: {exc}"}, status=502)
    result["agent_mode"] = "qwen"
    result["total_s"] = round(time.perf_counter() - started, 3)
    status = 200 if result.get("status") == "ok" else 422
    return web.json_response(result, status=status)


async def repair(request: web.Request) -> web.Response:
    """Bounded repair decision: one catalog key or abort."""
    try:
        body = await request.json()
    except json.JSONDecodeError:
        return web.json_response({"status": "bad_request", "error": "invalid JSON"}, status=400)
    try:
        config.task_for(int(body.get("task_id")))
    except (KeyError, TypeError, ValueError) as exc:
        return web.json_response({"status": "bad_request", "error": str(exc)}, status=400)
    if not isinstance(body.get("snapshot"), dict):
        return web.json_response({"status": "bad_request", "error": "snapshot is required"}, status=400)
    if not isinstance(body.get("candidates"), list):
        return web.json_response({"status": "bad_request", "error": "candidates is required"}, status=400)
    try:
        result = _repair_decision(body)
    except Exception as exc:
        return web.json_response({"status": "error", "error": f"{type(exc).__name__}: {exc}"}, status=502)
    result["agent_mode"] = "qwen"
    status = 200 if result.get("status") == "ok" else 422
    return web.json_response(result, status=status)


def make_app() -> web.Application:
    app = web.Application(client_max_size=8 * 1024 * 1024)
    app.router.add_get("/health", health)
    app.router.add_post("/plan", plan)
    app.router.add_post("/recover", recover)
    app.router.add_post("/repair", repair)
    return app


def main() -> None:
    service = config.engine_config()["services"]["agent"]
    print(f"[agent] qwen semantic router on {service['host']}:{service['port']}", flush=True)
    web.run_app(make_app(), host=service["host"], port=int(service["port"]), access_log=None)


if __name__ == "__main__":
    main()
