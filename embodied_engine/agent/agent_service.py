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
                    "one catalog choice, or an ordered pair of catalog choices; code looks up each "
                    "official whole_task prompt and checks pair executability"
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
    """Restrict the skill catalog to what the selected backend can execute."""
    catalog = config.skill_config()
    rows = list(catalog.get("skills") or [])
    name = str(backend or config.engine_config()["action_model"].get("backend") or "pi05").lower()
    wanted = "whole_task" if name == "openwam" else None
    if wanted:
        rows = [row for row in rows if row.get("name") == wanted]
    else:
        rows = [row for row in rows if row.get("name") != "whole_task"]
    return {**catalog, "skills": rows}


TRANSIENT_RETRY_CODES = {"chunk_budget_exhausted"}


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


def make_app() -> web.Application:
    app = web.Application(client_max_size=8 * 1024 * 1024)
    app.router.add_get("/health", health)
    app.router.add_post("/plan", plan)
    app.router.add_post("/recover", recover)
    return app


def main() -> None:
    service = config.engine_config()["services"]["agent"]
    print(f"[agent] qwen semantic router on {service['host']}:{service['port']}", flush=True)
    web.run_app(make_app(), host=service["host"], port=int(service["port"]), access_log=None)


if __name__ == "__main__":
    main()
