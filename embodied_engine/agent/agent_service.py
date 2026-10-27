#!/usr/bin/env python
"""Agent layer service: task understanding, structured planning, recovery.

Binds to 127.0.0.1:14700. The model is the local Qwen3.5-9B served by vLLM
on port 8000. Jev (TypeSafe System One) is disabled: only a waitlisted cloud
API is published, no local weights or credentials are available. The planner
receives scene metadata and simulation state, not camera pixels.
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

from agent import planner  # noqa: E402
from common import config  # noqa: E402


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


async def health(request: web.Request) -> web.Response:
    agent_cfg = config.engine_config()["agent"]
    identity = _model_identity()
    return web.json_response(
        {
            "service": "agent",
            "status": "ok",
            "model": agent_cfg["llm_model"],
            "llm_base_url": agent_cfg["llm_base_url"],
            "llm": identity,
            "planner_vision": False,
            "jev": {
                "enabled": False,
                "official_endpoint": agent_cfg["jev"]["endpoint"],
                "reason": "TypeSafe System One is a waitlisted cloud API; no local weights or credentials are present",
            },
            "role": "task understanding, object binding, skill selection, recovery decisions",
        }
    )


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
        result = planner.request_plan(
            instruction=instruction,
            task=task,
            snapshot=snapshot,
            skills=_skills_for(body.get("backend")),
            attempts=body.get("attempts") or [],
        )
    except Exception as exc:
        return web.json_response({"status": "error", "error": f"{type(exc).__name__}: {exc}"}, status=502)
    result["total_s"] = round(time.perf_counter() - started, 3)
    status = 200 if result["status"] == "ok" else 422
    return web.json_response(result, status=status)


def _skills_for(backend: str | None) -> dict:
    """Restrict the skill catalog to what the selected backend can execute.

    A whole-task policy cannot run pick_place/open_drawer sub-skills, and a
    sub-skill executor cannot run a whole-task policy: the catalog is the
    contract that keeps the planner from proposing an unexecutable step.
    """
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
    """Agent policy: one bounded retry for a transient execution timeout.

    This is a decision-layer rule, not a planner shortcut. It never declares
    success and never relaxes completion evidence; it only avoids spending a
    full language-model call to rediscover that a single chunk budget ran out.
    A second failure for the same step goes to the language model.
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
    result["total_s"] = round(time.perf_counter() - started, 3)
    status = 200 if result["status"] == "ok" else 422
    return web.json_response(result, status=status)


def make_app() -> web.Application:
    app = web.Application(client_max_size=8 * 1024 * 1024)
    app.router.add_get("/health", health)
    app.router.add_post("/plan", plan)
    app.router.add_post("/recover", recover)
    return app


def main() -> None:
    service = config.engine_config()["services"]["agent"]
    print(f"[agent] Qwen planner on {service['host']}:{service['port']}", flush=True)
    web.run_app(make_app(), host=service["host"], port=int(service["port"]), access_log=None)


if __name__ == "__main__":
    main()
