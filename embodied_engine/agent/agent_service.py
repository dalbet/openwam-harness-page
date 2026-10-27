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

from agent import d1_planner, planner  # noqa: E402
from common import config  # noqa: E402

AGENT_MODES = {"qwen", "d1", "d1_with_qwen_fallback"}


def _mode() -> str:
    mode = str(config.engine_config()["agent"].get("mode") or "qwen")
    return mode if mode in AGENT_MODES else "qwen"


def _d1_identity() -> dict:
    cfg = config.engine_config()["agent"]["d1"]
    health = d1_planner.D1Client(cfg["url"], timeout_s=2.0).health(timeout_s=0.75)
    return {
        "model": cfg["model"],
        "base_model": cfg["base_model"],
        "license": cfg["license"],
        "source": cfg["source"],
        "url": cfg["url"],
        "reachable": health.get("status") == "ok",
        "health": health if health.get("status") == "ok" else health.get("error") or health,
    }


def _active_model(mode: str, d1: dict, llm_model: str) -> str:
    if mode == "d1":
        return d1["model"]
    if mode == "d1_with_qwen_fallback":
        return f"{d1['model']} with {llm_model} fallback"
    return llm_model


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
    mode = _mode()
    d1 = _d1_identity()
    identity = _model_identity()
    return web.json_response(
        {
            "service": "agent",
            "status": "ok",
            "mode": mode,
            "model": _active_model(mode, d1, agent_cfg["llm_model"]),
            "llm_model": agent_cfg["llm_model"],
            "llm_base_url": agent_cfg["llm_base_url"],
            "llm": identity,
            "d1": d1,
            "planner_vision": False,
            "jev": {
                "enabled": False,
                "official_endpoint": agent_cfg["jev"]["endpoint"],
                "reason": "TypeSafe System One is a waitlisted cloud API; no local weights or credentials are present",
            },
            "role": "task understanding, object binding, skill selection, recovery decisions",
        }
    )


def _qwen_plan(instruction: str, task: dict, snapshot: dict, body: dict) -> dict:
    result = planner.request_plan(
        instruction=instruction,
        task=task,
        snapshot=snapshot,
        skills=_skills_for(body.get("backend")),
        attempts=body.get("attempts") or [],
    )
    result["agent_mode"] = "qwen"
    result.setdefault("plan_source", "qwen3.5-9b")
    return result


def _d1_accepted(d1_result: dict, mode: str) -> dict:
    return {
        "status": "ok",
        "plan": d1_result["plan"],
        "decision": d1_result.get("decision"),
        "evidence": d1_result.get("evidence"),
        "attempts": [],
        "errors": [],
        "repairs": 0,
        "agent_mode": mode,
        "plan_source": "d1-3b",
        "total_s": (d1_result.get("decision") or {}).get("total_s"),
    }


def _d1_rejected(d1_result: dict, mode: str) -> dict:
    return {
        "status": d1_result.get("status") or "rejected",
        "reason_code": d1_result.get("reason_code"),
        "reason": d1_result.get("reason"),
        "evidence": d1_result.get("evidence"),
        "plan": None,
        "attempts": [],
        "errors": [{"field": "plan", "code": d1_result.get("reason_code"), "value": d1_result.get("reason")}],
        "repairs": 0,
        "agent_mode": mode,
        "plan_source": "d1-3b",
    }


def _fallback_allowed(cfg: dict, d1_result: dict | None, d1_error: str | None) -> bool:
    """Only the failure kinds named in agent.d1.fallback_on hand the plan to Qwen.

    A semantic rejection (unsupported/ambiguous/missing information) is a
    decision, not an outage, and stays a rejection by default.
    """
    allowed = set(cfg.get("fallback_on") or [])
    if d1_error is not None:
        return bool(allowed & {"error", "unavailable"})
    return "rejection" in allowed


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
    mode = _mode()
    try:
        if mode in {"d1", "d1_with_qwen_fallback"}:
            d1_cfg = config.engine_config()["agent"]["d1"]
            d1_error = None
            d1_result = None
            try:
                d1_result = d1_planner.plan_with_d1(
                    instruction=instruction,
                    snapshot=snapshot,
                    skills=_skills_for(body.get("backend")),
                    d1_url=d1_cfg["url"],
                    task_id=int(task["task_id"]),
                    thresholds=d1_cfg.get("thresholds"),
                    timeout_s=float(d1_cfg.get("timeout_s") or 30.0),
                )
            except Exception as exc:
                d1_error = f"{type(exc).__name__}: {exc}"
            if d1_result is not None and d1_result.get("status") == "accepted":
                result = _d1_accepted(d1_result, mode)
            elif mode == "d1_with_qwen_fallback" and _fallback_allowed(d1_cfg, d1_result, d1_error):
                result = _qwen_plan(instruction, task, snapshot, body)
                result["agent_mode"] = mode
                result["fallback"] = {
                    "from": "d1-3b",
                    "status": d1_result.get("status") if d1_result else None,
                    "reason": d1_error or (d1_result or {}).get("reason_code"),
                }
                result["plan_source"] = (result.get("plan") or {}).get("plan_source") or "qwen3.5-9b"
            elif d1_result is not None:
                result = _d1_rejected(d1_result, mode)
            else:
                return web.json_response(
                    {"status": "error", "error": f"d1 decision failed: {d1_error}", "agent_mode": mode},
                    status=502,
                )
        else:
            result = _qwen_plan(instruction, task, snapshot, body)
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


def _d1_bounded_recovery(body: dict) -> dict:
    """d1 is a decision model, not a free-text re-planner.

    In pure d1 mode recovery is therefore a bounded execution policy: one
    retry for a whole-task step, then an explicit abort with the failure kept.
    No success is inferred and no plan is rewritten.
    """
    failed_step = body.get("failed_step") or {}
    prior = [
        row
        for row in (body.get("attempts") or [])
        if isinstance(row, dict) and row.get("step") == failed_step
    ]
    if not prior:
        return {
            "status": "ok",
            "decision": "retry",
            "reason": "d1 mode: one bounded retry of the whole-task step",
            "steps": [],
            "policy": "d1_bounded_retry",
            "attempts": [],
        }
    return {
        "status": "ok",
        "decision": "abort",
        "reason": "d1 mode: the whole-task step already failed once; keeping the recorded failure",
        "steps": [],
        "policy": "d1_bounded_retry",
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
    if _mode() == "d1":
        result = _d1_bounded_recovery(body)
        result["total_s"] = round(time.perf_counter() - started, 3)
        return web.json_response(result)
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
    result["agent_mode"] = _mode()
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
    print(f"[agent] mode={_mode()} on {service['host']}:{service['port']}", flush=True)
    web.run_app(make_app(), host=service["host"], port=int(service["port"]), access_log=None)


if __name__ == "__main__":
    main()
