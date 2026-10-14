"""One initial planner request plus at most one format repair.

A failed repair returns PLANNING_FAILED. It does not substitute a human plan.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from budget import OUTPUT_BUDGET, budget_check, slim_prompt
from schema import TOOL_SCHEMA, PlanError, example_plan, parse_model_message, submit_plan_tools, validate_plan

OUTPUT_FORMAT = (
    "Call submit_plan once. Its arguments are the plan object with plan_id, goal, and steps. "
    "Do not also print a second JSON plan. If tool calls are unavailable, return one JSON object "
    "and no markdown. goal must equal the task text. Do not invent skills."
)
SHORT_MEMORY = (
    "Earlier original-mode episodes failed before pi0_pick, from a planner timeout or a "
    "32768-token context overflow. Do not reload those transcripts."
)


def load_human_plan(path, scene_id: str) -> dict[str, Any]:
    catalog = json.loads(path.read_text(encoding="utf-8"))
    plan = catalog[scene_id]
    plan = validate_plan(plan, task=plan.get("goal"))
    plan["plan_source"] = "manual"
    return plan


def request_plan(
    *,
    task: str,
    scene_observation: str,
    client: Callable[..., dict[str, Any] | str],
    counter=None,
    short_memory: str = SHORT_MEMORY,
    image_ids: list[str] | None = None,
) -> dict[str, Any]:
    example = json.dumps(example_plan(), ensure_ascii=False)
    observation = scene_observation.strip()
    if image_ids:
        observation += "\nimage_ids: " + ", ".join(image_ids)
    parts = slim_prompt(
        task=task,
        scene_observation=observation,
        tool_schema=TOOL_SCHEMA,
        short_memory=short_memory,
        output_format=OUTPUT_FORMAT,
        example=example,
    )
    instruction_report = budget_check(parts, counter=counter)
    messages = [{"role": "user", "content": _join(parts)}]
    options = _request_options()
    attempts = []
    message = _normalize(client(messages, OUTPUT_BUDGET, options))
    attempts.append(_attempt(1, "initial", message, parts, counter))
    try:
        kind, plan = parse_model_message(message, task=task)
        plan["plan_source"] = "model"
        return _ok("model", plan, attempts, instruction_report, kind)
    except PlanError as exc:
        first_errors = exc.errors
    repair = _repair_text(message, first_errors, task)
    repair_parts = [*parts, ("repair", repair)]
    messages = [
        *messages,
        {"role": "assistant", "content": _replay(message)},
        {"role": "user", "content": repair},
    ]
    message = _normalize(client(messages, OUTPUT_BUDGET, options))
    attempts.append(_attempt(2, "format_repair", message, repair_parts, counter))
    try:
        kind, plan = parse_model_message(message, task=task)
        plan["plan_source"] = "model"
        return _ok("model_repair", plan, attempts, instruction_report, kind)
    except PlanError as exc:
        return {
            "status": "PLANNING_FAILED",
            "plan": None,
            "source": None,
            "plan_source": None,
            "attempts": attempts,
            "errors": exc.errors,
            "budgets": [item["preflight"] for item in attempts],
            "output_budget": OUTPUT_BUDGET,
            "repairs": 1,
        }


def _request_options() -> dict[str, Any]:
    return {
        "tools": submit_plan_tools(),
        "tool_choice": {"type": "function", "function": {"name": "submit_plan"}},
        "chat_template_kwargs": {"enable_thinking": False},
        "temperature": 0,
    }


def _normalize(result: dict[str, Any] | str) -> dict[str, Any]:
    if isinstance(result, str):
        return {
            "content": result,
            "tool_calls": None,
            "finish_reason": "stop",
            "usage": {},
            "reasoning": None,
        }
    return {
        "content": result.get("content"),
        "tool_calls": result.get("tool_calls"),
        "finish_reason": result.get("finish_reason") or "stop",
        "usage": result.get("usage") or {},
        "reasoning": result.get("reasoning"),
    }


def _attempt(index: int, role: str, message: dict[str, Any], parts, counter) -> dict[str, Any]:
    content = message.get("content") or ""
    tool_calls = message.get("tool_calls") or []
    if tool_calls and content.strip():
        kind = "tool_calls"
        ignored = "content ignored because tool_calls are present"
    elif tool_calls:
        kind = "tool_calls"
        ignored = None
    elif isinstance(content, str) and content.strip()[:1] in "{[":
        kind = "json"
        ignored = None
    else:
        kind = "text"
        ignored = None
    if message.get("finish_reason") == "length":
        kind = "truncated"
    reasoning = message.get("reasoning") or ""
    return {
        "index": index,
        "role": role,
        "response_kind": kind,
        "finish_reason": message.get("finish_reason"),
        "thinking_disabled_sent": True,
        "reasoning_present": bool(str(reasoning).strip()) or "<think>" in content,
        "ignored_content": ignored,
        "content": content,
        "tool_calls": tool_calls,
        "usage": message.get("usage") or {},
        "preflight": budget_check(parts, counter=counter),
    }


def _repair_text(message: dict[str, Any], errors: list[dict[str, Any]], task: str) -> str:
    original = _replay(message)
    return (
        "Format repair 1/1. Keep the task intent and do not add skills that are not registered. "
        f"The task text must stay exactly: {task}\n"
        f"Validation errors: {json.dumps(errors, ensure_ascii=False)}\n"
        "Fix only format or illegal fields. Call submit_plan again with the corrected arguments.\n"
        f"Original output:\n{original[:4000]}"
    )


def _replay(message: dict[str, Any]) -> str:
    calls = message.get("tool_calls") or []
    if calls:
        return json.dumps({"tool_calls": calls}, ensure_ascii=False)
    return str(message.get("content") or "")


def _ok(source: str, plan: dict[str, Any], attempts: list[dict], preflight: dict, kind: str) -> dict[str, Any]:
    return {
        "status": "ok",
        "plan": plan,
        "source": source,
        "plan_source": "model",
        "response_kind": kind,
        "attempts": attempts,
        "errors": [],
        "budgets": [item["preflight"] for item in attempts],
        "output_budget": OUTPUT_BUDGET,
        "repairs": 1 if source == "model_repair" else 0,
        "preflight": preflight,
    }


def _join(parts: list[tuple[str, str]]) -> str:
    return "\n".join(f"[{name}]\n{text}" for name, text in parts)


def make_openai_client(base_url: str, model: str = "Qwen3.5-9B", timeout_s: float = 420.0):
    """Chat client. The Authorization header is a local placeholder and is not recorded."""
    import urllib.error
    import urllib.request

    url = base_url.rstrip("/") + "/chat/completions"

    def client(messages, max_tokens, options=None):
        options = options or {}
        body = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": options.get("temperature", 0),
            "chat_template_kwargs": options.get("chat_template_kwargs") or {"enable_thinking": False},
        }
        if options.get("tools"):
            body["tools"] = options["tools"]
            body["tool_choice"] = options.get("tool_choice")
        return _post(url, body, timeout_s)

    return client


def _post(url: str, body: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    import urllib.error
    import urllib.request

    try:
        return _read_completion(_send(url, body, timeout_s))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        if exc.code == 400 and "tools" in body and _tools_rejected(detail):
            reduced = dict(body)
            reduced.pop("tools", None)
            reduced.pop("tool_choice", None)
            message = _read_completion(_send(url, reduced, timeout_s))
            message["tool_transport"] = "unsupported"
            return message
        raise RuntimeError(f"planner HTTP {exc.code}: {detail}") from exc


def _send(url: str, body: dict[str, Any], timeout_s: float):
    import urllib.request

    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
    )
    return urllib.request.urlopen(request, timeout=timeout_s)


def _read_completion(response) -> dict[str, Any]:
    payload = json.loads(response.read().decode("utf-8"))
    choice = payload["choices"][0]
    message = choice.get("message") or {}
    return {
        "content": message.get("content"),
        "tool_calls": message.get("tool_calls"),
        "finish_reason": choice.get("finish_reason"),
        "usage": payload.get("usage") or {},
        "reasoning": message.get("reasoning_content") or message.get("reasoning"),
    }


def _tools_rejected(detail: str) -> bool:
    text = detail.lower()
    return "tool" in text and any(word in text for word in ("unsupported", "unknown", "extra", "not allowed"))
