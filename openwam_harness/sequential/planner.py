"""One initial planner request plus at most one format repair.

A failed repair returns PLANNING_FAILED. It does not substitute a human plan.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from budget import OUTPUT_BUDGET, budget_check, slim_prompt
from schema import TOOL_SCHEMA, PlanError, parse_model_message, semantic_example, submit_plan_tools, validate_plan

OUTPUT_FORMAT = (
    "Call submit_plan once. Arguments are a short semantic plan: plan_id, goal, and steps. "
    "Each step skill is transfer, with object_ref, target_ref, relation, prompt, depends_on, "
    "completion_check object_placed, budget, and on_failure. Do not emit primitive skills or "
    "controller parameters. Do not also print a second JSON plan. goal must equal the task text."
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
    on_event: Callable[[str, str], None] | None = None,
) -> dict[str, Any]:
    example = json.dumps(semantic_example(), ensure_ascii=False)
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
    if on_event is not None:
        options["on_event"] = on_event
    attempts = []
    message = _normalize(client(messages, OUTPUT_BUDGET, options))
    _emit(options, "validating", "initial")
    attempts.append(_attempt(1, "initial", message, parts, counter))
    try:
        kind, plan = parse_model_message(message, task=task)
        plan["plan_source"] = "model"
        return _ok("model", plan, attempts, instruction_report, kind)
    except PlanError as exc:
        first_errors = exc.errors
        if any(item.get("code") in {"incomplete_stream", "truncated_output"} for item in first_errors):
            return _failed(attempts, first_errors, repairs=0)
    repair = _repair_text(message, first_errors, task)
    repair_parts = [*parts, ("repair", repair)]
    messages = [
        *messages,
        {"role": "assistant", "content": _replay(message)},
        {"role": "user", "content": repair},
    ]
    message = _normalize(client(messages, OUTPUT_BUDGET, options))
    _emit(options, "validating", "format_repair")
    attempts.append(_attempt(2, "format_repair", message, repair_parts, counter))
    try:
        kind, plan = parse_model_message(message, task=task)
        plan["plan_source"] = "model"
        return _ok("model_repair", plan, attempts, instruction_report, kind)
    except PlanError as exc:
        return _failed(attempts, exc.errors, repairs=1)


def _failed(attempts: list[dict], errors: list[dict], *, repairs: int) -> dict[str, Any]:
    return {
        "status": "PLANNING_FAILED",
        "plan": None,
        "source": None,
        "plan_source": None,
        "attempts": attempts,
        "errors": errors,
        "budgets": [item["preflight"] for item in attempts],
        "output_budget": OUTPUT_BUDGET,
        "repairs": repairs,
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
        "request_id": result.get("request_id"),
        "incomplete": bool(result.get("incomplete")),
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
        "request_id": message.get("request_id"),
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


def _emit(options: dict[str, Any], phase: str, detail: str) -> None:
    callback = options.get("on_event")
    if callback is not None:
        callback(phase, detail)


def make_openai_client(base_url: str, model: str = "Qwen3.5-9B", timeout_s: float = 180.0):
    """Streaming chat client. Closing the socket aborts only this request id.

    The 180s cap is a safety bound. It is not a performance fix. The previous
    420s value only waited out a 2048-token generation.
    """
    import urllib.error
    import urllib.request
    import uuid

    url = base_url.rstrip("/") + "/chat/completions"
    state: dict[str, Any] = {"response": None, "request_id": None}

    def cancel() -> None:
        response = state.get("response")
        if response is not None:
            try:
                response.close()
            except Exception:
                pass

    def client(messages, max_tokens, options=None):
        options = options or {}
        request_id = options.get("request_id") or f"seq-{uuid.uuid4().hex[:12]}"
        state["request_id"] = request_id
        body = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": options.get("temperature", 0),
            "stream": True,
            "stream_options": {"include_usage": True},
            "request_id": request_id,
            "chat_template_kwargs": options.get("chat_template_kwargs") or {"enable_thinking": False},
        }
        if options.get("tools"):
            body["tools"] = options["tools"]
            body["tool_choice"] = options.get("tool_choice")
        try:
            return _stream(url, body, timeout_s, request_id, options.get("on_event"), state)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            if exc.code == 400 and "tools" in body and _tools_rejected(detail):
                reduced = dict(body)
                reduced.pop("tools", None)
                reduced.pop("tool_choice", None)
                message = _stream(url, reduced, timeout_s, request_id, options.get("on_event"), state)
                message["tool_transport"] = "unsupported"
                return message
            raise RuntimeError(f"planner HTTP {exc.code}: {detail}") from exc

    client.cancel = cancel
    client.request_id = lambda: state.get("request_id")
    return client


def _stream(url, body, timeout_s, request_id, on_event, state) -> dict[str, Any]:
    import urllib.request

    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer EMPTY",
            "X-Request-Id": request_id,
        },
    )
    if on_event:
        on_event("queued", request_id)
    response = urllib.request.urlopen(request, timeout=timeout_s)
    state["response"] = response
    if on_event:
        on_event("processing_input", request_id)
    content: list[str] = []
    reasoning: list[str] = []
    tool_calls: dict[int, dict[str, Any]] = {}
    finish = None
    usage: dict[str, Any] = {}
    started = False
    try:
        for raw in response:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            if chunk.get("usage"):
                usage = chunk["usage"]
            choices = chunk.get("choices") or []
            if not choices:
                continue
            choice = choices[0]
            if choice.get("finish_reason"):
                finish = choice["finish_reason"]
            delta = choice.get("delta") or {}
            piece = delta.get("content") or delta.get("reasoning") or delta.get("reasoning_content")
            if not started and (piece or delta.get("tool_calls")):
                started = True
                if on_event:
                    on_event("generating", request_id)
            if delta.get("content"):
                content.append(delta["content"])
            if delta.get("reasoning") or delta.get("reasoning_content"):
                reasoning.append(delta.get("reasoning") or delta.get("reasoning_content"))
            for call in delta.get("tool_calls") or []:
                index = int(call.get("index") or 0)
                slot = tool_calls.setdefault(index, {"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                if call.get("id"):
                    slot["id"] = call["id"]
                fn = call.get("function") or {}
                if fn.get("name"):
                    slot["function"]["name"] += fn["name"]
                if fn.get("arguments"):
                    slot["function"]["arguments"] += fn["arguments"]
    finally:
        state["response"] = None
        response.close()
    calls = [tool_calls[index] for index in sorted(tool_calls)]
    return {
        "content": "".join(content),
        "tool_calls": calls or None,
        "finish_reason": finish,
        "usage": usage,
        "reasoning": "".join(reasoning),
        "request_id": request_id,
        "incomplete": finish is None,
    }


def _tools_rejected(detail: str) -> bool:
    text = detail.lower()
    return "tool" in text and any(word in text for word in ("unsupported", "unknown", "extra", "not allowed"))
