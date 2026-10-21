"""One initial planner request plus at most one format repair.

A failed repair returns PLANNING_FAILED. It does not substitute a human plan.
"""

from __future__ import annotations

import json
import time
from typing import Any, Callable

from budget import OUTPUT_BUDGET, budget_check, slim_prompt
from schema import (
    TOOL_SCHEMA,
    PlanError,
    failure_stage,
    free_submit_plan_tools,
    parse_model_message,
    free_semantic_example,
    semantic_example,
    submit_plan_tools,
    validate_plan,
)

OUTPUT_FORMAT = (
    "Call submit_plan once. Arguments are one flat plan: plan_id, goal, goal_relation, and steps. "
    "Each step is one transfer, and one transfer is the whole grasp, carry, place, and check. "
    "For a single placement, emit one transfer. Do not add a second transfer with the same "
    "object_ref, target_ref, and relation. Do not nest submit_plan inside arguments or steps. "
    "Do not send an objects map, primitive skills, controller parameters, or code. "
    "Bind each object to a candidate number: set object_n and target_n. "
    "The service derives object_ref and target_ref. "
    "Do not choose candidate 1 when several instances share a type. "
    "goal must equal the task text. "
    "This request contains scene metadata and image ids only, not image bytes. "
    "A sim_state binding is not a visual judgment."
)
FREE_OUTPUT_FORMAT = (
    "你是机器人任务理解与规划器。理解用户的中文或英文，并根据当前已加载场景选择对应的标准任务。"
    "用户可以使用近义词、简称和不同语序，不要求复述任务集原句。"
    "例如「碗放炉灶上」和「把黑碗放到灶台上」可能表达同一目标，但必须依据场景候选对象和目标区域确认。"
    "若表达对应 standard_tasks 中的单任务，canonical_instruction 填写该行英文原句，matched_task_key 填写该行 task_key。"
    "搬运使用已实现的 transfer，并填写 object_n 和 target_n。"
    "打开当前柜子的中层抽屉使用 open_drawer，relation 用 open；object_n 与 target_n 都填柜子的候选编号，后端只使用 target_n。"
    "不要把开抽屉写成 transfer，也不要改成上层抽屉、推动或旋钮。"
    "保留用户指定的位置关系，不能改选同类物体。goal 保留用户原句，标准任务不能覆盖用户原意。"
    "若存在多个合理实例或多个合理任务，返回一个具体澄清问题，不要猜测。"
    "若所需技能未实现，说明缺少的技能。不得虚构物体、区域或技能。不得生成或执行代码。不得为了匹配目录而切换场景。"
    "本轮只执行单任务。识别到多个独立目标时说明当前仅执行单任务，不要输出多步 transfer。"
    "不要输出 object_ref 或 target_ref。实例 ID 由后端根据候选编号派生。"
    "vla_prompt 只写短英文抓取句，例如 pick up the black bowl。不要写整句放置指令，不要写实例 ID，不要写候选编号。"
    "A self-reported done flag is not execution success. "
    "This request is scene metadata, vision=false, evidence_source=sim_state. image_id is not image content."
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
    scene_objects: dict[str, Any] | None = None,
    required_object_ref: str | None = None,
    numbered_candidates: list[dict[str, Any]] | None = None,
    on_event: Callable[[str, str], None] | None = None,
    execution_mode: str = "benchmark",
    predicates: list[dict[str, Any]] | None = None,
    ordered: bool = False,
) -> dict[str, Any]:
    example = json.dumps(free_semantic_example() if execution_mode == "free" else semantic_example(), ensure_ascii=False)
    output_format = OUTPUT_FORMAT
    if execution_mode == "free":
        output_format = FREE_OUTPUT_FORMAT
    observation = scene_observation.strip()
    if image_ids:
        observation += "\nimage_ids: " + ", ".join(image_ids)
    parts = slim_prompt(
        task=task,
        scene_observation=observation,
        tool_schema=TOOL_SCHEMA,
        short_memory=short_memory,
        output_format=output_format,
        example=example,
    )
    instruction_report = budget_check(parts, counter=counter)
    messages = [{"role": "user", "content": _join(parts)}]
    options = _request_options()
    if execution_mode == "free":
        options["tools"] = free_submit_plan_tools()
        options["tool_choice"] = "auto"
    if on_event is not None:
        options["on_event"] = on_event
    attempts = []
    message = _normalize(client(messages, OUTPUT_BUDGET, options))
    _emit(options, "validating", "initial")
    attempts.append(_attempt(1, "initial", message, parts, counter))
    question = _clarification(message) if execution_mode == "free" else None
    if question:
        return _clarify(question, attempts, instruction_report)
    try:
        kind, plan, resolution = _parsed(
            message,
            task=task,
            scene_objects=scene_objects,
            required_object_ref=required_object_ref,
            numbered_candidates=numbered_candidates,
            execution_mode=execution_mode,
            predicates=predicates,
            ordered=ordered,
        )
        plan["plan_source"] = "model"
        return _ok("model", plan, attempts, instruction_report, kind, resolution)
    except PlanError as exc:
        first_errors = exc.errors
        attempts[0]["errors"] = first_errors
        if any(item.get("code") in {"incomplete_stream", "truncated_output"} for item in first_errors):
            return _failed(attempts, first_errors, repairs=0)
        if first_errors and all(item.get("code") in {"target_region_missing", "target_region_ambiguous"} for item in first_errors):
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
    question = _clarification(message) if execution_mode == "free" else None
    if question:
        return _clarify(question, attempts, instruction_report)
    try:
        kind, plan, resolution = _parsed(
            message,
            task=task,
            scene_objects=scene_objects,
            required_object_ref=required_object_ref,
            numbered_candidates=numbered_candidates,
            execution_mode=execution_mode,
            predicates=predicates,
            ordered=ordered,
        )
        plan["plan_source"] = "model"
        return _ok("model_repair", plan, attempts, instruction_report, kind, resolution)
    except PlanError as exc:
        attempts[-1]["errors"] = exc.errors
        return _failed(attempts, exc.errors, repairs=1)


def _clarification(message: dict[str, Any]) -> str | None:
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        if function.get("name") != "ask_clarification":
            continue
        raw = function.get("arguments")
        payload = raw
        if isinstance(raw, str):
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                return None
        question = payload.get("question") if isinstance(payload, dict) else None
        if isinstance(question, str) and question.strip():
            return question.strip()
    return None


def _clarify(question: str, attempts: list[dict], instruction_report: dict) -> dict[str, Any]:
    return {
        "status": "clarify",
        "plan": None,
        "question": question,
        "source": "model",
        "plan_source": "model",
        "attempts": attempts,
        "errors": [],
        "budgets": [item["preflight"] for item in attempts],
        "output_budget": OUTPUT_BUDGET,
        "repairs": max(0, len(attempts) - 1),
        "stage": "clarify",
        "plan_from": None,
        "instruction_report": instruction_report,
    }


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
        "stage": failure_stage(errors),
        "plan_from": None,
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
            "telemetry": {},
        }
    return {
        "content": result.get("content"),
        "tool_calls": result.get("tool_calls"),
        "finish_reason": result.get("finish_reason") or "stop",
        "usage": result.get("usage") or {},
        "reasoning": result.get("reasoning"),
        "request_id": result.get("request_id"),
        "incomplete": bool(result.get("incomplete")),
        "telemetry": result.get("telemetry") or {},
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
        "telemetry": message.get("telemetry") or {},
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


def _parsed(message, **kwargs) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
    numbered = kwargs.pop("numbered_candidates", None)
    kind, plan = parse_model_message(message, **kwargs) if numbered is None else _parse_numbered(message, numbered, kwargs)
    if numbered is None:
        return kind, plan, []
    return kind, plan, plan.pop("_candidate_resolution", [])


def _parse_numbered(message, numbered, kwargs) -> tuple[str, dict[str, Any]]:
    """Resolve candidate numbers before schema validation, and keep the evidence beside the plan."""
    require_numbers = kwargs.get("execution_mode") == "free"
    kind, plan = parse_model_message(
        _with_resolved_arguments(message, numbered, require_numbers=require_numbers),
        **kwargs,
    )
    plan["_candidate_resolution"] = getattr(_with_resolved_arguments, "last_evidence", [])
    return kind, plan


def _with_resolved_arguments(message, numbered, require_numbers: bool = False):
    copied = dict(message)
    calls = []
    evidence: list[dict[str, Any]] = []
    for call in message.get("tool_calls") or []:
        item = dict(call)
        function = dict((call.get("function") or {}))
        raw = function.get("arguments")
        payload = raw
        if isinstance(raw, str):
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                payload = None
        if isinstance(payload, dict):
            resolved, found = _resolve_payload(payload, numbered, require_numbers)
            evidence.extend(found)
            function["arguments"] = resolved if not isinstance(raw, str) else json.dumps(resolved, ensure_ascii=False)
        item["function"] = function
        calls.append(item)
    if calls:
        copied["tool_calls"] = calls
    elif isinstance(message.get("content"), str):
        try:
            payload = json.loads(message["content"])
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            resolved, found = _resolve_payload(payload, numbered, require_numbers)
            evidence.extend(found)
            copied["content"] = json.dumps(resolved, ensure_ascii=False)
    _with_resolved_arguments.last_evidence = evidence
    return copied


def _resolve_payload(payload, numbered, require_numbers: bool):
    from candidates import resolve_numbers, resolve_plan_bindings
    from free_task import candidate_number_errors

    if require_numbers:
        errors = candidate_number_errors(payload)
        if errors:
            raise PlanError(errors)
        return resolve_plan_bindings(payload, _candidate_graph(numbered))
    rows = numbered if isinstance(numbered, list) else list((numbered or {}).get("nodes") or [])
    return resolve_numbers(payload, rows)


def _candidate_graph(numbered) -> dict[str, Any]:
    from candidates import candidate_set_version

    if isinstance(numbered, dict) and isinstance(numbered.get("nodes"), list):
        graph = dict(numbered)
        graph.setdefault(
            "candidate_set_version",
            candidate_set_version(graph.get("scene_epoch"), graph.get("observation_id"), graph["nodes"]),
        )
        return graph
    nodes = list(numbered or [])
    return {
        "nodes": nodes,
        "scene_epoch": None,
        "observation_id": None,
        "candidate_set_version": candidate_set_version(None, None, nodes),
    }


def _ok(
    source: str,
    plan: dict[str, Any],
    attempts: list[dict],
    preflight: dict,
    kind: str,
    resolution: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "status": "ok",
        "plan": plan,
        "source": source,
        "plan_source": "model",
        "response_kind": kind,
        "candidate_resolution": resolution or [],
        "attempts": attempts,
        "errors": [],
        "budgets": [item["preflight"] for item in attempts],
        "output_budget": OUTPUT_BUDGET,
        "repairs": 1 if source == "model_repair" else 0,
        "preflight": preflight,
        "stage": "ok",
        "plan_from": "format_repair" if source == "model_repair" else "initial",
    }


def _join(parts: list[tuple[str, str]]) -> str:
    return "\n".join(f"[{name}]\n{text}" for name, text in parts)


def _emit(options: dict[str, Any], phase: str, detail: str) -> None:
    callback = options.get("on_event")
    if callback is not None:
        callback(phase, detail)


def make_openai_client(
    base_url: str,
    model: str = "Qwen3.5-9B",
    timeout_s: float = 180.0,
    api_key: str = "EMPTY",
    input_mode: str = "scene_metadata",
):
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
            return _stream(
                url,
                body,
                timeout_s,
                request_id,
                options.get("on_event"),
                state,
                api_key=api_key,
                model=model,
                input_mode=options.get("input_mode") or input_mode,
            )
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            if exc.code == 400 and "tools" in body and _tools_rejected(detail):
                reduced = dict(body)
                reduced.pop("tools", None)
                reduced.pop("tool_choice", None)
                message = _stream(
                    url,
                    reduced,
                    timeout_s,
                    request_id,
                    options.get("on_event"),
                    state,
                    api_key=api_key,
                    model=model,
                    input_mode=options.get("input_mode") or input_mode,
                )
                message["tool_transport"] = "unsupported"
                return message
            raise RuntimeError(f"planner HTTP {exc.code}: {detail}") from exc

    client.cancel = cancel
    client.request_id = lambda: state.get("request_id")
    return client


def _stream(
    url,
    body,
    timeout_s,
    request_id,
    on_event,
    state,
    *,
    api_key: str = "EMPTY",
    model: str = "",
    input_mode: str = "scene_metadata",
) -> dict[str, Any]:
    import urllib.request

    from planner_profiles import build_telemetry

    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key or 'EMPTY'}",
            "X-Request-Id": request_id,
        },
    )
    sent_at = time.perf_counter()
    if on_event:
        on_event("queued", request_id)
    response = urllib.request.urlopen(request, timeout=timeout_s)
    response_at = time.perf_counter()
    state["response"] = response
    if on_event:
        on_event("processing_input", request_id)
    content: list[str] = []
    reasoning: list[str] = []
    tool_calls: dict[int, dict[str, Any]] = {}
    finish = None
    usage: dict[str, Any] = {}
    started = False
    first_token_at = None
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
                first_token_at = time.perf_counter()
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
    done_at = time.perf_counter()
    calls = [tool_calls[index] for index in sorted(tool_calls)]
    telemetry = build_telemetry(
        endpoint=url,
        model=model or str(body.get("model") or ""),
        input_mode=input_mode,
        usage=usage,
        sent_at=sent_at,
        response_at=response_at,
        first_token_at=first_token_at,
        done_at=done_at,
    )
    return {
        "content": "".join(content),
        "tool_calls": calls or None,
        "finish_reason": finish,
        "usage": usage,
        "reasoning": "".join(reasoning),
        "request_id": request_id,
        "incomplete": finish is None,
        "telemetry": telemetry,
    }


def _tools_rejected(detail: str) -> bool:
    text = detail.lower()
    return "tool" in text and any(word in text for word in ("unsupported", "unknown", "extra", "not allowed"))
