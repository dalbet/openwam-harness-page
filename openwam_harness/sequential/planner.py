"""One initial planner request plus at most one format repair."""

from __future__ import annotations

import json
from typing import Any, Callable

from budget import OUTPUT_BUDGET, budget_check, slim_prompt
from schema import TOOL_SCHEMA, PlanError, parse_plan_text, validate_plan


def load_human_plan(path, scene_id: str) -> dict[str, Any]:
    catalog = json.loads(path.read_text(encoding="utf-8"))
    plan = catalog[scene_id]
    return validate_plan(plan)


def request_plan(
    *,
    task: str,
    failure_evidence: str,
    human_plan: dict[str, Any],
    client: Callable[[list[dict[str, str]], int], str],
    counter=None,
) -> dict[str, Any]:
    parts = slim_prompt(task=task, failure_evidence=failure_evidence, tool_schema=TOOL_SCHEMA)
    instruction = (
        "Return one JSON object with a steps array. Use only the allowed actions. "
        "No markdown and no code."
    )
    messages = [{"role": "user", "content": _join(parts) + "\n" + instruction}]
    reports = [budget_check([*parts, ("instruction", instruction)], counter=counter)]
    raw = client(messages, OUTPUT_BUDGET)
    try:
        plan = parse_plan_text(raw)
        source = "model"
    except PlanError as exc:
        repair = (
            f"Format repair 1/1. Previous output failed: {exc}. "
            "Return only the JSON plan."
        )
        messages = [
            *messages,
            {"role": "assistant", "content": raw[:2000]},
            {"role": "user", "content": repair},
        ]
        reports.append(
            budget_check([*parts, ("instruction", instruction), ("repair", repair)], counter=counter)
        )
        raw = client(messages, OUTPUT_BUDGET)
        try:
            plan = parse_plan_text(raw)
            source = "model_repair"
        except PlanError:
            plan = validate_plan(human_plan)
            source = "human_fallback"
    return {"plan": plan, "source": source, "budgets": reports, "output_budget": OUTPUT_BUDGET}


def _join(parts: list[tuple[str, str]]) -> str:
    return "\n".join(f"[{name}]\n{text}" for name, text in parts)
