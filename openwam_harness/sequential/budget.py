"""Pre-request token budget for the sequential planner.

The estimator is intentionally conservative when no tokenizer is supplied:
two characters count as at least one token. An over-budget prompt fails
locally and is not sent.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT / "budget_config.json").read_text(encoding="utf-8"))
OUTPUT_BUDGET = int(CONFIG["planner_output_budget"])
CONTEXT_LIMIT = int(CONFIG["context_limit"])

SKILL_CONSTRAINTS = (
    "Pi0.5 only grasps through pi0_pick. Placement uses move_to and set_gripper. "
    "Do not emit executable code. Do not call pi0_doubled. "
    "pi0_doubled success is official termination only and is not a subtask result. "
    "Do not reset the scene between subtasks. "
    "A new observation after an action is not a request for another planner call. "
    "Use only registered skills and checker names. Object refs must be semantic scene names."
)


class BudgetError(RuntimeError):
    """Raised when input tokens plus the output budget exceed the server context."""


def estimate_tokens(text: str, counter: Callable[[str], int] | None = None) -> int:
    if counter is not None:
        return max(0, int(counter(text)))
    return max(1, (len(text) + 1) // 2) if text else 0


def budget_check(
    parts: list[tuple[str, str]],
    *,
    counter: Callable[[str], int] | None = None,
    output_budget: int = OUTPUT_BUDGET,
    context_limit: int = CONTEXT_LIMIT,
) -> dict:
    counted = []
    total = 0
    for name, text in parts:
        tokens = estimate_tokens(text, counter)
        counted.append({"name": name, "tokens": tokens, "chars": len(text)})
        total += tokens
    need = total + output_budget
    report = {
        "input_tokens": total,
        "output_budget": output_budget,
        "context_limit": context_limit,
        "projected": need,
        "parts": counted,
    }
    if need > context_limit:
        raise BudgetError(
            f"local budget exceeded: input {total} + output {output_budget} "
            f"= {need} > context {context_limit}"
        )
    return report


def slim_prompt(
    *,
    task: str,
    scene_observation: str,
    tool_schema: str,
    short_memory: str,
    output_format: str,
    example: str,
) -> list[tuple[str, str]]:
    """Task, current scene summary, skill schema, short memory, and format. No trajectory dump."""
    if not task.strip():
        raise ValueError("task must be kept")
    if not scene_observation.strip():
        raise ValueError("scene observation must be kept")
    if not tool_schema.strip():
        raise ValueError("tool schema must be kept")
    return [
        ("task", task.strip()),
        ("skill_constraints", SKILL_CONSTRAINTS),
        ("scene_observation", scene_observation.strip()),
        ("tool_schema", tool_schema.strip()),
        ("short_memory", short_memory.strip() or "No short memory for this request."),
        ("output_format", output_format.strip()),
        ("example", example.strip()),
    ]
