"""One short planner request against the already running local Qwen server.

This does not load LIBERO and does not execute a plan.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from budget import OUTPUT_BUDGET
from planner import load_human_plan, make_openai_client, request_plan

PLAN = Path(__file__).resolve().parent / "plans" / "two_object.json"
OUT = Path(__file__).resolve().parent / "records" / "planner_check.json"
URL = os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")


def _tokenizer():
    checkpoint = os.environ.get(
        "PLANNER_MODEL_PATH",
        "/root/Page_dmh/openwam_harness/checkpoints/Qwen3.5-9B",
    )
    try:
        from transformers import AutoTokenizer
    except ImportError:
        return None
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    return lambda text: len(tokenizer.encode(text))


def main() -> None:
    human = load_human_plan(PLAN, "libero_10_t0")
    scene = json.dumps(
        {
            "goal_relation": human["goal_relation"],
            "objects": human["objects"],
            "sim_object_names": ["alphabet_soup_1", "tomato_sauce_1", "basket_1"],
            "image_ids": ["agentview:none", "wrist:none"],
            "note": "schema check only; image bytes are not inlined",
        },
        ensure_ascii=False,
    )
    result = request_plan(
        task=human["goal"],
        scene_observation=scene,
        client=make_openai_client(URL),
        counter=_tokenizer(),
        image_ids=["agentview:none", "wrist:none"],
    )
    record = {
        "status": result["status"],
        "source": result["source"],
        "plan_source": result.get("plan_source"),
        "output_budget": OUTPUT_BUDGET,
        "repairs": result["repairs"],
        "errors": result["errors"],
        "requests": [
            {
                "role": item["role"],
                "response_kind": item["response_kind"],
                "finish_reason": item["finish_reason"],
                "thinking_disabled_sent": item["thinking_disabled_sent"],
                "reasoning_present": item["reasoning_present"],
                "ignored_content": item["ignored_content"],
                "content": item["content"],
                "tool_calls": item["tool_calls"],
                "usage": item["usage"],
                "preflight": item["preflight"],
            }
            for item in result["attempts"]
        ],
        "steps": None if result["plan"] is None else len(result["plan"]["steps"]),
        "note": (
            "The previous planner_check.json stored only token usage. "
            "That client kept message.content and dropped tool_calls and finish_reason, "
            "so the earlier two failures cannot be reconstructed beyond "
            "prompt 327/completion 266 and prompt 627/completion 338."
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(OUT, result["status"], result["source"], len(result["attempts"]))


if __name__ == "__main__":
    main()
