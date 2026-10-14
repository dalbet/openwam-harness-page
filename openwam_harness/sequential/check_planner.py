"""One short planner request against the already running local Qwen server."""

from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path

from budget import OUTPUT_BUDGET, budget_check, slim_prompt
from planner import load_human_plan, request_plan
from schema import TOOL_SCHEMA

PLAN = Path(__file__).resolve().parent / "plans" / "two_object.json"
OUT = Path(__file__).resolve().parent / "records" / "planner_check.json"
URL = os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1").rstrip("/")


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
    counter = _tokenizer()
    usage = []

    def client(messages, max_tokens):
        prompt = "\n".join(item["content"] for item in messages)
        parts = [("request", prompt)]
        report = budget_check(parts, counter=counter, output_budget=max_tokens)
        body = {
            "model": "Qwen3.5-9B",
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0,
        }
        request = urllib.request.Request(
            URL + "/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
        )
        with urllib.request.urlopen(request, timeout=180) as response:
            payload = json.loads(response.read().decode("utf-8"))
        usage.append({"preflight": report, "api_usage": payload.get("usage")})
        return payload["choices"][0]["message"].get("content") or ""

    # Budget the slim prompt before the client is allowed to send it.
    preflight = budget_check(
        slim_prompt(
            task=human["task"],
            failure_evidence="saved episodes hit planner timeout or a 32768 context error; pi0_pick was not called",
            tool_schema=TOOL_SCHEMA,
        ),
        counter=counter,
    )
    result = request_plan(
        task=human["task"],
        failure_evidence="saved episodes hit planner timeout or a 32768 context error; pi0_pick was not called",
        human_plan=human,
        client=client,
        counter=counter,
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "source": result["source"],
                "output_budget": OUTPUT_BUDGET,
                "preflight": preflight,
                "requests": usage,
                "steps": len(result["plan"]["steps"]),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(OUT, result["source"], len(usage))


if __name__ == "__main__":
    main()
