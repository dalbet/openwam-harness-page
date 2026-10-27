#!/usr/bin/env python
"""Offline evaluation of the compact Qwen semantic router.

The router is exercised through the live Agent HTTP endpoint (``/plan``), so
the measured latency includes request parsing, the vLLM call, JSON parsing,
catalog validation, live-scene checking and whole_task assembly. The legacy
full Qwen planner is called directly on the same positive cases for an A/B
latency comparison only; it is not on the production path.

Nothing here writes or reads an answer cache. vLLM prefix-cache counters are
reported as a delta so a cache hit cannot be mistaken for model inference.

Usage (RPent python, with Qwen on :8001 and the engine on :14800):
    python scripts/eval_offline.py --set config/semantic_eval_20261027.json \
        --out records/offline/semantic_eval_20261027.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from adapters.env_client import EnvClient  # noqa: E402
from agent import planner  # noqa: E402
from common import config, task_catalog  # noqa: E402

OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
OFFICIAL_BY_KEY = {
    row["key"]: row["official_instruction"] for row in task_catalog.tasks()
}


def _request(url: str, payload: dict[str, Any] | None, timeout_s: float) -> tuple[int, dict[str, Any]]:
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with OPENER.open(request, timeout=timeout_s) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        try:
            return exc.code, json.loads(detail)
        except json.JSONDecodeError:
            return exc.code, {"status": "error", "error": detail[:1000]}


def _get(url: str, timeout_s: float = 10.0) -> dict[str, Any]:
    with OPENER.open(request_url(url), timeout=timeout_s) as response:
        return json.loads(response.read().decode("utf-8"))


def request_url(url: str) -> urllib.request.Request:
    return urllib.request.Request(url, method="GET")


def _norm(text: Any) -> str:
    return " ".join(str(text or "").strip().lower().split())


def _official_choice(prompt: Any) -> str | None:
    wanted = _norm(prompt)
    for key, official in OFFICIAL_BY_KEY.items():
        if wanted == _norm(official):
            return key
    return None


def _latency(values: list[Any]) -> dict[str, Any]:
    clean = sorted(float(v) for v in values if isinstance(v, (int, float)) and v >= 0)
    if not clean:
        return {"n": 0}
    index = min(len(clean) - 1, max(0, int(round(0.95 * (len(clean) - 1)))))
    return {
        "n": len(clean),
        "median_s": round(statistics.median(clean), 4),
        "mean_s": round(statistics.fmean(clean), 4),
        "p95_s": round(clean[index], 4),
        "max_s": round(clean[-1], 4),
    }


def _rate(num: int, den: int) -> float | None:
    return round(num / den, 4) if den else None


def _token_stats(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    return _latency([row.get(field) for row in rows])


def _vllm_metric(name: str) -> float:
    try:
        with OPENER.open(request_url(f"{config.service_url('planner_llm').rstrip('/')}/metrics"), timeout=5) as response:
            text = response.read().decode("utf-8", "replace")
    except Exception:
        return 0.0
    total = 0.0
    for line in text.splitlines():
        if line.startswith(f"vllm:{name}") and "{" in line:
            try:
                total += float(line.rsplit(" ", 1)[1])
            except (IndexError, ValueError):
                continue
    return total


def _prefix_cache() -> dict[str, float]:
    return {
        "queries_total": round(_vllm_metric("prefix_cache_queries_total"), 3),
        "hits_total": round(_vllm_metric("prefix_cache_hits_total"), 3),
    }


def _attempt_tokens(attempts: list[dict[str, Any]]) -> dict[str, Any]:
    prompt = 0
    completion = 0
    ttft = None
    decode = None
    queue = None
    parse_s = 0.0
    validate_s = 0.0
    requests = None
    for row in attempts:
        timing = row.get("timing") or {}
        prompt += int(timing.get("prompt_tokens") or 0)
        completion += int(timing.get("completion_tokens") or 0)
        if timing.get("ttft_s") is not None:
            ttft = timing.get("ttft_s")
        if timing.get("decode_s") is not None:
            decode = timing.get("decode_s")
        if timing.get("vllm_queue_s") is not None:
            queue = timing.get("vllm_queue_s")
        parse_s += float(timing.get("parse_s") or 0.0)
        validate_s += float(timing.get("validate_s") or 0.0)
        if timing.get("requests_observed") is not None:
            requests = timing.get("requests_observed")
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "ttft_s": ttft,
        "decode_s": decode,
        "queue_s": queue,
        "parse_s": round(parse_s, 6),
        "validate_s": round(validate_s, 6),
        "requests_observed": requests,
    }


def _router_row(
    case: dict[str, Any],
    status: int,
    response: dict[str, Any],
    wall_s: float,
) -> dict[str, Any]:
    understanding = response.get("understanding") or {}
    attempts = response.get("attempts") or []
    token_row = _attempt_tokens(attempts)
    steps = ((response.get("plan") or {}).get("steps") or [])
    plan_prompt = (steps[0] or {}).get("prompt") if steps else None
    expected_choice = case["expected_choice"]
    choice = understanding.get("semantic_choice")
    return {
        "id": case["id"],
        "task_id": case["task_id"],
        "kind": case["kind"],
        "source": case["source"],
        "instruction": case["instruction"],
        "expected_choice": expected_choice,
        "expected_status": case["expected_status"],
        "http_status": status,
        "status": response.get("status"),
        "reason_code": response.get("reason_code"),
        "reason": response.get("reason"),
        "choice": choice,
        "choice_correct": choice == expected_choice,
        "status_correct": response.get("status") == case["expected_status"],
        "official_instruction": understanding.get("official_instruction"),
        "plan_prompt": plan_prompt,
        "prompt_equals_official": bool(plan_prompt)
        and _norm(plan_prompt) == _norm(understanding.get("official_instruction")),
        "scene_support": understanding.get("scene_support"),
        "needs_scene_switch": understanding.get("needs_scene_switch"),
        "assembly_s": understanding.get("assembly_s"),
        "router_s": understanding.get("router_latency_s"),
        "total_s": response.get("total_s"),
        "wall_s": round(wall_s, 4),
        "repairs": response.get("repairs"),
        "first_pass": (response.get("repairs") or 0) == 0,
        "format_ok": not (response.get("errors") or []),
        "errors": response.get("errors") or [],
        **token_row,
        "plan": response.get("plan"),
    }


def _legacy_row(case: dict[str, Any], result: dict[str, Any], wall_s: float, snapshot: dict[str, Any], task: dict[str, Any]) -> dict[str, Any]:
    plan = result.get("plan") or {}
    steps = plan.get("steps") or []
    prompt = (steps[0] or {}).get("prompt") if steps else None
    attempts = result.get("attempts") or []
    token_row = _attempt_tokens(attempts)
    recognized = _official_choice(prompt)
    return {
        "id": case["id"],
        "task_id": case["task_id"],
        "status": result.get("status"),
        "plan_prompt": prompt,
        "recognized_task_key": recognized,
        "matches_expected": recognized == case["expected_choice"],
        "total_s": result.get("total_s"),
        "wall_s": round(wall_s, 4),
        "repairs": result.get("repairs"),
        **token_row,
    }


def _summary(rows: list[dict[str, Any]], cases: list[dict[str, Any]], legacy: list[dict[str, Any]], cache: dict[str, float]) -> dict[str, Any]:
    positives = [row for row in rows if row["kind"] in {"positive", "scene_mismatch"}]
    pure_positives = [row for row in rows if row["kind"] == "positive"]
    negatives = [row for row in rows if row["kind"] == "negative"]
    scene_mismatch = [row for row in rows if row["kind"] == "scene_mismatch"]
    pos_ok = sum(1 for row in pure_positives if row["status"] == "ok")
    pos_choice_correct = sum(1 for row in pure_positives if row["choice_correct"])
    pos_prompt_correct = sum(
        1 for row in pure_positives
        if row["status"] == "ok" and row["prompt_equals_official"]
    )
    false_reject = len(pure_positives) - pos_ok
    negative_correct = sum(1 for row in negatives if row["status_correct"])
    negative_choice_correct = sum(1 for row in negatives if row["choice_correct"])
    false_accept = sum(1 for row in negatives if row["status"] == "ok")
    format_ok = sum(1 for row in rows if row["format_ok"] and row["first_pass"])
    warm = rows[5:]
    return {
        "cases": len(rows),
        "positives": len(pure_positives),
        "negatives": len(negatives),
        "scene_mismatch": len(scene_mismatch),
        "positive_ok": pos_ok,
        "positive_rate": _rate(pos_ok, len(pure_positives)),
        "positive_choice_correct": pos_choice_correct,
        "positive_choice_rate": _rate(pos_choice_correct, len(pure_positives)),
        "positive_prompt_official": pos_prompt_correct,
        "positive_prompt_official_rate": _rate(pos_prompt_correct, len(pure_positives)),
        "wrong_mapping": len(pure_positives) - pos_choice_correct,
        "wrong_mapping_rate": _rate(len(pure_positives) - pos_choice_correct, len(pure_positives)),
        "false_reject": false_reject,
        "false_reject_rate": _rate(false_reject, len(pure_positives)),
        "negative_correct": negative_correct,
        "negative_rate": _rate(negative_correct, len(negatives)),
        "negative_choice_correct": negative_choice_correct,
        "negative_choice_rate": _rate(negative_choice_correct, len(negatives)),
        "false_accept": false_accept,
        "scene_mismatch_correct": sum(
            1 for row in scene_mismatch if row["status"] == "task_mismatch" and row["choice_correct"]
        ),
        "scene_mismatch_rate": _rate(
            sum(1 for row in scene_mismatch if row["status"] == "task_mismatch" and row["choice_correct"]),
            len(scene_mismatch),
        ),
        "format_pass_rate": _rate(format_ok, len(rows)),
        "repairs_total": sum(int(row.get("repairs") or 0) for row in rows),
        "requests_observed_mismatch": sum(
            1 for row in rows if row.get("requests_observed") not in (None, 1)
        ),
        "latency_all": _latency([row["total_s"] for row in rows]),
        "latency_warm": _latency([row["total_s"] for row in warm]),
        "wall_all": _latency([row["wall_s"] for row in rows]),
        "wall_warm": _latency([row["wall_s"] for row in warm]),
        "ttft": _latency([row.get("ttft_s") for row in rows]),
        "decode": _latency([row.get("decode_s") for row in rows]),
        "queue": _latency([row.get("queue_s") for row in rows]),
        "prompt_tokens": _latency([row.get("prompt_tokens") for row in rows]),
        "completion_tokens": _latency([row.get("completion_tokens") for row in rows]),
        "parse_s": _latency([row.get("parse_s") for row in rows]),
        "validate_s": _latency([row.get("validate_s") for row in rows]),
        "assembly_s": _latency([row.get("assembly_s") for row in rows]),
        "prefix_cache": {
            **cache,
            "answer_cache": False,
            "note": "No answer cache. Prefix-cache counters are a vLLM delta over this run.",
        },
        "legacy_full": {
            "cases": len(legacy),
            "matches_expected": sum(1 for row in legacy if row.get("matches_expected")),
            "match_rate": _rate(sum(1 for row in legacy if row.get("matches_expected")), len(legacy)),
            "latency": _latency([row.get("total_s") for row in legacy]),
            "ttft": _latency([row.get("ttft_s") for row in legacy]),
            "decode": _latency([row.get("decode_s") for row in legacy]),
            "prompt_tokens": _latency([row.get("prompt_tokens") for row in legacy]),
            "completion_tokens": _latency([row.get("completion_tokens") for row in legacy]),
        },
        "by_source": {
            source: {
                "cases": len([row for row in rows if row["source"] == source]),
                "choice_correct": sum(
                    1 for row in rows if row["source"] == source and row["choice_correct"]
                ),
                "status_correct": sum(
                    1 for row in rows if row["source"] == source and row["status_correct"]
                ),
                "latency_warm": _latency(
                    [
                        row["total_s"]
                        for row in rows[5:]
                        if row["source"] == source
                    ]
                ),
            }
            for source in sorted({row["source"] for row in rows})
        },
        "by_task": {
            str(task_id): {
                "cases": len([row for row in rows if row["task_id"] == task_id]),
                "choice_correct": sum(
                    1 for row in rows if row["task_id"] == task_id and row["choice_correct"]
                ),
                "status_correct": sum(
                    1 for row in rows if row["task_id"] == task_id and row["status_correct"]
                ),
                "latency": _latency(
                    [row["total_s"] for row in rows if row["task_id"] == task_id]
                ),
            }
            for task_id in sorted({row["task_id"] for row in rows})
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", default="config/semantic_eval_20261027.json")
    parser.add_argument("--out", default="records/offline/semantic_eval_20261027.json")
    parser.add_argument("--tasks", default="", help="comma-separated task ids (default: all in the set)")
    parser.add_argument("--legacy", action="store_true", help="also run the legacy full planner on positives")
    parser.add_argument("--legacy-limit", type=int, default=0, help="0 means every positive")
    args = parser.parse_args()

    case_set = json.loads((PROJECT / args.set).read_text(encoding="utf-8"))
    cases = list(case_set["cases"])
    if args.tasks:
        wanted = {int(token) for token in args.tasks.split(",") if token.strip()}
        cases = [case for case in cases if int(case["task_id"]) in wanted]
    task_ids = sorted({int(case["task_id"]) for case in cases})
    cfg = config.engine_config()
    skills = config.skill_config()
    skills = {
        **skills,
        "skills": [row for row in (skills.get("skills") or []) if row.get("name") == "whole_task"],
    }
    env_client = EnvClient(config.service_url("env"))
    agent_url = f"{config.service_url('agent')}/plan"
    cache_before = _prefix_cache()
    rows: list[dict[str, Any]] = []
    legacy_rows: list[dict[str, Any]] = []
    snapshots: dict[int, dict[str, Any]] = {}
    load_s: dict[int, float] = {}

    legacy_budget = args.legacy_limit or 10**9
    for task_id in task_ids:
        print(f"[scene] task {task_id}", flush=True)
        started = time.perf_counter()
        status, loaded = _request(
            f"{config.service_url('engine')}/api/load", {"task_id": task_id}, timeout_s=320
        )
        load_s[task_id] = round(time.perf_counter() - started, 3)
        if loaded.get("status") != "ok":
            raise RuntimeError(f"engine load task {task_id} failed: {status} {loaded}")
        snapshot = env_client.state_snapshot()
        snapshots[task_id] = snapshot
        task = config.task_for(task_id)
        for case in [row for row in cases if int(row["task_id"]) == task_id]:
            body = {
                "instruction": case["instruction"],
                "task_id": task_id,
                "snapshot": snapshot,
                "attempts": [],
                "backend": "openwam",
            }
            started = time.perf_counter()
            http_status, response = _request(agent_url, body, timeout_s=180)
            wall_s = time.perf_counter() - started
            row = _router_row(case, http_status, response, wall_s)
            rows.append(row)
            print(
                f"  [{case['id']}] status={row['status']} choice={row['choice']} "
                f"expected={case['expected_choice']} wall={wall_s:.3f}s",
                flush=True,
            )
            if args.legacy and case["kind"] == "positive" and len(legacy_rows) < legacy_budget:
                legacy_started = time.perf_counter()
                try:
                    legacy_result = planner.request_plan(
                        instruction=case["instruction"],
                        task=task,
                        snapshot=snapshot,
                        skills=skills,
                    )
                    legacy_rows.append(
                        _legacy_row(case, legacy_result, time.perf_counter() - legacy_started, snapshot, task)
                    )
                except Exception as exc:
                    legacy_rows.append(
                        {
                            "id": case["id"],
                            "task_id": task_id,
                            "status": "error",
                            "error": f"{type(exc).__name__}: {exc}",
                            "wall_s": round(time.perf_counter() - legacy_started, 4),
                        }
                    )

    cache_after = _prefix_cache()
    cache = {
        "queries_total": round(cache_after["queries_total"] - cache_before["queries_total"], 3),
        "hits_total": round(cache_after["hits_total"] - cache_before["hits_total"], 3),
    }
    summary = _summary(rows, cases, legacy_rows, cache)
    output = {
        "schema": "embodied_engine.semantic_eval_result.v2",
        "set": args.set,
        "set_frozen": bool(case_set.get("frozen")),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "suite": cfg["env"]["suite"],
        "seed": cfg["env"]["seed"],
        "scene_load_s": load_s,
        "models": {
            "router": {
                "model": cfg["agent"]["llm_model"],
                "base_url": cfg["agent"]["llm_base_url"],
                "mode": "non_thinking",
                "max_tokens": int(((cfg["agent"].get("semantic_router") or {}).get("max_tokens")) or 24),
                "temperature": cfg["agent"]["temperature"],
            },
            "legacy_full": {
                "model": cfg["agent"]["llm_model"],
                "max_tokens": cfg["agent"]["max_tokens"],
                "note": "Legacy full planner, called directly for latency comparison only",
            },
            "action_model": cfg["action_model"]["openwam"]["name"],
        },
        "catalog": task_catalog.catalog_evidence(),
        "summary": summary,
        "cases": rows,
        "legacy_full_cases": legacy_rows,
    }
    out = PROJECT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out), "summary": summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
