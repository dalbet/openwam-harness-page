#!/usr/bin/env python
"""Route-only evaluation of natural-language composite requests.

Sends the *raw* instruction to the Agent ``/plan`` endpoint against a real
scene snapshot of the anchor task (task 8), and records what the model chose,
whether the ordered pair is executable, and how long routing took.

No simulator step is executed here: this isolates language mapping from
execution. The snapshot comes from the live env server (``env.state_snapshot``)
so scene/entity support is checked against the really loaded scene rather than
a hand-written object list.

Every case declares its expectation in the manifest:
    {"id": str, "split": "dev"|"test", "lang": "en"|"zh", "text": str,
     "expect_keys": [...]|null, "expect_status": "ok"|"clarify"|"unsupported",
     "expect_reason": str|null, "expect_executable": bool|null, "note": str}

usage: evaluate_composite_routes.py --manifest config/composite_routes_20261027.json \
           --split dev --out records/offline/composite_routes_dev.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from adapters.env_client import EnvClient  # noqa: E402
from common import config  # noqa: E402

OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
TERMINAL_OK = {"ok", "needs_clarification", "unsupported", "scene_not_supported", "task_mismatch"}


def post(url: str, body: dict[str, Any], timeout: float = 60.0) -> tuple[dict[str, Any], float]:
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    started = time.perf_counter()
    try:
        with OPENER.open(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        try:
            payload = json.loads(detail)
        except json.JSONDecodeError:
            payload = {"status": "error", "error": f"HTTP {exc.code}: {detail[:800]}"}
    return payload, time.perf_counter() - started


def _score(case: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    understanding = result.get("understanding") or {}
    keys = understanding.get("composite_keys") or (
        [understanding.get("task_key")] if understanding.get("task_key") else []
    )
    status = str(result.get("status") or "")
    expected_status = case.get("expect_status")
    expected_keys = case.get("expect_keys")
    expected_reason = case.get("expect_reason")
    expected_exec = case.get("expect_executable")

    checks: list[dict[str, Any]] = []

    def check(name: str, ok: bool, want: Any, got: Any) -> None:
        checks.append({"check": name, "ok": bool(ok), "expect": want, "got": got})

    check("status", status == expected_status, expected_status, status)
    if expected_keys is not None:
        check("keys", list(keys) == list(expected_keys), expected_keys, keys)
    if expected_reason is not None:
        reason = understanding.get("composite_reason_code") or understanding.get("reason_code")
        check("reason", reason == expected_reason, expected_reason, reason)
    if expected_exec is not None:
        executable = understanding.get("composite_executable")
        check("executable", executable == expected_exec, expected_exec, executable)
    check("no_errors", not (result.get("errors") or []), [], result.get("errors") or [])
    attempts = result.get("attempts") or []
    timings = [row.get("timing") or {} for row in attempts]
    last = timings[-1] if timings else {}
    return {
        "checks": checks,
        "passed": all(row["ok"] for row in checks),
        "status": status,
        "keys": keys,
        "reason_code": understanding.get("composite_reason_code"),
        "executable": understanding.get("composite_executable"),
        "plan_source": understanding.get("plan_source") or result.get("plan_source"),
        "attempts": len(attempts),
        "repairs": result.get("repairs") or 0,
        "ttft_s": last.get("ttft_s"),
        "prompt_tokens": last.get("prompt_tokens"),
        "completion_tokens": last.get("completion_tokens"),
        "decode_s": last.get("decode_s"),
        "inference_s": last.get("inference_s"),
        "understanding_s": understanding.get("understanding_s"),
        "assembly_s": understanding.get("assembly_s"),
        "agent_total_s": result.get("total_s"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="config/composite_routes_20261027.json")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--out", default="")
    parser.add_argument("--agent", default="http://127.0.0.1:14700")
    parser.add_argument("--env", default="http://127.0.0.1:14561")
    parser.add_argument("--task-id", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    manifest = json.loads((PROJECT / args.manifest).read_text(encoding="utf-8"))
    cases = [case for case in manifest["cases"] if case.get("split") == args.split]
    if args.limit:
        cases = cases[: args.limit]
    if not cases:
        print(f"no cases for split {args.split}")
        return 2

    config.ensure_dirs()
    snapshot = EnvClient(args.env).state_snapshot()
    objects = sorted((snapshot.get("sim") or {}).get("objects") or [])
    print(f"split={args.split} cases={len(cases)} scene_objects={len(objects)}")

    rows: list[dict[str, Any]] = []
    for case in cases:
        result, wall_s = post(
            f"{args.agent}/plan",
            {"instruction": case["text"], "task_id": args.task_id, "snapshot": snapshot},
        )
        scored = _score(case, result)
        row = {
            "id": case["id"],
            "lang": case.get("lang"),
            "text": case["text"],
            "note": case.get("note"),
            "wall_s": round(wall_s, 4),
            **scored,
        }
        rows.append(row)
        flag = "PASS" if row["passed"] else "FAIL"
        print(
            f"[{flag}] {row['id']:<28} {wall_s:5.2f}s status={row['status']:<18} "
            f"keys={row['keys']} reason={row['reason_code']} exec={row['executable']}"
        )
        for check in row["checks"]:
            if not check["ok"]:
                print(f"        ! {check['check']}: expect={check['expect']!r} got={check['got']!r}")

    def summarize(subset: list[dict[str, Any]]) -> dict[str, Any]:
        if not subset:
            return {}
        walls = sorted(row["wall_s"] for row in subset)
        return {
            "n": len(subset),
            "passed": sum(1 for row in subset if row["passed"]),
            "median_s": round(statistics.median(walls), 4),
            "p95_s": round(walls[min(len(walls) - 1, int(round(0.95 * (len(walls) - 1))))], 4),
            "max_s": round(walls[-1], 4),
            "ttft_median_s": round(
                statistics.median([row["ttft_s"] for row in subset if row.get("ttft_s") is not None]), 4
            )
            if any(row.get("ttft_s") is not None for row in subset)
            else None,
            "completion_tokens_median": statistics.median(
                [row["completion_tokens"] for row in subset if row.get("completion_tokens") is not None]
            )
            if any(row.get("completion_tokens") is not None for row in subset)
            else None,
            "attempts_total": sum(row["attempts"] for row in subset),
            "repairs_total": sum(row["repairs"] for row in subset),
        }

    composite_rows = [row for row in rows if row["keys"] and len(row["keys"]) == 2]
    report = {
        "schema": "embodied_engine.composite_routes.v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "manifest": args.manifest,
        "split": args.split,
        "task_id": args.task_id,
        "agent": args.agent,
        "scene_objects": objects,
        "summary": summarize(rows),
        "composite_summary": summarize(composite_rows),
        "by_lang": {
            lang: summarize([row for row in rows if row["lang"] == lang]) for lang in ("en", "zh")
        },
        "rows": rows,
    }
    out = Path(args.out) if args.out else PROJECT / "records" / "offline" / f"composite_routes_{args.split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summary": report["summary"], "by_lang": report["by_lang"], "out": str(out)}, ensure_ascii=False))
    return 0 if report["summary"]["passed"] == report["summary"]["n"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
