#!/usr/bin/env python3
"""Run the frozen bilingual flow-routing cases against the live router.

The expectation file (``config/flow_routing_cases.json``) was written before
the first run and is never edited to match an answer. This script only:

* reads one live snapshot (read-only ``env.state_snapshot``) so scene support
  is real, not assumed;
* asks the router for each instruction through the production code path
  (``semantic_router.decide`` + ``assemble_plan``);
* reports matched/missed per case, target omissions, latency and repairs;
* writes the raw rows next to the round records.

Usage:
    EMBODIED_CONFIG_DIR=/home/host/flashrt-exp/pi05_round/config_demo_b900 \
    envs/rpent/bin/python scripts/run_flow_cases.py --tag flows_20261105
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from agent import semantic_router  # noqa: E402
from common import config, task_catalog  # noqa: E402

RAW = Path("/home/host/flashrt-exp/pi05_round/raw/composite")
FLOW_KEYS = ["put_cream_cheese_in_bowl", "put_bowl_on_stove", "turn_on_stove"]


def _snapshot(url: str) -> dict:
    sys.path.insert(0, str(PROJECT))
    from adapters.env_client import EnvClient

    return EnvClient(url).state_snapshot()


def _actual(decision: dict, assembled: dict) -> dict:
    status = decision.get("status")
    keys = list(decision.get("composite_keys") or [])
    flow = (decision.get("flow") or {}).get("flow_id")
    task = decision.get("task") or {}
    if status != "ok":
        kind = "clarify" if status == "needs_clarification" else status
        return {"kind": kind, "keys": [], "flow_id": flow, "status": status}
    if keys:
        return {
            "kind": "flow" if flow else "composite",
            "keys": keys,
            "flow_id": flow,
            "status": status,
        }
    return {"kind": "single", "key": task.get("key"), "keys": [], "status": status}


def _matches(expected: dict, actual: dict) -> tuple[bool, str]:
    kind = expected.get("kind")
    if kind == "unsupported":
        return actual["status"] == "unsupported", actual["status"]
    if kind == "clarify":
        return actual["status"] == "needs_clarification", actual["status"]
    if kind == "single":
        ok = actual["status"] == "ok" and actual.get("kind") == "single" and actual.get("key") == expected.get("key")
        return ok, f"{actual.get('kind')}:{actual.get('key')}"
    if kind in {"flow", "composite"}:
        ok = (
            actual["status"] == "ok"
            and actual["kind"] == kind
            and actual.get("keys") == list(expected.get("keys") or [])
        )
        if kind == "flow":
            ok = ok and actual.get("flow_id") == expected.get("flow_id")
        return ok, f"{actual.get('kind')}:{actual.get('keys')}"
    return False, f"unknown expectation kind {kind}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="flows_20261105")
    parser.add_argument("--env", default="http://127.0.0.1:14591")
    parser.add_argument("--task-id", type=int, default=1)
    parser.add_argument("--cases", default=str(config.CONFIG_DIR / "flow_routing_cases.json"))
    args = parser.parse_args()

    cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    snapshot = _snapshot(args.env)
    loaded = task_catalog.task_for(args.task_id)
    started = time.time()
    rows = []
    for case in cases["cases"]:
        instruction = case["instruction"]
        decision = semantic_router.decide(instruction, snapshot)
        assembled = semantic_router.assemble_plan(
            decision,
            loaded_task_id=int(args.task_id),
            instruction=instruction,
            snapshot=snapshot,
            skills={"skills": [{"name": "whole_task"}]},
            loaded_task=loaded,
        )
        actual = _actual(decision, assembled)
        ok, observed = _matches(case["expect"], actual)
        omissions = []
        if case["expect"].get("kind") == "flow" and actual.get("keys") != FLOW_KEYS:
            missing = [key for key in FLOW_KEYS if key not in actual.get("keys", [])]
            if missing:
                omissions = missing
        rows.append(
            {
                "id": case["id"],
                "lang": case["lang"],
                "category": case["category"],
                "instruction": instruction,
                "expected": case["expect"],
                "actual": actual,
                "observed": observed,
                "matched": bool(ok),
                "goal_omissions": omissions,
                "status": decision.get("status"),
                "reason_code": decision.get("reason_code"),
                "reason": decision.get("reason"),
                "repairs": decision.get("repairs"),
                "router_s": decision.get("total_s"),
                "guard": decision.get("flow_guard"),
                "plan_ok": bool(assembled.get("status") == "ok"),
                "plan_reason_code": assembled.get("reason_code"),
                "plan_source": (assembled.get("plan") or {}).get("plan_source"),
            }
        )

    matched = sum(1 for row in rows if row["matched"])
    to_unsupported = sum(
        1 for row in rows if row["expected"]["kind"] in {"flow", "composite"} and not row["actual"].get("keys")
    )
    payload = {
        "schema": "embodied_engine.flow_case_run.v1",
        "tag": args.tag,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(started)),
        "cases_file": str(args.cases),
        "cases_version": cases.get("version"),
        "frozen_on": cases.get("frozen_on"),
        "catalog_version": task_catalog.version(),
        "catalog_sha256": task_catalog.fingerprint(),
        "flows_version": task_catalog.flows_version(),
        "task_id": int(args.task_id),
        "router": dict(semantic_router.router_options()),
        "summary": {
            "cases": len(rows),
            "matched": matched,
            "missed": len(rows) - matched,
            "goal_omissions": sum(1 for row in rows if row["goal_omissions"]),
            "flow_or_composite_downgraded": to_unsupported,
            "repairs": sum(int(row["repairs"] or 0) for row in rows),
            "router_s_total": round(sum(float(row["router_s"] or 0) for row in rows), 3),
            "guard_calls": sum(1 for row in rows if (row["guard"] or {}).get("asked")),
            "guard_flow_upgrades": sum(
                1 for row in rows if (row["guard"] or {}).get("scope") == "flow"
            ),
        },
        "rows": rows,
    }
    out_dir = RAW / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"flow_cases_{time.strftime('%Y%m%d-%H%M%S', time.localtime(started))}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"{'id':34} {'exp':10} {'got':46} ok")
    for row in rows:
        exp = row["expected"]["kind"] + (f":{row['expected'].get('flow_id')}" if row["expected"].get("kind") == "flow" else "")
        print(f"{row['id'][:34]:34} {exp[:10]:10} {row['observed'][:46]:46} {'Y' if row['matched'] else 'N'}")
    print(json.dumps(payload["summary"], ensure_ascii=False))
    print(f"written: {path}")
    return 0 if matched == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
