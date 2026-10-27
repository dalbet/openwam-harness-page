#!/usr/bin/env python
"""Twenty-round closed loop: raw instruction -> semantic route -> official prompt -> OpenWAM.

For every frozen round in ``config/language_rounds_<date>.json`` this runner:

1. loads the target scene (fresh ``/api/load``),
2. submits the *raw* user instruction to ``/api/run`` (the official instruction is
   never sent from here), so the Agent has to route the language,
3. waits for the episode to become terminal,
4. reads the episode record and reports the whole chain separately:
   user text -> model semantic choice -> catalog official instruction ->
   prompt the policy actually received -> execution outcome.

A round is never retried and never skipped. Language mapping, native task goal
and release evidence are reported as three independent columns; native success
while the gripper still holds the object is not counted as "released".
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parents[1]
TERMINAL = {"succeeded", "failed", "stopped"}
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

RELEASE_REQUIRED_KINDS = {"pick_place", "composite"}


def get(url: str, timeout: float = 15.0) -> dict[str, Any]:
    with OPENER.open(urllib.request.Request(url, method="GET"), timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def post(url: str, body: dict[str, Any], timeout: float = 320.0) -> dict[str, Any]:
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with OPENER.open(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:1000]
        try:
            return json.loads(detail)
        except json.JSONDecodeError:
            return {"status": "error", "error": f"HTTP {exc.code}: {detail}"}


def wait_terminal(engine: str, timeout_s: float) -> dict[str, Any]:
    started = time.time()
    last: dict[str, Any] = {}
    while time.time() - started < timeout_s:
        try:
            last = get(f"{engine}/api/status", timeout=10)
        except Exception as exc:  # noqa: BLE001 - engine may be restarting
            last = {"phase": "unreachable", "error": f"{type(exc).__name__}: {exc}"}
        if last.get("phase") in TERMINAL:
            return last
        time.sleep(1.0)
    return {**last, "phase": "timeout"}


def _norm(text: Any) -> str:
    return " ".join(str(text or "").strip().lower().split())


def _record(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _mapping_ok(
    *,
    status: str | None,
    chosen: str | None,
    expected: str | None,
    key: str | None,
    official: str | None,
    catalog_official: str | None,
    step_prompt: str | None,
    verification_prompt: str | None,
) -> bool:
    """Five-way agreement between the round, the route and the policy prompt.

    A manifest may name the expected route by catalog key
    (``put_bowl_on_stove``) or by the official instruction that key resolves to
    (``put the bowl on the stove``). Both express the same expectation; calling
    the phrase form a mismatch reports a routing error that never happened.
    """
    expected = str(expected or "")
    expected_match = chosen == expected or (
        _norm(expected) == _norm(official) and chosen == key
    )
    return bool(
        status == "ok"
        and expected_match
        and _norm(catalog_official) == _norm(official)
        and _norm(step_prompt) == _norm(official)
        and _norm(verification_prompt) == _norm(official)
    )


def _round_row(round_spec: dict[str, Any], task: dict[str, Any], status: dict[str, Any], wall_s: float) -> dict[str, Any]:
    result = status.get("last_result") or {}
    record = _record(result.get("record"))
    semantic = record.get("semantic") or {}
    steps = record.get("steps") or []
    step_prompt = (steps[0] or {}).get("prompt") if steps else None
    verification_prompt = ((steps[0] or {}).get("verification") or {}).get("prompt") if steps else None
    official = task["official_instruction"]
    chosen = semantic.get("semantic_choice")

    mapping_ok = _mapping_ok(
        status=semantic.get("decision_status"),
        chosen=chosen,
        expected=round_spec.get("expected_choice"),
        key=task.get("key"),
        official=official,
        catalog_official=semantic.get("official_instruction"),
        step_prompt=step_prompt,
        verification_prompt=verification_prompt,
    )

    release = record.get("release_evidence") or {}
    placement = record.get("placement") or {}
    requires_release = str(task.get("kind") or "") in RELEASE_REQUIRED_KINDS
    if requires_release:
        if placement:
            release_state = "verified" if placement.get("placement_verified") else "unknown"
        else:
            release_state = release.get("state") or "unknown"
    else:
        release_state = "not_applicable"

    native = bool(record.get("native_task_success"))
    user_goal = bool(record.get("user_goal_success"))
    if not requires_release:
        evidence_state = "native_only" if native else "absent"
    elif release_state == "verified":
        evidence_state = "released" if native else "released_without_native_success"
    else:
        evidence_state = "held_or_unknown" if native else "absent"

    return {
        "round": round_spec["round"],
        "task_id": round_spec["task_id"],
        "task_key": task["key"],
        "lang": round_spec["lang"],
        "user_instruction": round_spec["instruction"],
        "expected_choice": round_spec["expected_choice"],
        "semantic_choice": chosen,
        "decision_status": semantic.get("decision_status"),
        "catalog_official_instruction": semantic.get("official_instruction"),
        "task_official_instruction": official,
        "plan_step_prompt": step_prompt,
        "policy_verification_prompt": verification_prompt,
        "mapping_ok": mapping_ok,
        "mapping_reason_code": semantic.get("reason_code"),
        "router_latency_s": semantic.get("router_latency_s"),
        "planner_latency_s": record.get("plan", {}).get("latency_s") if record.get("plan") else None,
        "native_task_success": native,
        "native_goal_final": record.get("native_goal_final"),
        "user_goal_success": user_goal,
        "execution_status": record.get("execution_status") or result.get("execution_status"),
        "failure_code": record.get("failure_code") or result.get("failure_code"),
        "failure_reason": record.get("failure_reason") or result.get("reason"),
        "requires_release": requires_release,
        "release_state": release_state,
        "release_observed": bool(release.get("observed")),
        "release_final_gripper_opening_m": release.get("final_gripper_opening_m"),
        "full_goal_evidence": evidence_state,
        "release_executed": placement.get("release_executed"),
        "release_verified": placement.get("release_verified"),
        "placement_verified": placement.get("placement_verified"),
        "placement_status": placement.get("execution_status"),
        "evidence_source": placement.get("evidence_source"),
        "policy_owned": placement.get("policy_owned"),
        "retreat_status": placement.get("retreat_status"),
        "policy_actions_after_goal": (placement.get("policy") or {}).get("actions_after_goal"),
        "policy_stop_reason": (placement.get("policy") or {}).get("stop_reason"),
        "finalizer_actions": placement.get("extra_actions"),
        "finalizer_s": placement.get("extra_s"),
        "placement_reason": placement.get("reason_code"),
        "duration_s": record.get("duration_s"),
        "episode_sim_actions": record.get("episode_sim_actions"),
        "recovery_count": record.get("recovery_count"),
        "wall_s": round(wall_s, 3),
        "episode_id": record.get("episode_id") or result.get("episode_id"),
        "record": result.get("record"),
        "scene_epoch": record.get("scene_epoch"),
        "seed": (record.get("env") or {}).get("seed"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", default="http://127.0.0.1:14800")
    parser.add_argument("--rounds", default="config/language_rounds_20261027.json")
    parser.add_argument("--tasks", default="", help="comma list of task ids, default all")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--out", default="records/acceptance/language_rounds_20261027.json")
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="force /api/load before every round, so repeated task ids start from the frozen init",
    )
    parser.add_argument(
        "--rederive-mapping",
        action="store_true",
        help="recompute only the mapping_ok column of an existing --out file with the current rule; "
        "execution columns are left untouched and the rows are marked rederived",
    )
    args = parser.parse_args()

    if args.rederive_mapping:
        path = PROJECT / args.out
        payload = json.loads(path.read_text(encoding="utf-8"))
        changed = 0
        for row in payload.get("rows") or []:
            before = row.get("mapping_ok")
            after = _mapping_ok(
                status=row.get("decision_status"),
                chosen=row.get("semantic_choice"),
                expected=row.get("expected_choice"),
                key=row.get("task_key"),
                official=row.get("task_official_instruction"),
                catalog_official=row.get("catalog_official_instruction"),
                step_prompt=row.get("plan_step_prompt"),
                verification_prompt=row.get("policy_verification_prompt"),
            )
            if before != after:
                changed += 1
            row["mapping_ok"] = after
            row["mapping_ok_source"] = "rederived_from_recorded_fields"
        rows = payload.get("rows") or []
        payload["summary"] = {
            **(payload.get("summary") or {}),
            "mapping_ok": sum(1 for row in rows if row.get("mapping_ok")),
            "mapping_ok_rederived": changed,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"out": str(path), "rows": len(rows), "changed": changed}, ensure_ascii=False))
        return 0

    manifest = json.loads((PROJECT / args.rounds).read_text(encoding="utf-8"))
    rounds = list(manifest["rounds"])
    if args.tasks:
        wanted = {int(token) for token in args.tasks.split(",") if token.strip()}
        rounds = [row for row in rounds if int(row["task_id"]) in wanted]

    health = get(f"{args.engine}/health", timeout=20)
    identity = health.get("identity") or {}
    tasks = {int(row["task_id"]): row for row in json.loads((PROJECT / "config/tasks.json").read_text())["tasks"]}

    out_path = PROJECT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "schema": "embodied_engine.language_rounds.v1",
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "manifest": args.rounds,
        "manifest_frozen": bool(manifest.get("frozen")),
        "seed": manifest.get("seed"),
        "engine": args.engine,
        "agent_model": (identity.get("agent") or {}).get("model"),
        "agent_mode": (identity.get("agent") or {}).get("mode"),
        "action_model": (identity.get("action_model") or {}).get("name"),
        "inference_mode": (identity.get("inference") or {}).get("mode"),
        "catalog": (identity.get("catalog") or {}),
        "rounds_total": len(rounds),
    }
    rows: list[dict[str, Any]] = []

    def flush() -> None:
        summary = {
            "rounds": len(rows),
            "mapping_ok": sum(1 for row in rows if row["mapping_ok"]),
            "native_success": sum(1 for row in rows if row["native_task_success"]),
            "released": sum(1 for row in rows if row["full_goal_evidence"] == "released"),
            "held_or_unknown": sum(1 for row in rows if row["full_goal_evidence"] == "held_or_unknown"),
            "not_applicable_native_ok": sum(
                1 for row in rows if row["full_goal_evidence"] == "native_only"
            ),
            "absent": sum(1 for row in rows if row["full_goal_evidence"] == "absent"),
        }
        out_path.write_text(
            json.dumps({"meta": meta, "summary": summary, "rows": rows}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    for spec in rounds:
        task = tasks[int(spec["task_id"])]
        print(f"[round {spec['round']}] task {spec['task_id']} {spec['lang']}: {spec['instruction']}", flush=True)
        current = get(f"{args.engine}/api/status", timeout=20)
        same_scene = int(current.get("task_id") or -1) == int(spec["task_id"])
        if args.fresh:
            same_scene = False
        loaded = {"status": "ok", "reused": True} if same_scene else post(
            f"{args.engine}/api/load",
            {"task_id": int(spec["task_id"]), "seed": manifest.get("seed")},
            timeout=320,
        )
        if loaded.get("status") != "ok":
            rows.append(
                {
                    "round": spec["round"],
                    "task_id": spec["task_id"],
                    "lang": spec["lang"],
                    "user_instruction": spec["instruction"],
                    "mapping_ok": False,
                    "native_task_success": False,
                    "full_goal_evidence": "absent",
                    "failure_code": "load_failed",
                    "failure_reason": loaded.get("error") or str(loaded),
                }
            )
            flush()
            continue
        started = time.time()
        submitted = post(
            f"{args.engine}/api/run",
            {"task_id": int(spec["task_id"]), "instruction": spec["instruction"]},
            timeout=30,
        )
        if submitted.get("status") != "started":
            row = {
                "round": spec["round"],
                "task_id": spec["task_id"],
                "lang": spec["lang"],
                "user_instruction": spec["instruction"],
                "mapping_ok": False,
                "native_task_success": False,
                "full_goal_evidence": "absent",
                "failure_code": "run_rejected",
                "failure_reason": submitted.get("error") or submitted.get("reason") or str(submitted),
            }
        else:
            row = _round_row(spec, task, wait_terminal(args.engine, args.timeout), time.time() - started)
            row["wall_s"] = round(time.time() - started, 3)
        rows.append(row)
        print(
            f"  mapping={row['mapping_ok']} choice={row.get('semantic_choice')} "
            f"native={row.get('native_task_success')} evidence={row.get('full_goal_evidence')} "
            f"release={row.get('release_state')} {row.get('duration_s')}s "
            f"fail={row.get('failure_code')}",
            flush=True,
        )
        flush()

    meta["finished_at"] = datetime.now().isoformat(timespec="seconds")
    flush()
    summary = json.loads(out_path.read_text(encoding="utf-8"))["summary"]
    print(json.dumps({"out": str(out_path), "summary": summary}, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
