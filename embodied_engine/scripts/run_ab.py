#!/usr/bin/env python3
"""Interleaved A/B runner: Agent qwen vs d1 on fixed LIBERO initializations.

Each (task, seed) pair starts one env server pinned to that seed. LIBERO then
resets to the same fixed init state for every episode, so both Agents see the
same scene. A failed episode is recorded, never retried. The run aborts on an
infrastructure error instead of quietly continuing with a smaller sample.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from run_suite import archive_records, get, load_record, post, wait_terminal  # noqa: E402
DEFAULT_ENGINE = "http://127.0.0.1:14800"
TERMINAL = {"succeeded", "failed", "stopped"}


class AbortRun(RuntimeError):
    pass


def sha256_file(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def frozen_code() -> dict[str, str | None]:
    rows = {}
    for rel in (
        "agent/d1_planner.py",
        "agent/agent_service.py",
        "engine/engine_service.py",
        "engine/openwam_rollout.py",
        "common/plan_contract.py",
        "config/engine.json",
        "config/offline_eval.json",
    ):
        rows[rel] = sha256_file(PROJECT / rel)
    return rows


def official_expectations() -> dict[int, dict[str, Any]]:
    data = json.loads((PROJECT / "config" / "offline_eval.json").read_text(encoding="utf-8"))
    out: dict[int, dict[str, Any]] = {}
    for case in data.get("cases") or []:
        if str(case.get("id", "")).endswith("_official"):
            out[int(case["task_id"])] = {
                "instruction": case.get("instruction"),
                "expected": case.get("expected"),
            }
    return out


def _norm(value: Any) -> Any:
    return value if value is None else str(value)


def binding_checks(decision: dict[str, Any], expected: dict[str, Any]) -> dict[str, bool]:
    got = {key: _norm(decision.get(key)) for key in ("operation", "source", "target", "relation", "open_target")}
    checks = {}
    for key in ("operation", "source", "target", "relation"):
        if key in expected:
            checks[key] = got.get(key) == _norm(expected.get(key))
    if "open_target" in expected:
        checks["open_target"] = got.get("open_target") == _norm(expected.get("open_target"))
    return checks


def _covered(prompt: str, name: str | None, drawer: bool = False) -> bool:
    from agent.d1_planner import _covered as covered

    return bool(covered(prompt, name, drawer=drawer))


def _relation_in(prompt: str, relation: str) -> bool:
    from agent.d1_planner import _relation_in_instruction

    return bool(_relation_in_instruction(prompt, relation))


def prompt_checks(prompt: str | None, expected: dict[str, Any] | None) -> dict[str, bool]:
    if not prompt or not expected:
        return {}
    text = prompt.lower()
    operation = expected.get("operation")
    words = set(text.replace("'s", "").split())
    checks: dict[str, bool] = {}
    if operation == "open_drawer":
        checks["open"] = "open" in words
        checks["target"] = _covered(prompt, expected.get("target"), drawer=True)
        level = next((lv for lv in ("top", "middle", "bottom") if lv in words), None)
        if level:
            checks["level"] = level in text
        return checks
    if operation == "turn_on":
        checks["turn_on"] = _relation_in(prompt, "turn_on")
        checks["target"] = _covered(prompt, expected.get("target"))
        return checks
    relation = expected.get("relation") or "on"
    checks["source"] = _covered(prompt, expected.get("source"))
    checks["target"] = _covered(prompt, expected.get("target"), drawer=relation == "in")
    # push_object encodes the destination in the target region; the official
    # LIBERO instruction ("push the plate to the front of the stove") has no
    # lexical "on", so demanding one would mark a correct plan wrong.
    if operation != "push_object":
        checks["relation"] = _relation_in(prompt, relation)
    if operation == "composite":
        checks["open"] = "open" in words
        level = next((lv for lv in ("top", "middle", "bottom") if lv in words), None)
        if level:
            checks["level"] = level in text
    return checks


def episode_semantics(record: dict[str, Any], expected: dict[str, Any] | None, mode: str) -> dict[str, Any]:
    plan = record.get("plan") or {}
    steps = plan.get("steps") or []
    prompt = steps[0].get("prompt") if steps and isinstance(steps[0], dict) else None
    response = plan.get("agent_response") or {}
    decision = response.get("decision") or {}
    if decision:
        checks = binding_checks(decision, expected or {})
    else:
        checks = prompt_checks(prompt, expected)
    errors = response.get("errors") or []
    return {
        "plan_source": plan.get("source"),
        "agent_mode": plan.get("agent_mode"),
        "openwam_prompt": prompt,
        "decision": decision if mode == "d1" else None,
        "decision_status": (decision or {}).get("status") or response.get("status"),
        "decision_reason": (
            (decision or {}).get("reason_code")
            or (errors[0].get("code") if errors and isinstance(errors[0], dict) else None)
        ),
        "binding_checks": checks,
        "plan_semantic": bool(checks) and all(checks.values()),
    }


def switch_mode(engine: str, mode: str) -> dict[str, Any]:
    out = post(f"{engine}/api/agent_mode", {"mode": mode}, timeout=30)
    if out.get("status") != "ok":
        raise AbortRun(f"agent_mode {mode} rejected: {out}")
    status = get(f"{engine}/api/status", timeout=15)
    actual = ((status.get("identity") or {}).get("agent") or {}).get("mode")
    if actual != mode:
        raise AbortRun(f"agent_mode {mode} not reflected in identity: {actual}")
    return status


def settle(engine: str, timeout_s: float = 120.0) -> dict[str, Any]:
    """Wait for an episode to reach a terminal phase after a suite timeout.

    ``wait_terminal`` posts a stop when the episode overruns; the next episode
    must not start until that stop has settled, or the mode switch and the
    next run would be rejected while the old rollout still owns the env.
    """
    deadline = time.time() + timeout_s
    last: dict[str, Any] = {}
    while time.time() < deadline:
        last = get(f"{engine}/api/status", timeout=10)
        if last.get("phase") in TERMINAL:
            return last
        time.sleep(0.5)
    return last


def row_from_status(
    *,
    status: dict[str, Any],
    task_id: int,
    seed: int,
    init_index: int,
    order: int,
    mode: str,
    wall_s: float,
    expectations: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    result = status.get("last_result") or {}
    record = load_record(result.get("record"))
    expected_row = expectations.get(task_id) or {}
    expected = expected_row.get("expected")
    semantics = episode_semantics(record, expected, mode)
    openwam = record.get("openwam") or {}
    chunk = openwam.get("chunk_stats") or {}
    agent = ((status.get("identity") or {}).get("agent") or {})
    instruction = record.get("instruction") or (status.get("instruction") or "")
    prompt = semantics.get("openwam_prompt")
    requests = chunk.get("requests")
    chunk_ids = chunk.get("chunk_ids")
    if not requests:
        buffer_clear_ok = None
    else:
        buffer_clear_ok = bool(chunk_ids) and chunk_ids[0] == 0
    return {
        "task_id": task_id,
        "seed": seed,
        "init_index": init_index,
        "order": order,
        "mode": mode,
        "agent_mode": agent.get("mode"),
        "agent_model": agent.get("model"),
        "plan_source": semantics.get("plan_source"),
        "instruction": instruction,
        "openwam_prompt": prompt,
        "openwam_prompt_actual": openwam.get("prompt"),
        "prompt_matches_user": (prompt or "").strip() == (instruction or "").strip(),
        "expected": expected,
        "binding_checks": semantics.get("binding_checks"),
        "plan_semantic": semantics.get("plan_semantic"),
        "decision": semantics.get("decision"),
        "decision_status": semantics.get("decision_status"),
        "decision_reason": semantics.get("decision_reason"),
        "phase": status.get("phase"),
        "execution_status": result.get("execution_status") or record.get("execution_status"),
        "native_task_success": result.get("native_task_success", record.get("native_task_success")),
        "native_goal_final": record.get("native_goal_final"),
        "user_goal_success": result.get("success", record.get("user_goal_success")),
        "failure_code": result.get("failure_code") or record.get("failure_code"),
        "reason": result.get("reason") or record.get("failure_reason"),
        "duration_s": record.get("duration_s") or result.get("duration_s"),
        "wall_s": round(wall_s, 3),
        "episode_sim_actions": record.get("episode_sim_actions"),
        "timing": record.get("timing") or {},
        "openwam": openwam,
        "chunk_generations": chunk.get("chunk_generations"),
        "cache_hits": chunk.get("cache_hits"),
        "chunk_ids": chunk.get("chunk_ids"),
        "buffer_clear_ok": buffer_clear_ok,
        "scene_epoch": record.get("scene_epoch"),
        "initial_scene_fingerprint": result.get("initial_scene_fingerprint")
        or record.get("initial_scene_fingerprint"),
        "episode_id": result.get("episode_id") or record.get("episode_id"),
        "inference_mode": record.get("inference_mode"),
        "action_backend": record.get("action_backend"),
        "record": result.get("record"),
    }


def summarize(rows: list[dict[str, Any]], tasks: list[int], seeds: list[int]) -> dict[str, Any]:
    def rate(values: list[bool]) -> float | None:
        return round(sum(1 for v in values if v) / len(values), 4) if values else None

    def median(values: list[float]) -> float | None:
        clean = sorted(float(v) for v in values if isinstance(v, (int, float)))
        if not clean:
            return None
        mid = len(clean) // 2
        return round(clean[mid] if len(clean) % 2 else (clean[mid - 1] + clean[mid]) / 2, 3)

    out: dict[str, Any] = {"overall": {}, "per_task": {}, "per_task_seed": {}}
    for mode in sorted({row["mode"] for row in rows}):
        mode_rows = [row for row in rows if row["mode"] == mode]
        durations = [row.get("duration_s") for row in mode_rows]
        success_durations = [row.get("duration_s") for row in mode_rows if row.get("user_goal_success")]
        out["overall"][mode] = {
            "attempts": len(mode_rows),
            "completed": sum(1 for row in mode_rows if row.get("execution_status") == "completed"),
            "native_success": sum(1 for row in mode_rows if row.get("native_task_success")),
            "native_final_success": sum(1 for row in mode_rows if row.get("native_goal_final")),
            "user_goal_success": sum(1 for row in mode_rows if row.get("user_goal_success")),
            "plan_semantic": sum(1 for row in mode_rows if row.get("plan_semantic")),
            "decision_accepted": sum(
                1 for row in mode_rows if ((row.get("decision") or {}).get("status") == "accepted")
            ),
            "failures": sum(1 for row in mode_rows if row.get("execution_status") != "completed"),
            "buffer_clear_ok": sum(1 for row in mode_rows if row.get("buffer_clear_ok")),
            "plan_latency_median_s": median([(row.get("timing") or {}).get("planner_s") for row in mode_rows]),
            "duration_median_all_s": median(durations),
            "duration_median_success_s": median(success_durations),
            "plan_to_first_action_median_s": median(
                [(row.get("timing") or {}).get("plan_to_first_action_s") for row in mode_rows]
            ),
            "execution_median_s": median([(row.get("timing") or {}).get("execution_s") for row in mode_rows]),
            "verification_median_s": median(
                [(row.get("timing") or {}).get("verification_s") for row in mode_rows]
            ),
            "sim_actions_median": median([row.get("episode_sim_actions") for row in mode_rows]),
            "success_rate": rate([bool(row.get("user_goal_success")) for row in mode_rows]),
            "plan_semantic_rate": rate([bool(row.get("plan_semantic")) for row in mode_rows]),
        }
        for task_id in tasks:
            task_mode_rows = [row for row in mode_rows if int(row["task_id"]) == task_id]
            out["per_task"].setdefault(str(task_id), {})[mode] = {
                "attempts": len(task_mode_rows),
                "user_goal_success": sum(1 for row in task_mode_rows if row.get("user_goal_success")),
                "native_success": sum(1 for row in task_mode_rows if row.get("native_task_success")),
                "plan_semantic": sum(1 for row in task_mode_rows if row.get("plan_semantic")),
                "duration_median_all_s": median([row.get("duration_s") for row in task_mode_rows]),
                "duration_median_success_s": median(
                    [row.get("duration_s") for row in task_mode_rows if row.get("user_goal_success")]
                ),
            }
    for task_id in tasks:
        for seed in seeds:
            pair = [row for row in rows if int(row["task_id"]) == task_id and int(row["seed"]) == seed]
            if not pair:
                continue
            out["per_task_seed"][f"{task_id}/{seed}"] = {
                row["mode"]: {
                    "user_goal_success": row.get("user_goal_success"),
                    "native_task_success": row.get("native_task_success"),
                    "plan_semantic": row.get("plan_semantic"),
                    "duration_s": row.get("duration_s"),
                    "initial_scene_fingerprint": row.get("initial_scene_fingerprint"),
                }
                for row in pair
            }
            prints = {
                row["mode"]: row.get("initial_scene_fingerprint")
                for row in pair
                if row.get("initial_scene_fingerprint")
            }
            out["per_task_seed"][f"{task_id}/{seed}"]["same_init"] = (
                len(prints) == len({row["mode"] for row in pair}) and len(set(prints.values())) == 1
            )
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", default=DEFAULT_ENGINE)
    parser.add_argument("--tasks", default="all", help="comma list or all")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--modes", default="qwen,d1")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    tasks = list(range(10)) if args.tasks.strip() == "all" else [
        int(value) for value in args.tasks.split(",") if value.strip()
    ]
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    modes = [value.strip() for value in args.modes.split(",") if value.strip()]
    if not tasks or not seeds or not modes:
        raise SystemExit("tasks, seeds and modes must not be empty")
    expectations = official_expectations()
    started_at = datetime.now().isoformat(timespec="seconds")
    health = get(f"{args.engine}/health", timeout=15)
    identity = health.get("identity") or {}
    out_path = Path(args.out) if args.out else (
        PROJECT / "records" / "ab" / f"ab_{datetime.now():%Y%m%d_%H%M%S}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "schema": "embodied_engine.ab.v1",
        "started_at": started_at,
        "engine": args.engine,
        "tasks": tasks,
        "seeds": seeds,
        "modes": modes,
        "agent": identity.get("agent"),
        "action_model": identity.get("action_model"),
        "inference": identity.get("inference"),
        "frozen": frozen_code(),
        "precheck": len(tasks) < 10 or len(seeds) < 3,
    }
    rows: list[dict[str, Any]] = []
    print(f"A/B start tasks={tasks} seeds={seeds} modes={modes}", flush=True)

    try:
        for task_id in tasks:
            for init_index, seed in enumerate(seeds):
                loaded = post(f"{args.engine}/api/load", {"task_id": task_id, "seed": seed}, timeout=320)
                if loaded.get("status") != "ok" or int(loaded.get("seed", -1)) != seed:
                    raise AbortRun(f"load task={task_id} seed={seed} failed: {loaded}")
                order = modes if (task_id + init_index) % 2 == 0 else list(reversed(modes))
                print(f"[task {task_id} seed {seed}] order={order}", flush=True)
                for mode in order:
                    switch_mode(args.engine, mode)
                    started = time.time()
                    started_run = post(
                        f"{args.engine}/api/run",
                        {"task_id": task_id, "seed": seed},
                        timeout=30,
                    )
                    if started_run.get("status") != "started":
                        raise AbortRun(f"run task={task_id} seed={seed} mode={mode} rejected: {started_run}")
                    status = wait_terminal(args.engine, args.timeout)
                    timed_out = status.get("phase") not in TERMINAL
                    if timed_out:
                        status = settle(args.engine)
                        status["phase"] = "timeout"
                    row = row_from_status(
                        status=status,
                        task_id=task_id,
                        seed=seed,
                        init_index=init_index,
                        order=order.index(mode) + 1,
                        mode=mode,
                        wall_s=time.time() - started,
                        expectations=expectations,
                    )
                    if timed_out:
                        row["timed_out"] = True
                    rows.append(row)
                    print(
                        f"  {mode}: {row.get('execution_status')} native={row.get('native_task_success')} "
                        f"user={row.get('user_goal_success')} semantic={row.get('plan_semantic')} "
                        f"{row.get('duration_s')}s actions={row.get('episode_sim_actions')} "
                        f"fail={row.get('failure_code')}",
                        flush=True,
                    )
                    out_path.write_text(
                        json.dumps({"meta": meta, "rows": rows}, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
    except AbortRun as exc:
        payload = {
            "meta": meta,
            "rows": rows,
            "aborted_at": datetime.now().isoformat(timespec="seconds"),
            "abort_reason": str(exc),
        }
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"ABORT: {exc}", flush=True)
        return 2

    summary = {
        "meta": meta,
        "rows": rows,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "summary": summarize(rows, tasks, seeds),
    }
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    archived = archive_records(out_path, rows)
    summary["rows"] = rows
    summary["archived_records"] = archived
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out_path} ({len(rows)} episodes, archived={archived})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
