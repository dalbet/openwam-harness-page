#!/usr/bin/env python3
"""Run standard LIBERO-Goal tasks against the three-layer engine.

A failed episode is never retried by this runner. The Agent layer may perform
its own bounded recovery inside an episode, and that recovery is reported in
the episode record. Each task/run pair is a fresh ``/api/run`` call, which
resets the LIBERO scene before executing.
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
DEFAULT_ENGINE = "http://127.0.0.1:14800"
TERMINAL = {"succeeded", "failed", "stopped"}
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def get(url: str, timeout: float = 15.0) -> dict[str, Any]:
    with OPENER.open(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def post(url: str, body: dict[str, Any], timeout: float = 320.0) -> dict[str, Any]:
    data = json.dumps(body).encode("utf-8")
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
        except Exception as exc:  # noqa: BLE001 - engine may restart
            last = {"phase": "unreachable", "error": f"{type(exc).__name__}: {exc}"}
        if last.get("phase") in TERMINAL:
            return last
        time.sleep(1.0)
    post(f"{engine}/api/stop", {"reason": "suite_timeout"}, timeout=30)
    return {**last, "phase": "timeout"}


def load_record(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def summary_row(task_id: int, run: int, status: dict[str, Any]) -> dict[str, Any]:
    result = status.get("last_result") or {}
    record = load_record(result.get("record"))
    return {
        "task_id": task_id,
        "run": run,
        "episode_id": result.get("episode_id") or record.get("episode_id"),
        "phase": status.get("phase"),
        "execution_status": result.get("execution_status") or record.get("execution_status"),
        "native_task_success": result.get("native_task_success", record.get("native_task_success")),
        "user_goal_success": result.get("success", record.get("user_goal_success")),
        "failure_code": result.get("failure_code") or record.get("failure_code"),
        "reason": result.get("reason") or record.get("failure_reason"),
        "duration_s": record.get("duration_s") or result.get("duration_s"),
        "episode_sim_actions": record.get("episode_sim_actions") or record.get("sim_action_count"),
        "recovery_count": record.get("recovery_count"),
        "recovery_log": record.get("recovery_log"),
        "timing": record.get("timing") or {},
        "record": result.get("record"),
        "seed": ((record.get("env") or {}).get("seed")),
        "scene_epoch": record.get("scene_epoch"),
        "inference_mode": record.get("inference_mode"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", default=DEFAULT_ENGINE)
    parser.add_argument("--tasks", default="all", help="comma list or all")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    tasks = list(range(10)) if args.tasks.strip() == "all" else [
        int(value) for value in args.tasks.split(",") if value.strip()
    ]
    started_at = datetime.now().isoformat(timespec="seconds")
    health = get(f"{args.engine}/health", timeout=15)
    identity = health.get("identity") or {}
    out_path = Path(args.out) if args.out else (
        PROJECT / "records" / "acceptance" / f"suite_{datetime.now():%Y%m%d_%H%M%S}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "schema": "embodied_engine.suite.v1",
        "started_at": started_at,
        "engine": args.engine,
        "runs_per_task": args.runs,
        "agent_model": (identity.get("agent") or {}).get("model"),
        "action_model": (identity.get("action_model") or {}).get("name"),
        "inference_mode": (identity.get("inference") or {}).get("mode"),
        "env": identity.get("env"),
    }
    rows: list[dict[str, Any]] = []

    for task_id in tasks:
        loaded = post(f"{args.engine}/api/load", {"task_id": task_id}, timeout=320)
        if loaded.get("status") != "ok":
            rows.append(
                {
                    "task_id": task_id,
                    "run": 0,
                    "phase": "load_failed",
                    "user_goal_success": False,
                    "reason": loaded.get("error") or str(loaded),
                }
            )
            print(f"task {task_id}: load failed: {loaded}", flush=True)
            continue
        for run in range(1, args.runs + 1):
            print(f"task {task_id} run {run}/{args.runs}: starting", flush=True)
            started = time.time()
            started_run = post(f"{args.engine}/api/run", {"task_id": task_id}, timeout=30)
            if started_run.get("status") != "started":
                row = {
                    "task_id": task_id,
                    "run": run,
                    "phase": "run_rejected",
                    "user_goal_success": False,
                    "reason": started_run.get("error") or str(started_run),
                }
            else:
                row = summary_row(task_id, run, wait_terminal(args.engine, args.timeout))
            row["wall_s"] = round(time.time() - started, 3)
            rows.append(row)
            print(
                f"task {task_id} run {run}: {row.get('phase')} "
                f"native={row.get('native_task_success')} user={row.get('user_goal_success')} "
                f"{row.get('duration_s')}s actions={row.get('episode_sim_actions')} "
                f"fail={row.get('failure_code')}",
                flush=True,
            )
            out_path.write_text(
                json.dumps({"meta": meta, "rows": rows}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    summary = {
        "meta": meta,
        "rows": rows,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "success_count": sum(1 for row in rows if row.get("user_goal_success")),
        "attempts": len(rows),
    }
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out_path} ({summary['success_count']}/{summary['attempts']})", flush=True)
    archived = archive_records(out_path, rows)
    if archived:
        print(f"archived {archived} episode record(s) next to the suite JSON", flush=True)
    return 0


def archive_records(out_path: Path, rows: list[dict[str, Any]]) -> int:
    """Keep the records this suite depends on.

    The engine prunes ``records/runs`` to its newest ``MAX_RUNS`` files, so a
    suite JSON would slowly lose the evidence its rows point at. Copy the
    referenced records into ``records/acceptance/<suite>_runs/`` so the table
    and the raw episode stay together.
    """
    target = out_path.with_name(out_path.stem + "_runs")
    target.mkdir(parents=True, exist_ok=True)
    copied = 0
    for row in rows:
        source = Path(str(row.get("record") or ""))
        if not source.exists():
            continue
        destination = target / source.name
        if not destination.exists():
            destination.write_bytes(source.read_bytes())
            copied += 1
        row["archived_record"] = str(destination)
    return copied


if __name__ == "__main__":
    raise SystemExit(main())
