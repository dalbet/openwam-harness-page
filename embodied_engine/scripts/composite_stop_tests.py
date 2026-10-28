#!/usr/bin/env python
"""Real stop tests for the composite, synchronised on named stages.

Three points are covered, each triggered by something the engine actually
published rather than by a guessed sleep:

* ``step-1``   — while step 1 is executing (phase ``running`` + action count);
* ``handoff``  — the moment the engine enters the named ``handoff`` phase,
                 i.e. between the two steps, after step 1 was verified;
* ``finalizer``— the first ``finalizer_step`` journal row of phase ``release``
                 for this episode, i.e. the finalizer is mid-release.

After the stop ack the script keeps polling the sim action count: a stop that
still executes actions is a failure. A final ``/api/load`` proves the next
episode starts from zero instead of inheriting the stopped one.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

PROJECT = Path(__file__).resolve().parents[1]
JOURNAL = PROJECT / "records" / "run_journal.jsonl"
TERMINAL = {"succeeded", "failed", "stopped"}
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def get(url: str, timeout: float = 10.0) -> dict[str, Any]:
    with OPENER.open(urllib.request.Request(url, method="GET"), timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def post(url: str, body: dict[str, Any], timeout: float = 60.0) -> dict[str, Any]:
    data = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with OPENER.open(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:1200]
        try:
            return json.loads(detail)
        except json.JSONDecodeError:
            return {"status": "error", "error": f"HTTP {exc.code}: {detail}"}


class JournalTail:
    """Read new journal rows only, so the trigger is the engine's own record."""

    def __init__(self, path: Path = JOURNAL):
        self.path = path
        self.offset = path.stat().st_size if path.exists() else 0

    def new_rows(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        rows: list[dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8", errors="replace") as handle:
            handle.seek(self.offset)
            for line in handle:
                if not line.endswith("\n"):
                    break
                self.offset += len(line.encode("utf-8"))
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return rows


def wait_for(
    engine: str,
    *,
    predicate: Callable[[dict[str, Any]], bool],
    timeout_s: float,
    poll_s: float = 0.05,
) -> dict[str, Any]:
    started = time.time()
    last: dict[str, Any] = {}
    while time.time() - started < timeout_s:
        try:
            last = get(f"{engine}/api/status")
        except Exception as exc:  # noqa: BLE001 - engine may be restarting
            last = {"phase": "unreachable", "error": f"{type(exc).__name__}: {exc}"}
        if predicate(last):
            return last
        if last.get("phase") in TERMINAL:
            return last
        time.sleep(poll_s)
    return {**last, "phase": "sync_timeout"}


def actions_after_stop(engine: str, *, settle_s: float = 2.5) -> dict[str, Any]:
    started = time.time()
    counts: list[int] = []
    status: dict[str, Any] = {}
    while time.time() - started < settle_s:
        try:
            status = get(f"{engine}/api/status")
        except Exception:  # noqa: BLE001
            status = {}
        value = status.get("sim_action_count")
        if isinstance(value, int):
            counts.append(value)
        time.sleep(0.2)
    return {
        "samples": counts,
        "first": counts[0] if counts else None,
        "last": counts[-1] if counts else None,
        "advanced": bool(counts and counts[-1] != counts[0]),
        "phase": status.get("phase"),
    }


def _official_instruction(engine: str, task_id: int) -> str:
    """Official whole-task prompt for a task id, read from the live catalog."""
    payload = get(f"{engine}/api/tasks")
    for scene in payload.get("scenes") or []:
        if int(scene.get("task_id", -1)) != int(task_id):
            continue
        for key in ("prompt", "instruction", "official_prompt", "label"):
            value = scene.get(key)
            if isinstance(value, str) and value and not value.startswith(f"{task_id}:"):
                return value
    raise SystemExit(f"no official prompt for task {task_id} in /api/tasks")


def run_case(args, spec: dict[str, Any]) -> dict[str, Any]:
    engine = args.engine
    kind = spec["kind"]
    tail = JournalTail()
    loaded = post(
        f"{engine}/api/load",
        {"task_id": int(spec.get("load_task_id", spec["task_id"])), "seed": int(spec["seed"])},
        timeout=400,
    )
    if loaded.get("status") != "ok":
        return {"case": spec["id"], "kind": kind, "ok": False, "error": loaded}
    started_episode = post(
        f"{engine}/api/run",
        {
            "task_id": int(spec["task_id"]),
            "instruction": spec["instruction"],
            "seed": int(spec["seed"]),
        },
        timeout=120,
    )
    episode_id = started_episode.get("episode_id")
    print(f"[{spec['id']}] episode {episode_id} kind={kind}", flush=True)

    if kind == "step-1":
        status = wait_for(
            engine,
            predicate=lambda row: (
                str(row.get("phase")) in {"running", "verifying"}
                and "step 1/2" in str(row.get("detail") or "")
                and int(row.get("sim_action_count") or 0) >= int(spec.get("min_actions", 20))
            ),
            timeout_s=args.timeout,
        )
    elif kind == "step-2":
        status = wait_for(
            engine,
            predicate=lambda row: (
                str(row.get("phase")) in {"running", "verifying"}
                and "step 2/2" in str(row.get("detail") or "")
                and int(row.get("sim_action_count") or 0) >= int(spec.get("min_actions", 10))
            ),
            timeout_s=args.timeout,
            poll_s=0.02,
        )
    elif kind == "handoff":
        status = wait_for(
            engine,
            predicate=lambda row: str(row.get("phase")) == "handoff",
            timeout_s=args.timeout,
            poll_s=0.02,
        )
    elif kind in {"restore-before-write", "restore-after-write"}:
        # Both stages are published by the handoff itself, so the stop lands
        # where the case claims instead of after a guessed sleep:
        #   ``policy_buffer_reset`` — the action barrier has just closed, the
        #                             robot state has not been written yet;
        #   ``handoff``/ready      — the write, the settle and the fresh read
        #                             are all verified, step 2 has not run.
        deadline = time.time() + args.timeout
        status: dict[str, Any] = {}
        hit = None
        ended = None
        while time.time() < deadline:
            for row in tail.new_rows():
                if episode_id and row.get("episode_id") != episode_id:
                    continue
                if kind == "restore-before-write" and row.get("stage") == "policy_buffer_reset":
                    hit = row
                    break
                if (
                    kind == "restore-after-write"
                    and row.get("stage") == "handoff"
                    and (row.get("row") or {}).get("status") == "ready"
                ):
                    hit = row
                    break
                if row.get("stage") in {"episode_end", "succeeded", "failed", "stopped"}:
                    ended = row
            if hit:
                status = get(f"{engine}/api/status")
                break
            if ended is not None:
                status = {
                    "phase": "stage_not_reached",
                    "detail": f"episode ended with stage {ended.get('stage')} before {kind}",
                }
                break
            time.sleep(0.01)
        if not hit and not status:
            status = {"phase": "sync_timeout", "detail": f"no {kind} stage"}
        else:
            status = {**status, "trigger_row": hit}
    elif kind == "finalizer":
        deadline = time.time() + args.timeout
        status: dict[str, Any] = {}
        hit = None
        ended = None
        while time.time() < deadline:
            for row in tail.new_rows():
                if episode_id and row.get("episode_id") != episode_id:
                    continue
                if row.get("stage") == "finalizer_step":
                    # Any finalizer step is enough: the finalizer is now
                    # executing actions of its own, which is exactly the
                    # window the stop has to interrupt. Requiring a specific
                    # inner phase narrowed the trigger so much that the whole
                    # finalizer (22 actions, ~1.6 s) could finish first.
                    hit = row
                    break
                if row.get("stage") in {"episode_end", "succeeded", "failed", "stopped"}:
                    ended = row
            if hit:
                status = get(f"{engine}/api/status")
                break
            if ended is not None:
                # The episode ended without ever publishing a finalizer row:
                # the policy released through its own chunk. Report it now
                # instead of burning the whole timeout.
                status = {
                    "phase": "stage_not_reached",
                    "detail": f"episode ended with stage {ended.get('stage')}; the policy released on its own, so the finalizer never started",
                }
                break
            time.sleep(0.02)
        if not hit and not status:
            status = {"phase": "sync_timeout", "detail": "no finalizer release row"}
        else:
            status = {**status, "trigger_row": hit}
    else:
        raise SystemExit(f"unknown kind {kind}")

    if status.get("phase") in {"sync_timeout", "unreachable", "stage_not_reached"} or status.get("phase") in TERMINAL:
        # The episode ended before the stage was reached: keep that as the
        # result instead of inventing a stop.
        return {
            "case": spec["id"],
            "kind": kind,
            "episode_id": episode_id,
            "ok": False,
            "error": f"stage not reached: {status.get('phase')} {status.get('detail')}",
            "status": status,
        }

    ack = post(f"{engine}/api/stop", {"reason": f"stop-test:{spec['id']}"}, timeout=60)
    after = actions_after_stop(engine)
    final = get(f"{engine}/api/status")
    last_result = final.get("last_result") or {}
    record = {}
    path = last_result.get("record")
    if path and Path(path).exists():
        try:
            record = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            record = {}
    journal_rows = tail.new_rows()
    step2_started = any(
        row.get("stage") == "composite_step_start" and row.get("episode_id") == episode_id
        for row in journal_rows
    )
    ok = bool(
        ack.get("status") == "ok"
        and final.get("phase") == "stopped"
        and not after["advanced"]
        and record.get("execution_status") == "stopped"
    )

    # ---- no leak: a fresh load must not inherit the stopped episode --------
    leak: dict[str, Any] = {}
    try:
        reloaded = post(
            f"{engine}/api/load",
            {"task_id": int(spec.get("load_task_id", spec["task_id"])), "seed": int(spec["seed"]) + 7},
            timeout=400,
        )
        fresh = get(f"{engine}/api/status")
        leak = {
            "load_status": reloaded.get("status"),
            "sim_action_count_after_load": fresh.get("sim_action_count"),
            "scene_epoch_after_load": fresh.get("scene_epoch"),
            "episode_after_load": fresh.get("episode_id"),
            "phase_after_load": fresh.get("phase"),
            "last_result_after_load": fresh.get("last_result"),
            "no_actions_left": int(fresh.get("sim_action_count") or 0) == 0,
            "new_episode": str(fresh.get("episode_id")) != str(episode_id),
        }
        ok = ok and bool(leak["no_actions_left"] and leak["new_episode"])
    except Exception as exc:  # noqa: BLE001 - reported, not hidden
        leak = {"error": f"{type(exc).__name__}: {exc}"}
        ok = False

    return {
        "case": spec["id"],
        "kind": kind,
        "episode_id": episode_id,
        "ok": ok,
        "trigger_phase": status.get("phase"),
        "trigger_detail": status.get("detail"),
        "trigger_sim_actions": status.get("sim_action_count"),
        "stop_ack": ack,
        "after_stop": after,
        "final_phase": final.get("phase"),
        "final_detail": final.get("detail"),
        "step_2_started": step2_started,
        "record_execution_status": record.get("execution_status"),
        "record_failure": {
            "code": record.get("failure_code"),
            "reason": record.get("failure_reason"),
        },
        "record_sim_actions": record.get("episode_sim_actions"),
        "scene_epoch": record.get("scene_epoch"),
        "journal_stages": sorted({str(row.get("stage")) for row in journal_rows}),
        "leak_check": leak,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", default="http://127.0.0.1:14800")
    parser.add_argument("--out", default="records/acceptance/composite_stop_20261027.json")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--only", default="")
    parser.add_argument(
        "--handoff-mode",
        default="",
        help="switch the engine's step-handoff mode before the cases (action_retreat | robot_state_restore)",
    )
    parser.add_argument(
        "--include-restore-cases",
        action="store_true",
        help="add the two robot_state_restore stages (before/after the state write)",
    )
    parser.add_argument(
        "--finalizer-task",
        type=int,
        default=None,
        help="re-point the finalizer stop case at this task id (official prompt looked up from /api/tasks)",
    )
    args = parser.parse_args()

    cases = [
        {
            "id": "stop-step1",
            "kind": "step-1",
            "task_id": 8,
            "load_task_id": 8,
            "seed": 0,
            "min_actions": 25,
            "instruction": "Open the middle drawer of the cabinet, then put the bowl on the plate.",
        },
        {
            "id": "stop-handoff",
            "kind": "handoff",
            "task_id": 8,
            "load_task_id": 8,
            "seed": 0,
            "instruction": "Open the middle drawer of the cabinet, then put the bowl on the plate.",
        },
        {
            "id": "stop-step2",
            "kind": "step-2",
            "task_id": 8,
            "load_task_id": 8,
            "seed": 0,
            "min_actions": 20,
            "instruction": "Open the middle drawer of the cabinet, then put the bowl on the plate.",
        },
        {
            # The composite's step-2 finalizer only starts once the bowl is a
            # release candidate. Exercising the same finalizer code path on the
            # single-task instruction is the only way to interrupt it while it
            # is actually mid-release; the report says so.
            #
            # task 1 / seed 0 is the round whose earlier diagnostic run released
            # through ``execution_finalizer`` rather than through a policy chunk
            # (records/acceptance/placement_diag_finalizer2.json). task 8 / seed
            # 0 releases through a chunk in ~29 s, so it never publishes a
            # ``finalizer_step`` row and cannot serve as the synchronisation
            # point.
            "id": "stop-finalizer-single-control",
            "kind": "finalizer",
            "task_id": 1,
            "load_task_id": 1,
            "seed": 0,
            "instruction": "put the bowl on the stove",
        },
    ]
    if args.include_restore_cases:
        cases.extend(
            [
                {
                    # The action barrier has just closed (policy buffer reset,
                    # no request in flight); the robot state has not been
                    # written yet. A stop here must wait for the barrier and
                    # then leave no write behind.
                    "id": "stop-restore-before-write",
                    "kind": "restore-before-write",
                    "task_id": 8,
                    "load_task_id": 8,
                    "seed": 0,
                    "instruction": "Open the middle drawer of the cabinet, then put the bowl on the plate.",
                },
                {
                    # Restore applied, verified, settled and re-read; step 2
                    # has not issued an action yet. The ack may only come
                    # after the write barrier, and nothing may resume.
                    "id": "stop-restore-after-write",
                    "kind": "restore-after-write",
                    "task_id": 8,
                    "load_task_id": 8,
                    "seed": 0,
                    "instruction": "Open the middle drawer of the cabinet, then put the bowl on the plate.",
                },
            ]
        )
    if args.finalizer_task is not None:
        # Only the finalizer case is re-pointed: the stage exists only when the
        # policy fails to release on its own, so which task reaches it is a
        # property of the policy round, not of the stop machinery.
        official = _official_instruction(args.engine, int(args.finalizer_task))
        for case in cases:
            if case["kind"] == "finalizer":
                case["task_id"] = int(args.finalizer_task)
                case["load_task_id"] = int(args.finalizer_task)
                case["instruction"] = official
    if args.only:
        wanted = {value.strip() for value in args.only.split(",")}
        cases = [case for case in cases if case["id"] in wanted]
    mode_report: dict[str, Any] = {}
    if args.handoff_mode:
        mode_report = post(f"{args.engine}/api/handoff", {"mode": args.handoff_mode}, timeout=30)
        print(f"[handoff] {json.dumps(mode_report, ensure_ascii=False)[:400]}", flush=True)
        if mode_report.get("status") != "ok":
            raise SystemExit(f"could not set handoff mode: {mode_report}")

    report = {
        "schema": "embodied_engine.composite_stop.v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "engine": args.engine,
        "handoff_mode": args.handoff_mode or None,
        "handoff_mode_reply": mode_report or None,
        "cases": [],
    }
    out = PROJECT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    for spec in cases:
        row = run_case(args, spec)
        report["cases"].append(row)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(
            f"  -> {'PASS' if row.get('ok') else 'FAIL'} stop={row.get('stop_ack', {}).get('phase')} "
            f"after={row.get('after_stop', {}).get('advanced')} step2={row.get('step_2_started')}",
            flush=True,
        )
        time.sleep(2.0)
    passed = sum(1 for row in report["cases"] if row.get("ok"))
    print(json.dumps({"cases": len(report["cases"]), "passed": passed, "out": str(out)}))
    return 0 if passed == len(report["cases"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
