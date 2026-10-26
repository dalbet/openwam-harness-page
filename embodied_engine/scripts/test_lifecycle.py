#!/usr/bin/env python3
"""Lifecycle check against a running engine: start -> stop -> reset -> run.

Verifies that a stopped episode does not leak a result into the next one and
that reset returns to a clean ready state. Run after the acceptance suite.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

ENGINE = "http://127.0.0.1:14800"
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def call(path: str, payload: dict | None = None, timeout: float = 30.0) -> dict:
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        ENGINE + path, data=data, headers={"Content-Type": "application/json"}
    )
    with OPENER.open(req, timeout=timeout) as response:
        return json.loads(response.read())


def status() -> dict:
    return call("/api/status")


def wait(predicate, timeout_s: float, label: str) -> dict:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        row = status()
        if predicate(row):
            print(f"[ok] {label}: {row.get('phase')}")
            return row
        time.sleep(0.5)
    raise SystemExit(f"[fail] {label}: timeout; last={status()}")


def main() -> int:
    print("[1] load task 1")
    loaded = call("/api/load", {"task_id": 1}, timeout=320)
    if loaded.get("status") != "ok":
        raise SystemExit(f"load failed: {loaded}")
    print("[2] start")
    started = call("/api/run", {"task_id": 1})
    episode = started.get("episode_id")
    if started.get("status") != "started":
        raise SystemExit(f"run rejected: {started}")
    time.sleep(6)
    mid = status()
    print(f"    phase while running: {mid.get('phase')} episode={mid.get('episode_id')}")
    if mid.get("episode_id") != episode:
        raise SystemExit("[fail] status episode does not match the started episode")
    print("[3] stop")
    call("/api/stop", {"reason": "lifecycle_check"})
    stopped = wait(lambda row: row.get("phase") in {"stopped", "failed", "ready"}, 120, "stopped")
    result = stopped.get("last_result") or {}
    if result.get("episode_id") and result.get("episode_id") != episode:
        raise SystemExit("[fail] last_result belongs to another episode")
    if result.get("success"):
        raise SystemExit("[fail] a stopped episode must not be reported as success")
    print(f"    stopped result: execution={result.get('execution_status')} failure={result.get('failure_code')}")
    print("[4] reset")
    reset = call("/api/reset", {}, timeout=320)
    if reset.get("status") != "ok":
        raise SystemExit(f"reset failed: {reset}")
    clean = status()
    if clean.get("last_result"):
        raise SystemExit("[fail] reset kept the previous last_result")
    if clean.get("episode_id"):
        raise SystemExit("[fail] reset kept the previous episode_id")
    print(f"    clean phase={clean.get('phase')} scene_epoch={clean.get('scene_epoch')}")
    print("[5] run again after reset")
    again = call("/api/run", {"task_id": 1})
    if again.get("status") != "started":
        raise SystemExit(f"second run rejected: {again}")
    wait(lambda row: row.get("phase") not in {"loading", "preparing", "planning", "running"}, 240, "second episode terminal")
    second = status()
    print(f"    second episode={second.get('episode_id')} phase={second.get('phase')} "
          f"success={(second.get('last_result') or {}).get('success')}")
    print("lifecycle: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
