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
INFERENCE = "http://127.0.0.1:14600"
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


def inference_metrics() -> dict:
    # Read the inference service directly: the engine health endpoint caches
    # its identity for 5 s, which would hide a reset that just happened.
    with OPENER.open(INFERENCE + "/health", timeout=10) as response:
        return json.loads(response.read()).get("metrics") or {}


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
    time.sleep(12)
    mid = status()
    print(f"    phase while running: {mid.get('phase')} episode={mid.get('episode_id')}")
    if mid.get("episode_id") != episode:
        raise SystemExit("[fail] status episode does not match the started episode")
    print("[3] stop")
    before_stop = inference_metrics()
    sim_at_stop = mid.get("sim_action_count")
    stop_out = call("/api/stop", {"reason": "lifecycle_check"})
    budget = stop_out.get("stop") or {}
    print(f"    stop budget: {json.dumps(budget)}")
    stopped = wait(lambda row: row.get("phase") in {"stopped", "failed", "ready"}, 120, "stopped")
    result = stopped.get("last_result") or {}
    if result.get("episode_id") and result.get("episode_id") != episode:
        raise SystemExit("[fail] last_result belongs to another episode")
    if result.get("success"):
        raise SystemExit("[fail] a stopped episode must not be reported as success")
    print(f"    stopped result: execution={result.get('execution_status')} failure={result.get('failure_code')}")
    # The policy must not keep acting after a stop, and the model-side
    # receding-horizon buffer must be dropped rather than replayed.
    if not sim_at_stop:
        raise SystemExit("[fail] no action was executed before stop; the check would be vacuous")
    if int(budget.get("in_flight_actions", 0)) > 1:
        raise SystemExit(f"[fail] more than one in-flight action at stop: {budget}")
    time.sleep(3.0)
    after = status()
    frozen_at = int(budget.get("sim_at_ack") or sim_at_stop)
    if after.get("sim_action_count") != frozen_at:
        raise SystemExit(
            f"[fail] simulator advanced after stop ack: {frozen_at} -> {after.get('sim_action_count')}"
        )
    print(f"    no action issued after stop (in-flight at most 1); sim frozen at {frozen_at}")
    after_stop = inference_metrics()
    resets = int(after_stop.get("backend_resets", 0)) - int(before_stop.get("backend_resets", 0))
    if resets < 1:
        raise SystemExit(f"[fail] stop did not clear the model-side buffer (backend_resets delta={resets})")
    print(f"    model-side buffer cleared on stop (backend_resets +{resets})")
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
    if int(clean.get("scene_epoch") or 0) <= int(stopped.get("scene_epoch") or 0):
        raise SystemExit(
            f"[fail] reset did not create a new scene epoch: {stopped.get('scene_epoch')} -> {clean.get('scene_epoch')}"
        )
    print(f"    scene epoch advanced {stopped.get('scene_epoch')} -> {clean.get('scene_epoch')}")
    print("[5] run again after reset")
    again = call("/api/run", {"task_id": 1})
    if again.get("status") != "started":
        raise SystemExit(f"second run rejected: {again}")
    wait(lambda row: row.get("phase") in {"succeeded", "failed", "stopped"}, 300, "second episode terminal")
    second = status()
    print(f"    second episode={second.get('episode_id')} phase={second.get('phase')} "
          f"success={(second.get('last_result') or {}).get('success')}")
    print("lifecycle: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
