#!/usr/bin/env python
"""Frozen-round runner shared by the single-task smoke and composite rounds.

One manifest entry is one closed loop: fresh ``/api/load`` (the episode-level
reset happens here, never inside an episode), ``/api/run`` with the *raw user
instruction*, wait for a terminal phase, read the episode record and keep the
columns that separate the four different things that can go wrong:

    language mapping | step goal | native goal | release / final relation

A failed round is kept as-is. The raw episode record is copied next to the
summary immediately after the round, before the next one starts, because
``records/runs`` rotation can delete it.
"""

from __future__ import annotations

import argparse
import json
import shutil
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


def get(url: str, timeout: float = 20.0) -> dict[str, Any]:
    with OPENER.open(urllib.request.Request(url, method="GET"), timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def post(url: str, body: dict[str, Any], timeout: float = 320.0) -> dict[str, Any]:
    data = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with OPENER.open(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:1500]
        try:
            return json.loads(detail)
        except json.JSONDecodeError:
            return {"status": "error", "error": f"HTTP {exc.code}: {detail}"}


def wait_terminal(engine: str, timeout_s: float, poll_s: float = 1.0) -> dict[str, Any]:
    started = time.time()
    last: dict[str, Any] = {}
    while time.time() - started < timeout_s:
        try:
            last = get(f"{engine}/api/status", timeout=10)
        except Exception as exc:  # noqa: BLE001 - the engine may be restarting
            last = {"phase": "unreachable", "error": f"{type(exc).__name__}: {exc}"}
        if last.get("phase") in TERMINAL:
            return last
        time.sleep(poll_s)
    post(f"{engine}/api/stop", {"reason": "round_timeout"}, timeout=30)
    return {**last, "phase": "timeout"}


def norm(text: Any) -> str:
    return " ".join(str(text or "").strip().lower().split())


def load_record(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def archive(out_path: Path, row: dict[str, Any]) -> None:
    source = Path(str(row.get("record") or ""))
    if not source.exists():
        return
    target = out_path.with_name(out_path.stem + "_runs")
    target.mkdir(parents=True, exist_ok=True)
    destination = target / source.name
    if not destination.exists():
        shutil.copy2(source, destination)
    row["archived_record"] = str(destination)


def router_columns(semantic: dict[str, Any]) -> dict[str, Any]:
    attempts = semantic.get("attempts") or []
    first = (attempts[0] or {}).get("timing") if attempts else {}
    first = first or {}
    return {
        "router_status": semantic.get("decision_status"),
        "router_reason_code": semantic.get("reason_code"),
        "router_latency_s": semantic.get("router_latency_s"),
        "router_ttft_s": first.get("ttft_s"),
        "router_completion_tokens": first.get("completion_tokens"),
        "router_prompt_tokens": first.get("prompt_tokens"),
        "router_repairs": semantic.get("repairs"),
        "router_calls": len(attempts),
    }


def single_row(record: dict[str, Any], spec: dict[str, Any], catalog: dict[str, dict[str, Any]]) -> dict[str, Any]:
    semantic = record.get("semantic") or {}
    task = catalog[int(spec["task_id"])]
    steps = record.get("steps") or []
    step = steps[0] if steps else {}
    verification = (step.get("verification") or {}) if isinstance(step, dict) else {}
    placement = record.get("placement") or {}
    kind = str(task.get("kind") or "")
    requires_release = kind in RELEASE_REQUIRED_KINDS
    official = str(task.get("official_instruction") or "")
    chosen = semantic.get("semantic_choice")
    mapping_ok = bool(
        semantic.get("decision_status") == "ok"
        and chosen == spec.get("expected_choice")
        and norm(semantic.get("official_instruction")) == norm(official)
        and norm(step.get("prompt")) == norm(official)
    )
    native = bool(record.get("native_task_success"))
    final_steps = ((record.get("final_verification") or {}).get("steps") or [])
    final_check = (final_steps[0] or {}).get("verification") if final_steps else {}
    final_check = final_check or {}
    if requires_release:
        release_source = placement.get("evidence_source") or "none"
        release_state = "verified" if placement.get("placement_verified") else "unknown"
    else:
        release_source = "not_applicable"
        release_state = "not_applicable"
    return {
        "kind": "single",
        "task_id": spec["task_id"],
        "task_kind": kind,
        "task_key": task.get("key"),
        "lang": spec.get("lang"),
        "user_instruction": spec.get("instruction"),
        "expected_choice": spec.get("expected_choice"),
        "semantic_choice": chosen,
        **router_columns(semantic),
        "official_instruction": official,
        "plan_step_prompt": step.get("prompt") if isinstance(step, dict) else None,
        "mapping_ok": mapping_ok,
        "native_task_success": native,
        "native_goal_final": record.get("native_goal_final"),
        "user_goal_success": bool(record.get("user_goal_success")),
        "completion_predicate": verification.get("predicate") or verification.get("method"),
        "completion_ok": verification.get("ok"),
        "final_predicate": final_check.get("predicate") or final_check.get("method"),
        "final_ok": final_check.get("ok"),
        "final_qpos": final_check.get("qpos"),
        "final_native_samples": final_check.get("samples"),
        "requires_release": requires_release,
        "release_state": release_state,
        "release_source": release_source,
        "release_verified": placement.get("release_verified"),
        "placement_verified": placement.get("placement_verified"),
        "placement_status": placement.get("execution_status"),
        "policy_owned": placement.get("policy_owned"),
        "finalizer_actions": placement.get("extra_actions"),
        "execution_status": record.get("execution_status"),
        "failure_code": record.get("failure_code"),
        "failure_reason": record.get("failure_reason"),
        "episode_sim_actions": record.get("episode_sim_actions"),
        "duration_s": record.get("duration_s"),
        "recovery_count": record.get("recovery_count"),
        "episode_id": record.get("episode_id"),
        "scene_epoch": record.get("scene_epoch"),
        "seed": (record.get("env") or {}).get("seed"),
        "record": record.get("__path__"),
    }


def composite_row(record: dict[str, Any], spec: dict[str, Any], catalog: dict[str, dict[str, Any]]) -> dict[str, Any]:
    semantic = record.get("semantic") or {}
    steps = record.get("steps") or []
    composite = record.get("composite") or {}
    keys = list(semantic.get("composite_keys") or [])
    expected = list(spec.get("expected_keys") or [])
    expected_official = [
        str((catalog_by_key(catalog).get(key) or {}).get("official_instruction") or "") for key in expected
    ]
    prompts = [str((step or {}).get("prompt") or "") for step in steps]
    mapping_ok = bool(
        semantic.get("decision_status") == "ok"
        and keys == expected
        and len(prompts) == len(expected_official)
        and all(norm(a) == norm(b) for a, b in zip(prompts, expected_official))
    )
    step_rows = []
    for index, item in enumerate(composite.get("steps") or []):
        step_rows.append(
            {
                "index": item.get("index"),
                "step_id": item.get("step_id"),
                "task_id": item.get("task_id"),
                "task_key": item.get("task_key"),
                "prompt": item.get("prompt"),
                "goal_kind": item.get("goal_kind"),
                "ok": item.get("ok"),
                "ok_final": item.get("ok_final"),
                "native_current": item.get("native_current"),
                "native_ever": item.get("native_ever"),
                "sim_actions": item.get("sim_actions"),
                "duration_s": item.get("duration_s"),
                "failure_code": item.get("failure_code"),
                "release_state": item.get("release_state"),
                "release_source": item.get("release_source"),
                "evidence": item.get("evidence"),
            }
        )
    handoff = composite.get("handoff") or {}
    return {
        "kind": "composite",
        "task_id": spec["task_id"],
        "load_task_id": spec.get("load_task_id", spec["task_id"]),
        "task_key": ",".join(keys) or None,
        "lang": spec.get("lang"),
        "user_instruction": spec.get("instruction"),
        "expected_keys": expected,
        "composite_keys": keys,
        "semantic_choice": semantic.get("semantic_choice"),
        **router_columns(semantic),
        "user_official_instructions": expected_official,
        "plan_step_prompts": prompts,
        "mapping_ok": mapping_ok,
        "step_1_ok": composite.get("step_1_ok"),
        "step_1_final_ok": composite.get("step_1_final_ok"),
        "step_2_native": composite.get("step_2_native"),
        "step_2_final_native": composite.get("step_2_final_native"),
        "final_verification_read": composite.get("final_verification_read"),
        "native_goal_ever": composite.get("native_goal_ever"),
        "step_1_goal_ever": composite.get("step_1_goal_ever"),
        "scene_epoch_continuous": composite.get("scene_epoch_continuous"),
        "env_reset_between_steps": composite.get("env_reset_between_steps"),
        "env_reload_between_steps": composite.get("env_reload_between_steps"),
        "bddl_mutation": composite.get("bddl_mutation"),
        "object_teleport": composite.get("object_teleport"),
        "combo_verified": composite.get("combo_verified"),
        "native_task_success": bool(record.get("native_task_success")),
        "native_goal_final": record.get("native_goal_final"),
        "user_goal_success": bool(record.get("user_goal_success")),
        "steps": step_rows,
        "handoff_status": handoff.get("status"),
        "handoff_s": handoff.get("elapsed_s"),
        "handoff_retreat_actions": (handoff.get("retreat") or {}).get("actions"),
        "handoff_retreat_reason": (handoff.get("retreat") or {}).get("reason"),
        "handoff_buffer_reset": handoff.get("buffer_reset"),
        "handoff_scene_epoch_same": handoff.get("scene_epoch_same"),
        "handoff_actions_total": handoff.get("sim_actions"),
        "scene_epoch_start": composite.get("scene_epoch_start"),
        "scene_epoch_end": composite.get("scene_epoch_end"),
        "execution_status": record.get("execution_status"),
        "failure_code": record.get("failure_code"),
        "failure_reason": record.get("failure_reason"),
        "episode_sim_actions": record.get("episode_sim_actions"),
        "duration_s": record.get("duration_s"),
        "timing": record.get("timing"),
        "recovery_count": record.get("recovery_count"),
        "episode_id": record.get("episode_id"),
        "scene_epoch": record.get("scene_epoch"),
        "seed": (record.get("env") or {}).get("seed"),
        "record": record.get("__path__"),
    }


def catalog_by_key(catalog: dict[int, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("key")): row for row in catalog.values()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", default="http://127.0.0.1:14800")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    manifest = json.loads((PROJECT / args.manifest).read_text(encoding="utf-8"))
    rounds = list(manifest["rounds"])
    if args.limit:
        rounds = rounds[: args.limit]
    seed = manifest.get("seed")
    catalog = {int(row["task_id"]): row for row in json.loads((PROJECT / "config" / "tasks.json").read_text())["tasks"]}
    health = get(f"{args.engine}/health", timeout=20)
    identity = health.get("identity") or {}

    out_path = PROJECT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "schema": "embodied_engine.rounds.v1",
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "manifest": args.manifest,
        "manifest_sha256": __import__("hashlib").sha256((PROJECT / args.manifest).read_bytes()).hexdigest(),
        "seed": seed,
        "engine": args.engine,
        "agent_model": (identity.get("agent") or {}).get("model"),
        "agent_mode": (identity.get("agent") or {}).get("mode"),
        "action_model": (identity.get("action_model") or {}).get("name"),
        "inference_mode": (identity.get("inference") or {}).get("mode"),
        "catalog": identity.get("catalog") or (identity.get("agent") or {}).get("routing", {}).get("catalog"),
        "rounds_total": len(rounds),
        "retry_policy": "no round is ever retried or skipped",
    }
    rows: list[dict[str, Any]] = []

    def flush() -> None:
        out_path.write_text(
            json.dumps({"meta": meta, "rows": rows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    for spec in rounds:
        is_composite = bool(spec.get("expected_keys"))
        round_seed = int(spec.get("seed", seed if seed is not None else 0))
        label = f"[{spec.get('round')}] task {spec['task_id']} {spec.get('lang')} {'composite' if is_composite else 'single'}"
        print(f"{label}: {spec['instruction']} (seed {round_seed})", flush=True)
        loaded = post(
            f"{args.engine}/api/load",
            {"task_id": int(spec.get("load_task_id", spec["task_id"])), "seed": round_seed},
            timeout=400,
        )
        if loaded.get("status") != "ok":
            rows.append(
                {
                    "round": spec.get("round"),
                    "kind": "composite" if is_composite else "single",
                    "task_id": spec["task_id"],
                    "user_instruction": spec.get("instruction"),
                    "mapping_ok": False,
                    "failure_code": "load_failed",
                    "failure_reason": loaded.get("error") or str(loaded),
                }
            )
            flush()
            continue
        started = time.time()
        submitted = post(
            f"{args.engine}/api/run",
            {"task_id": int(spec.get("load_task_id", spec["task_id"])), "instruction": spec["instruction"]},
            timeout=60,
        )
        if submitted.get("status") != "started":
            rows.append(
                {
                    "round": spec.get("round"),
                    "kind": "composite" if is_composite else "single",
                    "task_id": spec["task_id"],
                    "user_instruction": spec.get("instruction"),
                    "mapping_ok": False,
                    "failure_code": "run_rejected",
                    "failure_reason": submitted.get("error") or str(submitted),
                }
            )
            flush()
            continue
        status = wait_terminal(args.engine, args.timeout)
        result = status.get("last_result") or {}
        record_path = result.get("record")
        record = load_record(record_path)
        record["__path__"] = record_path
        row = (
            composite_row(record, spec, catalog)
            if is_composite
            else single_row(record, spec, catalog)
        )
        row["round"] = spec.get("round")
        row["seed"] = round_seed
        row["wall_s"] = round(time.time() - started, 3)
        row["phase"] = status.get("phase")
        archive(out_path, row)
        rows.append(row)
        flush()
        print(
            f"  -> {row.get('execution_status')} map={row.get('mapping_ok')} "
            f"native={row.get('native_task_success')} user={row.get('user_goal_success')} "
            f"actions={row.get('episode_sim_actions')} dur={row.get('duration_s')} fail={row.get('failure_code')}",
            flush=True,
        )

    meta["finished_at"] = datetime.now().isoformat(timespec="seconds")
    meta["summary"] = {
        "rounds": len(rows),
        "mapping_ok": sum(1 for row in rows if row.get("mapping_ok")),
        "native_success": sum(1 for row in rows if row.get("native_task_success")),
        "user_goal_success": sum(1 for row in rows if row.get("user_goal_success")),
        "failed": sum(1 for row in rows if row.get("execution_status") == "failed"),
        "stopped": sum(1 for row in rows if row.get("execution_status") == "stopped"),
    }
    flush()
    print(json.dumps(meta["summary"], ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
