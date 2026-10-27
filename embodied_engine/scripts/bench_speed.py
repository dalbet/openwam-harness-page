#!/usr/bin/env python
"""Fixed-identity speed measurement for the Qwen + OpenWAM stack.

Three independent measurements, all on monotonic clocks:

1. Planner: repeated identical ``/plan`` requests. vLLM exposes queue, TTFT,
   decode and e2e histograms; a before/after scrape attributes one request to
   one delta (the agent serializes calls, and the metric count shows if another
   client interfered).
2. OpenWAM: repeated ``/infer`` calls on one frozen observation. The server's
   receding-horizon buffer means only 1 in ``execute_horizon`` calls runs the
   model; each row records ``chunk_generation``/``cache_hit``, chunk id,
   generated and consumed action counts, so the cached dequeue is never
   reported as model inference.
3. Action execution: one real closed-loop episode; the engine record carries
   per-action timings, wall control rate and sim real-time factor.

Usage (RPent python):
    python scripts/bench_speed.py --out records/acceptance/speed_qwen_openwam.json \
        --planner-reps 12 --openwam-actions 96
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
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
from adapters.openwam_contract import (  # noqa: E402
    build_openwam_obs,
    native_eef10_to_libero7d,
)
from common import config, numpy_json  # noqa: E402

OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def http(method: str, url: str, body: dict[str, Any] | None = None, timeout_s: float = 60.0) -> tuple[int, dict[str, Any]]:
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method=method)
    try:
        with OPENER.open(request, timeout=timeout_s) as response:
            return response.status, json.loads(response.read().decode("utf-8") or "{}")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        try:
            return exc.code, json.loads(detail)
        except json.JSONDecodeError:
            return exc.code, {"status": "error", "error": detail[:800]}


def get_json(url: str, timeout_s: float = 10.0) -> dict[str, Any]:
    try:
        with OPENER.open(urllib.request.Request(url), timeout=timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        return {"status": "unreachable", "error": f"{type(exc).__name__}: {exc}"}


def sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def cmdline(pid: int | str | None) -> list[str]:
    if pid is None:
        return []
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return []
    return [part for part in raw.decode("utf-8", "replace").split("\x00") if part]


def gpu_table() -> list[dict[str, Any]]:
    proc = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    rows = []
    for line in proc.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 6:
            rows.append(
                {
                    "index": int(parts[0]),
                    "uuid": parts[1],
                    "name": parts[2],
                    "memory_used_mib": int(parts[3]),
                    "memory_total_mib": int(parts[4]),
                    "utilization_pct": int(parts[5]),
                }
            )
    return rows


def summarize(values: list[Any]) -> dict[str, Any] | None:
    vals = sorted(float(v) for v in values if isinstance(v, (int, float)))
    if not vals:
        return None

    def at(p: float) -> float:
        return round(vals[min(len(vals) - 1, int(round(p * (len(vals) - 1))))], 4)

    return {
        "n": len(vals),
        "mean": round(statistics.fmean(vals), 4),
        "min": round(vals[0], 4),
        "median": at(0.5),
        "p95": at(0.95),
        "max": round(vals[-1], 4),
    }


def environment() -> dict[str, Any]:
    cfg_path = PROJECT / "config" / "engine.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    openwam_record = PROJECT / "records" / "openwam_server.json"
    return {
        "project": str(PROJECT),
        "git_head": subprocess.run(
            ["git", "-C", str(PROJECT), "rev-parse", "HEAD"], capture_output=True, text=True, check=False
        ).stdout.strip(),
        "config_sha256": sha256(cfg_path),
        "control_frequency_hz": cfg.get("action_model", {}).get("control_frequency_hz"),
        "pi05": {
            "checkpoint": cfg["action_model"].get("checkpoint"),
            "revision": cfg["action_model"].get("revision"),
        },
        "openwam": cfg["action_model"].get("openwam"),
        "openwam_launch_record": json.loads(openwam_record.read_text(encoding="utf-8"))
        if openwam_record.exists()
        else None,
        "agent": cfg.get("agent"),
        "env": cfg.get("env"),
        "inference": cfg.get("inference"),
        "gpus": gpu_table(),
        "processes": {
            "openwam": cmdline((PROJECT / "records" / "openwam_server.pid").read_text().strip())
            if (PROJECT / "records" / "openwam_server.pid").exists()
            else [],
            "inference": cmdline((PROJECT / "records" / "inference_service.pid").read_text().strip())
            if (PROJECT / "records" / "inference_service.pid").exists()
            else [],
            "agent": cmdline((PROJECT / "records" / "agent_service.pid").read_text().strip())
            if (PROJECT / "records" / "agent_service.pid").exists()
            else [],
            "engine": cmdline((PROJECT / "records" / "engine_service.pid").read_text().strip())
            if (PROJECT / "records" / "engine_service.pid").exists()
            else [],
        },
        "identities": {
            "engine": get_json(f"{config.service_url('engine')}/health"),
            "inference": get_json(f"{config.service_url('inference')}/health"),
            "agent": get_json(f"{config.service_url('agent')}/health"),
        },
    }


def planner_bench(*, instruction: str, task_id: int, snapshot: dict[str, Any], reps: int) -> dict[str, Any]:
    rows = []
    for index in range(reps):
        body = {
            "instruction": instruction,
            "task_id": task_id,
            "snapshot": snapshot,
            "attempts": [],
            "backend": "openwam",
        }
        started = time.perf_counter()
        status, response = http("POST", f"{config.service_url('agent')}/plan", body, timeout_s=180)
        wall_s = time.perf_counter() - started
        attempts = response.get("attempts") or []
        last = attempts[-1] if attempts else {}
        rows.append(
            {
                "n": index + 1,
                "cold": index == 0,
                "http_status": status,
                "wall_s": round(wall_s, 4),
                "status": response.get("status"),
                "agent_mode": response.get("agent_mode"),
                "total_s": response.get("total_s"),
                "attempts": len(attempts),
                "repairs": response.get("repairs"),
                "timing": last.get("timing"),
                "plan_steps": (response.get("plan") or {}).get("steps"),
                "errors": response.get("errors"),
                "response_keys": sorted(response.keys()),
            }
        )
        print(f"[planner] {index + 1}/{reps} wall={wall_s:.3f}s status={response.get('status')}", flush=True)
    warm = [row for row in rows if not row["cold"]]
    attempt_timings = [row["timing"] for row in rows if isinstance(row.get("timing"), dict)]
    return {
        "reps": reps,
        "cold": rows[0] if rows else None,
        "wall_s": summarize([row["wall_s"] for row in rows]),
        "warm_wall_s": summarize([row["wall_s"] for row in warm]),
        "elapsed_s": summarize([row.get("elapsed_s") for row in attempt_timings]),
        "prompt_tokens": summarize([row.get("prompt_tokens") for row in attempt_timings]),
        "completion_tokens": summarize([row.get("completion_tokens") for row in attempt_timings]),
        "vllm_queue_s": summarize([row.get("vllm_queue_s") for row in attempt_timings]),
        "ttft_s": summarize([row.get("ttft_s") for row in attempt_timings]),
        "decode_s": summarize([row.get("decode_s") for row in attempt_timings]),
        "server_e2e_s": summarize([row.get("server_e2e_s") for row in attempt_timings]),
        "parse_s": summarize([row.get("parse_s") for row in attempt_timings]),
        "validate_s": summarize([row.get("validate_s") for row in attempt_timings]),
        "requests_observed": summarize([row.get("requests_observed") for row in attempt_timings]),
        "rows": rows,
    }


def openwam_bench(*, obs: dict[str, Any], prompt: str, actions: int) -> dict[str, Any]:
    _, reset_body = http("POST", f"{config.service_url('inference')}/reset", {"reason": "bench_speed"}, timeout_s=30)
    token = str(reset_body.get("token") or "")
    if not token:
        raise RuntimeError(f"inference reset did not return a token: {reset_body}")
    encoded_obs = numpy_json.encode(obs)
    rows = []
    for index in range(actions):
        request = {
            "token": token,
            "episode_id": "bench-speed",
            "scene_epoch": None,
            "observation_id": None,
            "sim_action_count": index,
            "skill": "whole_task",
            "step_id": index,
            "prompt": prompt,
            "created_at": time.time(),
        }
        started = time.perf_counter()
        status, response = http(
            "POST",
            f"{config.service_url('inference')}/infer",
            {"request": request, "obs": encoded_obs},
            timeout_s=300,
        )
        wall_s = time.perf_counter() - started
        chunk = response.get("chunk") or {}
        rows.append(
            {
                "n": index + 1,
                "http_status": status,
                "wall_s": round(wall_s, 4),
                "status": response.get("status"),
                "inference_s": response.get("inference_s"),
                "total_s": response.get("total_s"),
                "queued_s": response.get("queued_s"),
                "encode_s": response.get("encode_s"),
                "send_s": response.get("send_s"),
                "recv_s": response.get("recv_s"),
                "server_latency_ms": response.get("server_latency_ms"),
                "server_step": response.get("server_step"),
                "chunk_generation": chunk.get("chunk_generation"),
                "cache_hit": chunk.get("cache_hit"),
                "chunk_id": chunk.get("chunk_id"),
                "buffer_index": chunk.get("buffer_index"),
                "generated_actions": chunk.get("generated_actions"),
                "consumed_actions": chunk.get("consumed_actions"),
                "action_raw": response.get("action_raw"),
                "error": response.get("error"),
            }
        )
        if index % 16 == 0 or index == actions - 1:
            print(
                f"[openwam] {index + 1}/{actions} gen={chunk.get('chunk_generation')} "
                f"server={response.get('server_latency_ms')}ms wall={wall_s:.3f}s",
                flush=True,
            )
    generated = [row for row in rows if row["chunk_generation"] is True]
    cached = [row for row in rows if row["cache_hit"] is True]
    conversion = []
    sample = next((row["action_raw"] for row in rows if row.get("action_raw")), None)
    if sample:
        import numpy as np

        raw = np.asarray(sample, dtype=np.float32)
        for _ in range(2000):
            started = time.perf_counter()
            native_eef10_to_libero7d(raw)
            conversion.append((time.perf_counter() - started) * 1e6)
    return {
        "actions": actions,
        "chunk_generations": len(generated),
        "cache_hits": len(cached),
        "generation": {
            "server_latency_ms": summarize([row["server_latency_ms"] for row in generated]),
            "client_inference_s": summarize([row["inference_s"] for row in generated]),
            "engine_total_s": summarize([row["total_s"] for row in generated]),
        },
        "cache": {
            "server_latency_ms": summarize([row["server_latency_ms"] for row in cached]),
            "client_inference_s": summarize([row["inference_s"] for row in cached]),
            "engine_total_s": summarize([row["total_s"] for row in cached]),
        },
        "first_generation": generated[0] if generated else None,
        "encode_s": summarize([row["encode_s"] for row in rows]),
        "send_s": summarize([row["send_s"] for row in rows]),
        "recv_s": summarize([row["recv_s"] for row in rows]),
        "conversion_us": summarize(conversion),
        "rows": rows,
    }


def run_episode(*, task_id: int, instruction: str, timeout_s: float = 600.0) -> dict[str, Any]:
    status, loaded = http("POST", f"{config.service_url('engine')}/api/load", {"task_id": task_id}, timeout_s=180)
    if loaded.get("status") != "ok":
        raise RuntimeError(f"engine load failed: {status} {loaded}")
    status, started = http(
        "POST",
        f"{config.service_url('engine')}/api/run",
        {"task_id": task_id, "instruction": instruction},
        timeout_s=30,
    )
    if started.get("status") != "started":
        raise RuntimeError(f"engine run failed: {status} {started}")
    episode_id = started.get("episode_id")
    deadline = time.monotonic() + timeout_s
    state: dict[str, Any] = {}
    while time.monotonic() < deadline:
        state = get_json(f"{config.service_url('engine')}/api/status")
        if state.get("episode_id") != episode_id:
            time.sleep(0.5)
            continue
        if state.get("phase") in {"succeeded", "failed", "stopped"}:
            break
        time.sleep(0.5)
    record_path = (state.get("last_result") or {}).get("record")
    record = None
    if record_path and Path(record_path).exists():
        record = json.loads(Path(record_path).read_text(encoding="utf-8"))
    return {
        "episode_id": episode_id,
        "phase": state.get("phase"),
        "record_path": record_path,
        "record": record,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(PROJECT / "records" / "acceptance" / "speed_qwen_openwam.json"))
    parser.add_argument("--task", type=int, default=1)
    parser.add_argument("--planner-reps", type=int, default=12)
    parser.add_argument("--openwam-actions", type=int, default=96)
    parser.add_argument("--skip-planner", action="store_true")
    parser.add_argument("--skip-openwam", action="store_true")
    parser.add_argument("--skip-episode", action="store_true")
    parser.add_argument(
        "--planner-mode",
        choices=["qwen", "d1", "d1_with_qwen_fallback"],
        default=None,
        help="switch the Agent mode before the planner bench, then restore the previous mode",
    )
    args = parser.parse_args()

    report: dict[str, Any] = {
        "schema": "embodied_engine.speed.v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "task_id": args.task,
        "environment": environment(),
    }
    task = config.task_for(args.task)

    status, loaded = http("POST", f"{config.service_url('engine')}/api/load", {"task_id": args.task}, timeout_s=180)
    if loaded.get("status") != "ok":
        raise RuntimeError(f"engine load failed: {status} {loaded}")
    env = EnvClient(config.service_url("env"))
    snapshot = env.state_snapshot()
    raw = env.raw_obs()
    obs = build_openwam_obs(raw)
    report["scene"] = {
        "scene_epoch": snapshot.get("scene_epoch"),
        "observation_id": snapshot.get("observation_id"),
        "sim_action_count": snapshot.get("sim_action_count"),
        "eef_pos": (snapshot.get("sim") or {}).get("eef_pos"),
        "env_identity": env.identity(),
        "env_meta": env.meta(),
    }
    report["instruction"] = {
        "official": task["instruction"],
        "model_prompt": task["instruction"],
        "source": "config/tasks.json official instruction",
    }

    if not args.skip_planner:
        previous_mode = None
        if args.planner_mode:
            _, current = http("GET", f"{config.service_url('engine')}/api/agent_mode")
            previous_mode = current.get("mode")
            if previous_mode != args.planner_mode:
                status, switched = http(
                    "POST",
                    f"{config.service_url('engine')}/api/agent_mode",
                    {"mode": args.planner_mode},
                    timeout_s=30,
                )
                if switched.get("status") != "ok":
                    raise RuntimeError(f"agent mode switch failed: {status} {switched}")
        try:
            report["planner"] = planner_bench(
                instruction=task["instruction"],
                task_id=args.task,
                snapshot=snapshot,
                reps=args.planner_reps,
            )
        finally:
            if previous_mode and previous_mode != args.planner_mode:
                http(
                    "POST",
                    f"{config.service_url('engine')}/api/agent_mode",
                    {"mode": previous_mode},
                    timeout_s=30,
                )
        report["planner"]["mode"] = args.planner_mode or previous_mode
        report["planner"]["restored_mode"] = previous_mode
    if not args.skip_openwam:
        report["openwam"] = openwam_bench(
            obs=obs,
            prompt=task["instruction"],
            actions=args.openwam_actions,
        )
    if not args.skip_episode:
        episode = run_episode(task_id=args.task, instruction=task["instruction"])
        record = episode.pop("record") or {}
        episode["summary"] = {
            key: record.get(key)
            for key in (
                "duration_s",
                "execution_status",
                "native_task_success",
                "user_goal_success",
                "episode_sim_actions",
            )
        }
        episode["timing"] = record.get("timing")
        episode["openwam"] = record.get("openwam")
        episode["plan"] = {
            key: (record.get("plan") or {}).get(key)
            for key in ("source", "latency_s", "steps", "interpretation")
        }
        report["episode"] = episode

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
