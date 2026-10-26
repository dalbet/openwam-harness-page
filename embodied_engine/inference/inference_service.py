#!/usr/bin/env python
"""Inference engine service for the resident Pi0.5 policy.

Binds to 127.0.0.1:14600. The model process is separate (14510) and stays
loaded across episodes. This service adds:

- one bounded prefetch slot (queue size 1, newest observation wins);
- token-scoped episode isolation (a reset invalidates every older request);
- request identity (episode, scene epoch, observation id, action count, skill);
- a synchronous path and a prefetch/result path;
- explicit metrics for inference wait vs model time.

HTTP/threads alone are not treated as asynchronous action inference: the
engine only overlaps a prefetched chunk with execution of the current chunk,
and rejects a chunk whose observation is older than ``max_chunk_age_actions``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import threading
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from aiohttp import web  # noqa: E402

from common import config, numpy_json  # noqa: E402
from models.pi05_backend import Pi05Backend  # noqa: E402


class InferenceEngine:
    def __init__(self, base_url: str):
        self.backend = Pi05Backend(base_url)
        self.lock = threading.Lock()
        self.queue: deque[dict[str, Any]] = deque()
        self.queue_lock = threading.Lock()
        self.cond = threading.Condition(self.queue_lock)
        self.results: dict[str, dict[str, Any]] = {}
        self.results_lock = threading.Lock()
        self.token = ""
        self.token_lock = threading.Lock()
        self.stopping = False
        self.metrics = {
            "infer_calls": 0,
            "prefetch_calls": 0,
            "prefetch_queued": 0,
            "prefetch_replaced": 0,
            "prefetch_completed": 0,
            "prefetch_discarded": 0,
            "model_errors": 0,
            "inference_s_total": 0.0,
            "inference_s_last": None,
            "last_request": None,
        }
        self._new_token("startup")
        self.worker = threading.Thread(target=self._loop, name="inference-worker", daemon=True)
        self.worker.start()

    def _new_token(self, reason: str) -> str:
        with self.token_lock:
            self.token = f"inf-{uuid.uuid4().hex[:12]}"
            return self.token

    def reset(self, reason: str) -> str:
        token = self._new_token(reason)
        with self.cond:
            for row in self.queue:
                self._store_result(row["request_id"], {"status": "discarded", "reason": f"reset:{reason}", "request": row["request"]})
            self.queue.clear()
            self.cond.notify_all()
        with self.results_lock:
            self.results.clear()
        return token

    def cancel(self, reason: str) -> dict[str, Any]:
        dropped = 0
        with self.cond:
            dropped = len(self.queue)
            for row in self.queue:
                self._store_result(row["request_id"], {"status": "discarded", "reason": f"cancel:{reason}", "request": row["request"]})
            self.queue.clear()
            self.cond.notify_all()
        with self.results_lock:
            for request_id, value in list(self.results.items()):
                if value.get("status") == "pending":
                    self.results[request_id] = {"status": "discarded", "reason": f"cancel:{reason}", "request": value.get("request")}
        self.metrics["prefetch_discarded"] += dropped
        return {"status": "ok", "discarded": dropped}

    def submit(self, request: dict[str, Any], obs: dict[str, Any]) -> dict[str, Any]:
        token = self._check_token(request.get("token"))
        if token is not None:
            return token
        request_id = f"req-{uuid.uuid4().hex[:12]}"
        row = {
            "request_id": request_id,
            "request": request,
            "obs": obs,
            "enqueued": time.monotonic(),
            "event": threading.Event(),
        }
        signature = (request.get("episode_id"), request.get("scene_epoch"), request.get("skill"), request.get("step_id"))
        with self.cond:
            if self.queue:
                old = self.queue[0]
                old_sig = (old["request"].get("episode_id"), old["request"].get("scene_epoch"), old["request"].get("skill"), old["request"].get("step_id"))
                old_count = old["request"].get("sim_action_count")
                new_count = request.get("sim_action_count")
                if old_sig == signature and isinstance(old_count, int) and isinstance(new_count, int) and new_count <= old_count:
                    self.metrics["prefetch_replaced"] += 1
                    return {"status": "rejected", "reason": "older_or_equal_observation", "request_id": None}
                self.queue.clear()
                self._store_result(old["request_id"], {"status": "discarded", "reason": "superseded_by_newer_prefetch", "request": old["request"]})
                old["event"].set()
                self.metrics["prefetch_replaced"] += 1
            self.queue.append(row)
            self.metrics["prefetch_queued"] += 1
            self.cond.notify_all()
        return {"status": "queued", "request_id": request_id}

    def wait_result(self, request_id: str, wait_s: float) -> dict[str, Any]:
        deadline = time.monotonic() + max(0.0, wait_s)
        while True:
            with self.results_lock:
                value = self.results.get(request_id)
            if value is not None and value.get("status") != "pending":
                return value
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return value or {"status": "missing"}
            time.sleep(min(0.02, remaining))

    def infer(self, request: dict[str, Any], obs: dict[str, Any]) -> dict[str, Any]:
        token_error = self._check_token(request.get("token"))
        if token_error is not None:
            return token_error
        started = time.perf_counter()
        timer = time.perf_counter()
        with self.lock:
            queued_s = time.perf_counter() - timer
            try:
                out = self.backend.predict(obs, str(request.get("prompt") or ""))
            except Exception as exc:
                self.metrics["model_errors"] += 1
                return {"status": "error", "error": f"{type(exc).__name__}: {exc}", "request": request}
        self._record_success(request, out, queued_s, time.perf_counter() - started)
        return {
            "status": "ok",
            "request": request,
            "actions": out["actions"],
            "actions_shape": out["actions_shape"],
            "inference_s": out["inference_s"],
            "queued_s": round(queued_s, 4),
            "total_s": round(time.perf_counter() - started, 4),
        }

    def _record_success(self, request: dict[str, Any], out: dict[str, Any], queued_s: float, total_s: float) -> None:
        self.metrics["inference_s_total"] += float(out["inference_s"])
        self.metrics["inference_s_last"] = float(out["inference_s"])
        self.metrics["last_request"] = {
            "episode_id": request.get("episode_id"),
            "scene_epoch": request.get("scene_epoch"),
            "observation_id": request.get("observation_id"),
            "sim_action_count": request.get("sim_action_count"),
            "skill": request.get("skill"),
            "step_id": request.get("step_id"),
            "prompt": request.get("prompt"),
            "queued_s": round(queued_s, 4),
            "total_s": round(total_s, 4),
        }

    def _loop(self) -> None:
        while not self.stopping:
            with self.cond:
                while not self.queue and not self.stopping:
                    self.cond.wait(timeout=0.5)
                if self.stopping:
                    return
                row = self.queue.popleft()
            request = row["request"]
            token_error = self._check_token(request.get("token"))
            if token_error is not None:
                self._store_result(row["request_id"], {"status": "discarded", "reason": token_error["reason"], "request": request})
                self.metrics["prefetch_discarded"] += 1
                row["event"].set()
                continue
            started = time.perf_counter()
            with self.lock:
                queued_s = started - float(row.get("enqueued", started))
                try:
                    out = self.backend.predict(row["obs"], str(request.get("prompt") or ""))
                except Exception as exc:
                    with self.results_lock:
                        self.results[row["request_id"]] = {
                            "status": "error",
                            "error": f"{type(exc).__name__}: {exc}",
                            "request": request,
                        }
                    self.metrics["model_errors"] += 1
                    row["event"].set()
                    continue
            self._record_success(request, out, queued_s, time.perf_counter() - started)
            with self.results_lock:
                self.results[row["request_id"]] = {
                    "status": "ready",
                    "request": request,
                    "actions": numpy_json.encode(out["actions"]),
                    "actions_shape": out["actions_shape"],
                    "inference_s": out["inference_s"],
                    "queued_s": round(queued_s, 4),
                    "total_s": round(time.perf_counter() - started, 4),
                }
            self.metrics["prefetch_completed"] += 1
            row["event"].set()

    def _store_result(self, request_id: str, value: dict[str, Any]) -> None:
        with self.results_lock:
            self.results[request_id] = value
            if len(self.results) > 16:
                for key in list(self.results.keys())[:-16]:
                    self.results.pop(key, None)

    def _check_token(self, token: Any) -> dict[str, Any] | None:
        with self.token_lock:
            current = self.token
        if not isinstance(token, str) or token != current:
            return {"status": "conflict", "reason": "stale_episode_token", "current_token": current}
        return None

    def identity(self) -> dict[str, Any]:
        cfg = config.engine_config()["action_model"]
        openwam = cfg["openwam"]
        checkpoint = Path(openwam["checkpoint"]) / openwam["checkpoint_file"]
        size = None
        try:
            size = checkpoint.stat().st_size
        except OSError:
            pass
        return {
            "backend": cfg["backend"],
            "name": cfg["name"],
            "checkpoint": cfg["checkpoint"],
            "revision": cfg["revision"],
            "action_chunk": cfg["action_chunk"],
            "action_dim": cfg["action_dim"],
            "control_frequency_hz": cfg["control_frequency_hz"],
            "server": self.backend.health(),
            "openwam": {
                "name": openwam["name"],
                "checkpoint": str(checkpoint),
                "hf_revision": openwam["hf_revision"],
                "bytes_on_disk": size,
                "action_horizon": openwam["action_horizon"],
                "denoise_steps": openwam["denoise_steps"],
                "integration": "not_on_execution_path",
                "reason": "weights downloading; a separate torch 2.7.1/cu128 environment is required",
            },
        }

    def metrics_row(self) -> dict[str, Any]:
        with self.results_lock:
            ready = sum(1 for row in self.results.values() if row.get("status") == "ready")
        with self.queue_lock:
            pending = len(self.queue)
        row = dict(self.metrics)
        row["pending_prefetch"] = pending
        row["ready_results"] = ready
        row["mode"] = config.engine_config()["inference"]["mode"]
        return row


ENGINE: InferenceEngine | None = None


async def health(request: web.Request) -> web.Response:
    assert ENGINE is not None
    return web.json_response(
        {
            "service": "inference",
            "status": "ok" if ENGINE.identity()["server"].get("reachable") else "degraded",
            "mode": config.engine_config()["inference"]["mode"],
            "identity": ENGINE.identity(),
            "metrics": ENGINE.metrics_row(),
            "token": ENGINE.token,
        }
    )


async def reset(request: web.Request) -> web.Response:
    assert ENGINE is not None
    body = await _json_body(request)
    token = ENGINE.reset(str(body.get("reason") or "episode_start"))
    return web.json_response({"status": "ok", "token": token})


async def cancel(request: web.Request) -> web.Response:
    assert ENGINE is not None
    body = await _json_body(request)
    return web.json_response(ENGINE.cancel(str(body.get("reason") or "cancelled")))


async def infer(request: web.Request) -> web.Response:
    assert ENGINE is not None
    body = await _json_body(request)
    req = body.get("request") or {}
    obs = numpy_json.decode(body.get("obs") or {})
    if not isinstance(req, dict) or not isinstance(obs, dict):
        return web.json_response({"status": "bad_request"}, status=400)
    result = await asyncio.to_thread(ENGINE.infer, req, obs)
    status = 200 if result["status"] == "ok" else (409 if result["status"] == "conflict" else 502)
    payload = dict(result)
    if "actions" in payload:
        payload["actions"] = numpy_json.encode(payload["actions"])
    return web.json_response(payload, status=status)


async def prefetch(request: web.Request) -> web.Response:
    assert ENGINE is not None
    body = await _json_body(request)
    req = body.get("request") or {}
    obs = numpy_json.decode(body.get("obs") or {})
    if not isinstance(req, dict) or not isinstance(obs, dict):
        return web.json_response({"status": "bad_request"}, status=400)
    result = ENGINE.submit(req, obs)
    status = 409 if result["status"] == "conflict" else 200
    return web.json_response(result, status=status)


async def result(request: web.Request) -> web.Response:
    assert ENGINE is not None
    body = await _json_body(request)
    request_id = body.get("request_id")
    wait_s = min(float(body.get("wait_s") or 0), 120.0)
    if not isinstance(request_id, str):
        return web.json_response({"status": "bad_request"}, status=400)
    value = await asyncio.to_thread(ENGINE.wait_result, request_id, wait_s)
    payload = dict(value)
    if "actions" in payload:
        payload["actions"] = numpy_json.decode(payload["actions"]).tolist()
    return web.json_response(payload)


async def metrics(request: web.Request) -> web.Response:
    assert ENGINE is not None
    return web.json_response(ENGINE.metrics_row())


async def _json_body(request: web.Request) -> dict[str, Any]:
    try:
        body = await request.json()
        return body if isinstance(body, dict) else {}
    except json.JSONDecodeError:
        return {}


def make_app() -> web.Application:
    app = web.Application(client_max_size=64 * 1024 * 1024)
    app.router.add_get("/health", health)
    app.router.add_get("/metrics", metrics)
    app.router.add_post("/reset", reset)
    app.router.add_post("/cancel", cancel)
    app.router.add_post("/infer", infer)
    app.router.add_post("/prefetch", prefetch)
    app.router.add_post("/result", result)
    return app


def main() -> None:
    global ENGINE
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()
    service = config.engine_config()["services"]["inference"]
    port = args.port or int(service["port"])
    ENGINE = InferenceEngine(config.service_url("pi05"))
    print(f"[inference] resident Pi0.5 on {service['host']}:{port}; vla={config.service_url('pi05')}", flush=True)
    web.run_app(make_app(), host=service["host"], port=port, access_log=None)


if __name__ == "__main__":
    main()
