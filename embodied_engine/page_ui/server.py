#!/usr/bin/env python
"""Page UI backend bridge for the three-layer embodied engine.

This is the restored historical Page frontend (layout, dual Agent/Wrist
cameras and WebSocket protocol) with its ManiSkill backend replaced by a thin
bridge to the engine on 14800. The browser protocol is unchanged:
text messages ``get_scenes`` / ``load`` / ``start`` / ``restart`` and binary
frames carrying a JSON header plus two JPEGs. ``stop`` / ``reset`` and an
identity message are additive.

The bridge never drives the environment itself. All load/run/stop/reset
calls go to the engine, which owns the LIBERO env, the Agent service, and the
Pi0.5 inference service.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import sys
import time
from pathlib import Path

from aiohttp import ClientSession, ClientTimeout, WSMsgType, web

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from common import config  # noqa: E402

LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [page] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "page.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("page")

CFG = config.engine_config()
ENGINE = config.service_url("engine")
POLL_S = 0.25
ACTIVE_PHASES = {"planning", "running", "recovering", "verifying", "stopping"}
TERMINAL_PHASES = {"succeeded", "failed", "stopped"}
SCENES: list[dict] = []
TASK_BY_SCENE: dict[str, dict] = {}
CURRENT_TASK: dict | None = None
SESSION: ClientSession | None = None


def build_frame_bytes(hdr: dict, agent_jpg: bytes, wrist_jpg: bytes) -> bytes:
    hdr = dict(hdr)
    hdr["a_len"] = len(agent_jpg)
    hdr["w_len"] = len(wrist_jpg)
    hdr_bytes = json.dumps(hdr, ensure_ascii=False).encode("utf-8")
    out = bytearray()
    out += len(hdr_bytes).to_bytes(4, "big")
    out += hdr_bytes
    out += agent_jpg
    out += wrist_jpg
    return bytes(out)


async def engine_get(path: str, timeout: float = 8.0) -> dict:
    assert SESSION is not None
    async with SESSION.get(ENGINE + path, timeout=ClientTimeout(total=timeout)) as response:
        response.raise_for_status()
        return await response.json()


async def engine_post(path: str, body: dict, timeout: float = 60.0) -> dict:
    assert SESSION is not None
    async with SESSION.post(ENGINE + path, json=body, timeout=ClientTimeout(total=timeout)) as response:
        try:
            payload = await response.json(content_type=None)
        except Exception:
            payload = {"status": "error", "error": f"HTTP {response.status}"}
        if response.status >= 400 and payload.get("status") not in {"error", "bad_request"}:
            payload = {"status": "error", "error": f"HTTP {response.status}"}
        return payload


async def engine_frame(view: str) -> bytes | None:
    assert SESSION is not None
    async with SESSION.get(
        ENGINE + f"/api/frame?view={view}", timeout=ClientTimeout(total=8)
    ) as response:
        if response.status != 200:
            return None
        return await response.read()


def identity_message(status: dict) -> dict:
    identity = status.get("identity") or {}
    agent = identity.get("agent") or {}
    action = identity.get("action_model") or {}
    inference = identity.get("inference") or {}
    openwam = action.get("openwam") or {}
    plan = status.get("plan") or {}
    return {
        "type": "identity",
        "agent_model": agent.get("model") or "unavailable",
        "agent_status": agent.get("status"),
        "agent_mode": agent.get("mode") or "semantic_router",
        "agent_router": agent.get("routing") or {},
        "action_model": action.get("name") or "unavailable",
        "action_backend": action.get("backend"),
        "world_model": openwam.get("name") or "not configured",
        "world_model_status": openwam.get("integration") or "unavailable",
        "inference_mode": inference.get("mode") or "unknown",
        "inference_backend": (inference.get("identity") or {}).get("backend") or "unknown",
        "phase": status.get("phase"),
        "failure_reason": status.get("failure_reason") or "",
        "raw_instruction": status.get("instruction") or "",
        "understanding": plan.get("understanding") or {},
    }


class Client:
    MAX_Q = 3

    def __init__(self, ws: web.WebSocketResponse):
        self.ws = ws
        self.q: asyncio.Queue = asyncio.Queue(maxsize=self.MAX_Q)
        self.sender_task: asyncio.Task | None = None
        self.dropped = 0

    async def sender(self) -> None:
        try:
            while True:
                kind, data = await self.q.get()
                if kind == "close":
                    return
                if kind == "text":
                    await self.ws.send_str(data)
                else:
                    await self.ws.send_bytes(data)
        except (asyncio.CancelledError, RuntimeError):
            return

    def enqueue_text(self, obj: dict) -> None:
        try:
            self.q.put_nowait(("text", json.dumps(obj, ensure_ascii=False)))
        except asyncio.QueueFull:
            self.dropped += 1

    def enqueue_frame(self, data: bytes) -> None:
        while self.q.full():
            try:
                self.q.get_nowait()
                self.dropped += 1
            except asyncio.QueueEmpty:
                break
        try:
            self.q.put_nowait(("bin", data))
        except asyncio.QueueFull:
            self.dropped += 1


class Hub:
    def __init__(self) -> None:
        self.clients: set[Client] = set()
        self._lock = asyncio.Lock()

    async def add(self, client: Client) -> None:
        async with self._lock:
            self.clients.add(client)

    async def remove(self, client: Client) -> None:
        async with self._lock:
            self.clients.discard(client)

    def _snapshot(self) -> list[Client]:
        return list(self.clients)

    async def text(self, obj: dict) -> None:
        for client in self._snapshot():
            client.enqueue_text(obj)

    async def frame(self, data: bytes) -> None:
        for client in self._snapshot():
            client.enqueue_frame(data)


class Monitor:
    """Single poller: engine status -> page text/binary frames."""

    def __init__(self, hub: Hub):
        self.hub = hub
        self.last_phase: str | None = None
        self.last_identity: str | None = None
        self.last_frame_key: tuple | None = None
        self.last_episode: str | None = None
        self.last_announced: set[str] = set()
        self.last_error_at = 0.0

    async def run(self) -> None:
        while True:
            try:
                status = await engine_get("/api/status", timeout=10)
                await self.tick(status)
            except Exception as exc:  # engine down or restarting
                now = time.time()
                if now - self.last_error_at > 5:
                    self.last_error_at = now
                    await self.hub.text(
                        {
                            "type": "state",
                            "phase": "引擎不可达",
                            "detail": f"{type(exc).__name__}: {exc}",
                            "cls": "bad",
                        }
                    )
            await asyncio.sleep(POLL_S)

    async def tick(self, status: dict) -> None:
        phase = str(status.get("phase") or "")
        eid = status.get("episode_id")
        identity = identity_message(status)
        key = json.dumps(identity, sort_keys=True)
        if key != self.last_identity:
            self.last_identity = key
            await self.hub.text(identity)

        if phase != self.last_phase:
            cls = "good" if phase == "succeeded" else "bad" if phase in {"failed"} else "warn" if phase in ACTIVE_PHASES else ""
            await self.hub.text(
                {
                    "type": "state",
                    "phase": phase,
                    "detail": status.get("failure_reason") or status.get("detail") or "",
                    "cls": cls,
                }
            )
            self.last_phase = phase

        frame_seq = status.get("frame_seq")
        terminal = phase in TERMINAL_PHASES
        active = phase in ACTIVE_PHASES
        display_eid = eid or (f"scene-{status.get('task_id')}" if status.get("task_id") is not None else None)
        if display_eid is None:
            return
        frame_key = (display_eid, frame_seq, phase if terminal else None)
        if frame_key == self.last_frame_key:
            return
        if not active and not terminal and phase not in {"ready", "loading"}:
            return

        agent_jpg, wrist_jpg = await asyncio.gather(
            engine_frame("agentview"), engine_frame("wrist")
        )
        if agent_jpg is None or wrist_jpg is None:
            return
        reason = ""
        if terminal:
            reason = status.get("failure_reason") or status.get("detail") or (
                "task completed" if phase == "succeeded" else "episode ended"
            )
        hdr = {
            "eid": display_eid,
            "fid": int(status.get("frame_seq") or 0),
            "phase": phase,
            "detail": status.get("detail") or "",
            "term": bool(terminal),
            "success": phase == "succeeded",
            "reason": reason,
            "aw": 0,
            "ah": 0,
            "ww": 0,
            "wh": 0,
            "sim_action_count": status.get("sim_action_count"),
            "scene_epoch": status.get("scene_epoch"),
        }
        await self.hub.frame(build_frame_bytes(hdr, agent_jpg, wrist_jpg))
        self.last_frame_key = frame_key

        if terminal and eid and eid not in self.last_announced:
            self.last_announced.add(eid)
            await self.hub.text(
                {
                    "type": "ended",
                    "episode_id": eid,
                    "success": phase == "succeeded",
                    "reason": reason,
                    "failure_code": status.get("failure_code"),
                    "duration_s": (status.get("last_result") or {}).get("duration_s"),
                }
            )


HUB: Hub | None = None
MONITOR: Monitor | None = None
LOAD_LOCK = asyncio.Lock()


async def refresh_scenes() -> list[dict]:
    global SCENES, TASK_BY_SCENE
    payload = await engine_get("/api/tasks", timeout=10)
    rows = payload.get("scenes") or []
    SCENES = [
        {
            "id": row["id"],
            "label": row["label"],
            "instruction": row.get("instruction") or "",
            "task_id": row.get("task_id"),
        }
        for row in rows
    ]
    TASK_BY_SCENE = {row["id"]: row for row in SCENES}
    return SCENES


async def do_load(scene_id: str) -> dict:
    global CURRENT_TASK
    row = TASK_BY_SCENE.get(scene_id)
    if row is None:
        await refresh_scenes()
        row = TASK_BY_SCENE.get(scene_id)
    if row is None:
        return {"status": "error", "error": f"unknown scene {scene_id}"}
    async with LOAD_LOCK:
        out = await engine_post("/api/load", {"task_id": int(row["task_id"])}, timeout=300)
        if out.get("status") == "ok":
            CURRENT_TASK = row
        return out


async def do_start(instruction: str) -> dict:
    if CURRENT_TASK is None:
        return {"status": "error", "error": "load a scene first"}
    body = {"task_id": int(CURRENT_TASK["task_id"])}
    if instruction:
        body["instruction"] = instruction
    return await engine_post("/api/run", body, timeout=30)


async def do_restart(instruction: str) -> None:
    await engine_post("/api/stop", {"reason": "page_restart"}, timeout=30)
    for _ in range(40):
        status = await engine_get("/api/status", timeout=8)
        if status.get("phase") in {"ready", "succeeded", "failed", "stopped"}:
            break
        await asyncio.sleep(0.25)
    out = await do_start(instruction)
    if out.get("status") == "started":
        await HUB.text({"type": "started", "episode_id": out.get("episode_id")})
    else:
        await HUB.text({"type": "state", "phase": "启动失败", "detail": out.get("error") or "run rejected", "cls": "bad"})


async def ws_handler(request: web.Request) -> web.WebSocketResponse:
    ws = web.WebSocketResponse(max_msg_size=0)
    await ws.prepare(request)
    client = Client(ws)
    client.sender_task = asyncio.create_task(client.sender())
    await HUB.add(client)
    try:
        client.enqueue_text(identity_message(await engine_get("/api/status", timeout=10)))
        client.enqueue_text({"type": "scenes", "scenes": SCENES})
    except Exception as exc:
        client.enqueue_text({"type": "state", "phase": "引擎不可达", "detail": str(exc), "cls": "bad"})
    try:
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                try:
                    payload = json.loads(msg.data)
                except json.JSONDecodeError:
                    continue
                await handle_msg(payload, client)
            elif msg.type == WSMsgType.ERROR:
                log.warning("ws error: %s", ws.exception())
    finally:
        if client.sender_task:
            client.sender_task.cancel()
        await HUB.remove(client)
    return ws


async def handle_msg(msg: dict, client: Client) -> None:
    kind = msg.get("type")
    if kind == "get_scenes":
        try:
            client.enqueue_text({"type": "scenes", "scenes": await refresh_scenes()})
        except Exception as exc:
            client.enqueue_text({"type": "state", "phase": "引擎不可达", "detail": str(exc), "cls": "bad"})
    elif kind == "load":
        await HUB.text({"type": "state", "phase": "加载中", "detail": str(msg.get("scene") or ""), "cls": "warn"})
        try:
            out = await do_load(str(msg.get("scene") or ""))
        except Exception as exc:
            out = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
        if out.get("status") == "ok":
            instruction = out.get("instruction") or (CURRENT_TASK or {}).get("instruction") or ""
            await HUB.text({"type": "loaded", "env_id": out.get("env_id"), "instruction": instruction})
            try:
                await HUB.text(identity_message(await engine_get("/api/status", timeout=10)))
            except Exception:
                pass
        else:
            await HUB.text({"type": "load_failed", "detail": out.get("error") or str(out)})
    elif kind == "start":
        out = await do_start(str(msg.get("task") or "").strip())
        if out.get("status") == "started":
            await HUB.text({"type": "started", "episode_id": out.get("episode_id")})
        else:
            await HUB.text({"type": "state", "phase": "启动失败", "detail": out.get("error") or str(out), "cls": "bad"})
    elif kind == "restart":
        await HUB.text({"type": "state", "phase": "重置中", "detail": "停止当前 episode 后重新开始", "cls": "warn"})
        asyncio.create_task(do_restart(str(msg.get("task") or "").strip()))
    elif kind == "stop":
        out = await engine_post("/api/stop", {"reason": "page_stop"}, timeout=30)
        await HUB.text({"type": "state", "phase": "停止中", "detail": out.get("phase") or "", "cls": "warn"})
    elif kind == "reset":
        out = await engine_post("/api/reset", {}, timeout=120)
        if out.get("status") == "ok":
            instruction = (CURRENT_TASK or {}).get("instruction") or ""
            env_id = (CURRENT_TASK or {}).get("bddl") or ""
            await HUB.text({"type": "loaded", "env_id": env_id, "instruction": instruction})
            await HUB.text({"type": "state", "phase": "已重置", "detail": f"task {out.get('scene_epoch')}", "cls": "good"})
        else:
            await HUB.text({"type": "state", "phase": "重置失败", "detail": out.get("error") or str(out), "cls": "bad"})
    elif kind == "get_status":
        client.enqueue_text(identity_message(await engine_get("/api/status", timeout=10)))
    elif kind == "client_perf":
        log.info(
            "perf client decode=%.0fms draw=%.0fms gap=%.0fms",
            float(msg.get("decode_ms") or 0),
            float(msg.get("draw_ms") or 0),
            float(msg.get("gap_ms") or 0),
        )
    elif kind == "ping":
        client.enqueue_text({"type": "pong"})


async def health(request: web.Request) -> web.Response:
    try:
        status = await engine_get("/api/status", timeout=8)
        return web.json_response({"status": "ok", "page": "page_ui", "engine_phase": status.get("phase"), "engine": ENGINE})
    except Exception as exc:
        return web.json_response({"status": "degraded", "page": "page_ui", "engine": ENGINE, "error": str(exc)})


async def index(request: web.Request) -> web.StreamResponse:
    return web.FileResponse(Path(__file__).resolve().parent / "static" / "index.html")


async def on_startup(app: web.Application) -> None:
    global SESSION, HUB, MONITOR
    SESSION = ClientSession(trust_env=False)
    HUB = Hub()
    MONITOR = Monitor(HUB)
    try:
        await refresh_scenes()
    except Exception as exc:
        log.warning("initial scene refresh failed: %s", exc)
    app["monitor_task"] = asyncio.create_task(MONITOR.run())


async def on_cleanup(app: web.Application) -> None:
    task = app.get("monitor_task")
    if task:
        task.cancel()
    if SESSION:
        await SESSION.close()


def make_app() -> web.Application:
    app = web.Application(client_max_size=0)
    app.router.add_get("/", index)
    app.router.add_get("/health", health)
    app.router.add_static("/static", Path(__file__).resolve().parent / "static")
    app.router.add_get("/ws", ws_handler)
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    return app


def main() -> None:
    page = CFG["services"]["page_ui"]
    host = os.environ.get("PAGE_HOST", page["host"])
    port = int(os.environ.get("PAGE_PORT", page["port"]))
    log.info("starting page bridge on %s:%d engine=%s", host, port, ENGINE)
    web.run_app(make_app(), host=host, port=port, access_log=None)


if __name__ == "__main__":
    main()
