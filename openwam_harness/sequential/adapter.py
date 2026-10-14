"""Thin WebSocket adapter. It speaks the existing Page UI frame protocol."""

from __future__ import annotations

import asyncio
import json
import os
import threading
from pathlib import Path
from queue import Empty, Queue

from aiohttp import web, WSMsgType

from executor import Session, run_plan
from planner import load_human_plan

ROOT = Path(__file__).resolve().parent
PAGE_JS = ROOT.parents[1] / "page_ui" / "static" / "app.js"
PLAN_PATH = ROOT / "plans" / "two_object.json"
HOST = os.environ.get("SEQUENTIAL_HOST", "127.0.0.1")
PORT = int(os.environ.get("SEQUENTIAL_PORT", "8766"))
LIVE = os.environ.get("HARNESS_LIVE", "0") == "1"
DIAGNOSTIC_S = float(os.environ.get("SEQUENTIAL_DIAGNOSTIC_S", "0"))

# 1x1 JPEG so the existing canvas path can decode a frame without a simulator.
TINY_JPEG = bytes.fromhex(
    "ffd8ffe000104a46494600010100000100010000ffdb004300080606070605080707070909080a0c140d0c0b0b0c1912130f141d1a1f1e1d1a1c1c20242e2720222c231c1c2837292c30313434341f27393d38323c2e333432ffc0000b080001000101011100ffc4001f0000010501010101010100000000000000000102030405060708090a0bffc400b5100002010303020403050504040000017d01020300041105122131410613516107227114328191a1082342b1c11552d1f02433627282090a161718191a25262728292a3435363738393a434445464748494a535455565758595a636465666768696a737475767778797a838485868788898a92939495969798999aa2a3a4a5a6a7a8a9aab2b3b4b5b6b7b8b9bac2c3c4c5c6c7c8c9cad2d3d4d5d6d7d8d9dae1e2e3e4e5e6e7e8e9eaf1f2f3f4f5f6f7f8f9faffda00080001000100003f00fbffd9"
)


def frame_bytes(header: dict, agent: bytes, wrist: bytes) -> bytes:
    header = dict(header)
    header["a_len"] = len(agent)
    header["w_len"] = len(wrist)
    payload = json.dumps(header, ensure_ascii=False).encode("utf-8")
    return len(payload).to_bytes(4, "big") + payload + agent + wrist


class DryBackend:
    """CPU stand-in. Live mode replaces this with the existing Harness tools."""

    def __init__(self) -> None:
        self.calls = []
        self._native = False
        self.fail_ids = set()

    def execute(self, action: str, args: dict):
        from executor import ToolOutcome

        self.calls.append((action, args))
        if action in self.fail_ids:
            return ToolOutcome(executed=False, error=f"{action} failed")
        xyz = [0.05, 0.22, 0.48] if action == "segment" else None
        grasp = True if action == "pi0_pick" else None
        return ToolOutcome(
            executed=True,
            grasp_signal=grasp,
            native_terminated=self._native,
            xyz=xyz,
            eef_pos=[0.0, 0.0, 0.62],
        )

    def observe(self) -> dict:
        return {"fresh": True, "llm": False}

    def native_success(self) -> bool:
        return self._native

    def frames(self):
        return TINY_JPEG, TINY_JPEG

    def close(self) -> None:
        return None


class App:
    def __init__(self) -> None:
        self.plans = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
        self.session = Session("none")
        self.backend = None
        self.confirm_q: Queue = Queue()
        self.clients = set()
        self.loop = None
        self.lock = threading.Lock()
        self.running = False
        self.step_detail = "未开始"

    def scenes(self) -> list[dict]:
        return [
            {"id": key, "label": value["task"], "instruction": value["task"]}
            for key, value in self.plans.items()
        ]

    def broadcast(self, message: dict) -> None:
        if self.loop is None:
            return
        data = json.dumps(message, ensure_ascii=False)
        for ws in list(self.clients):
            asyncio.run_coroutine_threadsafe(ws.send_str(data), self.loop)

    def broadcast_frame(self, header: dict) -> None:
        agent, wrist = (TINY_JPEG, TINY_JPEG)
        if self.backend is not None:
            got = self.backend.frames()
            agent = got[0] or TINY_JPEG
            wrist = got[1] or TINY_JPEG
            if agent[:2] != b"\xff\xd8":
                agent = _png_to_jpeg(agent)
            if wrist[:2] != b"\xff\xd8":
                wrist = _png_to_jpeg(wrist)
        payload = frame_bytes(header, agent, wrist)
        if self.loop is None:
            return
        for ws in list(self.clients):
            asyncio.run_coroutine_threadsafe(ws.send_bytes(payload), self.loop)

    def load(self, scene_id: str) -> None:
        plan = self.plans.get(scene_id)
        if plan is None:
            self.broadcast({"type": "load_failed", "detail": f"unknown scene {scene_id}"})
            return
        with self.lock:
            self.session.cancel()
            self._wake()
            self.session = Session(scene_id)
            episode = self.session.begin_episode()
            self._close_backend()
            if LIVE:
                from live import LiveBackend

                out = ROOT.parents[0] / "runs" / "sequential" / episode
                self.backend = LiveBackend(plan, out)
            else:
                self.backend = DryBackend()
        self.broadcast(
            {
                "type": "loaded",
                "instruction": plan["task"],
                "env_id": scene_id,
                "episode_id": episode,
                "scene_epoch": self.session.scene_epoch,
            }
        )
        self.broadcast_frame(
            {
                "eid": episode,
                "fid": 0,
                "phase": "场景已加载",
                "detail": scene_id,
                "term": False,
                "success": False,
                "reason": "",
            }
        )

    def start(self, kind: str) -> None:
        if self.session.scene_id == "none" or self.backend is None:
            self.broadcast({"type": "load_failed", "detail": "load a scene first"})
            return
        if kind == "restart":
            self.session.cancel()
            self._wake()
            episode = self.session.begin_episode()
            self._close_backend()
            plan = self.plans[self.session.scene_id]
            if LIVE:
                from live import LiveBackend

                out = ROOT.parents[0] / "runs" / "sequential" / episode
                self.backend = LiveBackend(plan, out)
            else:
                self.backend = DryBackend()
        else:
            episode = self.session.episode_id
        self.broadcast({"type": "started", "episode_id": episode})
        threading.Thread(target=self._run, args=(episode, self.session.scene_epoch), daemon=True).start()

    def stop(self) -> None:
        self.session.cancel()
        self._wake()
        self.broadcast(
            {
                "type": "ended",
                "episode_id": self.session.episode_id,
                "success": False,
                "reason": "stopped",
            }
        )

    def _run(self, episode: str, epoch: int) -> None:
        plan = load_human_plan(PLAN_PATH, self.session.scene_id)
        deadline = None
        if DIAGNOSTIC_S > 0:
            import time

            deadline = time.monotonic() + DIAGNOSTIC_S

        def confirm(subtask: str):
            self.step_detail = f"等待确认 {subtask}"
            self.broadcast_frame(
                {
                    "eid": episode,
                    "fid": len(getattr(self.backend, "calls", [])),
                    "phase": "等待确认",
                    "detail": subtask,
                    "term": False,
                    "success": False,
                    "reason": "subtask unknown until confirmed",
                }
            )
            with self.lock:
                queued = self.confirm_q
            verdict = queued.get()
            if self.session.scene_epoch != epoch or self.session.cancelled:
                return None
            return verdict

        result = run_plan(
            plan,
            self.backend,
            self.session,
            confirm,
            epoch=epoch,
            deadline_s=deadline,
        )
        if result["episode_id"] != episode or result["scene_epoch"] != epoch:
            return
        if result["status"] == "discarded":
            return
        native = result["native_success"]
        self.broadcast_frame(
            {
                "eid": episode,
                "fid": 9999,
                "phase": "结果",
                "detail": result["status"],
                "term": True,
                "success": bool(native),
                "reason": result["reason"] or result["status"],
            }
        )
        self.broadcast(
            {
                "type": "ended",
                "episode_id": episode,
                "success": bool(native),
                "reason": result["reason"] or result["status"],
                "native_success": native,
            }
        )

    def _wake(self) -> None:
        queued = self.confirm_q
        self.confirm_q = Queue()
        queued.put(None)

    def _close_backend(self) -> None:
        if self.backend is not None:
            self.backend.close()
            self.backend = None


def _png_to_jpeg(raw: bytes) -> bytes:
    import io

    from PIL import Image

    image = Image.open(io.BytesIO(raw)).convert("RGB")
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


APP = App()


async def index(_request):
    return web.FileResponse(ROOT / "static" / "index.html")


async def app_js(_request):
    return web.FileResponse(PAGE_JS)


async def hook_js(_request):
    return web.FileResponse(ROOT / "static" / "hook.js")


async def websocket(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    APP.clients.add(ws)
    APP.loop = asyncio.get_running_loop()
    try:
        async for msg in ws:
            if msg.type != WSMsgType.TEXT:
                continue
            data = json.loads(msg.data)
            kind = data.get("type")
            if kind == "get_scenes":
                await ws.send_str(json.dumps({"type": "scenes", "scenes": APP.scenes()}))
            elif kind == "ping":
                await ws.send_str(json.dumps({"type": "pong"}))
            elif kind == "load":
                await asyncio.to_thread(APP.load, data.get("scene"))
            elif kind in {"start", "restart"}:
                await asyncio.to_thread(APP.start, kind)
            elif kind == "stop":
                APP.stop()
            elif kind == "confirm":
                APP.confirm_q.put(data.get("verdict"))
            elif kind == "client_perf":
                continue
    finally:
        APP.clients.discard(ws)
    return ws


def main() -> None:
    app = web.Application()
    app.router.add_get("/", index)
    app.router.add_get("/static/app.js", app_js)
    app.router.add_get("/static/hook.js", hook_js)
    app.router.add_get("/ws", websocket)
    web.run_app(app, host=HOST, port=PORT)


if __name__ == "__main__":
    main()
