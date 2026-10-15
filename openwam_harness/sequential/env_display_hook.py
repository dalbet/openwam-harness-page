"""Display snapshots from the environment thread during an action step.

The control step still returns only its own observation. A throttled 512 render
is copied here and encoded on another thread. A slow receiver does not call
``env.step`` again and does not render concurrently.
"""

from __future__ import annotations

import json
import os
import queue
import socket
import threading
import time

DISPLAY_EDGE = 512
TARGET_INTERVAL_S = 0.125
_installed = False
_queue: queue.Queue | None = None
_last_capture = 0.0
_interval = TARGET_INTERVAL_S
_stats = {"captures": 0, "renders": 0, "render_ms": [], "action_ms": [], "sent": 0, "dropped": 0}


def install() -> None:
    global _installed
    if _installed or not os.environ.get("SEQUENTIAL_DISPLAY_PORT"):
        return
    import builtins

    real_import = builtins.__import__

    def hooked(name, globals=None, locals=None, fromlist=(), level=0):
        module = real_import(name, globals, locals, fromlist, level)
        # Only the LIBERO env class. getattr on unrelated modules re-enters
        # pydantic's lazy importer and crashes the env process.
        if "libero" not in str(getattr(module, "__name__", "")):
            return module
        env_cls = module.__dict__.get("LiberoEnv")
        if isinstance(env_cls, type):
            _wrap(env_cls)
        return module

    builtins.__import__ = hooked
    _installed = True
    _start_sender()


def stats() -> dict:
    render = _stats["render_ms"]
    action = _stats["action_ms"]
    return {
        "captures": _stats["captures"],
        "renders": _stats["renders"],
        "sent": _stats["sent"],
        "dropped": _stats["dropped"],
        "interval_s": round(_interval, 4),
        "render_ms_max": None if not render else round(max(render), 2),
        "action_ms_max": None if not action else round(max(action), 2),
        "edge": DISPLAY_EDGE,
    }


def _wrap(cls) -> None:
    original = cls.step
    if getattr(original, "_display_wrapped", False):
        return

    def step(self, *args, **kwargs):
        started = time.perf_counter()
        result = original(self, *args, **kwargs)
        action_ms = (time.perf_counter() - started) * 1000
        try:
            _maybe_capture(self, action_ms)
        except Exception:
            pass
        return result

    step._display_wrapped = True
    cls.step = step


def _maybe_capture(env, action_ms: float) -> None:
    global _last_capture, _interval
    _stats["action_ms"].append(action_ms)
    _stats["action_ms"] = _stats["action_ms"][-40:]
    now = time.perf_counter()
    if now - _last_capture < _interval:
        return
    _last_capture = now
    _stats["captures"] += 1
    started = time.perf_counter()
    agent, wrist, source = _render_pair(env)
    render_ms = (time.perf_counter() - started) * 1000
    _stats["renders"] += 1
    _stats["render_ms"].append(render_ms)
    _stats["render_ms"] = _stats["render_ms"][-40:]
    if render_ms > 40:
        _interval = max(TARGET_INTERVAL_S, (render_ms / 1000) * 4)
    _enqueue({
        "agent": agent,
        "wrist": wrist,
        "source": source,
        "render_ms": round(render_ms, 2),
        "action_ms": round(action_ms, 2),
        "timestamp": time.time(),
        "frame_id": _stats["captures"],
    })


def _render_pair(env):
    from display_view import prepare_camera

    rendered = []
    for camera, source in (
        ("agentview", "render_camera.agentview"),
        ("robot0_eye_in_hand", "render_camera.robot0_eye_in_hand"),
    ):
        image = env.render_camera(camera_name=camera, height=DISPLAY_EDGE, width=DISPLAY_EDGE, depth=False)
        rendered.append(prepare_camera(image, source=source))
    return rendered[0], rendered[1], rendered[0]["source"]


def _enqueue(item: dict) -> None:
    global _queue
    if _queue is None:
        return
    if _queue.full():
        try:
            _queue.get_nowait()
            _stats["dropped"] += 1
        except queue.Empty:
            pass
    try:
        _queue.put_nowait(item)
    except queue.Full:
        _stats["dropped"] += 1


def _start_sender() -> None:
    global _queue
    _queue = queue.Queue(maxsize=1)
    threading.Thread(target=_send_loop, name="display-outlet", daemon=True).start()


def _send_loop() -> None:
    port = int(os.environ["SEQUENTIAL_DISPLAY_PORT"])
    sock = None
    while True:
        item = _queue.get()
        try:
            payload = _pack(item)
            if sock is None:
                sock = socket.create_connection(("127.0.0.1", port), timeout=0.4)
            sock.sendall(payload)
            _stats["sent"] += 1
        except Exception:
            _stats["dropped"] += 1
            if sock is not None:
                try:
                    sock.close()
                except Exception:
                    pass
            sock = None


def _pack(item: dict) -> bytes:
    from display_view import encode_display

    agent, agent_meta = encode_display(item["agent"])
    wrist, wrist_meta = encode_display(item["wrist"])
    header = {
        "type": "frame",
        "stage": "motion",
        "source": agent_meta["source"],
        "flip": agent_meta["flip"],
        "jpeg_quality": agent_meta["jpeg_quality"],
        "aw": agent_meta["out_w"],
        "ah": agent_meta["out_h"],
        "ww": wrist_meta["out_w"],
        "wh": wrist_meta["out_h"],
        "src_aw": agent_meta["src_w"],
        "src_ah": agent_meta["src_h"],
        "src_ww": wrist_meta["src_w"],
        "src_wh": wrist_meta["src_h"],
        "frame_id": item["frame_id"],
        "timestamp": item["timestamp"],
        "render_ms": item["render_ms"],
        "action_ms": item["action_ms"],
        "live_chunk": True,
        "primitive": "env_step",
        "a_len": len(agent),
        "w_len": len(wrist),
    }
    payload = json.dumps(header).encode("utf-8")
    return len(payload).to_bytes(4, "big") + payload + agent + wrist
