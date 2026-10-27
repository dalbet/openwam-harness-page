#!/usr/bin/env python
"""Resident d1-3B decision service.

d1 is a single-forward-pass decision model: ``system_one(state, questions)``
returns a typed answer per named question and generates no tokens. This
service keeps one copy of the checkpoint in memory on its own GPU/port so the
Agent layer can ask staged planning questions without reloading weights.

It is deliberately not an OpenAI-compatible chat endpoint: there is no
free-text generation path here. The HTTP server is the standard library's
because the model call is synchronous and serialized anyway.
"""

from __future__ import annotations

import base64
import io
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from common import config  # noqa: E402

MAX_BODY = 32 * 1024 * 1024
_LOCK = threading.Lock()
_STATS = {
    "calls": 0,
    "errors": 0,
    "total_s": 0.0,
    "max_s": 0.0,
    "last_s": 0.0,
    "last_input_tokens": 0,
    "started_at": time.time(),
}
_MODEL: Any = None
_LOAD_S = 0.0


def _cfg() -> dict[str, Any]:
    return config.engine_config()["agent"]["d1"]


def _load_model() -> Any:
    import torch
    from transformers import AutoModel

    cfg = _cfg()
    model = AutoModel.from_pretrained(cfg["model_path"], trust_remote_code=True, dtype=torch.bfloat16)
    if not torch.cuda.is_available():
        raise RuntimeError("d1 needs a CUDA device; CUDA_VISIBLE_DEVICES selected none")
    model.to("cuda")
    model.eval()
    return model


def _decode_image(value: Any) -> Any:
    from PIL import Image

    if not isinstance(value, str):
        raise ValueError("each image must be a base64/data-URL string")
    raw = value.split(",", 1)[1] if value.startswith("data:") else value
    return Image.open(io.BytesIO(base64.b64decode(raw))).convert("RGB")


def _health() -> dict[str, Any]:
    import torch

    cfg = _cfg()
    calls = _STATS["calls"]
    return {
        "service": "d1",
        "status": "ok" if _MODEL is not None else "starting",
        "model": cfg["model"],
        "base_model": cfg["base_model"],
        "license": cfg["license"],
        "source": cfg["source"],
        "model_path": cfg["model_path"],
        "model_sha256": cfg["model_sha256"],
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "dtype": "bfloat16",
        "load_s": round(_LOAD_S, 3),
        "uptime_s": round(time.time() - _STATS["started_at"], 1),
        "stats": {
            "calls": calls,
            "errors": _STATS["errors"],
            "mean_s": round(_STATS["total_s"] / calls, 5) if calls else None,
            "last_s": round(_STATS["last_s"], 5),
            "max_s": round(_STATS["max_s"], 5),
            "last_input_tokens": _STATS["last_input_tokens"],
        },
        "output_tokens_per_decision": 0,
    }


def _system_one(body: dict[str, Any]) -> dict[str, Any]:
    state = body.get("state")
    questions = body.get("questions")
    if not isinstance(questions, dict) or not questions:
        raise ValueError("questions must be a non-empty object")
    images_in = body.get("images") or []
    if not isinstance(images_in, list):
        raise ValueError("images must be a list")
    images = [_decode_image(value) for value in images_in]
    with _LOCK:
        started = time.perf_counter()
        try:
            response = _MODEL.system_one(state, questions, images=images or None)
        except Exception:
            _STATS["errors"] += 1
            raise
        elapsed = time.perf_counter() - started
    usage = response.get("usage") or {}
    _STATS["calls"] += 1
    _STATS["total_s"] += elapsed
    _STATS["last_s"] = elapsed
    _STATS["max_s"] = max(_STATS["max_s"], elapsed)
    _STATS["last_input_tokens"] = int(usage.get("input_tokens") or 0)
    return {"status": "ok", "answers": response.get("answers") or {}, "usage": usage, "latency_s": round(elapsed, 5)}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - stdlib signature
        return

    def _send(self, code: int, payload: dict[str, Any]) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802 - stdlib signature
        if self.path.split("?", 1)[0] == "/health":
            self._send(200, _health())
        else:
            self._send(404, {"status": "not_found"})

    def do_POST(self) -> None:  # noqa: N802 - stdlib signature
        if self.path.split("?", 1)[0] != "/system_one":
            self._send(404, {"status": "not_found"})
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        if length <= 0 or length > MAX_BODY:
            self._send(400, {"status": "bad_request", "error": "invalid body size"})
            return
        try:
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(body, dict):
                raise ValueError("body must be a JSON object")
        except Exception as exc:
            self._send(400, {"status": "bad_request", "error": f"invalid JSON: {exc}"})
            return
        try:
            result = _system_one(body)
        except ValueError as exc:
            self._send(400, {"status": "bad_request", "error": str(exc)})
        except Exception as exc:
            self._send(500, {"status": "error", "error": f"{type(exc).__name__}: {exc}"[:800]})
        else:
            self._send(200, result)


def main() -> None:
    global _MODEL, _LOAD_S
    cfg = _cfg()
    service = config.engine_config()["services"]["d1"]
    print(
        f"[d1] loading {cfg['model']} from {cfg['model_path']} "
        f"on CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}",
        flush=True,
    )
    started = time.perf_counter()
    _MODEL = _load_model()
    _LOAD_S = time.perf_counter() - started
    server = ThreadingHTTPServer((service["host"], int(service["port"])), Handler)
    server.daemon_threads = True
    print(f"[d1] ready in {_LOAD_S:.2f}s on http://{service['host']}:{service['port']}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
