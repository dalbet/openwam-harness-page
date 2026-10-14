"""Exercise the running sequential adapter the same way the Page UI does."""

from __future__ import annotations

import asyncio
import json
import sys

import aiohttp


def _read_frame(raw: bytes) -> dict:
    length = int.from_bytes(raw[:4], "big")
    return json.loads(raw[4 : 4 + length].decode("utf-8"))


async def main() -> None:
    async with aiohttp.ClientSession() as session:
        async with session.get("http://127.0.0.1:8766/health") as response:
            health = await response.json()
        assert health["port"] == 8766, health
        assert health["silent_dry_fallback"] is False
        async with session.ws_connect("http://127.0.0.1:8766/ws") as ws:
            hello = json.loads(await _text(ws))
            assert hello["type"] == "health", hello
            assert hello["mode"] == health["mode"]
            await ws.send_json({"type": "get_scenes"})
            scenes = json.loads(await _text(ws))
            assert scenes["type"] == "scenes", scenes
            assert {item["id"] for item in scenes["scenes"]} == {"libero_10_t0", "libero_10_t1"}
            await ws.send_json({"type": "load", "scene": "libero_10_t0"})
            messages = []
            while not any(item.get("type") in {"loaded", "load_failed"} for item in messages):
                messages.append(json.loads(await _text(ws)))
            loaded = next(item for item in messages if item["type"] in {"loaded", "load_failed"})
            if health["mode"] == "dry":
                assert loaded["type"] == "loaded" and loaded["placeholder"] is True, loaded
                failed = json.loads(await _text(ws))
                assert failed["type"] == "state" and failed["phase"] == "failed", failed
                await ws.send_json({"type": "start", "plan_source": "manual", "task": loaded["instruction"]})
                started = json.loads(await _text(ws))
                assert started["type"] == "started" and started["mode"] == "dry", started
                refused = json.loads(await _text(ws))
                assert refused["phase"] == "failed", refused
                ended = json.loads(await _text(ws))
                assert ended["type"] == "ended" and ended["reason"] == "dry_run_refused", ended
            else:
                assert loaded["type"] == "loaded" and loaded["placeholder"] is False, loaded
                assert loaded["scene"]["agent_bytes"] > 1000, loaded["scene"]
                frame = _read_frame(await _binary(ws))
                assert frame["placeholder"] is False and frame["aw"] > 1 and frame["ah"] > 1, frame
    print("UI_PROTOCOL_OK", health["mode"])


async def _text(ws):
    while True:
        msg = await ws.receive(timeout=30)
        if msg.type == aiohttp.WSMsgType.TEXT:
            return msg.data


async def _binary(ws):
    while True:
        msg = await ws.receive(timeout=30)
        if msg.type == aiohttp.WSMsgType.BINARY:
            return msg.data


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as exc:
        print(f"UI_PROTOCOL_FAIL {exc}", file=sys.stderr)
        raise
