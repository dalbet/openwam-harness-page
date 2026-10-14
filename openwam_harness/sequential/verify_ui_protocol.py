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
        async with session.ws_connect("http://127.0.0.1:8766/ws") as ws:
            await ws.send_json({"type": "get_scenes"})
            scenes = json.loads(await _text(ws))
            assert scenes["type"] == "scenes", scenes
            assert {item["id"] for item in scenes["scenes"]} == {
                "libero_10_t0",
                "libero_10_t1",
            }
            await ws.send_json({"type": "load", "scene": "libero_10_t0"})
            loaded = json.loads(await _text(ws))
            assert loaded["type"] == "loaded", loaded
            first = _read_frame(await _binary(ws))
            assert first["phase"] == "场景已加载" and first["term"] is False
            await ws.send_json({"type": "start", "task": loaded["instruction"]})
            started = json.loads(await _text(ws))
            assert started["type"] == "started", started
            pause = _read_frame(await _binary(ws))
            assert pause["phase"] == "等待确认" and pause["detail"] == "soup_place"
            assert pause["eid"] == started["episode_id"]
            await ws.send_json({"type": "confirm", "verdict": "unknown"})
            pause = _read_frame(await _binary(ws))
            assert pause["detail"] == "sauce_place", pause
            await ws.send_json({"type": "stop"})
            ended = json.loads(await _text(ws))
            assert ended["type"] == "ended" and ended["reason"] == "stopped", ended
    print("UI_PROTOCOL_OK")


async def _text(ws):
    while True:
        msg = await ws.receive(timeout=10)
        if msg.type == aiohttp.WSMsgType.TEXT:
            return msg.data


async def _binary(ws):
    while True:
        msg = await ws.receive(timeout=10)
        if msg.type == aiohttp.WSMsgType.BINARY:
            return msg.data


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as exc:
        print(f"UI_PROTOCOL_FAIL {exc}", file=sys.stderr)
        raise
