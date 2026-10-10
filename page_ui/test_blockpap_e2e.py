#!/usr/bin/env python3
"""Page UI 端到端冒烟：通过 WebSocket 加载官方场景并跑一个 episode。

用法: .venv-maniskill/bin/python test_blockpap_e2e.py [scene_id] [--run]
默认只验证场景列表与加载；--run 跑完整 episode 并报告 success。
"""
from __future__ import annotations

import asyncio
import json
import struct
import sys

import aiohttp

WS = "ws://127.0.0.1:8765/ws"


async def main(scene_id: str, run_episode: bool) -> int:
    async with aiohttp.ClientSession() as sess:
        async with sess.ws_connect(WS, max_msg_size=0) as ws:
            await ws.send_json({"type": "get_scenes"})
            loaded = False
            started = False
            frames = 0
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.BINARY:
                    # 二进制帧: 4字节头长度 + JSON 头 + agent jpeg + wrist jpeg
                    frames += 1
                    continue
                if msg.type != aiohttp.WSMsgType.TEXT:
                    continue
                m = json.loads(msg.data)
                t = m.get("type")
                if t == "scenes":
                    ids = [s["id"] for s in m["scenes"]]
                    print("scenes:", ids)
                    if scene_id not in ids:
                        print(f"FAIL: {scene_id} 不在场景列表")
                        return 1
                    await ws.send_json({"type": "load", "scene": scene_id})
                elif t == "loaded":
                    print("loaded:", m["env_id"], "|", m["instruction"])
                    loaded = True
                    if not run_episode:
                        return 0
                    await ws.send_json({"type": "start", "task": m["instruction"]})
                    started = True
                elif t == "load_failed":
                    print("FAIL load_failed:", m.get("detail"))
                    return 1
                elif t == "state":
                    print("state:", m.get("phase"), "|", m.get("detail"))
                elif t == "ended":
                    print(f"ENDED success={m['success']} reason={m['reason']} frames={frames}")
                    return 0 if (started and loaded) else 1
    return 1


if __name__ == "__main__":
    scene = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else "BlockPAP-v1"
    run = "--run" in sys.argv
    rc = asyncio.run(main(scene, run))
    sys.exit(rc)
