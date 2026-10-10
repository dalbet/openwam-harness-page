#!/usr/bin/env python3
"""抓取 Page UI 广播的展示帧，验证浏览器端实际看到的画面。"""
import asyncio, json, sys
import aiohttp

async def main(n=3):
    out = []
    async with aiohttp.ClientSession() as sess:
        async with sess.ws_connect("ws://127.0.0.1:8765/ws", max_msg_size=0) as ws:
            await ws.send_json({"type": "get_scenes"})
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.BINARY:
                    b = msg.data
                    hl = int.from_bytes(b[:4], "big")
                    hdr = json.loads(b[4:4+hl])
                    p = 4 + hl
                    ag = b[p:p+hdr["a_len"]]
                    wr = b[p+hdr["a_len"]:p+hdr["a_len"]+hdr["w_len"]]
                    i = len(out)
                    open(f"/tmp/ui_disp_agent_{i}.jpg", "wb").write(ag)
                    open(f"/tmp/ui_disp_wrist_{i}.jpg", "wb").write(wr)
                    out.append(hdr)
                    print("frame", hdr["fid"], hdr["phase"], hdr["detail"], "term" if hdr["term"] else "")
                    if len(out) >= n:
                        return 0

asyncio.run(main(int(sys.argv[1]) if len(sys.argv) > 1 else 3))
