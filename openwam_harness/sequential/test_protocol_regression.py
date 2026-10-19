"""Websocket regressions against the adapter command path.

The environment is a stub. Nothing in this file opens LIBERO or port 8766.
There is no browser, so a passing result is not a click on the start button.
"""

from __future__ import annotations

import asyncio
import json
import threading
import unittest
from pathlib import Path

from aiohttp import WSMsgType, web
from aiohttp.test_utils import TestClient, TestServer

import adapter
import live
from executor import ToolOutcome
from placement import RECORDED_RETREAT, legacy_retreat_eef
from run_context import RunContext
from schema import _in_workspace

ROOT = Path(__file__).resolve().parent
PLANNER_LAST = ROOT / "records" / "planner_last.json"
REPORT = ROOT / "records" / "protocol_regression_report.json"
JPEG = b"\xff\xd8\xff\xd9"
META = {"out_w": 1, "out_h": 1, "src_w": 1, "src_h": 1, "source": "stub", "flip": "none", "jpeg_quality": 80}


class StubEnv:
    fail_execute = False
    executes = 0

    def __init__(self, spec, out, on_phase=None):
        self.spec = spec
        self.out = out
        self.scene_epoch = 0
        self.timings = {"service_start": 0, "env_create": 0, "reset": 0}
        self.output_dir = out

    def scene_key(self):
        return (self.spec["suite"], self.spec["task_id"], self.spec["seed"])

    def scene_info(self):
        return {
            "task_language": self.spec.get("goal"),
            "env_step": 0,
            "object_names": ["akita_black_bowl_1", "flat_stove_1"],
        }

    def sim_state(self):
        return {
            "objects": {
                "akita_black_bowl_1": {"pos": [0.0, 0.0, 0.9]},
                "flat_stove_1": {"pos": [0.05, 0.0, 0.88]},
            },
            "eef_pos": [0.0, 0.0, 1.0],
            "terminated": False,
            "truncated": False,
        }

    def warm_reset(self):
        return None

    def retarget(self, spec, out):
        self.spec = spec

    def close(self):
        return None

    def set_execution_mode(self, mode):
        self.mode = mode

    def component_health(self):
        return [{"name": "env", "alive": True}, {"name": "vla", "alive": True}]

    def capture_display(self):
        return {"agent": JPEG, "wrist": JPEG, "agent_meta": META, "wrist_meta": META, "encode_ms": 0, "capture_ms": 0}

    def execute(self, action, args):
        StubEnv.executes += 1
        if StubEnv.fail_execute:
            raise RuntimeError("stub producer failed")
        if action == "segment":
            return ToolOutcome(executed=True, real=True, xyz=None, error="score 0.10 below 0.50", raw={"score": 0.10, "found": False})
        return ToolOutcome(executed=True, real=True)

    def observe(self):
        return {
            "sim": self.sim_state(),
            "cameras": {"agentview": {"id": "a"}, "wrist": {"id": "w"}},
            "observation_id": "stub",
            "env_step": 0,
            "timestamp": 1,
        }

    def native_success(self):
        return False

    def episode_continuable(self):
        return True

    def env_step(self):
        return 0


def _plan(goal: str, steps: list[dict]) -> dict:
    return {
        "status": "ok",
        "stage": "ok",
        "repairs": 0,
        "errors": [],
        "attempts": [{"request_id": "stub-protocol"}],
        "plan": {
            "plan_id": "protocol",
            "goal": goal,
            "goal_relation": "on",
            "plan_source": "model",
            "objects": {
                "bowl": {"sim_name": "akita_black_bowl_1", "phrases": ["akita black bowl"], "role": "movable"},
            },
            "steps": steps,
        },
    }


def _segment_plan(goal: str) -> dict:
    return _plan(goal, [
        {
            "step_id": "seg_bowl",
            "skill": "segment",
            "args": {"prompt": "akita black bowl", "camera": "agentview", "object_ref": "bowl"},
            "depends_on": [],
            "preconditions": [],
            "completion_check": "object_localized",
            "budget": {"max_s": 40, "max_attempts": 1},
            "on_failure": "stop",
            "object_ref": "bowl",
            "subtask": "bowl",
        }
    ])


def _move_plan(goal: str) -> dict:
    return _plan(goal, [
        {
            "step_id": "move_bowl",
            "skill": "move_to",
            "args": {"xyz": [0.0, 0.0, 0.9], "gripper": 1, "object_ref": "bowl", "max_steps": 5},
            "depends_on": [],
            "preconditions": [],
            "completion_check": "object_transported",
            "budget": {"max_s": 10, "max_attempts": 1},
            "on_failure": "stop",
            "object_ref": "bowl",
        }
    ])


class ProtocolRegressionTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.saved_planner = PLANNER_LAST.read_bytes() if PLANNER_LAST.exists() else None
        self.saved_app = adapter.APP
        self.saved_live = adapter.LIVE
        self.saved_backend = live.LiveBackend
        self.saved_plan = adapter.request_plan
        self.saved_wait = RunContext.wait_confirm
        self.saved_probe = dict(adapter.PROBE.get("edge") or {})
        self.box = {"mode": "raise"}
        adapter.LIVE = True
        live.LiveBackend = StubEnv
        adapter.APP = adapter.App()
        adapter.PROBE["edge"] = {"status": "ok", "detail": "stub", "model_ids": ["Qwen3.5-9B"]}

        def request_plan(**_kwargs):
            mode = self.box["mode"]
            goal = adapter.APP.scene_spec.get("goal") or ""
            if mode == "hold":
                self.box["entered"].set()
                if not self.box["release"].wait(5):
                    raise RuntimeError("hold expired")
                raise RuntimeError("hold released")
            if mode == "raise":
                raise RuntimeError("planner stub failed")
            if mode == "segment":
                return _segment_plan(goal)
            if mode == "move":
                return _move_plan(goal)
            raise AssertionError(mode)

        def wait_confirm(run, timeout=90.0):
            del timeout
            return self.saved_wait(run, 1.2)

        adapter.request_plan = request_plan
        RunContext.wait_confirm = wait_confirm
        from suites import load_suites

        adapter.APP.suites = load_suites()
        StubEnv.fail_execute = False
        StubEnv.executes = 0
        app = web.Application()
        app.router.add_get("/ws", adapter.websocket)
        self.server = TestServer(app)
        self.client = TestClient(self.server)
        await self.client.start_server()
        self.assertNotEqual(self.server.port, 8766)
        self.report_port = self.server.port
        self.results = []

    async def asyncTearDown(self):
        await self.client.close()
        adapter.APP = self.saved_app
        adapter.LIVE = self.saved_live
        live.LiveBackend = self.saved_backend
        adapter.request_plan = self.saved_plan
        RunContext.wait_confirm = self.saved_wait
        adapter.PROBE["edge"] = self.saved_probe
        if self.saved_planner is None:
            PLANNER_LAST.unlink(missing_ok=True)
        else:
            PLANNER_LAST.write_bytes(self.saved_planner)
        REPORT.parent.mkdir(exist_ok=True)
        REPORT.write_text(json.dumps({
            "libero_connected": False,
            "browser_click": False,
            "port": getattr(self, "report_port", None),
            "driver": "aiohttp TestClient websocket to adapter.websocket; LiveBackend and request_plan are stubs",
            "paths": getattr(self, "results", []),
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    async def test_command_paths(self):
        ws = await self.client.ws_connect("/ws")
        await self._load(ws, "load-refresh")
        epoch = adapter.APP.session.scene_epoch
        episode = adapter.APP.session.episode_id
        await ws.close()
        ws = await self.client.ws_connect("/ws")
        refreshed = await self._until(ws, lambda rows: any(row.get("type") == "snapshot" and row.get("can_start") for row in rows) and any(row.get("type") == "binary" for row in rows))
        snap = next(row for row in refreshed if row.get("type") == "snapshot" and row.get("can_start"))
        self.assertEqual(snap["scene_epoch"], epoch)
        self.assertTrue(snap["scene_loaded"])
        self.assertEqual(snap["block_reason"], "")
        self.assertTrue(any(row.get("type") == "binary" for row in refreshed))
        self._ok("加载→刷新→开始前", "刷新后的快照 can_start 与空 block_reason 一致，并带最后一帧")

        adapter.PROBE["edge"] = {"status": "failed", "detail": "offline"}
        await ws.send_str(json.dumps(_start(epoch, "offline")))
        offline = await self._until(ws, lambda rows: any(row.get("type") == "start_ack" and row.get("code") == "profile_offline" for row in rows))
        self.assertTrue(any(row.get("code") == "profile_offline" for row in offline))
        adapter.PROBE["edge"] = {"status": "ok", "detail": "stub", "model_ids": ["Qwen3.5-9B"]}
        self._ok("Planner离线→恢复", "离线时 start_ack 为 profile_offline；探测恢复后允许下一次开始")

        self.box["mode"] = "raise"
        await ws.send_str(json.dumps(_start(epoch, "plan-fail")))
        failed = await self._until(ws, lambda rows: any(row.get("type") == "ended" and "planner stub failed" in str(row.get("reason")) for row in rows), timeout=8)
        self.assertTrue(any("planner stub failed" in str(row.get("reason")) for row in failed))
        self.assertFalse(adapter.APP.runs.producer_alive())
        await self._load(ws, "reload-after-plan")
        self.assertNotEqual(adapter.APP.session.episode_id, episode)
        self.assertTrue(adapter.APP.snapshot()["can_start"])
        self.assertTrue(any(item.get("root_error") and "planner stub failed" in item["root_error"] for item in adapter.APP.snapshot()["history"]))
        self._ok("Planner失败→重新加载", "失败留在 history，新场景 can_start")

        epoch = adapter.APP.session.scene_epoch
        self.box["entered"] = threading.Event()
        self.box["release"] = threading.Event()
        self.box["mode"] = "hold"
        await ws.send_str(json.dumps(_start(epoch, "double-a")))
        self.assertTrue(await asyncio.to_thread(self.box["entered"].wait, 5))
        await ws.send_str(json.dumps(_start(epoch, "double-b")))
        acks = await self._until(ws, lambda rows: sum(row.get("type") == "start_ack" for row in rows) >= 2)
        statuses = [row.get("status") for row in acks if row.get("type") == "start_ack"]
        self.assertIn("accepted", statuses)
        self.assertIn("rejected", statuses)
        self.box["release"].set()
        await self._until(ws, lambda rows: any(row.get("type") == "snapshot" and not row.get("producer_alive") for row in rows), timeout=6)
        self._ok("双击开始", "第一次计划还停在 request_plan 时，第二次 start_ack 被拒绝")

        await self._load(ws, "reload-before-segment")
        epoch = adapter.APP.session.scene_epoch
        self.box["mode"] = "segment"
        await ws.send_str(json.dumps(_start(epoch, "segment-a")))
        waiting = await self._until(ws, lambda rows: any(row.get("type") == "snapshot" and row.get("phase") == "waiting_confirmation" for row in rows), timeout=6)
        self.assertTrue(any(row.get("phase") == "waiting_confirmation" for row in waiting))
        await ws.send_str(json.dumps({"type": "stop", "client_request_id": "stop-1"}))
        await ws.send_str(json.dumps({"type": "stop", "client_request_id": "stop-2"}))
        stops = await self._until(ws, lambda rows: sum(row.get("type") == "stop_ack" for row in rows) >= 2)
        codes = [row.get("code") for row in stops if row.get("type") == "stop_ack"]
        self.assertIn("already_stopping", codes)
        await self._until(ws, lambda rows: any(row.get("type") == "snapshot" and row.get("can_load") and not row.get("producer_alive") for row in rows), timeout=6)
        self._ok("分割低分→停止等待确认", "低分分割进入等待；第一次停止接受，重复停止是 already_stopping")
        self._ok("重复停止", "第二条 stop_ack code 为 already_stopping")

        await self._load(ws, "reload-before-timeout")
        epoch = adapter.APP.session.scene_epoch
        self.box["mode"] = "segment"
        await ws.send_str(json.dumps(_start(epoch, "segment-timeout")))
        timed = await self._until(ws, lambda rows: any("确认超时" in str(row.get("root_error") or row.get("reason") or "") for row in rows), timeout=6)
        self.assertTrue(any("确认超时" in str(row.get("root_error") or row.get("reason") or "") for row in timed))
        self.assertFalse(adapter.APP.runs.producer_alive())
        self._ok("分割低分→确认超时", "没有确认消息，wait_confirm 返回超时并结束这一轮")

        await self._load(ws, "reload-before-exception")
        epoch = adapter.APP.session.scene_epoch
        current_episode = adapter.APP.session.episode_id
        StubEnv.fail_execute = True
        self.box["mode"] = "move"
        await ws.send_str(json.dumps(_start(epoch, "boom")))
        blew = await self._until(ws, lambda rows: any("stub producer failed" in str(row.get("reason") or row.get("root_error") or "") for row in rows), timeout=8)
        self.assertTrue(any("stub producer failed" in str(row) for row in blew))
        self.assertFalse(adapter.APP.runs.producer_alive())
        self.assertGreater(StubEnv.executes, 0)
        await self._load(ws, "reload-after-exception")
        self.assertNotEqual(adapter.APP.session.episode_id, current_episode)
        self.assertTrue(adapter.APP.snapshot()["can_start"])
        self._ok("执行异常→生产者退出→重新加载", "execute 抛出后生产者退出，再次 load_ack accepted")

        old_epoch = adapter.APP.session.scene_epoch
        old_episode = adapter.APP.session.episode_id
        revision = adapter.APP.state_revision
        adapter.APP.broadcast({
            "type": "ended",
            "episode_id": "stale-episode",
            "scene_epoch": old_epoch - 1,
            "run_id": 1,
            "reason": "old ended",
            "success": False,
        })
        stale = await self._until(ws, lambda rows: any(row.get("type") == "ended" and row.get("reason") == "old ended" for row in rows))
        late = next(row for row in stale if row.get("reason") == "old ended")
        self.assertLess(late["scene_epoch"], old_epoch)
        self.assertGreater(late["state_revision"], revision)
        snap = adapter.APP.snapshot()
        self.assertEqual(snap["scene_epoch"], old_epoch)
        self.assertEqual(snap["scene"]["episode_id"], old_episode)
        self.assertNotEqual(snap["phase"], "failed")
        source = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
        self.assertIn("m.scene_epoch < sceneEpoch", source)
        self.assertIn("m.episode_id !== episode", source)
        self.assertIn("m.run_id < currentRun", source)
        self._ok("旧 ended 消息迟到", "服务端快照不被这条消息改写；app.js 有对应过滤。没有执行浏览器")

        ws2 = await self.client.ws_connect("/ws")
        again = await self._until(ws2, lambda rows: any(row.get("type") == "snapshot" and row.get("has_frame") for row in rows) and any(row.get("type") == "binary" for row in rows))
        self.assertTrue(any(row.get("can_start") for row in again if row.get("type") == "snapshot"))
        await ws2.close()
        self._ok("WS重连", "新连接收到快照和最后一帧")

        eef = legacy_retreat_eef(
            RECORDED_RETREAT["support_origin"],
            RECORDED_RETREAT["fixed_xy_offset"],
            RECORDED_RETREAT["z_offset_m"],
            RECORDED_RETREAT["hold_offset"],
        )
        self.assertEqual(eef, RECORDED_RETREAT["rejected_eef_origin"])
        self.assertFalse(_in_workspace(eef))
        self.results.append({
            "name": "退离几何回归输入",
            "result": "pass",
            "driver": "placement.legacy_retreat_eef(RECORDED_RETREAT)",
            "libero": False,
            "browser": False,
            "detail": "记录中的 retreat_step2 仍复现 z=1.156，且不在工作空间内",
        })
        self.assertEqual([item["name"] for item in self.results], [
            "加载→刷新→开始前",
            "Planner离线→恢复",
            "Planner失败→重新加载",
            "双击开始",
            "分割低分→停止等待确认",
            "重复停止",
            "分割低分→确认超时",
            "执行异常→生产者退出→重新加载",
            "旧 ended 消息迟到",
            "WS重连",
            "退离几何回归输入",
        ])

    async def _load(self, ws, request_id: str):
        await ws.send_str(json.dumps({
            "type": "load",
            "suite": "libero_goal",
            "task_id": 1,
            "seed": 0,
            "client_request_id": request_id,
        }))
        loaded = await self._until(
            ws,
            lambda rows: any(row.get("type") == "load_ack" and row.get("status") == "accepted" for row in rows)
            and any(row.get("type") == "snapshot" and row.get("can_start") and row.get("scene_loaded") for row in rows),
            timeout=20,
        )
        self.assertTrue(any(row.get("type") == "loaded" and row.get("mode") == "live" for row in loaded))
        return loaded

    async def _until(self, ws, pred, timeout=8):
        rows = []
        deadline = asyncio.get_running_loop().time() + timeout
        while asyncio.get_running_loop().time() < deadline:
            try:
                msg = await asyncio.wait_for(ws.receive(), timeout=0.25)
            except asyncio.TimeoutError:
                if rows and pred(rows):
                    return rows
                continue
            if msg.type == WSMsgType.TEXT:
                rows.append(json.loads(msg.data))
            elif msg.type == WSMsgType.BINARY:
                rows.append({"type": "binary", "n": len(msg.data)})
            if pred(rows):
                return rows
        brief = [
            {
                "type": row.get("type"),
                "status": row.get("status"),
                "code": row.get("code"),
                "phase": row.get("phase"),
                "can_start": row.get("can_start"),
                "block": row.get("block_reason"),
                "reason": str(row.get("reason") or row.get("detail") or "")[:120],
            }
            for row in rows[-15:]
        ]
        raise AssertionError(json.dumps(brief, ensure_ascii=False))

    def _ok(self, name: str, detail: str):
        self.results.append({
            "name": name,
            "result": "pass",
            "driver": "websocket stub",
            "libero": False,
            "browser": False,
            "detail": detail,
        })


def _start(epoch: int, request_id: str) -> dict:
    return {
        "type": "start",
        "client_request_id": request_id,
        "scene_epoch": epoch,
        "task": "",
        "execution_mode": "benchmark",
        "planner_profile": "edge",
        "plan_source": "model",
    }


if __name__ == "__main__":
    unittest.main()
