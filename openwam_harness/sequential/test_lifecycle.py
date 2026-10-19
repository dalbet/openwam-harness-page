"""Lifecycle gates on the real command methods. No browser click is claimed."""

from __future__ import annotations

import threading
import unittest

import adapter
from executor import Session


def _scene_app():
    app = adapter.App()
    app.sent = []
    app.broadcast = app.sent.append
    app.scene_spec = {
        "suite": "libero_goal",
        "task_id": 1,
        "seed": 0,
        "goal": "put the bowl on the stove",
        "summary_zh": "把碗放到炉灶上。",
    }
    app.session = Session("libero_goal_t1_s0")
    app.session.scene_epoch = 3
    app.suites = [{"id": "libero_goal", "tasks": []}]
    adapter.PROBE["edge"] = {"status": "ok", "detail": "HTTP 200", "model_ids": ["Qwen3.5-9B"]}
    return app


class LifecycleTest(unittest.TestCase):
    def test_revision_increases_and_history_does_not_block_start(self):
        app = _scene_app()
        app.runs.begin(4, "old", 2, "先把奶酪放到盘子上", "free")
        app.runs.note_error("retreat z=1.156", "workspace")
        app.runs.finish("failed", "retreat z=1.156", "workspace")
        app.run_history.append(dict(app.runs.public(), archived=True))
        app.runs = adapter.RunContext()
        first = app._stamp({"type": "snapshot"})
        second = app._stamp({"type": "snapshot"})
        self.assertGreater(second["state_revision"], first["state_revision"])
        self.assertEqual(first["server_instance_id"], second["server_instance_id"])
        gates = app.command_gates()
        self.assertEqual(gates["block_reason"], "")
        self.assertTrue(gates["can_start"])
        self.assertTrue(gates["can_load"])
        self.assertFalse(gates["can_stop"])
        snap = app.snapshot()
        self.assertEqual(snap["block_reason"], "")
        self.assertEqual(snap["history"][0]["root_error"], "retreat z=1.156")
        self.assertEqual(snap["scene_epoch"], 3)
        self.assertFalse(snap["producer_alive"])

    def test_producer_release_waits_until_the_thread_exits(self):
        app = _scene_app()
        started = threading.Event()
        release_now = threading.Event()
        released = threading.Event()

        def work():
            app.runs.finish("failed", "segment low score", "TARGET_NOT_IDENTIFIED")
            started.set()
            release_now.wait(1)
            app.runs.complete_from_producer(lambda ok: released.set() if ok else None)

        thread = threading.Thread(target=work)
        app.runs.begin(1, "ep", 3, "任务", "free")
        app.runs.attach(thread)
        thread.start()
        started.wait(1)
        self.assertTrue(app.runs.producer_alive())
        self.assertEqual(app.command_gates()["block_reason"], "正在停止")
        self.assertFalse(app.command_gates()["can_start"])
        self.assertFalse(app.command_gates()["can_load"])
        release_now.set()
        self.assertTrue(released.wait(1))
        self.assertFalse(app.runs.producer_alive())
        self.assertEqual(app.command_gates()["block_reason"], "")

    def test_failed_load_does_not_mark_the_episode_ended(self):
        app = _scene_app()
        app.load({"type": "load", "suite": "not_a_suite", "client_request_id": "bad-load"})
        self.assertFalse(app.session.cancelled)
        self.assertEqual(app.command_gates()["block_reason"], "")
        rejected = [item for item in app.sent if item.get("type") == "load_failed"]
        self.assertTrue(rejected)

    def test_load_rejection_keeps_the_run_root_error(self):
        app = _scene_app()
        app.runs.begin(7, "ep", 3, "把奶酪放到盘子上", "free")
        app.runs.note_error("top score 0.028 is below min_score 0.200", "TARGET_NOT_IDENTIFIED")
        app.runs.phase = "waiting_confirmation"
        app.load({"type": "load", "suite": "libero_goal", "client_request_id": "load-busy"})
        ack = next(item for item in app.sent if item.get("type") == "load_ack")
        self.assertEqual(ack["status"], "rejected")
        self.assertEqual(ack["root_error"], "top score 0.028 is below min_score 0.200")
        self.assertEqual(app.runs.root_error, ack["root_error"])
        self.assertNotEqual(ack["detail"], ack["root_error"])

    def test_double_start_and_repeated_stop_have_their_own_acks(self):
        app = _scene_app()
        app._run = lambda *args, **kwargs: None
        message = {
            "type": "start",
            "task": "",
            "plan_source": "model",
            "planner_profile": "edge",
            "client_request_id": "one",
            "scene_epoch": 3,
        }
        app.start("start", message)
        app.start("start", {**message, "client_request_id": "two"})
        acks = [item for item in app.sent if item.get("type") == "start_ack"]
        self.assertEqual([item["status"] for item in acks], ["accepted", "rejected"])
        self.assertEqual(acks[1]["code"], "busy_running")
        app.stop({"client_request_id": "stop-1"})
        app.stop({"client_request_id": "stop-2"})
        stops = [item for item in app.sent if item.get("type") == "stop_ack"]
        self.assertEqual(stops[0]["status"], "accepted")
        self.assertEqual(stops[1]["status"], "rejected")
        self.assertEqual(stops[1]["code"], "already_stopping")
        self.assertEqual(stops[1]["client_request_id"], "stop-2")

    def test_frontend_uses_server_reason_and_schedules_draw(self):
        script = (adapter.ROOT / "static" / "app.js").read_text(encoding="utf-8")
        page = (adapter.ROOT / "static" / "index.html").read_text(encoding="utf-8")
        self.assertIn("lastSnapshot.block_reason", script)
        self.assertIn("未连接", script)
        self.assertIn("scheduleDraw", script)
        self.assertIn("state_revision", script)
        self.assertNotIn('runBusy ? "正在执行"', script)
        self.assertIn(f"static/app.js?v={adapter.STATIC_VERSION}", page)
        self.assertIn('id="start-reason"', page)


if __name__ == "__main__":
    unittest.main()
