"""Start contract: the button no longer supplies task or seed."""

from __future__ import annotations

import unittest
from pathlib import Path

import adapter
from capabilities import builtin_matrix
from instruction import resolve_instruction
from recovery import _actions

ROOT = Path(__file__).resolve().parent


def _scene_app():
    row = next(item for item in builtin_matrix() if item["suite"] == "libero_spatial" and item["task_id"] == 0)
    app = adapter.App()
    app.scene_spec = {
        "suite": "libero_spatial",
        "task_id": 0,
        "seed": 0,
        "init_state_id": 0,
        "goal": row["language"],
        "summary_zh": row["summary_zh"],
        "goal_relation": "on",
    }
    app.session = adapter.Session("libero_spatial_t0_s0")
    app.session.scene_epoch = 1
    app.sent = []
    app.broadcast = lambda message: app.sent.append(message)
    app._run = lambda *args, **kwargs: None
    adapter.PROBE["edge"] = {"status": "ok", "detail": "HTTP 200", "model_ids": ["Qwen3.5-9B"]}
    adapter.PROBE["cloud"] = {"status": "unconfigured", "detail": "未配置", "model_ids": []}
    return app


def _join_run(app):
    thread = app.runs.thread
    if thread is not None:
        thread.join(2)
    if app.runs.active() and not app.runs.producer_alive():
        app.runs.finish("cancelled", "test")
    if app.runs.thread is not None and not app.runs.producer_alive():
        app.runs.release(app.runs.thread)


def _start(task="", epoch=1, profile="edge", request_id="req-1"):
    return {
        "type": "start",
        "task": task,
        "plan_source": "model",
        "planner_profile": profile,
        "client_request_id": request_id,
        "scene_epoch": epoch,
    }


class StartContractTest(unittest.TestCase):
    def test_page_start_does_not_send_removed_controls(self):
        script = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
        page = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
        self.assertIn("client_request_id", script)
        self.assertIn("scene_epoch: sceneEpoch", script)
        self.assertIn("task: taskIn.value", script)
        self.assertIn('id="start-reason"', page)
        self.assertNotIn('id="task-id"', page)
        self.assertNotIn('id="seed"', page)
        self.assertIn(f"static/app.js?v={adapter.STATIC_VERSION}", page)

    def test_empty_task_uses_loaded_goal_and_keeps_typed_text(self):
        app = _scene_app()
        app.start("start", _start("   ", request_id="empty"))
        ack = next(item for item in reversed(app.sent) if item.get("type") == "start_ack")
        self.assertEqual(ack["status"], "accepted")
        self.assertTrue(ack["used_loaded_goal"])
        self.assertIn("黑碗", ack["summary_zh"])
        _join_run(app)
        typed = "请按当前目标执行"
        app.start("start", _start(typed, request_id="typed"))
        kept = next(item for item in reversed(app.sent) if item.get("type") == "start_ack")
        self.assertEqual(kept["user_text"], typed)
        self.assertFalse(kept["used_loaded_goal"])

    def test_stale_epoch_offline_profile_and_double_click(self):
        app = _scene_app()
        app.start("start", _start(epoch=0, request_id="stale"))
        self.assertFalse(app.running)
        self.assertEqual(app.run_journal[-1]["code"], "epoch_stale")
        adapter.PROBE["edge"] = {"status": "down", "detail": "connection refused", "model_ids": []}
        app.start("start", _start(request_id="offline"))
        self.assertFalse(app.running)
        self.assertEqual(app.run_journal[-1]["code"], "profile_offline")
        self.assertIn("connection refused", app.run_journal[-1]["detail"])
        adapter.PROBE["edge"] = {"status": "ok", "detail": "HTTP 200", "model_ids": ["Qwen3.5-9B"]}
        real_profiles = adapter.profiles
        def _no_cloud():
            data = real_profiles()
            data["cloud"] = {**data["cloud"], "available": False}
            return data
        adapter.profiles = _no_cloud
        self.addCleanup(lambda: setattr(adapter, "profiles", real_profiles))
        app.start("start", _start(profile="cloud", request_id="cloud"))
        self.assertEqual(app.run_journal[-1]["code"], "profile_unconfigured")
        self.assertIn("云端未配置", app.run_journal[-1]["detail"])
        app.start("start", _start(request_id="one"))
        self.assertTrue(app.running)
        app.start("start", _start(request_id="two"))
        self.assertEqual(app.run_journal[-1]["code"], "busy_running")
        self.assertEqual(app.run_id, app.request_seq - 1)
        app.load({"type": "load", "suite": "libero_spatial"})
        self.assertTrue(any(item.get("detail") == "正在执行，不能同时加载。" for item in app.sent))
        self.assertFalse(app.loading)
        _join_run(app)
        app.start("start", _start(request_id="after"))
        self.assertTrue(app.running)

    def test_current_goal_rewrite_reaches_planner_path(self):
        row = next(item for item in builtin_matrix() if item["suite"] == "libero_spatial" and item["task_id"] == 0)
        loaded = {"suite": "libero_spatial", "task_id": 0, "goal": row["language"], "summary_zh": row["summary_zh"]}
        rewrite = resolve_instruction("请按当前目标执行", loaded)
        self.assertEqual(rewrite["status"], "matched")
        self.assertFalse(rewrite["changed"])
        self.assertEqual(rewrite["user_text"], "请按当前目标执行")
        self.assertEqual(rewrite["language"], row["language"])
        unrelated = resolve_instruction("飞到月球", loaded)
        self.assertEqual(unrelated["status"], "unsupported")

    def test_recovery_does_not_pick_the_target_plate(self):
        step = {
            "object_ref": "black_bowl",
            "target_ref": "plate",
            "args": {"prompt": "pick up the black bowl"},
        }
        actions = _actions("PLACEMENT_UNVERIFIED", step, holding=True, released=False, attempt_index=0)
        self.assertEqual(actions[0]["args"]["xyz_from"], "segment:plate")
        self.assertTrue(all(item["args"]["object_ref"] == "black_bowl" for item in actions))
        self.assertNotIn("basket", actions[0]["args"]["xyz_from"])
        same = {"object_ref": "plate", "target_ref": "plate", "args": {}}
        self.assertEqual(_actions("PLACEMENT_UNVERIFIED", same, holding=False, released=True, attempt_index=0), [])

    def test_recovery_regrasp_uses_the_natural_phrase_not_the_instance_id(self):
        step = {
            "step_id": "place_step1",
            "object_ref": "cream_cheese_1",
            "target_ref": "akita_black_bowl_1",
            "grounding_query": "cream cheese",
            "args": {"gripper": -1, "steps": 8, "object_ref": "cream_cheese_1"},
        }
        actions = _actions("PLACEMENT_UNVERIFIED", step, holding=False, released=True, attempt_index=0)
        self.assertEqual([item["skill"] for item in actions], ["segment", "pi0_pick"])
        self.assertEqual(actions[0]["args"]["prompt"], "cream cheese")
        self.assertEqual(actions[1]["args"]["prompt"], "pick up the cream cheese")
        self.assertTrue(all(item["args"]["object_ref"] == "cream_cheese_1" for item in actions))
        bare = {"step_id": "p", "object_ref": "cream_cheese_1", "target_ref": "akita_black_bowl_1", "args": {}}
        fallback = _actions("PLACEMENT_UNVERIFIED", bare, holding=False, released=True, attempt_index=0)
        self.assertEqual(fallback[1]["args"]["prompt"], "pick up the cream_cheese_1")


if __name__ == "__main__":
    unittest.main()
