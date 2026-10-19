"""Run ownership, name admission, and one sim-assisted point retry."""

from __future__ import annotations

import threading
import unittest
from pathlib import Path

from localize import pixel_for_instance
from run_context import RunContext
from schema import expand_semantic_plan, validate_semantic_plan

ROOT = Path(__file__).resolve().parent


class RunLifeTest(unittest.TestCase):
    def test_finish_is_idempotent_and_release_waits_for_the_producer(self):
        run = RunContext()
        run.begin(1, "ep", 1, "把奶酪放到盘子上", "free")
        self.assertTrue(run.active())
        held = threading.Event()
        release = threading.Event()

        def work():
            held.set()
            release.wait(1)

        thread = threading.Thread(target=work)
        run.attach(thread)
        thread.start()
        held.wait(1)
        self.assertFalse(run.release(thread))
        self.assertTrue(run.producer_alive())
        self.assertTrue(run.finish("failed", "segment low score", "TARGET_NOT_IDENTIFIED"))
        release.set()
        thread.join(1)
        self.assertTrue(run.release(thread))
        self.assertFalse(run.producer_alive())
        self.assertFalse(run.finish("failed", "later"))
        self.assertEqual(run.root_error, "segment low score")
        self.assertEqual(run.phase, "failed")

    def test_confirm_timeout_and_cancel_do_not_wait_forever(self):
        run = RunContext()
        run.begin(2, "ep", 1, "任务", "free")
        self.assertEqual(run.wait_confirm(0.01), "timeout")
        self.assertEqual(run.root_code, "confirm_timeout")
        run.begin(3, "ep", 1, "任务", "free")
        run.cancel()
        self.assertIsNone(run.wait_confirm(1))
        self.assertEqual(run.phase, "cancelling")

    def test_busy_load_keeps_the_root_error(self):
        import adapter

        app = adapter.App()
        app.sent = []
        app.broadcast = app.sent.append
        app.runs.begin(7, "ep", 1, "把奶酪放到盘子上", "free")
        app.runs.note_error("top score 0.028 is below min_score 0.200", "TARGET_NOT_IDENTIFIED")
        app.runs.phase = "waiting_confirmation"
        app.load({"type": "load", "suite": "libero_goal", "client_request_id": "load-1"})
        rejected = [item for item in app.sent if item.get("type") == "load_ack" and item.get("status") == "rejected"]
        self.assertEqual(rejected[0]["client_request_id"], "load-1")
        self.assertEqual(rejected[0]["root_error"], app.runs.root_error)
        self.assertFalse(any(item.get("type") == "state" for item in app.sent))
        self.assertEqual(app.runs.phase, "waiting_confirmation")
        self.assertIn("top score 0.028", app.snapshot()["run"]["root_error"])
        self.assertNotIn("running", app.health())

    def test_same_target_uses_a_second_drop_point(self):
        objects = {
            "cream_cheese_1": {"sim_name": "cream_cheese_1", "phrases": ["cream cheese"], "movable": True, "role": "movable"},
            "akita_black_bowl_1": {"sim_name": "akita_black_bowl_1", "phrases": ["black bowl"], "movable": True, "role": "movable"},
            "plate_1": {"sim_name": "plate_1", "phrases": ["plate"], "movable": False, "can_be_target": True, "target_relations": ["on"], "role": "support"},
        }
        plan = validate_semantic_plan(
            {
                "plan_id": "two",
                "goal": "先把奶酪放到盘子上，再把碗放到盘子上",
                "steps": [
                    _transfer("cheese", "cream_cheese_1", []),
                    _transfer("bowl", "akita_black_bowl_1", ["cheese"]),
                ],
            },
            scene_objects=objects,
            execution_mode="free",
            ordered=True,
        )
        expanded = expand_semantic_plan(plan)
        moves = [step for step in expanded["steps"] if step["skill"] == "move_to" and step["args"].get("motion") == "travel"]
        self.assertNotIn("xy_offset", moves[0]["args"])
        self.assertNotIn("xy_offset", moves[1]["args"])
        self.assertEqual(moves[0]["args"]["placement"], "support_region")
        self.assertEqual(moves[1]["args"]["placement"], "support_region")
        retreats = [step for step in expanded["steps"] if step["args"].get("motion") == "retreat"]
        self.assertTrue(retreats)
        self.assertNotIn("hold_offset", retreats[0]["args"])
        self.assertNotIn("carry_offset", retreats[0]["args"])
        pick = next(step for step in expanded["steps"] if step["skill"] == "pi0_pick")
        self.assertEqual(pick["vla_prompt"], "pick up the cream cheese")
        self.assertNotIn("奶酪", pick["vla_prompt"])

    def test_occluded_projection_is_not_a_point(self):
        hidden = pixel_for_instance([[[0.0, 0.0, 0.2]]], [0.0, 0.0, 0.9], [[0.0, 0.0, 0.2]])
        self.assertIsNone(hidden)
        foreground = pixel_for_instance([[[0.0, 0.0, 0.2]]], [0.0, 0.0, 0.22], [[0.0, 0.0, 0.2]])
        self.assertIsNone(foreground)
        found = pixel_for_instance([[[0.0, 0.0, 0.9], [0.0, 0.0, 0.2]]], [0.0, 0.0, 0.2], [[0.0, 0.0, 0.9]])
        self.assertEqual((found["row"], found["col"]), (0, 1))
        self.assertEqual(found["evidence_source"], "sim_state")
        self.assertFalse(found["vision"])

    def test_server_does_not_latch_termination(self):
        server = (ROOT / ".." / "RPent" / "robots" / "libero" / "env_server.py").resolve().read_text(encoding="utf-8")
        client = (ROOT / ".." / "RPent" / "robots" / "libero" / "env_client.py").resolve().read_text(encoding="utf-8")
        self.assertIn('"auto_reset": False', server)
        self.assertIn('"ignore_terminations": False', server)
        self.assertIn("env.step called after the episode signaled term/trunc", client)
        live = (ROOT / "live.py").read_text(encoding="utf-8")
        self.assertIn("env.native_goal", live)
        self.assertNotIn("env.terminated = False", live)


def _transfer(step_id, obj, depends):
    return {
        "step_id": step_id,
        "skill": "transfer",
        "object_n": 1,
        "target_n": 3,
        "object_ref": obj,
        "target_ref": "plate_1",
        "relation": "on",
        "prompt": "把奶酪放到盘子上",
        "user_phrase": "奶酪" if obj.startswith("cream") else "碗",
        "depends_on": depends,
        "completion_check": "object_placed",
        "budget": {"max_s": 30, "max_attempts": 1},
        "on_failure": "recover",
    }


if __name__ == "__main__":
    unittest.main()
