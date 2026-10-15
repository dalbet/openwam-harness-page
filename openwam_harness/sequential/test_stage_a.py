"""Stage A: a start requests a new plan, and native success stops the environment."""

from __future__ import annotations

import os
import unittest

from executor import Session, run_plan
from planner import load_human_plan
from planner_profiles import build_telemetry, comparison_record, profiles, public_profile
from test_contract import PLAN, Scripted, _run
from test_stagec import _pick_plan
from verify import native_success


class StageATest(unittest.TestCase):
    def test_profiles_do_not_echo_a_cloud_key(self):
        os.environ["CLOUD_PLANNER_API_KEY"] = "super-secret"
        os.environ["CLOUD_PLANNER_BASE_URL"] = "https://cloud.example/v1"
        os.environ["CLOUD_PLANNER_MODEL"] = "cloud-model"
        try:
            catalog = profiles()
            shown = public_profile(catalog["cloud"])
            self.assertNotIn("super-secret", str(shown))
            self.assertTrue(catalog["cloud"]["available"])
            self.assertEqual(catalog["edge"]["input_mode"], catalog["cloud"]["input_mode"])
            record = comparison_record(catalog["edge"], {"evidence_source": "sim_state"}, "same-input")
            other = comparison_record(catalog["cloud"], {"evidence_source": "sim_state"}, "same-input")
            self.assertEqual(record["input_sha256"], other["input_sha256"])
            self.assertEqual(record["held_constant"], ["vla", "perception", "executor", "planning_input"])
            self.assertEqual(record["swapped_modules"], ["planner"])
            self.assertEqual(record["binding_evidence_source"], "sim_state")
        finally:
            os.environ.pop("CLOUD_PLANNER_API_KEY", None)
            os.environ.pop("CLOUD_PLANNER_BASE_URL", None)
            os.environ.pop("CLOUD_PLANNER_MODEL", None)

    def test_telemetry_records_queue_first_token_and_total(self):
        row = build_telemetry(
            endpoint="http://127.0.0.1:8000/v1/chat/completions",
            model="Qwen3.5-9B",
            input_mode="scene_metadata",
            usage={"prompt_tokens": 10, "completion_tokens": 2, "queue_time": 0.01},
            sent_at=1.0,
            response_at=1.2,
            first_token_at=1.5,
            done_at=2.0,
        )
        self.assertEqual(row["queue_s"], 0.2)
        self.assertEqual(row["ttft_s"], 0.5)
        self.assertEqual(row["total_s"], 1.0)
        self.assertEqual(row["server_queue_s"], 0.01)
        self.assertEqual(row["usage"]["prompt_tokens"], 10)
        self.assertEqual(row["input_mode"], "scene_metadata")

    def test_start_rejects_saved_and_manual_plans(self):
        from adapter import App

        app = App()
        messages = []
        app.broadcast = messages.append
        app.session = Session("libero_object_t0_s0")
        app.session.begin_episode()
        app.backend = object()
        for source in ("manual", "replay"):
            messages.clear()
            app.start("start", {"plan_source": source, "task": "pick up the alphabet soup and place it in the basket"})
            self.assertFalse(app.running)
            self.assertTrue(messages)
            self.assertIn("保存计划", messages[-1]["detail"])

    def test_native_success_does_not_sample_or_close(self):
        class NativeAfterPick(Scripted):
            def __init__(self):
                super().__init__()
                self.holds = 0

            def execute(self, action, args):
                outcome = super().execute(action, args)
                if action == "pi0_pick":
                    self.native = True
                return outcome

            def hold_for_grasp(self, steps=4):
                self.holds += 1
                return []

        backend = NativeAfterPick()
        result = _run(_pick_plan(), backend, lambda _subtask: "success")
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["reason"], "NATIVE_SUCCESS")
        self.assertTrue(result["native_success"])
        self.assertEqual(backend.holds, 0)
        self.assertNotIn("set_gripper", backend.calls)
        self.assertEqual(backend.calls.count("pi0_pick"), 1)

    def test_truncated_does_not_finish_the_episode(self):
        self.assertFalse(native_success(False, True))
        self.assertFalse(native_success(True, True))

        class Truncated(Scripted):
            def native_success(self):
                return False

        backend = Truncated()
        result = run_plan(
            load_human_plan(PLAN, "libero_10_t0"),
            backend,
            Session("libero_10_t0"),
            lambda _subtask: "unknown",
            task=load_human_plan(PLAN, "libero_10_t0")["goal"],
        )
        self.assertNotEqual(result["reason"], "NATIVE_SUCCESS")
        self.assertFalse(result["native_success"])


if __name__ == "__main__":
    unittest.main()
