"""Stage D: VLA async stays aligned, and a display thread is not VLA async."""

from __future__ import annotations

import time
import unittest

from vla_async import CALL_ORDER, RTC_ASSESSMENT, VLAController, classify_span, control_stalls, replay_queue


class StageDTest(unittest.TestCase):
    def test_predict_does_not_overlap_chunk_step_and_display_is_not_vla(self):
        self.assertFalse(CALL_ORDER[0]["overlaps_env_step"])
        self.assertEqual(CALL_ORDER[0]["kind"], "vla_infer")
        self.assertEqual(CALL_ORDER[2]["kind"], "env_step")
        self.assertFalse(RTC_ASSESSMENT["rtc"])
        self.assertFalse(RTC_ASSESSMENT["thread_prefetch_is_rtc"])
        self.assertEqual(RTC_ASSESSMENT["unaligned_policy"], "sync")
        self.assertEqual(classify_span("frame_bus publish"), "display")
        stalls = control_stalls(
            [
                {"name": "model.predict", "ms": 800},
                {"name": "vla.predict RPC", "ms": 40},
                {"name": "env.chunk_step", "ms": 120},
                {"name": "render 512", "ms": 350},
                {"name": "frame_bus", "ms": 900},
            ]
        )
        self.assertEqual([item["kind"] for item in stalls], ["vla_infer", "render_512"])
        self.assertTrue(all(item["vla_async"] is False for item in stalls))

    def test_queue_records_alignment_and_drops_stale_results(self):
        ctrl = VLAController(mode="async", max_depth=2)
        token = ("ep", 1, 1, "pi0_pick")
        ctrl.arm(*token)
        self.assertTrue(ctrl.begin_predict(0, token))
        self.assertFalse(ctrl.begin_predict(0, token))
        ctrl.mark_env(0)
        self.assertTrue(ctrl.finish_predict([[1], [2], [3]], 0, token))
        self.assertEqual(len(ctrl.queue), 2)
        first = ctrl.pop_action(token)
        self.assertEqual(first["action_index"], 0)
        self.assertEqual(first["obs_version"], 0)
        self.assertEqual(first["queue_depth_after"], 1)
        self.assertFalse(ctrl.finish_predict([[9]], 0, ("ep", 1, 2, "pi0_pick")))
        ctrl.clear("terminated")
        self.assertIsNone(ctrl.pop_action(token))
        self.assertTrue(ctrl.closed)
        summary = replay_queue(ctrl.events)
        self.assertIn("terminated", summary["clears"])
        self.assertNotIn(True, summary["rtc_labels"])

    def test_unaligned_prefetch_is_not_appended_and_sync_still_steps(self):
        ctrl = VLAController(mode="async")
        token = ("ep", 1, 1, "pi0_pick")
        ctrl.arm(*token)
        ctrl.mark_env(5)
        ctrl.begin_predict(0, token)
        self.assertFalse(ctrl.finish_predict([[1, 0, 0, 0, 0, 0, 0]], 0, token))
        self.assertEqual(ctrl.queue, [])
        stepped = []
        ctrl.mode = "sync"
        ctrl.closed = False
        ctrl.token = token
        ctrl.begin_predict(5, token)
        self.assertTrue(ctrl.finish_predict([[1, 0, 0, 0, 0, 0, 0]], 5, token))
        result = ctrl.owner_step(lambda actions: stepped.append(list(actions)) or "ok", [[1, 0, 0, 0, 0, 0, 0]], token, 5)
        self.assertEqual(result, "ok")
        self.assertEqual(len(stepped), 1)
        ctrl.clear("cancel")
        self.assertIsNone(ctrl.owner_step(lambda actions: stepped.append(actions), [[0]], token, 6))
        self.assertEqual(len(stepped), 1)
        self.assertEqual(replay_queue(ctrl.events)["discarded_unaligned"], 1)

    def test_discard_prefetch_is_not_issued(self):
        from live import _install_vla_experiment

        class Env:
            def __init__(self):
                self.steps = []

            def chunk_step(self, actions, *args, **kwargs):
                self.steps.append(list(actions))
                time.sleep(0.02)
                return "owner"

        class Model:
            def __init__(self):
                self.calls = 0

            def predict(self, obs, options=None):
                self.calls += 1
                time.sleep(0.05)
                return [[9, 9, 9, 9, 9, 9, 9]]

        class Prim:
            def __init__(self):
                self.env = Env()
                self.model = Model()
                self._last_obs = {"task_descriptions": "pick"}
                self._env_action_count = 0

            def _vlm_chunk(self, instruction):
                return self.model.predict(self._last_obs, options={"mode": "eval"})

        prim = Prim()
        ctrl = VLAController(mode="async")
        ctrl.arm("ep", 1, 1, "pi0_pick")
        _install_vla_experiment(prim, ctrl)
        owner = prim.env.chunk_step([[0, 0, 0, 0, 0, 0, 1]])
        self.assertEqual(prim.model.calls, 0)
        follow = prim._vlm_chunk("pick up the soup")
        self.assertEqual(owner, "owner")
        self.assertEqual(prim.env.steps, [[[0, 0, 0, 0, 0, 0, 1]]])
        self.assertEqual(ctrl.queue, [])
        self.assertEqual(prim.model.calls, 1)
        self.assertFalse(any(item.get("kind") == "prefetch_error" for item in ctrl.events))
        self.assertEqual(follow[0][0], 9)


if __name__ == "__main__":
    unittest.main()
