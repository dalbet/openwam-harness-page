"""Recorded retreat_step2 z=1.156, and the free-mode termination split."""

from __future__ import annotations

import unittest

from episode_service import classify_step, publish_mode, read_mode
from episode_service import _wrap as wrap_facade
from executor import Session, run_plan
from placement import RECORDED_RETREAT, choose_object_target, legacy_retreat_eef, plan_retreat, resolve_motion_step
from schema import THRESHOLDS, _in_workspace


class RetreatGeometryTest(unittest.TestCase):
    def test_recorded_hold_offset_is_outside_the_workspace(self):
        row = RECORDED_RETREAT
        eef = legacy_retreat_eef(row["support_origin"], row["fixed_xy_offset"], row["z_offset_m"], row["hold_offset"])
        self.assertEqual(eef, row["rejected_eef_origin"])
        self.assertGreater(eef[2], 1.1)
        self.assertFalse(_in_workspace(eef))

    def test_retreat_starts_at_the_release_eef_and_does_not_clamp(self):
        row = RECORDED_RETREAT
        decision = plan_retreat(
            row["release_eef_origin"],
            row["released_object_origin"],
            [],
            THRESHOLDS["workspace_m"],
        )
        self.assertTrue(decision["ok"])
        self.assertFalse(decision["applied_hold_offset"])
        self.assertEqual(decision["frame"], "eef_origin")
        self.assertLessEqual(decision["eef_origin"][2], 1.1)
        self.assertNotEqual(decision["eef_origin"], [round(min(1.1, row["rejected_eef_origin"][2]), 4), *row["rejected_eef_origin"][:2]])
        self.assertNotEqual([round(value, 4) for value in row["rejected_eef_origin"]], decision["eef_origin"])
        resolved = resolve_motion_step(
            {"object_ref": "bowl", "args": {"motion": "retreat", "gripper": -1, "object_ref": "bowl"}},
            {},
            {
                "sim": {
                    "eef_pos": row["release_eef_origin"],
                    "objects": {"bowl_1": {"pos": row["released_object_origin"]}},
                }
            },
            {"objects": {"bowl": {"sim_name": "bowl_1"}}},
        )
        self.assertTrue(resolved["ok"])
        self.assertFalse(resolved["applied_hold_offset"])
        self.assertNotIn("hold_offset", resolved["args"])
        self.assertTrue(_in_workspace(resolved["args"]["xyz"]))

    def test_second_object_needs_a_free_support_point(self):
        support = [0.0, 0.0, 0.9]
        first = choose_object_target(support, [], radius=0.06)
        self.assertTrue(first["ok"])
        blocked = choose_object_target(support, [[0.0, 0.0, 0.9], [0.04, 0.0, 0.9], [0.0, 0.04, 0.9], [-0.04, 0.0, 0.9], [0.0, -0.04, 0.9], [0.028, 0.028, 0.9], [-0.028, 0.028, 0.9], [0.028, -0.028, 0.9], [-0.028, -0.028, 0.9]], radius=0.06)
        self.assertFalse(blocked["ok"])
        self.assertEqual(blocked["code"], "support_occupied")
        self.assertNotIn("object_origin", blocked)

    def test_retreat_error_does_not_assign_the_user_goal(self):
        plan = {
            "plan_id": "retreat",
            "goal": "把奶酪放到盘子上",
            "goal_relation": "on",
            "execution_mode": "free",
            "objects": {
                "cheese": {"sim_name": "cream_cheese_1", "phrases": ["cream cheese"], "role": "movable"},
                "plate": {"sim_name": "plate_1", "phrases": ["plate"], "role": "support"},
            },
            "predicates": [{"object_ref": "cream_cheese_1", "target_ref": "plate_1", "relation": "on", "hold": "end"}],
            "steps": [
                {
                    "step_id": "retreat_step2",
                    "skill": "move_to",
                    "args": {"motion": "retreat", "gripper": -1, "object_ref": "cheese"},
                    "depends_on": [],
                    "preconditions": [],
                    "completion_check": "retreated",
                    "budget": {"max_s": 30, "max_attempts": 1},
                    "on_failure": "stop",
                    "object_ref": "cheese",
                    "target_ref": "plate",
                    "relation": "on",
                    "subtask": "step2",
                }
            ],
        }
        backend = _Corner()
        session = Session("scene")
        session.begin_episode()
        result = run_plan(plan, backend, session, lambda _name: None, task=plan["goal"])
        self.assertEqual(result["execution_status"], "retreat_error")
        self.assertEqual(result["user_goal_status"], "met")
        self.assertTrue(result["retreat_error"])
        self.assertEqual(result["evidence_source"], "sim_state")
        self.assertEqual(backend.steps, 0)
        self.assertIsNotNone(result["final_check"])
        self.assertFalse(result["final_check"]["extra_step"])

    def test_service_keeps_benchmark_termination_and_free_native_goal(self):
        benchmark = classify_step(True, False, "benchmark")
        combo = classify_step(True, False, "free", task_scope="combo")
        official = classify_step(True, False, "free")
        limited = classify_step(True, True, "free")
        self.assertTrue(benchmark["terminated"])
        self.assertFalse(benchmark["continuable"])
        self.assertFalse(combo["terminated"])
        self.assertTrue(combo["native_goal"])
        self.assertTrue(combo["continuable"])
        self.assertTrue(official["terminated"])
        self.assertFalse(official["continuable"])
        self.assertFalse(official["clear_terminated"])
        self.assertTrue(limited["truncated"])
        self.assertFalse(limited["continuable"])
        self.assertFalse(limited["terminated"])

    def test_facade_records_native_goal_without_reset_or_client_clear(self):
        publish_mode("free", task_scope="combo")
        self.assertEqual(read_mode(), "free")

        class Facade:
            def __init__(self):
                self.resets = 0
                self._rpc = {}
                self._readonly_methods = set()

            def step(self, action):
                return {"obs": action}, 0.0, True, False, {"solved": True}

            def chunk_step(self, actions):
                return {"obs": actions}, 0.0, True, False, [{"solved": True}]

            def reset(self):
                self.resets += 1
                return "reset"

            def _register_rpc(self):
                return None

        wrap_facade(Facade)
        env = Facade()
        env._register_rpc()
        obs, _reward, term, trunc, info = env.step([0.0])
        self.assertFalse(term)
        self.assertFalse(trunc)
        self.assertTrue(info["native_goal"])
        self.assertTrue(info["episode_continuable"])
        self.assertEqual(env.resets, 0)
        self.assertTrue(env.native_goal_status()["native_goal"])
        self.assertEqual(env.reset(), "reset")
        self.assertFalse(env.native_goal_status()["native_goal"])
        _obs, _reward, chunk_term, chunk_trunc, rows = env.chunk_step([[0.0]])
        self.assertFalse(chunk_term)
        self.assertFalse(chunk_trunc)
        self.assertTrue(rows[-1]["native_goal"])
        publish_mode("benchmark")
        _obs, _reward, bench_term, bench_trunc, bench_info = env.step([0.0])
        self.assertTrue(bench_term)
        self.assertFalse(bench_trunc)
        self.assertFalse(bench_info["episode_continuable"])


class _Corner:
    def __init__(self):
        self.steps = 0
        self.calls = []

    def execute(self, action, args):
        self.steps += 1
        self.calls.append((action, args))
        from executor import ToolOutcome

        return ToolOutcome(executed=False, error="should not step", real=False)

    def observe(self):
        return {"sim": self.sim_state(), "cameras": {"agentview": {"id": "a"}, "wrist": {"id": "w"}}, "observation_id": "corner", "env_step": 0, "timestamp": 1}

    def sim_state(self):
        return {
            "objects": {
                "cream_cheese_1": {"pos": [0.30, 0.46, 1.015958]},
                "plate_1": {"pos": [0.30, 0.46, 0.97]},
            },
            "eef_pos": [0.35, 0.5, 1.1],
        }

    def native_success(self):
        return False

    def env_step(self):
        return 0


if __name__ == "__main__":
    unittest.main()
