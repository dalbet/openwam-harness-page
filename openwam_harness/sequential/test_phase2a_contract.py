"""PHASE2A: native On while holding is not placement, and release is bounded."""

from __future__ import annotations

import copy
import unittest

from adapter import completion_phase
from executor import Session, ToolOutcome, run_plan
from finalize import finalize_run


def _plan(*, native_window=True):
    steps = [
        {
            "step_id": "trigger",
            "skill": "view_env_state",
            "args": {"step": -1},
            "depends_on": [],
            "preconditions": [],
            "completion_check": "observe_ready",
            "budget": {"max_s": 30, "max_attempts": 1},
            "on_failure": "stop",
            "subtask": "bowl_on_stove",
            "transfer_phase": "transporting",
            "object_ref": "bowl",
        }
    ]
    if native_window:
        steps.extend(
            [
                {
                    "step_id": "place",
                    "skill": "set_gripper",
                    "args": {"gripper": -1, "steps": 8, "object_ref": "bowl"},
                    "depends_on": ["trigger"],
                    "preconditions": [],
                    "completion_check": "object_placed",
                    "budget": {"max_s": 30, "max_attempts": 1},
                    "on_failure": "stop",
                    "object_ref": "bowl",
                    "target_ref": "stove",
                    "relation": "on",
                    "subtask": "bowl_on_stove",
                    "transfer_phase": "releasing",
                },
                {
                    "step_id": "retreat",
                    "skill": "move_to",
                    "args": {"gripper": -1, "object_ref": "bowl", "motion": "retreat"},
                    "depends_on": ["place"],
                    "preconditions": [],
                    "completion_check": "retreated",
                    "budget": {"max_s": 30, "max_attempts": 1},
                    "on_failure": "stop",
                    "object_ref": "bowl",
                    "target_ref": "stove",
                    "relation": "on",
                    "subtask": "bowl_on_stove",
                    "transfer_phase": "retreat",
                },
            ]
        )
    steps.append(
        {
            "step_id": "next_transfer",
            "skill": "view_env_state",
            "args": {"step": -1},
            "depends_on": [steps[-1]["step_id"]],
            "preconditions": [],
            "completion_check": "observe_ready",
            "budget": {"max_s": 30, "max_attempts": 1},
            "on_failure": "stop",
            "subtask": "cheese_on_plate",
            "transfer_phase": "transporting",
            "object_ref": "bowl",
        }
    )
    return {
        "plan_id": "phase2a",
        "goal": "put the bowl on the stove",
        "goal_relation": "on",
        "execution_mode": "free",
        "objects": {
            "bowl": {"sim_name": "bowl_1", "phrases": ["bowl"], "movable": True, "role": "movable"},
            "stove": {
                "sim_name": "stove_1",
                "phrases": ["stove"],
                "movable": False,
                "can_be_target": True,
                "target_relations": ["on"],
                "role": "support",
            },
        },
        "predicates": [{"object_ref": "bowl", "target_ref": "stove", "relation": "on", "hold": "end"}],
        "steps": steps,
    }


class FreeBackend:
    def __init__(
        self,
        *,
        trigger_native=False,
        episode_ok=True,
        drop_on_retreat=False,
        fail_retreat=False,
        cancel_on_trigger=False,
        reset_on_trigger=False,
        session=None,
        omit_final_count=False,
    ):
        self.trigger_native = trigger_native
        self.continuable = episode_ok
        self.drop_on_retreat = drop_on_retreat
        self.fail_retreat = fail_retreat
        self.cancel_on_trigger = cancel_on_trigger
        self.reset_on_trigger = reset_on_trigger
        self.session = session
        self.omit_final_count = omit_final_count
        self.native = False
        self.steps = 0
        self.query = 0
        self.calls = []
        self.object_pos = [0.10, 0.20, 0.45]
        self.target_pos = [0.10, 0.20, 0.45]
        self.eef = [0.10, 0.20, 0.55]
        self.opening = 0.02
        self.in_action = False

    def execute(self, action, args):
        self.calls.append((action, dict(args)))
        if action == "view_env_state":
            if self.cancel_on_trigger and self.session is not None:
                self.session.cancel()
            if self.reset_on_trigger and self.session is not None:
                self.session.begin_episode()
            if self.trigger_native:
                self.native = True
            self.steps += 1
            return self._outcome()
        if action == "set_gripper" and float(args.get("gripper") or 0) < 0:
            self.steps += 1
            self.opening = 0.078
            self.object_pos = list(self.target_pos)
            return self._outcome()
        if action == "move_to" and "xyz" in args:
            if self.fail_retreat:
                return ToolOutcome(executed=False, error="retreat blocked by clamp", real=True, error_class="physical")
            self.steps += 1
            self.eef = [float(value) for value in args["xyz"]]
            if self.drop_on_retreat:
                self.object_pos = [0.30, 0.30, 0.45]
            return self._outcome()
        return ToolOutcome(executed=False, error=f"unexpected {action}", real=False, error_class="plan")

    def _outcome(self):
        return ToolOutcome(
            executed=True,
            native_terminated=self.native,
            eef_pos=list(self.eef),
            raw={},
            real=True,
            env_step=self.steps,
            state_after=self.snapshot(),
        )

    def snapshot(self):
        self.query += 1
        return {
            "observation_id": f"action-{self.steps}-{self.query}",
            "env_step": self.steps,
            "sim_action_count": self.steps,
            "step_clock": "sim_action",
            "state_source": "action_return",
            "cameras": {
                "agentview": {"id": f"agentview:{self.steps}"},
                "wrist": {"id": f"wrist:{self.steps}"},
            },
            "sim": self.sim(),
        }

    def observe(self):
        self.query += 1
        count = None if self.omit_final_count else self.steps
        return {
            "observation_id": f"query-{self.steps}-{self.query}",
            "env_step": self.steps,
            "sim_action_count": count,
            "step_clock": "sim_action",
            "state_source": "readonly_rpc",
            "cameras": {
                "agentview": {"id": f"agentview-q{self.query}"},
                "wrist": {"id": f"wrist-q{self.query}"},
            },
            "sim": self.sim(),
        }

    def sim(self):
        return {
            "objects": {
                "bowl_1": {"pos": list(self.object_pos), "identified": True},
                "stove_1": {"pos": list(self.target_pos), "identified": True},
            },
            "eef_pos": list(self.eef),
            "gripper_opening_m": self.opening,
            "evidence_source": "sim_state",
        }

    def native_success(self):
        return self.native

    def episode_continuable(self):
        return self.continuable

    def env_step(self):
        return self.steps


def _run(plan, backend, session=None):
    session = session or Session("libero_goal_t1")
    session.begin_episode()
    return run_plan(plan, backend, session, lambda _subtask: None, task=plan["goal"])


def _finalize(result, plan, backend, exit_kind="native_goal_stopped"):
    payload = copy.deepcopy(result)
    for key in ("finalized", "user_goal_status", "user_goal_success", "subgoal_results", "release_evidence", "unknown_reason", "missing", "finalization"):
        payload.pop(key, None)
    return finalize_run(payload, backend=backend, plan=plan, exit_kind=exit_kind)


class Phase2AContractTest(unittest.TestCase):
    def test_native_on_while_still_holding_is_not_a_placed_user_goal(self):
        backend = FreeBackend(trigger_native=True)
        result = _run(_plan(native_window=False), backend)
        self.assertEqual(result["status"], "native_goal_stopped")
        self.assertTrue(result["native_success"])
        self.assertFalse(result["release"]["command_executed"])
        self.assertIsNone(result["native_finish_window"])
        self.assertEqual([action for action, _args in backend.calls], ["view_env_state"])

        report = _finalize(result, _plan(native_window=False), backend)
        self.assertEqual(report["user_goal_status"], "unknown")
        self.assertIsNone(report["user_goal_success"])
        self.assertIn("释放后证据不足", report["unknown_reason"])
        self.assertTrue(report["subgoal_results"][0]["geometry_held_at_end"])
        self.assertFalse(report["subgoal_results"][0]["held_at_end"])
        self.assertFalse(report["release"]["verified"])

    def test_release_and_retreat_finish_the_current_transfer_and_not_the_next_one(self):
        backend = FreeBackend(trigger_native=True)
        plan = _plan()
        result = _run(plan, backend)
        actions = [action for action, _args in backend.calls]
        self.assertEqual(actions.count("set_gripper"), 3)
        self.assertIn("move_to", actions)
        self.assertEqual(actions.count("view_env_state"), 1)

        self.assertEqual(result["status"], "native_goal_stopped")
        self.assertTrue(result["native_finish_window"])
        self.assertEqual(result["native_finish_window"]["scope"], "bowl_on_stove")
        self.assertEqual(result["native_finish_window"]["budget_s"], 60)
        self.assertEqual(result["release"]["command_executed"], True)
        self.assertTrue(result["release"]["verified"])
        self.assertTrue(result["retreat_complete"])

        report = _finalize(result, plan, backend)
        self.assertEqual(report["user_goal_status"], "met")
        self.assertTrue(report["user_goal_success"])
        self.assertEqual(report["release_evidence"][("bowl", "stove", "on")]["status"], "verified")
        self.assertEqual(completion_phase(report, plan, "free")[0], "completed")

    def test_post_release_relation_false_is_not_success(self):
        backend = FreeBackend(trigger_native=True, drop_on_retreat=True)
        plan = _plan()
        result = _run(plan, backend)
        self.assertTrue(result["release"]["verified"])
        report = _finalize(result, plan, backend)
        self.assertEqual(report["user_goal_status"], "unmet")
        self.assertFalse(report["user_goal_success"])
        self.assertEqual(report["release_evidence"][("bowl", "stove", "on")]["status"], "verified")
        self.assertFalse(report["subgoal_results"][0]["geometry_held_at_end"])
        self.assertEqual(report.get("unknown_reason"), None)
        self.assertNotEqual(completion_phase(report, plan, "free")[0], "completed")

    def test_missing_post_release_observation_count_is_unknown(self):
        backend = FreeBackend(trigger_native=True)
        plan = _plan()
        result = _run(plan, backend)
        backend.omit_final_count = True
        report = _finalize(result, plan, backend)
        self.assertEqual(report["user_goal_status"], "unknown")
        self.assertIsNone(report["user_goal_success"])
        self.assertIn("post_release_observation_missing", report["unknown_reason"])
        self.assertNotEqual(completion_phase(report, plan, "free")[0], "completed")

    def test_env_not_continuable_executes_no_action(self):
        backend = FreeBackend(episode_ok=False)
        backend.native = True
        result = _run(_plan(), backend)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["reason"], "episode_not_continuable")
        self.assertEqual(backend.calls, [])
        self.assertIsNone(result["native_finish_window"])

    def test_user_goal_met_and_retreat_failure_are_both_preserved(self):
        backend = FreeBackend(trigger_native=True, fail_retreat=True)
        plan = _plan()
        result = _run(plan, backend)
        self.assertEqual(result["execution_status"], "retreat_error")
        self.assertTrue(result["retreat_error"])
        self.assertTrue(result["release"]["verified"])
        self.assertFalse(result["retreat_complete"])
        self.assertEqual(result["user_goal_status"], "met")
        self.assertTrue(result["user_goal_success"])
        self.assertEqual(result["finalization"], "user_goal_met_retreat_error")

        phase, status, reason = completion_phase(result, plan, "free")
        self.assertEqual(phase, "incomplete")
        self.assertIn("用户目标已完成", reason)
        self.assertIn("退离未完成", reason)

        report = _finalize(result, plan, backend, exit_kind="retreat_error")
        self.assertEqual(report["user_goal_status"], "met")
        self.assertTrue(report["retreat_error"])
        self.assertFalse(report["final_check"]["retreat_error"] is None)

    def test_stop_before_release_runs_no_release_or_retreat(self):
        session = Session("libero_goal_t1")
        backend = FreeBackend(trigger_native=True, cancel_on_trigger=True, session=session)
        result = _run(_plan(), backend, session)
        self.assertEqual(result["status"], "discarded")
        self.assertEqual([action for action, _args in backend.calls], ["view_env_state"])
        self.assertFalse(any(action == "set_gripper" for action, _args in backend.calls))
        self.assertFalse(any(action == "move_to" for action, _args in backend.calls))

    def test_reset_drops_the_old_episode_native_result(self):
        session = Session("libero_goal_t1")
        backend = FreeBackend(trigger_native=True, reset_on_trigger=True, session=session)
        old = _run(_plan(), backend, session)
        self.assertEqual(old["status"], "discarded")
        self.assertEqual([action for action, _args in backend.calls], ["view_env_state"])

        backend.reset_on_trigger = False
        backend.trigger_native = False
        backend.native = False
        backend.calls.clear()
        fresh = _run(_plan(), backend, session)
        actions = [action for action, _args in backend.calls]
        self.assertEqual(fresh["status"], "completed")
        self.assertTrue(any(action == "set_gripper" for action in actions))
        self.assertTrue(any(action == "move_to" for action in actions))


if __name__ == "__main__":
    unittest.main()
