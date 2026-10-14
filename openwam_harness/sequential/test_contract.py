"""CPU contracts for the sequential budget, plan schema, and executor."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from budget import BudgetError, SKILL_CONSTRAINTS, budget_check, slim_prompt
from executor import Session, ToolOutcome, run_plan
from planner import load_human_plan, request_plan
from schema import PlanError, parse_plan_text, split_move, validate_plan

PLAN = Path(__file__).resolve().parent / "plans" / "two_object.json"


class Scripted:
    def __init__(self, fail_times=None, native=False):
        self.fail_times = dict(fail_times or {})
        self.native = native
        self.calls = []
        self.observations = 0

    def execute(self, action, args):
        self.calls.append(action)
        left = self.fail_times.get(action, 0)
        if left > 0:
            self.fail_times[action] = left - 1
            return ToolOutcome(executed=False, error=f"{action} failed")
        xyz = [0.06, 0.23, 0.49] if action == "segment" else None
        return ToolOutcome(
            executed=True,
            grasp_signal=True if action == "pi0_pick" else None,
            native_terminated=self.native,
            xyz=xyz,
            eef_pos=[0.0, 0.0, 0.62],
            raw={"success": True, "task_success": True} if action == "pi0_pick" else {},
        )

    def observe(self):
        self.observations += 1
        return {"llm": False}

    def native_success(self):
        return self.native


class ContractTest(unittest.TestCase):
    def test_human_plan_rejects_code_and_pi0_doubled(self):
        plan = load_human_plan(PLAN, "libero_10_t0")
        self.assertNotIn("pi0_doubled", {step["action"] for step in plan["steps"]})
        bad = json.loads(json.dumps(plan))
        bad["steps"].append({"id": "hack", "action": "pi0_doubled", "args": {}})
        with self.assertRaises(PlanError):
            validate_plan(bad)
        coded = json.loads(json.dumps(plan))
        coded["steps"][0]["args"] = {"code": "print(1)"}
        with self.assertRaises(PlanError):
            validate_plan(coded)

    def test_budget_keeps_task_and_fails_locally(self):
        parts = slim_prompt(
            task="put both the alphabet soup and the tomato sauce in the basket",
            failure_evidence="previous grasp missed; gripper stayed open",
            tool_schema="pi0_pick, move_to, set_gripper",
        )
        text = "\n".join(part for _, part in parts)
        self.assertIn("alphabet soup", text)
        self.assertIn(SKILL_CONSTRAINTS, text)
        self.assertIn("gripper stayed open", text)
        self.assertNotIn("MEMORY.md", text)
        report = budget_check(parts, counter=lambda item: len(item.split()))
        self.assertEqual(report["output_budget"], 2048)
        self.assertGreater(report["input_tokens"], 0)
        with self.assertRaises(BudgetError):
            budget_check([("task", "x" * 100)], counter=lambda item: 40000)

    def test_plan_once_then_one_repair_then_human(self):
        human = load_human_plan(PLAN, "libero_10_t0")
        calls = []

        def client(messages, max_tokens):
            calls.append(max_tokens)
            return "not json" if len(calls) == 1 else "still not json"

        result = request_plan(
            task=human["task"],
            failure_evidence="context error on the previous episode",
            human_plan=human,
            client=client,
            counter=lambda item: 10,
        )
        self.assertEqual(calls, [2048, 2048])
        self.assertEqual(result["source"], "human_fallback")
        self.assertEqual(len(result["budgets"]), 2)

    def test_valid_model_plan_does_not_repair(self):
        human = load_human_plan(PLAN, "libero_10_t1")

        def client(messages, max_tokens):
            return json.dumps({"steps": human["steps"]})

        result = request_plan(
            task=human["task"],
            failure_evidence="planner timed out before pi0_pick",
            human_plan=human,
            client=client,
            counter=lambda item: 8,
        )
        self.assertEqual(result["source"], "model")

    def test_two_object_pauses_and_does_not_call_llm(self):
        plan = load_human_plan(PLAN, "libero_10_t0")
        backend = Scripted()
        session = Session("libero_10_t0")
        session.begin_episode()
        verdicts = []

        def confirm(subtask):
            verdicts.append(subtask)
            return "unknown"

        result = run_plan(plan, backend, session, confirm)
        self.assertEqual(verdicts, ["soup_place", "sauce_place"])
        self.assertEqual(result["llm_calls_during_execution"], 0)
        self.assertFalse(result["native_success"])
        self.assertGreater(backend.observations, 0)
        picks = [step for step in result["steps"] if step["action"] == "pi0_pick"]
        self.assertTrue(all(step["grasp_signal"] for step in picks))
        self.assertTrue(all(step["subtask_status"] == "unknown" for step in picks))
        self.assertNotIn("reset", backend.calls)
        places = [step for step in result["steps"] if step["action"] == "pause_for_confirm"]
        self.assertEqual([step["subtask_status"] for step in places], ["unknown", "unknown"])

    def test_no_progress_and_stale_confirm(self):
        plan = {
            "steps": [
                {"id": "pick", "action": "pi0_pick", "args": {"prompt": "pick up the soup"}, "subtask": "soup_pick"}
            ]
        }
        backend = Scripted(fail_times={"pi0_pick": 2})
        session = Session("libero_10_t0")
        session.begin_episode()
        result = run_plan(plan, backend, session, lambda _subtask: "success")
        self.assertEqual(result["status"], "no_progress")
        self.assertEqual(backend.calls.count("pi0_pick"), 2)

        session = Session("libero_10_t0")
        session.begin_episode()
        backend = Scripted()

        def confirm(_subtask):
            session.scene_epoch += 1
            return "success"

        plan = load_human_plan(PLAN, "libero_10_t0")
        result = run_plan(plan, backend, session, confirm)
        self.assertEqual(result["status"], "discarded")

    def test_long_move_is_split_without_code(self):
        points = split_move([0.0, 0.0, 0.6], [0.0, 0.8, 0.6])
        self.assertGreater(len(points), 1)
        self.assertEqual(points[-1], [0.0, 0.8, 0.6])
        self.assertLessEqual(max(
            abs(points[i][1] - (points[i - 1][1] if i else 0.0)) for i in range(len(points))
        ), 0.30)

    def test_parse_plan_text(self):
        plan = load_human_plan(PLAN, "libero_10_t1")
        parsed = parse_plan_text("note\n" + json.dumps(plan))
        self.assertEqual(parsed["task"], plan["task"])


if __name__ == "__main__":
    unittest.main()
