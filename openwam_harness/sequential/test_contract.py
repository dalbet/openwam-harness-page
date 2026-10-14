"""CPU contracts for the sequential budget, plan schema, and executor."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from budget import BudgetError, SKILL_CONSTRAINTS, budget_check, slim_prompt
from executor import Session, ToolOutcome, run_plan
from planner import load_human_plan, request_plan
from schema import PlanError, example_plan, parse_model_message, parse_plan_text, split_move, validate_plan

PLAN = Path(__file__).resolve().parent / "plans" / "two_object.json"


SIM = {
    "alphabet_soup": "alphabet_soup_1",
    "tomato_sauce": "tomato_sauce_1",
    "cream_cheese": "cream_cheese_1",
    "butter": "butter_1",
    "basket": "basket_1",
}


class Scripted:
    def __init__(self, fail_times=None, native=False, real=True, empty=False, blind=False):
        self.fail_times = dict(fail_times or {})
        self.native = native
        self.real = real
        self.empty = empty
        self.blind = blind
        self.calls = []
        self.observations = 0
        self.objects = {
            "alphabet_soup_1": [0.10, -0.10, 0.45],
            "tomato_sauce_1": [-0.07, 0.04, 0.45],
            "basket_1": [0.06, 0.23, 0.49],
            "cream_cheese_1": [0.08, -0.16, 0.45],
            "butter_1": [-0.02, -0.18, 0.45],
        }
        self.eef = [0.0, 0.0, 0.70]
        self.opening = 0.078
        self.held = None

    def execute(self, action, args):
        self.calls.append(action)
        left = self.fail_times.get(action, 0)
        if left > 0:
            self.fail_times[action] = left - 1
            return ToolOutcome(executed=False, error=f"{action} failed", real=self.real, error_class="physical")
        sim = SIM.get(args.get("object_ref"))
        if self.real and action == "pi0_pick" and sim and not self.empty:
            self.objects[sim] = [0.0, 0.0, 0.62]
            self.eef = [0.0, 0.0, 0.62]
            self.opening = 0.02
            self.held = sim
        elif self.real and action == "pi0_pick" and self.empty:
            self.opening = 0.001
            self.eef = [0.0, 0.0, 0.62]
        elif self.real and action == "move_to" and self.held:
            self.eef = [float(value) for value in args["xyz"]]
            self.objects[self.held] = list(self.eef)
        elif self.real and action == "set_gripper" and float(args.get("gripper", 0)) < 0 and self.held:
            self.opening = 0.078
            self.objects[self.held] = [0.06, 0.23, 0.50]
            self.held = None
        xyz = list(self.objects[sim]) if action == "segment" and sim else None
        return ToolOutcome(
            executed=True,
            grasp_signal=True if action == "pi0_pick" else None,
            native_terminated=self.native,
            xyz=xyz,
            eef_pos=list(self.eef),
            raw={"success": True} if action == "pi0_pick" else {},
            real=self.real,
        )

    def observe(self):
        self.observations += 1
        objects = {}
        for name, pos in self.objects.items():
            objects[name] = {"pos": list(pos), "identified": not self.blind}
        return {
            "llm": False,
            "cameras": {"agentview": {"id": "agentview:0"}, "wrist": {"id": "wrist:0"}},
            "sim": {
                "objects": objects,
                "eef_pos": list(self.eef),
                "gripper_opening_m": self.opening,
                "evidence_source": "sim_state_assisted",
            },
        }

    def native_success(self):
        return self.native


def _run(plan, backend, confirm):
    session = Session(plan.get("scene_id") or "libero_10_t0")
    session.begin_episode()
    return run_plan(plan, backend, session, confirm, task=plan.get("goal"))


class ContractTest(unittest.TestCase):
    def test_human_plan_rejects_code_and_pi0_doubled(self):
        plan = load_human_plan(PLAN, "libero_10_t0")
        self.assertEqual(plan["plan_source"], "manual")
        self.assertNotIn("pi0_doubled", {step["skill"] for step in plan["steps"]})
        bad = json.loads(json.dumps(plan))
        bad["steps"].append({
            "step_id": "hack",
            "skill": "pi0_doubled",
            "args": {},
            "depends_on": [],
            "preconditions": [],
            "completion_check": "grasp_held",
            "budget": {"max_s": 10, "max_attempts": 1},
            "on_failure": "stop",
        })
        with self.assertRaises(PlanError) as raised:
            validate_plan(bad, task=bad["goal"])
        self.assertIn("skill_rejected", str(raised.exception))
        coded = json.loads(json.dumps(plan))
        coded["steps"][0]["args"] = {"code": "print(1)"}
        with self.assertRaises(PlanError) as raised:
            validate_plan(coded, task=coded["goal"])
        self.assertIn("executable_field", str(raised.exception))

    def test_budget_keeps_task_and_fails_locally(self):
        parts = slim_prompt(
            task="put both the alphabet soup and the tomato sauce in the basket",
            scene_observation="objects: alphabet_soup, tomato_sauce, basket; image_ids: agentview:0",
            tool_schema="pi0_pick, move_to, set_gripper",
            short_memory="previous grasp missed; gripper stayed open",
            output_format="call submit_plan",
            example=json.dumps(example_plan()),
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

    def test_plan_once_then_one_repair_then_stop(self):
        calls = []

        def client(messages, max_tokens, options=None):
            calls.append(max_tokens)
            self.assertIn("enable_thinking", json.dumps(options))
            return "not json" if len(calls) == 1 else "still not json"

        result = request_plan(
            task="put both the alphabet soup and the tomato sauce in the basket",
            scene_observation="image_ids: none",
            client=client,
            counter=lambda item: 10,
        )
        self.assertEqual(calls, [2048, 2048])
        self.assertEqual(result["status"], "PLANNING_FAILED")
        self.assertIsNone(result["plan"])
        self.assertEqual(result["repairs"], 1)
        self.assertNotEqual(result["source"], "human_fallback")
        self.assertTrue(any(item["code"] == "json_decode" for item in result["errors"]))

    def test_valid_model_plan_does_not_repair(self):
        human = load_human_plan(PLAN, "libero_10_t1")

        def client(messages, max_tokens, options=None):
            return json.dumps(human)

        result = request_plan(
            task=human["goal"],
            scene_observation="image_ids: agentview:0, wrist:0",
            client=client,
            counter=lambda item: 8,
        )
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["source"], "model")
        self.assertEqual(result["plan_source"], "model")
        self.assertEqual(result["repairs"], 0)

    def test_tool_call_is_not_parsed_twice(self):
        human = load_human_plan(PLAN, "libero_10_t0")
        human.pop("plan_source", None)

        def client(messages, max_tokens, options=None):
            return {
                "content": "this text is not a second plan",
                "tool_calls": [{
                    "function": {"name": "submit_plan", "arguments": json.dumps(human)}
                }],
                "finish_reason": "tool_calls",
                "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
            }

        result = request_plan(
            task=human["goal"],
            scene_observation="image_ids: agentview:0",
            client=client,
            counter=lambda item: 4,
        )
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["response_kind"], "tool_calls")
        self.assertEqual(result["attempts"][0]["ignored_content"], "content ignored because tool_calls are present")

    def test_truncated_and_illegal_plans_are_specific(self):
        def client(messages, max_tokens, options=None):
            return {"content": "{\"plan_id\":", "finish_reason": "length", "tool_calls": None, "usage": {}}

        result = request_plan(
            task="put both the alphabet soup and the tomato sauce in the basket",
            scene_observation="image_ids: none",
            client=client,
            counter=lambda item: 3,
        )
        self.assertEqual(result["status"], "PLANNING_FAILED")
        self.assertTrue(all(item["code"] == "truncated_output" for item in result["errors"]))

        bad = json.loads(json.dumps(load_human_plan(PLAN, "libero_10_t0")))
        bad["steps"][4]["skill"] = "fly"
        bad["steps"][4]["action"] = "fly"
        with self.assertRaises(PlanError) as raised:
            validate_plan(bad, task=bad["goal"])
        self.assertIn("skill_not_registered", str(raised.exception))
        self.assertIn("fly", str(raised.exception))

        cycled = json.loads(json.dumps(load_human_plan(PLAN, "libero_10_t0")))
        cycled["steps"][0]["depends_on"] = ["confirm_alphabet_soup"]
        with self.assertRaises(PlanError) as raised:
            validate_plan(cycled, task=cycled["goal"])
        self.assertIn("cycle", str(raised.exception))

        wrong = json.loads(json.dumps(load_human_plan(PLAN, "libero_10_t0")))
        wrong["steps"][4]["args"]["max_chunks"] = 99
        with self.assertRaises(PlanError) as raised:
            validate_plan(wrong, task=wrong["goal"])
        self.assertIn("max_chunks", str(raised.exception))
        self.assertIn("99", str(raised.exception))

    def test_two_object_pauses_only_after_real_actions(self):
        plan = load_human_plan(PLAN, "libero_10_t0")
        backend = Scripted()
        verdicts = []
        phases = []

        def confirm(subtask):
            verdicts.append(subtask)
            return "unknown"

        result = _run(plan, backend, confirm)
        self.assertEqual(verdicts, ["soup_place", "sauce_place"])
        self.assertEqual(result["llm_calls_during_execution"], 0)
        self.assertFalse(result["native_success"])
        self.assertGreater(backend.observations, 0)
        picks = [step for step in result["steps"] if step["action"] == "pi0_pick"]
        self.assertTrue(all(step["grasp_signal"] for step in picks))
        self.assertTrue(all(step["verification"]["status"] == "success" for step in picks))
        self.assertTrue(all(step["verification"]["reason_code"] == "GRASP_HELD" for step in picks))
        self.assertTrue(all(step["verification"]["evidence_source"] == "sim_state_assisted" for step in picks))
        self.assertNotIn("reset", backend.calls)
        places = [step for step in result["steps"] if step["action"] == "pause_for_confirm"]
        self.assertEqual([step["confirmation_source"] for step in places], ["manual", "manual"])

        dry = Scripted(real=False)
        refused = _run(plan, dry, lambda _subtask: phases.append("confirm") or "success")
        self.assertEqual(refused["status"], "failed")
        self.assertEqual(refused["reason"], "confirmation_before_real_action")
        self.assertEqual(phases, [])

    def test_no_progress_and_stale_confirm(self):
        plan = json.loads(json.dumps(load_human_plan(PLAN, "libero_10_t0")))
        plan["steps"] = [plan["steps"][4]]
        plan["steps"][0]["depends_on"] = []
        plan["steps"][0]["preconditions"] = []
        backend = Scripted(fail_times={"pi0_pick": 2})
        result = _run(plan, backend, lambda _subtask: "success")
        self.assertEqual(result["status"], "no_progress")
        self.assertEqual(backend.calls.count("pi0_pick"), 2)

        session = Session("libero_10_t0")
        session.begin_episode()
        backend = Scripted()

        def confirm(_subtask):
            session.scene_epoch += 1
            return "success"

        plan = load_human_plan(PLAN, "libero_10_t0")
        result = run_plan(plan, backend, session, confirm, task=plan["goal"])
        self.assertEqual(result["status"], "discarded")

    def test_long_move_is_split_without_code(self):
        points = split_move([0.0, 0.0, 0.6], [0.0, 0.8, 0.6])
        self.assertGreater(len(points), 1)
        self.assertEqual(points[-1], [0.0, 0.8, 0.6])
        self.assertLessEqual(max(
            abs(points[i][1] - (points[i - 1][1] if i else 0.0)) for i in range(len(points))
        ), 0.30)

    def test_parse_plan_text_rejects_partial_fence(self):
        plan = load_human_plan(PLAN, "libero_10_t1")
        parsed = parse_plan_text("```json\n" + json.dumps(plan) + "\n```", task=plan["goal"])
        self.assertEqual(parsed["goal"], plan["goal"])
        with self.assertRaises(PlanError) as raised:
            parse_plan_text("note {\"plan_id\": \"x\"}", task=plan["goal"])
        self.assertIn("json_decode", str(raised.exception))
        validate_plan(example_plan())

    def test_semantic_transfer_expands_and_incomplete_stream_is_rejected(self):
        from schema import prepare_plan, semantic_example

        expanded = prepare_plan(semantic_example(), task=semantic_example()["goal"])
        skills = [step["skill"] for step in expanded["steps"]]
        self.assertNotIn("transfer", skills)
        self.assertIn("pi0_pick", skills)
        self.assertIn("set_gripper", skills)
        with self.assertRaises(PlanError) as raised:
            parse_model_message({"content": "", "finish_reason": None, "incomplete": True})
        self.assertIn("incomplete_stream", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
