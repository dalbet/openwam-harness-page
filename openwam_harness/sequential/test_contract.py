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
        raw = {}
        if action == "pi0_pick":
            raw = {"success": True}
            if self.real and self.held and not getattr(self, "skip_samples", False):
                raw["samples"] = self._grasp_samples()
        return ToolOutcome(
            executed=True,
            grasp_signal=True if action == "pi0_pick" else None,
            native_terminated=self.native,
            xyz=xyz,
            eef_pos=list(self.eef),
            raw=raw,
            real=self.real,
        )

    def _grasp_samples(self):
        sim = {
            "objects": {
                name: {"pos": list(pos), "identified": True}
                for name, pos in self.objects.items()
            },
            "eef_pos": list(self.eef),
            "gripper_opening_m": self.opening,
            "contact": None,
            "evidence_source": "sim_state",
        }
        second = json.loads(json.dumps(sim))
        return [
            {"observation_id": "pick-1", "env_action_count": 5, "timestamp": 1.0, "sim": sim},
            {"observation_id": "pick-2", "env_action_count": 10, "timestamp": 2.0, "sim": second},
        ]

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
        self.assertTrue(all(step["verification"]["evidence_source"] == "sim_state" for step in picks))
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

        from schema import example_scene_objects

        expanded = prepare_plan(
            semantic_example(),
            task=semantic_example()["goal"],
            scene_objects=example_scene_objects(),
        )
        skills = [step["skill"] for step in expanded["steps"]]
        self.assertNotIn("transfer", skills)
        self.assertIn("pi0_pick", skills)
        self.assertIn("set_gripper", skills)
        with self.assertRaises(PlanError) as raised:
            parse_model_message({"content": "", "finish_reason": None, "incomplete": True})
        self.assertIn("incomplete_stream", str(raised.exception))
        saved = json.loads((Path(__file__).resolve().parent / "records" / "planner_speed.json").read_text(encoding="utf-8"))
        arguments = saved["plan_check"]["attempts"][0]["tool_calls"][0]["function"]["arguments"]
        with self.assertRaises(PlanError) as saved_error:
            parse_model_message(
                {"tool_calls": [{"function": {"name": "submit_plan", "arguments": arguments}}], "finish_reason": "stop"},
                task=semantic_example()["goal"],
                scene_objects=example_scene_objects(),
            )
        self.assertIn("unknown_field", str(saved_error.exception))

    def test_chunk_trace_counts_env_actions_separately_from_records(self):
        from chunk_trace import attach_chunk_trace, take_chunk_trace

        class Env:
            def chunk_step(self, actions):
                count = len(actions)
                return ({}, 0, [False] * count, [False] * count, {})

        primitives = type("P", (), {"env": Env()})()
        attach_chunk_trace(primitives)
        primitives.env.chunk_step([[0] * 7] * 5)
        primitives.env.chunk_step([[0] * 7] * 5)
        rows = take_chunk_trace(primitives)
        self.assertEqual([row["actions_sent"] for row in rows], [5, 5])
        self.assertEqual(rows[-1]["env_actions_after"], 10)
        self.assertEqual(rows[-1]["observation_id"], "env-actions-5:10")

    def test_bowl_capabilities_binding_and_saved_planner_output(self):
        from binding import actions_allowed, bind_task
        from schema import failure_stage, parse_model_message, prepare_plan
        from suites import load_suites
        from task_zh import SUITE_LABELS, TASK_ZH

        task = "pick up the black bowl between the plate and the ramekin and place it on the plate"
        scene = {
            "akita_black_bowl_1": _bowl("akita_black_bowl_1"),
            "akita_black_bowl_2": _bowl("akita_black_bowl_2"),
            "plate_1": {
                "sim_name": "plate_1",
                "semantic_type": "plate",
                "phrases": ["plate"],
                "movable": True,
                "can_be_target": True,
                "target_relations": ["on"],
                "capability_source": "bddl:objects",
            },
            "wooden_cabinet_1": {
                "sim_name": "wooden_cabinet_1",
                "semantic_type": "wooden_cabinet",
                "phrases": ["wooden cabinet", "cabinet"],
                "movable": False,
                "can_be_target": True,
                "target_relations": ["on", "inside_drawer"],
                "capability_source": "bddl:fixtures",
            },
        }
        prepared = prepare_plan(_transfer("held", "akita_black_bowl_1", "plate_1", task), task=task, scene_objects=scene, required_object_ref="akita_black_bowl_1")
        self.assertIn("pi0_pick", [step["skill"] for step in prepared["steps"]])
        self.assertEqual(scene["akita_black_bowl_1"]["capability_source"], "bddl:objects")
        with self.assertRaises(PlanError) as fixed:
            prepare_plan(_transfer("fixed", "wooden_cabinet_1", "plate_1", task), task=task, scene_objects=scene)
        self.assertIn("not_movable", str(fixed.exception))
        self.assertIn("bddl:fixtures", str(fixed.exception))

        wrapped = {"submit_plan": _transfer("inner", "akita_black_bowl_1", "plate_1", task)}
        with self.assertRaises(PlanError) as nested:
            parse_model_message(
                {"tool_calls": [{"function": {"name": "submit_plan", "arguments": json.dumps(wrapped)}}], "finish_reason": "stop"},
                task=task,
                scene_objects=scene,
            )
        self.assertIn("nested_tool", str(nested.exception))
        self.assertEqual(failure_stage(nested.exception.errors), "parse_failed")

        duplicated = _transfer("pick", "akita_black_bowl_1", "plate_1", task)
        duplicated["steps"].append(dict(duplicated["steps"][0], step_id="place"))
        with self.assertRaises(PlanError) as duplicate:
            prepare_plan(duplicated, task=task, scene_objects=scene, required_object_ref="akita_black_bowl_1")
        self.assertIn("duplicate_transfer", str(duplicate.exception))

        plate = [0.06, 0.20, 0.45]
        ramekin = [-0.20, 0.20, 0.45]
        objects = dict(scene)
        objects["glazed_rim_porcelain_ramekin_1"] = {
            "semantic_type": "glazed_rim_porcelain_ramekin",
            "phrases": ["ramekin"],
            "movable": True,
            "can_be_target": True,
            "target_relations": ["on", "in"],
            "capability_source": "bddl:objects",
        }
        ambiguous = bind_task(
            task,
            objects,
            {"plate_1": plate, "glazed_rim_porcelain_ramekin_1": ramekin, "akita_black_bowl_1": [-0.05, 0.20, 0.45], "akita_black_bowl_2": [-0.07, 0.20, 0.45]},
            observation_id="obs-1",
        )
        self.assertEqual(ambiguous["status"], "ambiguous")
        self.assertFalse(actions_allowed(ambiguous))
        resolved = bind_task(
            task,
            objects,
            {"plate_1": plate, "glazed_rim_porcelain_ramekin_1": ramekin, "akita_black_bowl_1": [-0.05, 0.20, 0.45], "akita_black_bowl_2": [-0.18, 0.32, 0.45]},
            observation_id="obs-2",
        )
        self.assertEqual(resolved["status"], "resolved")
        self.assertEqual(resolved["object_ref"], "akita_black_bowl_1")
        self.assertEqual(resolved["evidence_source"], "sim_state")
        self.assertFalse(resolved["region_seen"])

        saved = json.loads((Path(__file__).resolve().parent / "records" / "planner_bowl_failure.json").read_text(encoding="utf-8"))
        self.assertFalse(saved["initial"]["stored"])
        self.assertIsNone(saved["initial"]["request_id"])
        arguments = saved["repair"]["arguments"]
        self.assertTrue(all(step.get("skill") != "submit_plan" for step in arguments["steps"]))
        with self.assertRaises(PlanError) as replay:
            parse_model_message(
                {"tool_calls": [{"id": saved["repair"]["tool_call_id"], "function": {"name": "submit_plan", "arguments": json.dumps(arguments)}}], "finish_reason": "stop"},
                task=task,
                scene_objects=objects,
            )
        self.assertIn("duplicate_transfer", str(replay.exception))
        self.assertIn("object_ref", str(replay.exception))

        suites = load_suites()
        labels = [item["label"] for item in suites]
        self.assertEqual(labels, ["空间任务", "对象任务", "目标任务", "长程任务"])
        languages = []
        for suite in suites:
            self.assertEqual(suite["label"], SUITE_LABELS[suite["id"]])
            for item in suite["tasks"]:
                languages.append(item["language"])
                self.assertEqual(item["summary_zh"], TASK_ZH[item["language"]])
                self.assertNotEqual(item["summary_zh"], item["language"])
                self.assertTrue(item["task_key"].startswith(suite["id"]))
        self.assertEqual(len(languages), 40)
        spatial = suites[0]["tasks"][0]
        self.assertEqual(spatial["summary_zh"], "抓起位于盘子与小碗之间的黑碗，并放到盘子上。")
        self.assertEqual(spatial["language"], task)
        self.assertIn("akita_black_bowl_1", [row["name"] for row in spatial["instances"]])


    def test_segment_miss_keeps_the_original_error(self):
        from live import attach_saved_pick_result, interpret_exception, interpret_tool_data, tool_arguments
        from schema import prepare_plan
        from verify import verify_step

        task = "pick up the black bowl between the plate and the ramekin and place it on the plate"
        scene = {
            "akita_black_bowl_1": _bowl("akita_black_bowl_1"),
            "akita_black_bowl_2": _bowl("akita_black_bowl_2"),
            "plate_1": {
                "sim_name": "plate_1",
                "semantic_type": "plate",
                "phrases": ["plate"],
                "movable": True,
                "can_be_target": True,
                "target_relations": ["on"],
                "capability_source": "bddl:objects",
            },
        }
        prepared = prepare_plan(
            _transfer("transfer_bowl", "akita_black_bowl_1", "plate_1", task),
            task=task,
            scene_objects=scene,
            required_object_ref="akita_black_bowl_1",
        )
        prepared["planner_stage"] = "ok"
        prepared["request_id"] = "seq-fdb0bf5e8750"
        segment = next(step for step in prepared["steps"] if step["skill"] == "segment" and step["object_ref"] == "akita_black_bowl_1")
        pick = next(step for step in prepared["steps"] if step["skill"] == "pi0_pick")
        self.assertEqual(segment["grounding_query"], "black bowl")
        self.assertEqual(segment["args"]["prompt"], "black bowl")
        self.assertNotIn("akita_black_bowl", tool_arguments(segment["args"])["prompt"])
        self.assertNotIn("object_ref", tool_arguments(segment["args"]))
        self.assertEqual(pick["vla_prompt"], "pick up the black bowl")
        self.assertEqual(tool_arguments(pick["args"])["prompt"], "pick up the black bowl")
        self.assertNotEqual(tool_arguments(pick["args"])["prompt"], "akita_black_bowl_1")

        miss = interpret_tool_data("segment", {
            "found": False,
            "score": 0.08,
            "world_xyz": None,
            "error": "top score 0.080 is below min_score 0.200",
        })
        self.assertTrue(miss["executed"])
        self.assertEqual(miss["category"], "subtask_incomplete")
        self.assertIn("0.080", miss["error"])
        grasp = interpret_tool_data("pi0_pick", {"success": False, "terminated": True})
        self.assertTrue(grasp["executed"])
        self.assertFalse(grasp["grasp_signal"])
        self.assertIsNone(grasp["category"])
        restored = attach_saved_pick_result(
            {"env_action_count": 55},
            {"success": True, "chunks_used": 11, "peak_lift_m": 0.065, "terminated": True},
        )
        self.assertNotIn("terminated", restored)
        self.assertEqual(restored["tool_result"]["terminated"], True)
        self.assertEqual(restored["observation"]["env_action_count"], 55)
        self.assertNotIn("success", restored["observation"])
        restored_grasp = interpret_tool_data("pi0_pick", restored)
        self.assertTrue(restored_grasp["executed"])
        self.assertTrue(restored_grasp["grasp_signal"])
        blown = interpret_exception(RuntimeError("segment() got an unexpected keyword argument 'object_ref'"))
        self.assertFalse(blown["executed"])
        self.assertEqual(blown["category"], "tool_exception")

        class Miss:
            def __init__(self):
                self.calls = []

            def execute(self, action, args):
                self.calls.append(action)
                if action == "segment":
                    return ToolOutcome(
                        executed=True,
                        error="top score 0.080 is below min_score 0.200",
                        xyz=None,
                        category="subtask_incomplete",
                        raw={"found": False, "score": 0.08},
                    )
                return ToolOutcome(executed=True)

            def observe(self):
                return {"cameras": {"agentview": {"id": "a"}, "wrist": {"id": "w"}}, "sim": {"objects": {}}}

            def native_success(self):
                return False

        backend = Miss()
        events = []
        result = run_plan(
            prepared,
            backend,
            Session("scene"),
            lambda _subtask: "unknown",
            task=task,
            on_event=events.append,
        )
        waiting = [event for event in events if event["phase"] == "waiting_confirmation"]
        self.assertEqual(result["status"], "no_progress")
        self.assertIn("0.080", result["reason"])
        self.assertNotIn("tool_failed", result["reason"])
        self.assertEqual(waiting[0]["reason_code"], "TARGET_NOT_IDENTIFIED")
        self.assertIn("0.080", waiting[0]["reason"])
        self.assertEqual(result["execution"]["category"], "subtask_incomplete")
        self.assertEqual(result["execution"]["request_id"], "seq-fdb0bf5e8750")
        self.assertEqual(backend.calls.count("segment"), 1)
        self.assertNotIn("pi0_pick", backend.calls)
        self.assertNotIn("move_to", backend.calls)

        class Boom:
            def execute(self, action, args):
                if action == "segment":
                    return ToolOutcome(
                        executed=False,
                        error="segment() got an unexpected keyword argument 'object_ref'",
                        error_class="service",
                        category="tool_exception",
                    )
                return ToolOutcome(executed=True)

            def observe(self):
                return {"cameras": {"agentview": {"id": "a"}, "wrist": {"id": "w"}}, "sim": {"objects": {}}}

            def native_success(self):
                return False

        raised = run_plan(prepared, Boom(), Session("scene"), lambda _subtask: None, task=task)
        self.assertEqual(raised["execution"]["reason_code"], "TOOL_EXCEPTION")
        self.assertIn("unexpected keyword", raised["reason"])
        self.assertNotIn("tool_failed", raised["reason"])

        def obs(points):
            return {
                "observation_id": "obs-1",
                "scene_epoch": 1,
                "env_step": 1,
                "cameras": {"agentview": {"id": "a"}, "wrist": {"id": "w"}},
                "sim": {"objects": {name: {"pos": pos, "identified": True} for name, pos in points.items()}},
            }

        local = {
            "step_id": "seg_akita_black_bowl_1",
            "skill": "segment",
            "object_ref": "akita_black_bowl_1",
            "completion_check": "object_localized",
            "args": {"prompt": "black bowl"},
            "grounding_query": "black bowl",
        }
        points = {
            "akita_black_bowl_1": [-0.05, 0.20, 0.45],
            "akita_black_bowl_2": [-0.18, 0.32, 0.45],
        }
        matched = verify_step(
            local,
            ToolOutcome(executed=True, xyz=[-0.05, 0.20, 0.46]),
            [obs(points)],
            scene_epoch=1,
            min_step=0,
            objects=scene,
            support_z={},
            goal_relation="on",
        )
        self.assertEqual(matched["reason_code"], "OBJECT_LOCALIZED")
        self.assertEqual(matched["evidence_source"], "sim_state")
        self.assertEqual(matched["metrics"]["object_ref"], "akita_black_bowl_1")
        other = verify_step(
            local,
            ToolOutcome(executed=True, xyz=[-0.18, 0.32, 0.45]),
            [obs(points)],
            scene_epoch=1,
            min_step=0,
            objects=scene,
            support_z={},
            goal_relation="on",
        )
        self.assertEqual(other["reason_code"], "visual_instance_ambiguous")
        self.assertEqual(other["status"], "clarify")
        self.assertEqual(other["metrics"]["object_ref"], "akita_black_bowl_1")
        self.assertIn("不更换", other["metrics"]["error"])

        # A tall instance: the grounded point sits on the visible body, above
        # the base origin. The xy plane still confirms the bound instance.
        bottle_scene = {"wine_bottle_1": {
            "sim_name": "wine_bottle_1",
            "semantic_type": "wine_bottle",
            "phrases": ["wine bottle"],
            "movable": True,
            "can_be_target": False,
        }}
        bottle_step = {
            "step_id": "seg_wine_bottle_1",
            "skill": "segment",
            "object_ref": "wine_bottle_1",
            "completion_check": "object_localized",
            "args": {"prompt": "wine bottle"},
            "grounding_query": "wine bottle",
        }
        bottle_obs = obs({"wine_bottle_1": [-0.1887, -0.0495, 0.8989]})
        tall = verify_step(
            bottle_step,
            ToolOutcome(executed=True, xyz=[-0.1764, -0.05, 0.9702]),
            [bottle_obs],
            scene_epoch=1,
            min_step=0,
            objects=bottle_scene,
            support_z={},
            goal_relation="on",
        )
        self.assertEqual(tall["reason_code"], "OBJECT_LOCALIZED")
        self.assertEqual(tall["metrics"]["distance_plane"], "xy")
        self.assertAlmostEqual(tall["metrics"]["bound_distance_m"], 0.0123, places=3)
        self.assertGreater(tall["metrics"]["bound_distance_3d_m"], 0.05)

        missed = verify_step(
            bottle_step,
            ToolOutcome(executed=True, xyz=[0.15, 0.2, 0.9702]),
            [bottle_obs],
            scene_epoch=1,
            min_step=0,
            objects=bottle_scene,
            support_z={},
            goal_relation="on",
        )
        self.assertEqual(missed["reason_code"], "localization_unconfirmed")


def _bowl(name: str) -> dict:
    return {
        "sim_name": name,
        "semantic_type": "akita_black_bowl",
        "phrases": ["black bowl", "akita black bowl"],
        "movable": True,
        "can_be_target": True,
        "target_relations": ["in", "on"],
        "capability_source": "bddl:objects",
    }


def _transfer(step_id: str, obj: str, target: str, task: str) -> dict:
    return {
        "plan_id": step_id,
        "goal": task,
        "goal_relation": "on",
        "steps": [{
            "step_id": step_id,
            "skill": "transfer",
            "object_ref": obj,
            "target_ref": target,
            "relation": "on",
            "prompt": "pick up the black bowl",
            "depends_on": [],
            "preconditions": [],
            "completion_check": "object_placed",
            "budget": {"max_s": 180, "max_attempts": 2},
            "on_failure": "recover",
        }],
    }


if __name__ == "__main__":
    unittest.main()
