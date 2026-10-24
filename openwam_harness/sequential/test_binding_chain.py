"""Replay the saved bowl/stove plan and the binding contract. No model and no env.step."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from candidates import resolve_plan_bindings, scene_graph
from executor import Session, ToolOutcome, run_plan
from finalize import finalize_run
from schema import PlanError, prepare_plan
from site_paths import site
from suites import _catalog, objects_for_names

ROOT = Path(__file__).resolve().parent
SAVED = ROOT / "fixtures" / "planner_seq_68c884edeb80.json"
STATES = ROOT / "fixtures" / "goal_task1_state_names.json"
BDDL = site("liberopro", "liberopro", "bddl_files", "libero_goal", "put_the_bowl_on_the_stove.bddl")


def _saved_arguments(role: str) -> dict:
    payload = json.loads(SAVED.read_text(encoding="utf-8"))
    for attempt in payload["attempts"]:
        if attempt.get("role") == role or (role == "format_repair" and attempt.get("request_id") == "seq-68c884edeb80"):
            return json.loads(attempt["tool_calls"][0]["function"]["arguments"])
    raise AssertionError(role)


def _goal_scene():
    catalog = _catalog(str(BDDL))
    task = {
        "instances": catalog["instances"],
        "fixtures": catalog["fixtures"],
        "place_regions": catalog["place_regions"],
    }
    state = json.loads(STATES.read_text(encoding="utf-8"))
    names = list(state["steps"][-1]["state"]["object_names"])
    for item in task["instances"] + task["fixtures"]:
        if item["name"] not in names:
            names.append(item["name"])
    objects = objects_for_names(names, task)
    graph = scene_graph(objects, {}, scene_epoch=2, observation_id="874f6a80bddf:epoch2")
    return objects, graph, names


class BindingReplayTest(unittest.TestCase):
    def test_saved_repair_binds_bowl_and_stove_site(self):
        objects, graph, names = _goal_scene()
        self.assertEqual(names[0], "akita_black_bowl_1")
        self.assertEqual(names[6], "flat_stove_1")
        self.assertEqual(objects["flat_stove_1"]["place_site"], "flat_stove_1_cook_region")
        from scene_request import resolve_load
        from suites import load_suites

        loaded = resolve_load({"suite": "libero_goal", "task_id": 1, "seed": 0}, load_suites())
        self.assertEqual(
            objects_for_names(["akita_black_bowl_1", "flat_stove_1"], loaded)["flat_stove_1"]["place_site"],
            "flat_stove_1_cook_region",
        )
        arguments = _saved_arguments("format_repair")
        self.assertEqual(arguments["steps"][0]["object_n"], 1)
        self.assertEqual(arguments["steps"][0]["target_n"], 7)
        resolved, bindings = resolve_plan_bindings(arguments, graph)
        by_role = {item["role"]: item for item in bindings}
        self.assertEqual(by_role["object"]["instance_ref"], "akita_black_bowl_1")
        self.assertEqual(by_role["object"]["candidate_id"], 1)
        self.assertEqual(by_role["target"]["instance_ref"], "flat_stove_1")
        self.assertEqual(by_role["target"]["query_key"], "flat_stove_1_cook_region")
        self.assertEqual(by_role["object"]["pose"]["status"], "not_sampled")
        self.assertTrue(resolved["bindings_checked"])
        prepared = prepare_plan(resolved, task=arguments["goal"], scene_objects=objects, execution_mode="free")
        segment_targets = [step for step in prepared["steps"] if step["skill"] == "segment"]
        self.assertEqual(segment_targets[0]["args"]["object_ref"], "akita_black_bowl_1")
        self.assertEqual(segment_targets[1]["query_key"], "flat_stove_1_cook_region")
        page = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
        self.assertIn("Planner输入/输出", page)
        self.assertNotIn("确认成功", page)
        self.assertNotIn("执行诊断", page)

    def test_contract_rejects_type_name_and_stale_version(self):
        objects, graph, _names = _goal_scene()
        step = {
            "step_id": "bowl",
            "skill": "transfer",
            "object_n": 1,
            "target_n": 7,
            "object_ref": "akita_black_bowl",
            "target_ref": "flat_stove_1",
            "relation": "on",
            "prompt": "pick up the black bowl",
            "depends_on": [],
            "completion_check": "object_placed",
            "budget": {"max_s": 30, "max_attempts": 1},
            "on_failure": "stop",
        }
        plan = {"plan_id": "bad-type", "goal": "把碗放在炉灶上", "steps": [step]}
        with self.assertRaises(PlanError) as raised:
            resolve_plan_bindings(plan, graph)
        self.assertTrue(any(item["code"] == "type_name_not_instance" for item in raised.exception.errors))
        good, _bindings = resolve_plan_bindings(
            {**plan, "steps": [{**step, "object_ref": "akita_black_bowl_1"}]},
            graph,
        )
        self.assertEqual(good["steps"][0]["object_ref"], "akita_black_bowl_1")
        self.assertEqual(good["steps"][0]["target_ref"], "flat_stove_1")
        stale = dict(good)
        stale["candidate_set_version"] = "not-this-scene"
        with self.assertRaises(PlanError) as old:
            resolve_plan_bindings(stale, graph)
        self.assertEqual(old.exception.errors[0]["code"], "stale_candidate_set")
        self.assertIn("flat_stove_1", objects)

    def test_missing_pose_does_not_ask_for_grasp_confirmation(self):
        asked = []
        plan = {
            "plan_id": "pose",
            "goal": "把碗放在炉灶上",
            "goal_relation": "on",
            "execution_mode": "free",
            "predicates": [{"object_ref": "akita_black_bowl_1", "target_ref": "flat_stove_1", "relation": "on", "hold": "end"}],
            "objects": {
                "akita_black_bowl_1": {
                    "sim_name": "akita_black_bowl_1",
                    "phrases": ["black bowl"],
                    "movable": True,
                    "role": "movable",
                    "semantic_type": "akita_black_bowl",
                },
                "flat_stove_1": {
                    "sim_name": "flat_stove_1",
                    "phrases": ["stove"],
                    "movable": False,
                    "role": "support",
                    "semantic_type": "flat_stove",
                    "place_site": "flat_stove_1_cook_region",
                    "place_site_source": "bddl_goal_atom",
                },
            },
            "steps": [{
                "step_id": "seg_flat_stove_1",
                "skill": "segment",
                "args": {"prompt": "stove", "camera": "agentview", "object_ref": "flat_stove_1"},
                "object_ref": "flat_stove_1",
                "depends_on": [],
                "preconditions": [],
                "completion_check": "object_localized",
                "budget": {"max_s": 30, "max_attempts": 1},
                "on_failure": "stop",
            }],
        }
        result = run_plan(plan, _Pose(), Session("scene"), lambda name: asked.append(name) or "success", task=plan["goal"])
        self.assertEqual(asked, [])
        self.assertEqual(result["execution"]["reason_code"], "pose_not_provided")
        self.assertEqual(result["user_goal_status"], "unknown")
        self.assertNotEqual(result["execution_status"], "discarded")
        self.assertFalse(result["final_check"]["extra_step"])

    def test_timeout_and_stop_keep_a_full_report(self):
        plan = {"execution_mode": "free", "predicates": [], "objects": {}}
        timeout = finalize_run({"status": "discarded", "reason": "stale confirmation discarded", "steps": []}, backend=None, plan=plan, exit_kind="confirm_timeout")
        self.assertEqual(timeout["execution_status"], "confirm_timeout")
        self.assertEqual(timeout["user_goal_status"], "unknown")
        self.assertIsNone(timeout["native_task_success"])
        self.assertIsNone(timeout["retreat_error"])
        self.assertEqual(timeout["evidence_source"], "unavailable")
        self.assertTrue(timeout["unknown_reason"])
        again = finalize_run(timeout, backend=_Explode(), plan=plan, exit_kind="confirm_timeout")
        self.assertEqual(again["execution_status"], "confirm_timeout")
        stopped = finalize_run({"status": "discarded", "steps": [{"step_id": "place", "stage": "place", "verification": {"reason_code": "PLACED", "status": "success"}}]}, backend=None, plan=plan, exit_kind="stopped", first_error="已取消")
        self.assertEqual(stopped["execution_status"], "stopped")
        self.assertEqual(stopped["user_goal_status"], "unknown")
        self.assertEqual(stopped["completed_subgoals"], ["place"])
        self.assertEqual(stopped["first_error"], "已取消")
        fresh = __import__("run_context", fromlist=["RunContext"]).RunContext()
        fresh.user_goal_status = "met"
        fresh.begin(2, "next", 3, "", "free")
        self.assertIsNone(fresh.user_goal_status)


class _Pose:
    def __init__(self):
        self.steps = 0

    def execute(self, action, args):
        self.steps += 1
        return ToolOutcome(executed=True, xyz=[-0.2, 0.2, 0.9], real=True)

    def observe(self):
        return {
            "observation_id": "pose-1",
            "env_step": 1,
            "timestamp": 1,
            "cameras": {"agentview": {"id": "a"}, "wrist": {"id": "w"}},
            "sim": {"objects": {"akita_black_bowl_1": {"pos": [0, 0, 0.9], "identified": True}}},
            "named_poses": {"bodies": {"flat_stove_1": [0, 0, 0]}, "sites": {}},
        }

    def sim_state(self):
        return self.observe()["sim"]

    def native_success(self):
        return False

    def env_step(self):
        return 1


class _Explode:
    def observe(self):
        raise AssertionError("finalize must not read twice")

    def sim_state(self):
        raise AssertionError("finalize must not read twice")


if __name__ == "__main__":
    unittest.main()
