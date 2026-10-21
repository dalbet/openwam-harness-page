"""Offline checks for catalog context, grasp prompts, and step rows. No planner and no env.step."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from candidates import resolve_plan_bindings, scene_graph
from free_task import prepare_free_request
from planner import FREE_OUTPUT_FORMAT
from schema import prepare_plan
from step_display import apply_display_event, display_steps
from suites import _catalog, objects_for_names
from task_catalog import lookup_task, planner_task_context, scene_examples, single_task_block

ROOT = Path(__file__).resolve().parent
BDDL = Path(
    "/root/Page_dmh/openwam_harness/envs/rpent/lib/python3.10/site-packages/"
    "liberopro/liberopro/bddl_files/libero_goal/put_the_bowl_on_the_stove.bddl"
)
GOAL_NAMES = [
    "akita_black_bowl_1",
    "cream_cheese_1",
    "plate_1",
    "wine_bottle_1",
    "flat_stove_1",
    "wine_rack_1",
]


def _objects():
    catalog = _catalog(str(BDDL))
    task = {
        "instances": catalog["instances"],
        "fixtures": catalog["fixtures"],
        "place_regions": catalog["place_regions"],
    }
    return objects_for_names(GOAL_NAMES, task)


class TaskCatalogTest(unittest.TestCase):
    def test_context_keeps_registry_english_and_regions(self):
        rows = planner_task_context("libero_goal", GOAL_NAMES)
        by_key = {row["task_key"]: row for row in rows}
        self.assertEqual(by_key["libero_goal:t1"]["instruction_en"], "put the bowl on the stove")
        self.assertEqual(by_key["libero_goal:t1"]["target_region"], "flat_stove_1_cook_region")
        self.assertIn("akita_black_bowl_1", [row["source_object"] for row in rows])
        self.assertIn("canonical_instruction", FREE_OUTPUT_FORMAT)
        self.assertNotIn("resolve_instruction", FREE_OUTPUT_FORMAT)

    def test_paraphrase_is_not_rejected_before_the_planner(self):
        ready = prepare_free_request("碗放炉灶上", _objects())
        self.assertEqual(ready["status"], "ready")
        self.assertFalse(ready["pre_gate"])
        self.assertEqual(ready["user_text"], "碗放炉灶上")

    def test_canonical_sentence_stays_out_of_the_grasp_prompt(self):
        objects = _objects()
        graph = scene_graph(objects, {}, scene_epoch=1, observation_id="catalog:epoch1")
        plan = {
            "plan_id": "bowl-stove",
            "goal": "碗放炉灶上",
            "canonical_instruction": "put the bowl on the stove",
            "matched_task_key": "libero_goal:t1",
            "user_instruction": "碗放炉灶上",
            "steps": [
                {
                    "step_id": "bowl",
                    "skill": "transfer",
                    "object_n": 1,
                    "target_n": 5,
                    "relation": "on",
                    "prompt": "put the bowl on the stove",
                    "vla_prompt": "put the bowl on the stove",
                    "grounding_query": "black bowl",
                    "user_phrase": "碗放炉灶上",
                    "depends_on": [],
                    "completion_check": "object_placed",
                    "budget": {"max_s": 30, "max_attempts": 1},
                    "on_failure": "stop",
                }
            ],
        }
        resolved, bindings = resolve_plan_bindings(plan, graph)
        self.assertEqual(bindings[0]["instance_ref"], "akita_black_bowl_1")
        prepared = prepare_plan(resolved, scene_objects=objects, execution_mode="free")
        self.assertEqual(prepared["canonical_instruction"], "put the bowl on the stove")
        self.assertEqual(prepared["user_instruction"], "碗放炉灶上")
        pick = next(step for step in prepared["steps"] if step["skill"] == "pi0_pick")
        self.assertEqual(pick["vla_prompt"], "pick up the black bowl")
        self.assertNotIn("akita_black_bowl_1", pick["vla_prompt"])
        self.assertNotIn("put the bowl on the stove", pick["vla_prompt"])
        self.assertGreater(prepared["semantic_step_count"], 0)
        leaked = dict(plan)
        leaked["steps"] = [{**plan["steps"][0], "prompt": "pick up akita_black_bowl_1", "vla_prompt": "pick up akita_black_bowl_1"}]
        prepared_leak = prepare_plan(resolve_plan_bindings(leaked, graph)[0], scene_objects=objects, execution_mode="free")
        again = next(step for step in prepared_leak["steps"] if step["skill"] == "pi0_pick")
        self.assertNotIn("akita_black_bowl_1", again["vla_prompt"])

    def test_display_follows_compiled_events_only(self):
        objects = _objects()
        graph = scene_graph(objects, {}, scene_epoch=1, observation_id="catalog:epoch1")
        plan = {
            "plan_id": "bowl-stove",
            "goal": "把碗放到炉灶上",
            "canonical_instruction": "put the bowl on the stove",
            "matched_task_key": "libero_goal:t1",
            "steps": [
                {
                    "step_id": "bowl",
                    "skill": "transfer",
                    "object_n": 1,
                    "target_n": 5,
                    "relation": "on",
                    "prompt": "pick up the black bowl",
                    "vla_prompt": "pick up the black bowl",
                    "grounding_query": "black bowl",
                    "depends_on": [],
                    "completion_check": "object_placed",
                    "budget": {"max_s": 30, "max_attempts": 1},
                    "on_failure": "stop",
                }
            ],
        }
        prepared = prepare_plan(resolve_plan_bindings(plan, graph)[0], scene_objects=objects, execution_mode="free")
        rows = display_steps(prepared["steps"])
        self.assertTrue(all(row["status"] == "pending" for row in rows))
        self.assertIn("安全退离", [row["text_zh"] for row in rows])
        pick = next(step for step in prepared["steps"] if step["skill"] == "pi0_pick")
        running = apply_display_event(rows, {"phase": "executing", "step_id": pick["step_id"], "skill": "pi0_pick"})
        self.assertEqual(next(row for row in running if row["kind"] == "pick")["status"], "running")
        self.assertTrue(all(row["status"] == "pending" for row in running if row["kind"] != "pick"))
        held = apply_display_event(running, {"phase": "executing", "step_id": pick["step_id"], "reason_code": "GRASP_HELD"})
        self.assertEqual(next(row for row in held if row["kind"] == "pick")["status"], "success")
        self.assertEqual(next(row for row in held if row["kind"] == "verify")["status"], "pending")

    def test_unsupported_skill_and_combo_stay_explicit(self):
        wine = lookup_task("libero_goal:t9", None, "libero_goal", GOAL_NAMES)
        self.assertEqual(wine["executor_support"], "unsupported_skill")
        self.assertIn("wine_rack_1_top_region", wine["support_reason"])
        blocked = single_task_block({"semantic_step_count": 1, "canonical_instruction": wine["instruction_en"]}, wine)
        self.assertIn("wine_rack_1_top_region", blocked)
        combo = single_task_block({"semantic_step_count": 2}, wine)
        self.assertIn("当前仅执行单任务", combo)
        missing = single_task_block(
            {"semantic_step_count": 1},
            {"executor_support": "unresolved_region", "support_reason": "没有放置区域 stove_missing。"},
        )
        self.assertIn("没有放置区域", missing)
        examples = scene_examples("libero_goal", GOAL_NAMES)
        texts = [item["input_zh"] for item in examples]
        self.assertIn("把碗放到炉灶上。", texts)
        self.assertNotIn("把酒瓶放到架子上。", texts)
        page = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
        self.assertIn("renderTaskView", page)
        self.assertIn("taskIn.value", page)
        self.assertNotIn('send({type:"start"', page)
        html = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
        self.assertIn("任务与执行步骤", html)
        self.assertIn("view16", html)
        self.assertNotIn("官方任务选择", html)
        self.assertNotIn("selectOfficial", page)
        self.assertIn("task_text", page)
        self.assertNotIn("输入保留", page)
        self.assertNotIn("自由模式", page)
        self.assertNotIn("当前场景示例", page)
        self.assertIn("开始不会切换场景", (ROOT / "adapter.py").read_text(encoding="utf-8"))
        self.assertIsNone(single_task_block({"semantic_step_count": 1, "canonical_instruction": "Put the bowl on the stove."}, None))

    def test_refresh_payload_keeps_summary_and_failure(self):
        from adapter import _task_view

        objects = _objects()
        graph = scene_graph(objects, {}, scene_epoch=1, observation_id="catalog:epoch1")
        plan = {
            "plan_id": "bowl-stove",
            "goal": "把碗放到炉灶上",
            "canonical_instruction": "put the bowl on the stove",
            "matched_task_key": "libero_goal:t1",
            "steps": [
                {
                    "step_id": "bowl",
                    "skill": "transfer",
                    "object_n": 1,
                    "target_n": 5,
                    "relation": "on",
                    "prompt": "pick up the black bowl",
                    "vla_prompt": "pick up the black bowl",
                    "grounding_query": "black bowl",
                    "depends_on": [],
                    "completion_check": "object_placed",
                    "budget": {"max_s": 30, "max_attempts": 1},
                    "on_failure": "stop",
                }
            ],
        }
        prepared = prepare_plan(resolve_plan_bindings(plan, graph)[0], scene_objects=objects, execution_mode="free")
        prepared["request_id"] = "seq-test"
        view = _task_view(prepared, {"suite": "libero_goal"}, objects, "把碗放到炉灶上")
        view["failure"] = {"reason_code": "localization_unconfirmed", "reason": "距离 0.20 大于阈值 0.1325"}
        payload = {"task_view": view, "run": {"root_error": view["failure"]["reason"], "user_text": view["user_instruction"]}}
        encoded = json.dumps(payload, ensure_ascii=False)
        self.assertIn("put the bowl on the stove", encoded)
        self.assertIn("把碗放到炉灶上", encoded)
        self.assertIn("pick up the black bowl", encoded)
        self.assertIn("localization_unconfirmed", encoded)
        self.assertNotEqual(view["user_instruction"], view["canonical_instruction"])


if __name__ == "__main__":
    unittest.main()
