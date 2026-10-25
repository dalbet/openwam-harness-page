"""Load routing, catalog admission, and transport handoff. No planner call and no env.step."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from adapter import App, completion_phase
from candidates import resolve_plan_bindings, scene_graph
from executor import Session, ToolOutcome
from placement import resolve_motion_step
from run_context import RunContext
from schema import PlanError, failure_stage, prepare_plan
from suites import _catalog, objects_for_names
from site_paths import site
from task_catalog import cached_catalog, goal_alignment, lookup_task, match_task_text, normalize_phrase, single_task_block
from verify import verify_step

ROOT = Path(__file__).resolve().parent
BDDL = site("liberopro", "liberopro", "bddl_files", "libero_goal", "put_the_bowl_on_the_stove.bddl")
JOURNAL = ROOT / "records" / "run_journal.jsonl"
TRANSPORTS = (
    "put the bowl on the stove",
    "put the bowl on the plate",
    "put the cream cheese in the bowl",
    "put the bowl on top of the cabinet",
    "put the wine bottle on top of the cabinet",
)


def _goal_rows():
    return [row for row in cached_catalog() if row.get("suite") == "libero_goal"]


def _tasks():
    return [
        {
            "id": row["task_id"],
            "language": row["instruction_en"],
            "summary_zh": row["instruction_zh"],
            "task_key": row["task_key"],
            "paraphrases_zh": row.get("paraphrases_zh") or [],
        }
        for row in _goal_rows()
    ]


def _objects():
    catalog = _catalog(str(BDDL))
    task = {
        "instances": catalog["instances"],
        "fixtures": catalog["fixtures"],
        "place_regions": catalog["place_regions"],
    }
    names = [item["name"] for item in task["instances"] + task["fixtures"] if item.get("name")]
    return objects_for_names(names, task)


def _number(graph, name: str) -> int:
    for node in graph["nodes"]:
        if node.get("object_ref") == name or node.get("sim_name") == name:
            return int(node["n"])
    raise AssertionError(name)


def _semantic(graph, objects, source: str, target: str, english: str, relation: str = "on") -> dict:
    phrase = ((objects.get(source) or {}).get("phrases") or ["object"])[0]
    return {
        "plan_id": english,
        "goal": english,
        "canonical_instruction": english,
        "candidate_set_version": graph["candidate_set_version"],
        "scene_epoch": graph["scene_epoch"],
        "steps": [
            {
                "step_id": "carry",
                "skill": "transfer",
                "object_n": _number(graph, source),
                "target_n": _number(graph, target),
                "relation": relation,
                "prompt": f"pick up the {phrase}",
                "vla_prompt": f"pick up the {phrase}",
                "grounding_query": phrase,
                "depends_on": [],
                "completion_check": "object_placed",
                "budget": {"max_s": 30, "max_attempts": 1},
                "on_failure": "stop",
            }
        ],
    }


class GoalHandoffTest(unittest.TestCase):
    def test_phrase_normalization_does_not_merge_relations(self):
        self.assertEqual(normalize_phrase("  Put the bowl on the stove. "), "put the bowl on the stove")
        self.assertNotEqual(normalize_phrase("put the bowl in the stove"), normalize_phrase("put the bowl on the stove"))
        names = list(_objects())
        row = lookup_task(None, "Put the bowl on the stove.", "libero_goal", names)
        self.assertEqual(row["task_key"], "libero_goal:t1")
        self.assertIsNone(single_task_block({"semantic_step_count": 1}, row))
        self.assertIsNone(lookup_task(None, "put the bowl in the stove", "libero_goal", names))
        self.assertIsNone(single_task_block({"semantic_step_count": 1, "canonical_instruction": "put the bowl in the stove"}, None))

    def test_load_text_matches_standard_names_without_guessing(self):
        tasks = _tasks()
        stove = match_task_text("libero_goal", "把碗放到炉灶上。", tasks)
        self.assertEqual(stove["status"], "matched")
        self.assertEqual(stove["task_id"], 1)
        self.assertEqual(stove["source"], "instruction_zh")
        plate = match_task_text("libero_goal", "put the bowl on the plate", tasks)
        self.assertEqual(plate["task_id"], 8)
        self.assertEqual(plate["source"], "instruction_en")
        cheese = match_task_text("libero_goal", "  put the cream cheese in the bowl. ", tasks)
        self.assertEqual(cheese["task_id"], 6)
        empty = match_task_text("libero_goal", "   ", tasks)
        self.assertEqual(empty["status"], "empty")
        self.assertEqual(empty["task_id"], 1)
        self.assertEqual(empty["source"], "default")
        unknown = match_task_text("libero_goal", "把碗放进炉灶", tasks)
        self.assertEqual(unknown["status"], "unknown")
        cloned = [dict(item) for item in tasks]
        cloned[0]["summary_zh"] = cloned[1]["summary_zh"]
        ambiguous = match_task_text("libero_goal", cloned[1]["summary_zh"], cloned)
        self.assertEqual(ambiguous["status"], "ambiguous")
        self.assertIsNone(ambiguous["task_id"])

    def test_requested_goal_is_not_the_loaded_native_success(self):
        spec = {"goal": "put the bowl on the stove", "summary_zh": "把碗放到炉灶上"}
        self.assertEqual(goal_alignment("把碗放到炉灶上。", spec), "native")
        self.assertEqual(goal_alignment("把碗放到盘子上", spec), "requested")
        native = completion_phase(
            {"status": "completed", "reason": "NATIVE_SUCCESS", "native_task_success": True, "user_goal_status": "unmet"},
            {"goal_alignment": "native"},
            "free",
        )
        self.assertEqual(native[0], "incomplete")
        self.assertIn("放置未完成", native[2])
        held = completion_phase(
            {
                "status": "completed",
                "reason": "NATIVE_SUCCESS",
                "native_task_success": True,
                "user_goal_status": "met",
                "release": {"verified": True},
            },
            {"goal_alignment": "native"},
            "free",
        )
        self.assertEqual(held[0], "completed")
        benchmark = completion_phase(
            {"status": "completed", "reason": "NATIVE_SUCCESS", "native_task_success": True},
            {"goal_alignment": "native"},
            "benchmark",
        )
        self.assertEqual(benchmark[0], "completed")
        other = completion_phase(
            {"status": "completed", "reason": "NATIVE_SUCCESS", "native_task_success": True, "user_goal_status": "unmet"},
            {"goal_alignment": "requested"},
            "free",
        )
        self.assertEqual(other[0], "incomplete")
        self.assertIn("请求目标未完成", other[2])
        unknown = completion_phase(
            {"status": "completed", "native_task_success": False, "user_goal_status": "unknown"},
            {"goal_alignment": "requested"},
            "free",
        )
        self.assertEqual(unknown[2], "请求目标未验证")
        self.assertNotEqual(unknown[0], "completed")
        failed = completion_phase(
            {"status": "failed", "reason": "grasp missed", "native_task_success": False, "user_goal_status": "unknown"},
            {"goal_alignment": "requested"},
            "free",
        )
        self.assertEqual(failed[0], "failed")

    def test_new_run_drops_the_previous_planner_stage(self):
        ctx = RunContext()
        ctx.begin(1, "ep", 1, "把碗放到炉灶上", "free")
        ctx.planner_stage = "ok"
        ctx.note_error("标准指令不在当前场景的任务目录中，不会切换场景。", "task_not_in_scene")
        ctx.note_error("later", "other")
        self.assertEqual(ctx.root_error, "标准指令不在当前场景的任务目录中，不会切换场景。")
        ctx.begin(2, "ep", 1, "把碗放到盘子上", "free")
        self.assertEqual(ctx.root_error, "")
        self.assertEqual(ctx.planner_stage, "")
        self.assertFalse(ctx.execution_started)

    def test_start_does_not_reset_or_switch_the_loaded_task(self):
        import adapter as adapter_mod

        app = App()
        app.broadcast = lambda _message: None
        app.session = Session("libero_goal_t1_s0")
        app.session.begin_episode()
        app.execution_mode = "free"
        app.scene_spec = {
            "suite": "libero_goal",
            "task_id": 1,
            "seed": 0,
            "goal": "put the bowl on the stove",
            "summary_zh": "把碗放到炉灶上",
            "scene_id": "libero_goal_t1_s0",
        }

        class Backend:
            def __init__(self):
                self.resets = 0

            def warm_reset(self):
                self.resets += 1

            def retarget(self, *_args, **_kwargs):
                raise AssertionError("start retargeted the scene")

            def scene_info(self):
                return {"object_names": [], "env_step": 0}

            def sim_state(self):
                return {"objects": {}}

            def set_execution_mode(self, _mode):
                return None

        app.backend = Backend()
        app.runs.begin(1, app.session.episode_id, app.session.scene_epoch, "把碗放到盘子上", "free")
        previous = adapter_mod.LIVE
        adapter_mod.LIVE = True
        try:
            app._run_inner("start", "model", 1, "single", True, None, None, "把碗放到盘子上")
        finally:
            adapter_mod.LIVE = previous
        self.assertEqual(app.scene_spec["task_id"], 1)
        self.assertEqual(app.scene_spec["goal"], "put the bowl on the stove")
        self.assertEqual(app.backend.resets, 0)

    def test_missing_object_and_unimplemented_skill_stay_specific(self):
        objects = _objects()
        graph = scene_graph(objects, {}, scene_epoch=1, observation_id="handoff")
        plan = _semantic(graph, objects, "akita_black_bowl_1", "flat_stove_1", "put the bowl on the stove")
        plan["steps"][0]["object_n"] = 0
        with self.assertRaises(PlanError) as caught:
            resolve_plan_bindings(plan, graph)
        self.assertIn("candidate_number", {item["code"] for item in caught.exception.errors})
        unsupported = {
            "plan_id": "knob",
            "goal": "turn on the stove",
            "steps": [
                {
                    "step_id": "knob",
                    "skill": "turn_on_stove",
                    "object_ref": "akita_black_bowl_1",
                    "target_ref": "flat_stove_1",
                    "relation": "on",
                    "prompt": "turn on the stove",
                    "depends_on": [],
                    "preconditions": [],
                    "completion_check": "object_placed",
                    "budget": {"max_s": 30, "max_attempts": 1},
                    "on_failure": "stop",
                }
            ],
        }
        with self.assertRaises(PlanError) as skill_error:
            prepare_plan(unsupported, scene_objects=objects, execution_mode="free")
        self.assertEqual(failure_stage(skill_error.exception.errors), "skill_unsupported")

    def test_transport_plans_reach_motion_args_and_the_checker(self):
        objects = _objects()
        graph = scene_graph(objects, {}, scene_epoch=1, observation_id="handoff")
        catalog = {row["instruction_en"]: row for row in _goal_rows()}
        for english in TRANSPORTS:
            row = catalog[english]
            plan = _semantic(graph, objects, row["source_object"], row["target_object"], english, row["goal_relation"] or "on")
            bound, _bindings = resolve_plan_bindings(plan, graph)
            prepared = prepare_plan(bound, scene_objects=objects, execution_mode="free")
            self.assertIsNone(single_task_block(prepared, lookup_task(None, english + ".", "libero_goal", objects)))
            descend = next(step for step in prepared["steps"] if step.get("transfer_phase") == "descending")
            obs = _observation(prepared, row["source_object"])
            self.assertEqual(obs["sim_action_count"], 9)
            self.assertEqual(obs["env_step"], 4)
            self.assertNotEqual(obs["sim_action_count"], obs["env_step"])
            xyz_from = (descend.get("args") or {}).get("xyz_from")
            bindings = {xyz_from: [0.0, 0.0, 0.8]} if isinstance(xyz_from, str) else {}
            motion = resolve_motion_step(descend, bindings, obs, prepared)
            self.assertTrue(motion["ok"], (english, motion))
            self.assertIn("xyz", motion["args"])
            outcome = ToolOutcome(executed=True, env_step=4, raw={"sim_action_count": 9})
            verdict = verify_step(
                descend,
                outcome,
                [obs],
                scene_epoch=1,
                min_step=None,
                objects=prepared["objects"],
                support_z={},
                goal_relation=prepared.get("goal_relation") or "on",
            )
            self.assertIn("reason_code", verdict)

    def test_recorded_failures_match_the_journal(self):
        if not JOURNAL.is_file():
            self.skipTest("run journal is not in this checkout")
        rows = [json.loads(line) for line in JOURNAL.read_text(encoding="utf-8").splitlines() if line.strip()]
        by_request = {}
        for row in rows:
            by_request.setdefault(row.get("client_request_id"), []).append(row)
        if "req-1792568525389-0c9438" not in by_request:
            self.skipTest("historical journal anchors are not in this checkout")
        stove = by_request["req-1792568525389-0c9438"]
        self.assertEqual(stove[1]["detail"], "把碗放到炉灶上。")
        self.assertEqual(stove[-1]["code"], "task_not_in_scene")
        self.assertNotIn("first_tool", {item["stage"] for item in stove})
        plate = by_request["req-1792568620958-8de216"]
        self.assertEqual(plate[1]["detail"], "把碗放到炉灶上。")
        self.assertEqual(next(item["detail"] for item in plate if item["stage"] == "task_check" and item["status"] == "running"), "把碗放到盘子上")
        self.assertEqual(plate[-1]["code"], "task_not_in_scene")
        drawer = by_request["req-1792569005948-bb1378"]
        self.assertEqual(next(item["detail"] for item in drawer if item["stage"] == "task_check" and item["status"] == "running"), "打开柜子的中间抽屉")
        self.assertEqual(drawer[-1]["detail"], "PLANNING_FAILED")
        self.assertNotIn("first_tool", {item["stage"] for item in drawer})
        wine_texts = {"把酒瓶放到柜子顶上", "把酒瓶放到柜子顶上。"}
        first_wine = "req-1792574025419-db0cf2"
        order = [row.get("client_request_id") for row in rows]
        before_audit = [row for row in rows[: order.index(first_wine)] if row.get("client_request_id") != first_wine] if first_wine in order else rows
        self.assertFalse(any(item.get("detail") in wine_texts for item in before_audit))
        # The historical rejection is anchored by request id. After the prompt
        # and schema alignment the same sentence is a supported task: a newer
        # journal entry may legitimately reach first_tool and then be judged by
        # the executor. Do not turn that new behaviour back into a failure here.
        historical_wine = by_request[first_wine]
        stages = {item["stage"] for item in historical_wine}
        self.assertNotIn("first_tool", stages, first_wine)
        self.assertIn(historical_wine[-1]["code"], {"task_not_in_scene", "validation_failed"}, first_wine)


def _observation(plan: dict, object_name: str) -> dict:
    sites = {}
    geometry = {}
    for item in plan.get("plan_bindings") or []:
        key = item.get("query_key")
        if isinstance(key, str):
            sites[key] = [0.0, 0.0, 0.8]
            geometry[key] = {"size": [0.05, 0.05, 0.01], "mat": [1, 0, 0, 0, 1, 0, 0, 0, 1]}
    return {
        "observation_id": "query-handoff",
        "scene_epoch": 1,
        "env_step": 4,
        "sim_action_count": 9,
        "step_clock": "tool_record",
        "timestamp": 4,
        "cameras": {"agentview": {"id": "a"}, "wrist": {"id": "w"}},
        "named_poses": {"sites": sites, "site_geometry": geometry},
        "sim": {
            "objects": {object_name: {"pos": [0.0, 0.0, 0.9], "identified": True}},
            "eef_pos": [0.0, 0.0, 0.95],
            "evidence_source": "sim_state",
        },
    }


if __name__ == "__main__":
    unittest.main()
