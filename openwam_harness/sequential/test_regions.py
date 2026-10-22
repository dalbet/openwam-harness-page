"""Scene regions, reset isolation, and observation clocks. No model and no env.step."""

from __future__ import annotations

import json
import threading
import unittest
from pathlib import Path

import adapter
from candidates import number_candidates, resolve_plan_bindings, scene_graph
from executor import Session, _recover_or_confirm
from planner import request_plan
from regions import clear_pose_cache, on_site, parse_bddl_regions, placement_mode, query_region
from store import new_observation
from suites import _catalog, objects_for_names
from verify import observation_ok

ROOT = Path(__file__).resolve().parent
BDDL = Path(
    "/root/Page_dmh/openwam_harness/envs/rpent/lib/python3.10/site-packages/"
    "liberopro/liberopro/bddl_files/libero_goal"
)
WINE_PLAN = {
    "plan_id": "bowl_to_stove",
    "goal": "把碗放在炉灶上，然后将红酒瓶放在架子上",
    "steps": [
        {
            "step_id": "step1",
            "skill": "transfer",
            "object_n": 1,
            "target_n": 7,
            "relation": "on",
            "prompt": "pick up the black bowl and place it on the stove",
            "grounding_query": "black bowl",
            "vla_prompt": "pick up the black bowl",
            "depends_on": [],
            "preconditions": [],
            "completion_check": "object_placed",
            "budget": {"max_s": 180, "max_attempts": 2},
            "on_failure": "recover",
        },
        {
            "step_id": "step2",
            "skill": "transfer",
            "object_n": 4,
            "target_n": 8,
            "relation": "on",
            "prompt": "pick up the wine bottle and place it on the wine rack",
            "grounding_query": "wine bottle",
            "vla_prompt": "pick up the wine bottle",
            "depends_on": ["step1"],
            "preconditions": [],
            "completion_check": "object_placed",
            "budget": {"max_s": 180, "max_attempts": 2},
            "on_failure": "recover",
        },
    ],
}


def _names():
    state = json.loads((ROOT.parent / "runs" / "sequential" / "874f6a80bddf" / "states.json").read_text(encoding="utf-8"))
    names = list(state["steps"][-1]["state"]["object_names"])
    catalog = _catalog(str(BDDL / "put_the_bowl_on_the_stove.bddl"))
    for item in catalog["instances"] + catalog["fixtures"]:
        if item["name"] not in names:
            names.append(item["name"])
    return names, catalog


class RegionRegistryTest(unittest.TestCase):
    def test_both_bddl_files_keep_both_top_regions(self):
        for name in ("put_the_bowl_on_the_stove.bddl", "put_the_wine_bottle_on_the_rack.bddl"):
            rows = parse_bddl_regions((BDDL / name).read_text(encoding="utf-8"))
            ids = [item["region_id"] for item in rows if item["status"] == "declared"]
            self.assertIn("wooden_cabinet_1_top_region", ids)
            self.assertIn("wine_rack_1_top_region", ids)
            self.assertEqual(ids.count("wine_rack_1_top_region"), 1)
            init = next(item for item in rows if item["region_id"] == "main_table_wine_rack_region")
            self.assertEqual(init["semantic_relation"], "init_support")
            rack = next(item for item in rows if item["region_id"] == "wine_rack_1_top_region")
            self.assertEqual(rack["semantic_relation"], "fixture_site")
            self.assertEqual(rack["query_key"], "wine_rack_1_top_region")

    def test_stove_task_resolves_the_rack_outside_its_native_goal(self):
        names, catalog = _names()
        objects = objects_for_names(names, catalog)
        self.assertEqual(objects["flat_stove_1"]["place_site"], "flat_stove_1_cook_region")
        self.assertEqual(objects["wine_rack_1"]["place_site"], "wine_rack_1_top_region")
        self.assertFalse(objects["wine_rack_1"]["native_goal_region"])
        self.assertGreater(len(objects["wooden_cabinet_1"]["place_sites"]), 1)
        self.assertNotIn("place_site", objects["wooden_cabinet_1"])
        graph = scene_graph(objects, {}, scene_epoch=2, observation_id="regions")
        resolved, bindings = resolve_plan_bindings(WINE_PLAN, graph)
        by_step = {}
        for item in bindings:
            by_step.setdefault(item["step_id"], {})[item["role"]] = item
        self.assertEqual(by_step["step1"]["target"]["query_key"], "flat_stove_1_cook_region")
        self.assertEqual(by_step["step2"]["target"]["instance_ref"], "wine_rack_1")
        self.assertEqual(by_step["step2"]["target"]["query_key"], "wine_rack_1_top_region")
        self.assertTrue(resolved["bindings_checked"])

    def test_query_errors_and_rigid_cache_do_not_drop_the_definition(self):
        clear_pose_cache()
        names, catalog = _names()
        registry = catalog["scene_regions"]
        declared = {item["region_id"] for item in registry["regions"]}
        missing = query_region("not_a_region", {"named_poses": {"sites": {}}}, registry)
        self.assertEqual(missing["reason_code"], "region_not_declared")
        init = query_region("main_table_wine_rack_region", {"named_poses": {"sites": {}}}, registry)
        self.assertEqual(init["reason_code"], "region_not_placeable")
        obs = {
            "scene_epoch": 4,
            "observation_id": "a",
            "step_clock": "tool_record",
            "env_step": 1,
            "named_poses": {
                "bodies": {"wine_rack_1": [0.1, 0.2, 0.3]},
                "sites": {"wine_rack_1_top_region": [0.2, 0.3, 0.9]},
                "site_geometry": {"wine_rack_1_top_region": {"quat_wxyz": [0.854912, -0.518773, 0.0, 0.0], "size": [0.10087, 0.022, 0.00251]}},
            },
        }
        first = query_region("wine_rack_1_top_region", obs, registry)
        self.assertEqual(first["status"], "ok")
        self.assertEqual(first["motion_class"], "rigid")
        refreshed = {
            "scene_epoch": 4,
            "named_poses": {"bodies": {"wine_rack_1": [0.1, 0.2, 0.3]}, "sites": {}},
        }
        cached = query_region("wine_rack_1_top_region", refreshed, registry)
        self.assertEqual(cached["source"], "cached_rigid_site")
        self.assertEqual(cached["xyz"], first["xyz"])
        self.assertEqual(declared, {item["region_id"] for item in registry["regions"]})
        moved = {
            "scene_epoch": 4,
            "named_poses": {"bodies": {"wine_rack_1": [0.4, 0.2, 0.3]}, "sites": {}},
        }
        self.assertEqual(query_region("wine_rack_1_top_region", moved, registry)["reason_code"], "region_site_missing")
        other_epoch = {
            "scene_epoch": 5,
            "named_poses": {"bodies": {"wine_rack_1": [0.1, 0.2, 0.3]}, "sites": {}},
        }
        self.assertEqual(query_region("wine_rack_1_top_region", other_epoch, registry)["reason_code"], "region_site_missing")
        mode = placement_mode(first)
        self.assertFalse(mode["ok"])
        self.assertEqual(mode["code"], "orientation_not_commandable")
        self.assertGreater(mode["tilt_rad"], 0.2)
        stove = placement_mode({"quat_wxyz": [1, 0, 0, 0], "xyz": [0, 0, 1]})
        self.assertTrue(stove["ok"])
        self.assertEqual(stove["mode"], "translation")
        yaw = placement_mode({"quat_wxyz": [0.0, 0.0, 0.0, 1.0], "mat": [-1, 0, 0, 0, -1, 0, 0, 0, 1], "xyz": [0, 0, 1]})
        self.assertTrue(yaw["ok"], yaw)
        self.assertEqual(yaw["mode"], "translation")
        self.assertLess(yaw["tilt_rad"], 0.02)
        judged = on_site([0.0, 0.0, 1.01], [0.0, 0.0, 1.0], [1, 0, 0, 0, 1, 0, 0, 0, 1], [0.075, 0.075, 0.0025])
        self.assertTrue(judged["ok"])
        self.assertEqual(judged["predicate"], "SiteObject.under")

    def test_registry_error_does_not_request_a_format_repair(self):
        names, catalog = _names()
        objects = objects_for_names(names, catalog)
        numbered = scene_graph(objects, {}, scene_epoch=2, observation_id="regions")
        cabinet = next(item["n"] for item in numbered["nodes"] if item["object_ref"] == "wooden_cabinet_1")
        plan = json.loads(json.dumps(WINE_PLAN))
        plan["steps"] = [plan["steps"][0]]
        plan["steps"][0]["target_n"] = cabinet
        calls = []

        def client(messages, max_tokens, options=None):
            calls.append(messages)
            return {
                "content": None,
                "tool_calls": [{"id": "call", "type": "function", "function": {"name": "submit_plan", "arguments": json.dumps(plan)}}],
                "finish_reason": "stop",
                "usage": {"prompt_tokens": 3, "completion_tokens": 2},
            }

        result = request_plan(
            task=plan["goal"],
            scene_observation="candidates",
            client=client,
            scene_objects=objects,
            numbered_candidates=numbered,
            execution_mode="free",
        )
        self.assertEqual(result["status"], "PLANNING_FAILED")
        self.assertEqual(result["repairs"], 0)
        self.assertEqual(len(calls), 1)
        self.assertTrue(any(item["code"] == "target_region_ambiguous" for item in result["errors"]))

    def test_clocks_reread_once_and_reject_another_scene(self):
        same = {
            "scene_epoch": 2,
            "env_step": 3,
            "step_clock": "env_action_count",
            "cameras": {"agentview": {}, "wrist": {}},
        }
        ok, code = observation_ok(same, scene_epoch=2, min_step=9, min_clock="tool_record")
        self.assertTrue(ok)
        self.assertEqual(code, "")
        behind = dict(same, step_clock="tool_record", env_step=3)
        ok, code = observation_ok(behind, scene_epoch=2, min_step=9, min_clock="tool_record")
        self.assertFalse(ok)
        self.assertEqual(code, "stale_snapshot")
        foreign = dict(behind, scene_epoch=1, env_step=20)
        ok, code = observation_ok(foreign, scene_epoch=2, min_step=None)
        self.assertEqual(code, "cross_scene")
        kept = new_observation({"scene_epoch": 1, "env_step": 4}, scene_epoch=2)
        self.assertEqual(kept["scene_epoch"], 1)
        self.assertTrue(kept["cross_scene"])

        class Backend:
            def __init__(self, epoch):
                self.calls = 0
                self.epoch = epoch

            def observe(self):
                self.calls += 1
                return {
                    "observation_id": "fresh",
                    "scene_epoch": self.epoch,
                    "env_step": 9,
                    "step_clock": "tool_record",
                    "cameras": {"agentview": {}, "wrist": {}},
                    "sim": {"objects": {}},
                }

        backend = Backend(2)
        step = {"completion_check": "object_transported", "object_ref": "akita_black_bowl_1", "args": {}, "_state_reread": False}
        verdict, _outcome, post = _recover_or_confirm(
            step=step,
            outcome=type("O", (), {"executed": True, "real": True, "error": None, "raw": {}})(),
            verification={"reason_code": "stale_snapshot", "status": "failure"},
            post=[],
            backend=backend,
            bindings={},
            cache=None,
            recovery=None,
            support_z={},
            plan={"objects": {}, "execution_mode": "free"},
            started_epoch=2,
            before_step=8,
            confirm=None,
            emit=None,
            stale=None,
            session=Session("scene"),
            eef=None,
        )
        self.assertEqual(backend.calls, 1)
        self.assertNotEqual(verdict["reason_code"], "stale_snapshot")
        self.assertEqual(post[-1]["env_step"], 9)
        again = Backend(1)
        rejected, _outcome, _post = _recover_or_confirm(
            step={"completion_check": "object_transported", "object_ref": "akita_black_bowl_1", "args": {}},
            outcome=type("O", (), {"executed": True, "real": True, "error": None, "raw": {}})(),
            verification={"reason_code": "STALE_OBSERVATION", "status": "failure", "scene_epoch": 1},
            post=[],
            backend=again,
            bindings={},
            cache=None,
            recovery=None,
            support_z={},
            plan={"objects": {}, "execution_mode": "free"},
            started_epoch=2,
            before_step=8,
            confirm=None,
            emit=None,
            stale=None,
            session=Session("scene"),
            eef=None,
        )
        self.assertEqual(again.calls, 0)
        self.assertEqual(rejected["reason_code"], "STALE_OBSERVATION")


class ResetIsolationTest(unittest.TestCase):
    def test_reset_bumps_epoch_and_keeps_history_and_task(self):
        app = adapter.App()
        app.sent = []
        app.broadcast = app.sent.append
        app.scene_spec = {"suite": "libero_goal", "task_id": 1, "seed": 0, "goal": "put the bowl on the stove", "summary_zh": "把碗放到炉灶上。", "user_text": "把酒瓶放到酒架上"}
        app.session = Session("libero_goal_t1_s0")
        app.session.scene_epoch = 3
        app.suites = [{"id": "libero_goal", "tasks": []}]
        adapter.PROBE["edge"] = {"status": "ok", "detail": "HTTP 200", "model_ids": ["Qwen3.5-9B"]}
        app.runs.begin(4, "old-episode", 3, "把碗放到炉灶上", "free")
        app.runs.finish("failed", "旧结果", "workspace")
        app.binding = {"object_ref": "akita_black_bowl_1"}
        app._active_plan = {"plan_id": "old"}
        app._reset_scene({"client_request_id": "reset-1"})
        self.assertEqual(app.session.scene_epoch, 4)
        self.assertNotEqual(app.session.episode_id, "old-episode")
        self.assertIsNone(app.binding)
        self.assertIsNone(app._active_plan)
        self.assertEqual(app.scene_spec["user_text"], "把酒瓶放到酒架上")
        self.assertEqual(app.run_history[-1]["root_error"], "旧结果")
        self.assertEqual(app.runs.phase, "idle")
        self.assertFalse(app.session.cancelled)
        ack = next(item for item in app.sent if item.get("type") == "reset_ack" and item.get("code") == "dry_reset")
        self.assertEqual(ack["previous_scene_epoch"], 3)
        self.assertTrue(app.command_gates()["can_reset"])
        self.assertIn("重置场景", (ROOT / "static" / "index.html").read_text(encoding="utf-8"))
        self.assertNotIn('begin("restart")', (ROOT / "static" / "app.js").read_text(encoding="utf-8"))

    def test_second_reset_while_the_first_is_running_does_not_bump_again(self):
        app = adapter.App()
        app.sent = []
        app.broadcast = app.sent.append
        app.scene_spec = {"suite": "libero_goal", "task_id": 1, "seed": 0, "goal": "goal", "summary_zh": "目标", "user_text": "保留"}
        app.session = Session("libero_goal_t1_s0")
        app.session.scene_epoch = 2
        app.suites = [{"id": "libero_goal", "tasks": []}]
        adapter.PROBE["edge"] = {"status": "ok", "detail": "HTTP 200", "model_ids": ["Qwen3.5-9B"]}
        entered = threading.Event()
        release = threading.Event()
        original = app.session.begin_episode

        def slow():
            entered.set()
            release.wait(1)
            return original()

        app.session.begin_episode = slow
        app._resetting = False
        thread = threading.Thread(target=app._reset_scene, args=({"client_request_id": "reset-a"},))
        thread.start()
        self.assertTrue(entered.wait(1))
        app._reset_scene({"client_request_id": "reset-b"})
        release.set()
        thread.join(1)
        self.assertEqual(app.session.scene_epoch, 3)
        codes = [item.get("code") for item in app.sent if item.get("type") == "reset_ack"]
        self.assertIn("already_resetting", codes)


if __name__ == "__main__":
    unittest.main()
