"""Middle-drawer skill contract. Stubs stand in for _vlm_chunk. No env.step and no planner call."""

from __future__ import annotations

import re
import unittest
from pathlib import Path

from candidates import resolve_plan_bindings, scene_graph
from executor import ToolOutcome, _calls_for
from libero.libero.envs.objects.articulated_objects import WoodenCabinet
from open_drawer import MAX_CHUNKS, MAX_S, asset_joint, official_instruction, official_region, read_joint, run_open_drawer
from schema import PlanError, failure_stage, prepare_plan, resolve_args
from skills import implemented_skills, skill_rejection
from suites import _catalog, objects_for_names
from task_catalog import lookup_task, single_task_block, tasks_for_scene
from verify import verify_step


ROOT = Path(__file__).resolve().parent
STOVE = Path(
    "/root/Page_dmh/openwam_harness/envs/rpent/lib/python3.10/site-packages/"
    "liberopro/liberopro/bddl_files/libero_goal/put_the_bowl_on_the_stove.bddl"
)


def _objects():
    catalog = _catalog(str(STOVE))
    task = {"instances": catalog["instances"], "fixtures": catalog["fixtures"], "place_regions": catalog["place_regions"]}
    names = [item["name"] for item in task["instances"] + task["fixtures"] if item.get("name")]
    return objects_for_names(names, task)


def _number(graph, name: str) -> int:
    for node in graph["nodes"]:
        if node.get("object_ref") == name:
            return int(node["n"])
    raise AssertionError(name)


class _Site:
    joints = ["wooden_cabinet_1_middle_level"]
    parent_name = "wooden_cabinet_1"


class _Sim:
    def __init__(self, qpos):
        self.qpos = qpos
        self.model = self
        self.data = self

    def get_joint_qpos_addr(self, name):
        if name != "wooden_cabinet_1_middle_level":
            raise KeyError(name)
        return 0


class _Env:
    def __init__(self, qpos=0.0):
        self.qpos = [qpos]
        self.object_sites_dict = {"wooden_cabinet_1_middle_region": _Site()}
        self._parent = WoodenCabinet("wooden_cabinet_1")
        self.sim = _Sim(self.qpos)
        self.terminated = False
        self.truncated = False
        self.success = False
        self.steps = 0

    def get_object(self, name):
        if name != "wooden_cabinet_1":
            return None
        return self._parent

    def step(self, *_args, **_kwargs):
        self.steps += 1
        raise AssertionError("env.step")


class _Primitives:
    def __init__(self, env, script, cancel_after=None):
        self.env = env
        self.script = list(script)
        self.cancel_after = cancel_after
        self.chunks = []

    def _check_cancelled(self):
        if self.cancel_after is not None and len(self.chunks) >= self.cancel_after:
            raise RuntimeError("cancelled")

    def _vlm_chunk(self, instruction):
        self.chunks.append(instruction)
        action = self.script[len(self.chunks) - 1]
        if action == "term":
            self.env.terminated = True
            self.env.success = True
            return
        self.env.qpos[0] = float(action)
        self.env.success = True


class _RpcClient:
    """The readonly state snapshot RPC the live backend already exposes."""

    def __init__(self, qpos):
        self.joints = {"wooden_cabinet_1_middle_level": qpos}
        self.calls = []

    def call(self, method, timeout_s=None):
        self.calls.append(method)
        if method != "env.state_snapshot":
            raise KeyError(method)
        return {
            "joints": dict(self.joints),
            "named_poses": {"sites": {}},
            "sim": {"objects": {}, "evidence_source": "readonly_rpc"},
            "evidence_source": "readonly_rpc",
        }


class _RpcEnv:
    """The live backend's env is an RPC client without ``object_sites_dict``."""

    def __init__(self, qpos=0.0):
        self._client = _RpcClient(qpos)
        self.terminated = False
        self.truncated = False
        self.steps = 0

    def step(self, *_args, **_kwargs):
        self.steps += 1
        raise AssertionError("env.step")


class _RpcPrimitives:
    def __init__(self, env, script):
        self.env = env
        self.script = list(script)
        self.chunks = []

    def _check_cancelled(self):
        return None

    def _vlm_chunk(self, instruction):
        self.chunks.append(instruction)
        self.env._client.joints["wooden_cabinet_1_middle_level"] = float(self.script[len(self.chunks) - 1])


class OpenDrawerTest(unittest.TestCase):
    def test_joint_threshold_and_region_come_from_local_files(self):
        asset = asset_joint()
        self.assertEqual(asset["local_name"], "middle_level")
        self.assertEqual(asset["axis"], "0 1 0")
        self.assertTrue(str(asset["range"]).startswith("-0.16"))
        self.assertEqual(official_region(), "wooden_cabinet_1_middle_region")
        self.assertEqual(official_instruction(), "open the middle drawer of the cabinet")
        cabinet = WoodenCabinet("wooden_cabinet_1")
        ranges = cabinet.object_properties["articulation"]["default_open_ranges"]
        threshold = max(ranges)
        self.assertFalse(cabinet.is_open(threshold))
        self.assertTrue(cabinet.is_open(threshold - 0.001))
        source = (ROOT / "open_drawer.py").read_text(encoding="utf-8")
        self.assertIsNone(re.search(r"\.step\s*\(", source))
        self.assertIsNone(re.search(r"\bpi0_doubled\s*\(", source))
        self.assertNotIn("middle_joint", source)

    def test_compile_reaches_action_args_and_the_checker(self):
        objects = _objects()
        graph = scene_graph(objects, {}, scene_epoch=3, observation_id="drawer")
        plan = {
            "plan_id": "middle",
            "goal": "打开柜子的中间抽屉",
            "canonical_instruction": "open the middle drawer of the cabinet",
            "candidate_set_version": graph["candidate_set_version"],
            "scene_epoch": graph["scene_epoch"],
            "steps": [
                {
                    "step_id": "open",
                    "skill": "open_drawer",
                    "object_n": 0,
                    "target_n": _number(graph, "wooden_cabinet_1"),
                    "relation": "open",
                    "prompt": "open the middle drawer of the cabinet",
                    "depends_on": [],
                    "completion_check": "object_placed",
                    "budget": {"max_s": 30, "max_attempts": 1},
                    "on_failure": "stop",
                }
            ],
        }
        bound, bindings = resolve_plan_bindings(plan, graph)
        self.assertEqual([item["role"] for item in bindings], ["target"])
        self.assertEqual(bound["steps"][0]["target_ref"], "wooden_cabinet_1")
        self.assertNotIn("object_ref", bound["steps"][0])
        prepared = prepare_plan(bound, scene_objects=objects, execution_mode="free", task="打开柜子的中间抽屉")
        step = prepared["steps"][0]
        self.assertEqual(step["skill"], "open_drawer")
        self.assertEqual(step["completion_check"], "drawer_open")
        args = resolve_args(step["args"], {})
        self.assertEqual(_calls_for("open_drawer", args, None), [("open_drawer", args)])
        self.assertEqual(args["instruction"], official_instruction())
        self.assertEqual(args["region"], "wooden_cabinet_1_middle_region")
        self.assertEqual(args["instance"], "wooden_cabinet_1")
        self.assertEqual(args["max_chunks"], MAX_CHUNKS)
        self.assertEqual(args["max_s"], MAX_S)
        self.assertIsNone(single_task_block(prepared, lookup_task(None, official_instruction(), "libero_goal", objects)))
        opened = _evidence(opened=True, qpos=-0.15, stop="predicate_open")
        verdict = verify_step(step, opened, [_obs()], scene_epoch=3, min_step=None, objects=prepared["objects"], support_z={}, goal_relation="open")
        self.assertEqual(verdict["status"], "success")
        self.assertEqual(verdict["reason_code"], "DRAWER_OPEN")
        self.assertEqual(_obs()["sim_action_count"], 9)
        self.assertNotEqual(_obs()["sim_action_count"], _obs()["env_step"])
        closed = _evidence(opened=False, qpos=0.0, stop="chunk_budget", success_flag=True)
        missed = verify_step(step, closed, [_obs()], scene_epoch=3, min_step=None, objects=prepared["objects"], support_z={}, goal_relation="open")
        self.assertEqual(missed["status"], "failure")
        self.assertNotEqual(missed["reason_code"], "DRAWER_OPEN")
        unknown = ToolOutcome(executed=True, env_step=4, native_terminated=True, raw={"opened": True, "pi0_success_used": False})
        unverified = verify_step(step, unknown, [_obs()], scene_epoch=3, min_step=None, objects=prepared["objects"], support_z={}, goal_relation="open")
        self.assertEqual(unverified["status"], "unknown")
        self.assertEqual(unverified["reason_code"], "DRAWER_UNVERIFIED")

    def test_top_drawer_missing_object_and_other_contact_stay_specific(self):
        objects = _objects()
        graph = scene_graph(objects, {}, scene_epoch=3, observation_id="drawer")
        top = {
            "plan_id": "top",
            "goal": "打开上层抽屉",
            "candidate_set_version": graph["candidate_set_version"],
            "steps": [
                {
                    "step_id": "open",
                    "skill": "open_drawer",
                    "target_n": _number(graph, "wooden_cabinet_1"),
                    "target_region": "wooden_cabinet_1_top_region",
                    "relation": "open",
                    "prompt": "open the top drawer",
                    "depends_on": [],
                    "completion_check": "drawer_open",
                    "budget": {"max_s": 30, "max_attempts": 1},
                    "on_failure": "stop",
                }
            ],
        }
        bound, _rows = resolve_plan_bindings(top, graph)
        with self.assertRaises(PlanError) as rejected:
            prepare_plan(bound, scene_objects=objects, execution_mode="free")
        self.assertIn("region_not_middle", {item["code"] for item in rejected.exception.errors})
        missing = {
            "plan_id": "gone",
            "goal": "打开柜子的中间抽屉",
            "steps": [
                {
                    "step_id": "open",
                    "skill": "open_drawer",
                    "target_ref": "wooden_cabinet_9",
                    "relation": "open",
                    "prompt": "open",
                    "depends_on": [],
                    "completion_check": "drawer_open",
                    "budget": {"max_s": 30, "max_attempts": 1},
                    "on_failure": "stop",
                }
            ],
        }
        with self.assertRaises(PlanError) as absent:
            prepare_plan(missing, scene_objects=objects, execution_mode="free")
        self.assertIn("object_missing", {item["code"] for item in absent.exception.errors})
        for skill in ("push", "close", "turn_on_stove"):
            self.assertIsNotNone(skill_rejection(skill))
            with self.assertRaises(PlanError) as unsupported:
                prepare_plan(
                    {
                        "plan_id": skill,
                        "goal": skill,
                        "steps": [{"step_id": "x", "skill": skill, "target_ref": "flat_stove_1"}],
                    },
                    scene_objects=objects,
                    execution_mode="free",
                )
            self.assertEqual(failure_stage(unsupported.exception.errors), "skill_unsupported")
        self.assertIn("open_drawer", implemented_skills())
        self.assertNotIn("push", implemented_skills())
        present = tasks_for_scene("libero_goal", objects)
        self.assertTrue(any(row["task_key"] == "libero_goal:t0" for row in present))
        self.assertFalse(any(row["task_key"] == "libero_goal:t0" for row in tasks_for_scene("libero_goal", {"akita_black_bowl_1"})))

    def test_chunk_loop_stops_on_cancel_terminal_predicate_and_budget(self):
        env = _Env()
        opening = _Primitives(env, [-0.15, -0.15])
        opened = run_open_drawer(opening, {"region": official_region()})
        self.assertEqual(env.steps, 0)
        self.assertTrue(opened["opened"])
        self.assertEqual(opened["stop_reason"], "predicate_open")
        self.assertEqual(opened["chunks_used"], 1)
        self.assertFalse(opened["pi0_success_used"])
        self.assertLess(opened["current_qpos"], opened["threshold"])
        self.assertEqual(opened["joint"], "wooden_cabinet_1_middle_level")
        self.assertEqual(opened["comparison"], "qpos < max(default_open_ranges)")
        self.assertEqual(opened["evidence_source"], "WoodenCabinet.is_open")

        closed = _Env()
        stopped = _Primitives(closed, ["term", -0.15])
        result = run_open_drawer(stopped, {"region": official_region()})
        self.assertEqual(len(stopped.chunks), 1)
        self.assertFalse(result["opened"])
        self.assertEqual(result["stop_reason"], "terminated")
        self.assertTrue(closed.success)

        cancelled = _Env()
        holder = _Primitives(cancelled, [0.0, -0.15], cancel_after=1)
        with self.assertRaises(RuntimeError):
            run_open_drawer(holder, {"region": official_region()})
        self.assertEqual(len(holder.chunks), 1)
        self.assertEqual(cancelled.steps, 0)

        stalled = _Env()
        many = _Primitives(stalled, [0.0] * 20)
        exhausted = run_open_drawer(many, {"region": official_region()})
        self.assertEqual(exhausted["chunks_used"], MAX_CHUNKS)
        self.assertEqual(len(many.chunks), MAX_CHUNKS)
        self.assertEqual(exhausted["stop_reason"], "chunk_budget")
        self.assertIn(str(MAX_CHUNKS), exhausted["reason"])
        self.assertFalse(exhausted["opened"])

        timed = _Env()
        clock = iter([0.0, 0.0, MAX_S + 1])
        import open_drawer

        original = open_drawer.time.monotonic
        open_drawer.time.monotonic = lambda: next(clock)
        try:
            timed_out = run_open_drawer(_Primitives(timed, [0.0, 0.0]), {"region": official_region()})
        finally:
            open_drawer.time.monotonic = original
        self.assertEqual(timed_out["stop_reason"], "time_budget")
        self.assertEqual(timed_out["chunks_used"], 1)
        self.assertIn("时间预算", timed_out["reason"])

        other = run_open_drawer(_Primitives(_Env(), []), {"region": "wooden_cabinet_1_top_region"})
        self.assertEqual(other["stop_reason"], "region_not_middle")
        self.assertEqual(other["chunks_used"], 0)

    def test_read_joint_uses_state_snapshot_when_env_is_an_rpc_client(self):
        env = _RpcEnv(qpos=-0.15)
        row = read_joint(env, official_region())
        self.assertTrue(row["ok"])
        self.assertTrue(row["opened"])
        self.assertEqual(row["joint"], "wooden_cabinet_1_middle_level")
        self.assertEqual(row["parent"], "wooden_cabinet_1")
        self.assertEqual(row["threshold"], -0.14)
        self.assertEqual(row["comparison"], "qpos < max(default_open_ranges)")
        self.assertIn("state_snapshot", row["evidence_source"])
        self.assertEqual(env._client.calls, ["env.state_snapshot"])
        self.assertEqual(env.steps, 0)

        closed = _RpcEnv(qpos=0.0)
        self.assertFalse(read_joint(closed, official_region())["opened"])

        explicit = read_joint(_RpcEnv(qpos=-0.2), official_region(), instance="wooden_cabinet_1")
        self.assertTrue(explicit["opened"])

        missing = _RpcEnv(qpos=0.0)
        missing._client.joints = {}
        unreadable = read_joint(missing, official_region())
        self.assertFalse(unreadable["ok"])
        self.assertEqual(unreadable["code"], "joint_unreadable")

    def test_run_open_drawer_over_the_state_snapshot_rpc(self):
        env = _RpcEnv()
        prim = _RpcPrimitives(env, [-0.15, -0.15])
        result = run_open_drawer(prim, {"region": official_region(), "instance": "wooden_cabinet_1"})
        self.assertEqual(env.steps, 0)
        self.assertTrue(result["opened"])
        self.assertEqual(result["stop_reason"], "predicate_open")
        self.assertEqual(result["chunks_used"], 1)
        self.assertLess(result["current_qpos"], result["threshold"])
        self.assertEqual(result["joint"], "wooden_cabinet_1_middle_level")
        self.assertIn("state_snapshot", result["evidence_source"])

        stalled = _RpcEnv()
        many = _RpcPrimitives(stalled, [0.0] * 20)
        exhausted = run_open_drawer(many, {"region": official_region()})
        self.assertFalse(exhausted["opened"])
        self.assertEqual(exhausted["stop_reason"], "chunk_budget")


def _obs() -> dict:
    return {
        "observation_id": "query-drawer",
        "scene_epoch": 3,
        "env_step": 4,
        "sim_action_count": 9,
        "step_clock": "tool_record",
        "timestamp": 4,
        "cameras": {"agentview": {"id": "a"}, "wrist": {"id": "w"}},
        "sim": {"objects": {}, "evidence_source": "sim_state"},
    }


def _evidence(*, opened: bool, qpos: float, stop: str, success_flag: bool = False) -> ToolOutcome:
    return ToolOutcome(
        executed=True,
        env_step=4,
        native_terminated=success_flag,
        raw={
            "opened": opened,
            "joint": "wooden_cabinet_1_middle_level",
            "initial_qpos": 0.0,
            "current_qpos": qpos,
            "threshold": -0.14,
            "comparison": "qpos < max(default_open_ranges)",
            "predicate": "Open",
            "evidence_source": "WoodenCabinet.is_open",
            "stop_reason": stop,
            "chunks_used": 1,
            "pi0_success_used": False,
            "success": success_flag,
        },
    )


if __name__ == "__main__":
    unittest.main()
