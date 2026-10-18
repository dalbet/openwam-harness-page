"""Free mode keeps the loaded scene and does not treat the catalog as a gate."""

from __future__ import annotations

import unittest

from executor import native_execution_gate
from free_task import EpisodeFlags, judge_user_goal, prepare_free_request
from instruction import resolve_instruction
from schema import PlanError, validate_semantic_plan


def _goal_objects():
    def item(name, phrases, movable, relations):
        return {
            "sim_name": name,
            "semantic_type": name.rsplit("_", 1)[0],
            "phrases": phrases,
            "movable": movable,
            "can_be_target": bool(relations),
            "target_relations": relations,
        }

    return {
        "akita_black_bowl_1": item("akita_black_bowl_1", ["black bowl", "bowl"], True, ["on", "in"]),
        "cream_cheese_1": item("cream_cheese_1", ["cream cheese"], True, []),
        "plate_1": item("plate_1", ["plate"], True, ["on"]),
        "flat_stove_1": item("flat_stove_1", ["stove"], False, ["on"]),
    }


def _step(step_id, obj, target, relation, depends):
    return {
        "step_id": step_id,
        "skill": "transfer",
        "object_ref": obj,
        "target_ref": target,
        "relation": relation,
        "prompt": f"pick up the {obj}",
        "depends_on": depends,
        "completion_check": "object_placed",
        "budget": {"max_s": 30, "max_attempts": 1},
        "on_failure": "recover",
    }


class FreeTaskTest(unittest.TestCase):
    def test_catalog_miss_can_plan_without_switching(self):
        prepared = prepare_free_request(
            "请把奶油奶酪放到盘子上，再把碗放到炉灶上",
            _goal_objects(),
        )
        self.assertEqual(prepared["status"], "ready")
        self.assertFalse(prepared["changed"])
        self.assertEqual(
            [(item["object_ref"], item["target_ref"]) for item in prepared["predicates"]],
            [("cream_cheese_1", "plate_1"), ("akita_black_bowl_1", "flat_stove_1")],
        )
        loaded = {"suite": "libero_goal", "task_id": 1, "goal": "put the bowl on the stove", "summary_zh": "把碗放到炉灶上。"}
        benchmark = resolve_instruction("飞到月球", loaded)
        self.assertEqual(benchmark["status"], "unsupported")

    def test_unimplemented_skill_missing_object_and_ambiguity(self):
        objects = _goal_objects()
        objects["akita_black_bowl_2"] = dict(objects["akita_black_bowl_1"])
        objects["akita_black_bowl_2"]["sim_name"] = "akita_black_bowl_2"
        blocked = prepare_free_request("打开中间抽屉", objects)
        self.assertEqual(blocked["code"], "skill_unimplemented")
        self.assertIn("open_drawer", blocked["reason"])
        missing = prepare_free_request("把苹果放到炉灶上", objects)
        self.assertEqual(missing["code"], "object_missing")
        ambiguous = prepare_free_request("把碗放到盘子上", objects)
        self.assertEqual(ambiguous["status"], "clarify")
        self.assertIn("akita_black_bowl_2", ambiguous["question"])
        self.assertNotIn("默认第一个", ambiguous["question"].split("。")[0] + "选择")
        sided = prepare_free_request(
            "把左边的碗放到盘子上",
            objects,
            {"akita_black_bowl_1": [0.2, 0.0, 0.4], "akita_black_bowl_2": [-0.2, 0.0, 0.4], "plate_1": [0.0, 0.2, 0.4]},
        )
        self.assertEqual(sided["predicates"][0]["object_ref"], "akita_black_bowl_2")

    def test_sequence_is_not_a_duplicate_and_native_does_not_cover_user_goal(self):
        objects = _goal_objects()
        predicates = prepare_free_request("先把碗放到炉灶上，再把奶油奶酪放到盘子上", objects)["predicates"]
        plan = {
            "plan_id": "combo",
            "goal": "先把碗放到炉灶上，再把奶油奶酪放到盘子上",
            "steps": [
                _step("bowl", "akita_black_bowl_1", "flat_stove_1", "on", []),
                _step("cheese", "cream_cheese_1", "plate_1", "on", ["bowl"]),
            ],
        }
        validated = validate_semantic_plan(
            plan,
            task="put the bowl on the stove",
            scene_objects=objects,
            execution_mode="free",
            predicates=predicates,
            ordered=True,
        )
        self.assertEqual(validated["execution_mode"], "free")
        self.assertEqual(len(validated["steps"]), 2)
        broken = dict(plan)
        broken["steps"] = [dict(plan["steps"][0]), dict(plan["steps"][1], depends_on=[])]
        with self.assertRaises(PlanError) as raised:
            validate_semantic_plan(
                broken,
                scene_objects=objects,
                execution_mode="free",
                predicates=predicates,
                ordered=True,
            )
        self.assertTrue(any(item["code"] == "sequence_dependency" for item in raised.exception.errors))
        self.assertFalse(any(item["code"] == "duplicate_transfer" for item in raised.exception.errors))
        judged = judge_user_goal(
            predicates,
            {
                "objects": {
                    "akita_black_bowl_1": {"pos": [0.0, 0.0, 0.45]},
                    "flat_stove_1": {"pos": [0.0, 0.0, 0.45]},
                    "cream_cheese_1": {"pos": [0.3, 0.0, 0.45]},
                    "plate_1": {"pos": [0.0, 0.2, 0.45]},
                }
            },
        )
        self.assertFalse(judged["user_goal_success"])
        self.assertTrue(judged["subgoal_results"][0]["held_at_end"])
        self.assertFalse(judged["subgoal_results"][1]["held_at_end"])
        self.assertEqual(judged["evidence_source"], "sim_state")

    def test_free_episode_stays_steppable_after_native_goal(self):
        free = EpisodeFlags("free")
        native = free.absorb(True, False)
        self.assertTrue(native["clear_terminated"])
        self.assertTrue(free.continuable())
        limited = EpisodeFlags("free")
        self.assertFalse(limited.absorb(False, True)["continuable"])
        benchmark = EpisodeFlags("benchmark")
        self.assertFalse(benchmark.absorb(True, False)["clear_terminated"])
        self.assertFalse(benchmark.continuable())

        class Backend:
            def __init__(self, native_flag, continuable):
                self.native_flag = native_flag
                self._continuable = continuable

            def native_success(self):
                return self.native_flag

            def episode_continuable(self):
                return self._continuable

        self.assertEqual(native_execution_gate({"execution_mode": "benchmark"}, Backend(True, True)), "stop_native")
        self.assertEqual(native_execution_gate({"execution_mode": "free"}, Backend(True, True)), "record_native")
        self.assertEqual(native_execution_gate({"execution_mode": "free"}, Backend(True, False)), "stop_uncontinuable")


if __name__ == "__main__":
    unittest.main()
