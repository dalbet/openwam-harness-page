"""The composite contract: what is recognized, what is executable, what is not.

These tests never touch a simulator. They pin the boundary the acceptance
report depends on:

* the loaded scene must be the *last* step's task (its BDDL is the end state);
* only explicitly listed ordered pairs are executable;
* each step's prompt is the catalog's official instruction, never the user's
  words and never the other step's prompt;
* a pair is refused, with a specific reason code, when it is reversed,
  unknown, duplicated, longer than two steps, or anchored elsewhere.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from common import composite_plan, config, task_catalog  # noqa: E402

CFG = config.engine_config()
CATALOG = task_catalog.task_by_key()
TASK0 = config.task_for(0)
TASK8 = config.task_for(8)


class SupportTest(unittest.TestCase):
    def test_supported_pair_in_loaded_anchor_scene(self):
        check = composite_plan.support(
            ["open_middle_drawer", "put_bowl_on_plate"], loaded_task=TASK8, catalog=CATALOG, cfg=CFG
        )
        self.assertTrue(check["ok"], check)
        self.assertTrue(check["executable"])
        self.assertEqual(check["anchor"]["task_id"], 8)
        self.assertEqual([row["goal"]["kind"] for row in check["steps"]], ["drawer_open", "native_goal"])

    def test_reversed_order_is_not_executable(self):
        check = composite_plan.support(
            ["put_bowl_on_plate", "open_middle_drawer"], loaded_task=TASK8, catalog=CATALOG, cfg=CFG
        )
        self.assertFalse(check["ok"])
        self.assertEqual(check["reason_code"], "unsupported_combo")

    def test_scene_switch_required_when_anchor_is_not_loaded(self):
        check = composite_plan.support(
            ["open_middle_drawer", "put_bowl_on_plate"], loaded_task=TASK0, catalog=CATALOG, cfg=CFG
        )
        self.assertFalse(check["ok"])
        self.assertEqual(check["reason_code"], "requires_scene_switch")

    def test_unknown_pair_is_refused_not_attempted(self):
        check = composite_plan.support(
            ["open_middle_drawer", "put_bowl_on_stove"], loaded_task=TASK8, catalog=CATALOG, cfg=CFG
        )
        self.assertFalse(check["ok"])
        self.assertEqual(check["reason_code"], "unsupported_combo")
        self.assertIn(["open_middle_drawer", "put_bowl_on_plate"], check["supported_pairs"])

    def test_three_steps_exceed_the_limit(self):
        check = composite_plan.support(
            ["open_middle_drawer", "put_bowl_on_plate", "put_bowl_on_stove"],
            loaded_task=TASK8,
            catalog=CATALOG,
            cfg=CFG,
        )
        self.assertFalse(check["ok"])
        self.assertEqual(check["reason_code"], "step_limit_exceeded")

    def test_duplicate_step_is_refused(self):
        check = composite_plan.support(
            ["open_middle_drawer", "open_middle_drawer"], loaded_task=TASK8, catalog=CATALOG, cfg=CFG
        )
        self.assertFalse(check["ok"])
        self.assertEqual(check["reason_code"], "duplicate_step")

    def test_unknown_key_is_refused(self):
        check = composite_plan.support(
            ["open_middle_drawer", "not_a_capability"], loaded_task=TASK8, catalog=CATALOG, cfg=CFG
        )
        self.assertFalse(check["ok"])
        self.assertEqual(check["reason_code"], "key_not_in_catalog")


class PlanTest(unittest.TestCase):
    def _plan(self):
        return composite_plan.build_plan(
            ["open_middle_drawer", "put_bowl_on_plate"],
            catalog=CATALOG,
            loaded_task=TASK8,
            instruction="open the middle drawer then put the bowl on the plate",
            cfg=CFG,
        )

    def test_plan_uses_catalog_official_prompts_in_order(self):
        plan = self._plan()
        self.assertEqual(plan["task_scope"], "combo")
        self.assertEqual([step["step_id"] for step in plan["steps"]], ["c1", "c2"])
        self.assertEqual(plan["steps"][0]["prompt"], TASK0["official_instruction"])
        self.assertEqual(plan["steps"][1]["prompt"], TASK8["official_instruction"])
        self.assertEqual(plan["composite"]["anchor_task_id"], 8)
        self.assertFalse(plan["composite"]["bddl_mutation"])
        self.assertFalse(plan["composite"]["env_reset_between_steps"])

    def test_validate_accepts_the_built_plan(self):
        errors = composite_plan.validate_plan(self._plan(), catalog=CATALOG, loaded_task=TASK8, cfg=CFG)
        self.assertEqual(errors, [])

    def test_validate_rejects_a_replaced_prompt(self):
        plan = self._plan()
        plan["steps"][0]["prompt"] = "open the top drawer"
        errors = composite_plan.validate_plan(plan, catalog=CATALOG, loaded_task=TASK8, cfg=CFG)
        self.assertTrue(any(row["code"] == "not_catalog_official" for row in errors), errors)

    def test_validate_rejects_an_anchor_whose_goal_is_not_native(self):
        plan = self._plan()
        plan["steps"][-1]["completion"] = {"kind": "drawer_open", "predicate": "drawer_open"}
        errors = composite_plan.validate_plan(plan, catalog=CATALOG, loaded_task=TASK8, cfg=CFG)
        self.assertTrue(any(row["code"] == "anchor_goal_must_be_native" for row in errors), errors)

    def test_validate_rejects_a_plan_anchored_on_another_task(self):
        plan = self._plan()
        errors = composite_plan.validate_plan(plan, catalog=CATALOG, loaded_task=TASK0, cfg=CFG)
        self.assertTrue(any(row["code"] == "not_loaded_scene" for row in errors), errors)

    def test_validate_rejects_duplicate_step_ids(self):
        plan = self._plan()
        plan["steps"][1]["step_id"] = plan["steps"][0]["step_id"]
        errors = composite_plan.validate_plan(plan, catalog=CATALOG, loaded_task=TASK8, cfg=CFG)
        self.assertTrue(any(row["code"] == "not_unique" for row in errors), errors)

    def test_step_goal_is_predicate_only_for_the_early_step(self):
        plan = self._plan()
        self.assertEqual(composite_plan.step_goal(plan["steps"][0])["target_n"], "wooden_cabinet_1_middle_region")
        self.assertIsNone(composite_plan.step_goal(plan["steps"][1]))

    def test_release_required_only_for_placement_steps(self):
        self.assertFalse(composite_plan.release_required(TASK0))
        self.assertTrue(composite_plan.release_required(TASK8))

    def test_budgets_are_bounded_per_step(self):
        opts = composite_plan.options(CFG)
        self.assertEqual(opts["max_steps"], 2)
        self.assertGreater(composite_plan.step_budget(1, CFG), 0)
        self.assertGreater(composite_plan.step_budget(2, CFG), 0)
        self.assertLessEqual(composite_plan.step_budget(1, CFG) + composite_plan.step_budget(2, CFG), 2000)


class RouterParseTest(unittest.TestCase):
    def test_pair_is_parsed_as_an_ordered_list(self):
        from agent import semantic_router

        choice, errors = semantic_router.parse_choice(
            '{"choice":["open_middle_drawer","put_bowl_on_plate"]}'
        )
        self.assertEqual(errors, [])
        self.assertEqual(choice["kind"], "composite")
        self.assertEqual(choice["keys"], ["open_middle_drawer", "put_bowl_on_plate"])

    def test_single_key_still_parses_as_before(self):
        from agent import semantic_router

        choice, errors = semantic_router.parse_choice('{"choice":"put_bowl_on_plate"}')
        self.assertEqual(errors, [])
        self.assertEqual(choice["kind"], "single")
        self.assertEqual(choice["choice"], "put_bowl_on_plate")

    def test_one_element_list_is_a_format_error_not_a_silent_single(self):
        from agent import semantic_router

        choice, errors = semantic_router.parse_choice('{"choice":["put_bowl_on_plate"]}')
        self.assertIsNone(choice)
        self.assertEqual(errors[0]["code"], "single_key_list")

    def test_special_tokens_are_preserved(self):
        from agent import semantic_router

        for token in ("clarify", "unsupported"):
            choice, errors = semantic_router.parse_choice(f'{{"choice":"{token}"}}')
            self.assertEqual(errors, [])
            self.assertEqual(choice["kind"], "special")
            self.assertEqual(choice["choice"], token)


class _FakeChat:
    """Replays a fixed model answer so no Qwen call is needed."""

    def __init__(self, content: str):
        self.content = content
        self.calls = 0

    def __call__(self, messages, *, max_tokens=None):  # noqa: ARG002
        self.calls += 1
        return self.content, {"elapsed_s": 0.0, "completion_tokens": 8}


class RouterDecisionTest(unittest.TestCase):
    def test_pair_decision_reports_composite_keys_without_the_loaded_task(self):
        from agent import semantic_router

        snapshot = {
            "sim": {
                "objects": {
                    "akita_black_bowl_1": {"pos": [0, 0, 0]},
                    "cream_cheese_1": {"pos": [0, 0, 0]},
                    "wine_bottle_1": {"pos": [0, 0, 0]},
                    "plate_1": {"pos": [0, 0, 0]},
                }
            },
            "named_poses": {
                "sites": {
                    "wooden_cabinet_1_middle_region": {},
                    "plate_1": {},
                },
                "bodies": {"main_table": {}, "wooden_cabinet_1": {}, "flat_stove_1": {}, "wine_rack_1": {}},
            },
            "joints": {"wooden_cabinet_1_middle_level": {}},
        }
        decision = semantic_router.decide(
            "open the middle drawer then put the bowl on the plate",
            snapshot,
            chat_fn=_FakeChat('{"choice":["open_middle_drawer","put_bowl_on_plate"]}'),
        )
        self.assertEqual(decision["status"], "ok")
        self.assertEqual(decision["composite_keys"], ["open_middle_drawer", "put_bowl_on_plate"])
        self.assertFalse(decision["router"]["snapshot_in_prompt"])
        assembled = semantic_router.assemble_plan(
            decision,
            loaded_task_id=8,
            instruction="open the middle drawer then put the bowl on the plate",
            snapshot=snapshot,
            skills={"skills": []},
            loaded_task=TASK8,
        )
        self.assertEqual(assembled["status"], "ok")
        self.assertEqual(
            [step["prompt"] for step in assembled["plan"]["steps"]],
            [TASK0["official_instruction"], TASK8["official_instruction"]],
        )

    def test_recognized_but_unsupported_pair_is_refused_by_code(self):
        from agent import semantic_router

        snapshot = {
            "sim": {
                "objects": {
                    "akita_black_bowl_1": {"pos": [0, 0, 0]},
                    "cream_cheese_1": {"pos": [0, 0, 0]},
                    "wine_bottle_1": {"pos": [0, 0, 0]},
                    "plate_1": {"pos": [0, 0, 0]},
                }
            },
            "named_poses": {
                "sites": {
                    "wooden_cabinet_1_middle_region": {},
                    "flat_stove_1_cook_region": {},
                    "plate_1": {},
                },
                "bodies": {"main_table": {}, "wooden_cabinet_1": {}, "flat_stove_1": {}, "wine_rack_1": {}},
            },
            "joints": {"wooden_cabinet_1_middle_level": {}},
        }
        decision = semantic_router.decide(
            "open the middle drawer then put the bowl on the stove",
            snapshot,
            chat_fn=_FakeChat('{"choice":["open_middle_drawer","put_bowl_on_stove"]}'),
        )
        self.assertEqual(decision["status"], "ok")
        self.assertEqual(decision["composite_keys"], ["open_middle_drawer", "put_bowl_on_stove"])
        assembled = semantic_router.assemble_plan(
            decision,
            loaded_task_id=8,
            instruction="open the middle drawer then put the bowl on the stove",
            snapshot=snapshot,
            skills={"skills": []},
            loaded_task=TASK8,
        )
        self.assertEqual(assembled["status"], "unsupported")
        self.assertEqual(assembled["reason_code"], "unsupported_combo")
        self.assertIsNone(assembled["plan"])

    def test_three_step_answer_is_refused_before_assembly(self):
        from agent import semantic_router

        decision = semantic_router.decide(
            "open the drawer, put the bowl on the plate and switch on the stove",
            {"sim": {"objects": {}}},            chat_fn=_FakeChat(
                '{"choice":["open_middle_drawer","put_bowl_on_plate","turn_on_stove"]}'
            ),
        )
        self.assertEqual(decision["status"], "unsupported")
        self.assertEqual(decision["reason_code"], "step_limit_exceeded")

    def test_clarify_and_unsupported_are_unchanged(self):
        from agent import semantic_router

        clarify = semantic_router.decide("put it on the plate", {}, chat_fn=_FakeChat('{"choice":"clarify"}'))
        self.assertEqual(clarify["status"], "needs_clarification")
        outside = semantic_router.decide(
            "put the bowl behind the stove", {}, chat_fn=_FakeChat('{"choice":"unsupported"}')
        )
        self.assertEqual(outside["status"], "unsupported")


if __name__ == "__main__":
    unittest.main()
