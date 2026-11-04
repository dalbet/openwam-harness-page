"""The ordered composite contract is N steps (2 or 3), not "a pair".

These are pure-data checks: no simulator, no model.
"""

from __future__ import annotations

from common import composite_plan

CATALOG = {
    "put_cream_cheese_in_bowl": {
        "task_id": 6,
        "key": "put_cream_cheese_in_bowl",
        "instruction": "put the cream cheese in the bowl",
        "official_instruction": "put the cream cheese in the bowl",
        "kind": "pick_place",
        "objects": ["cream_cheese_1"],
        "targets": ["akita_black_bowl_1"],
        "semantics": {"operation": "pick_place", "relation": "in"},
    },
    "put_bowl_on_stove": {
        "task_id": 1,
        "key": "put_bowl_on_stove",
        "instruction": "put the bowl on the stove",
        "official_instruction": "put the bowl on the stove",
        "kind": "pick_place",
        "objects": ["akita_black_bowl_1"],
        "targets": ["flat_stove_1_cook_region"],
        "semantics": {"operation": "pick_place", "relation": "on"},
    },
    "turn_on_stove": {
        "task_id": 7,
        "key": "turn_on_stove",
        "instruction": "turn on the stove",
        "official_instruction": "turn on the stove",
        "kind": "turn_on",
        "objects": ["flat_stove_1"],
        "targets": ["flat_stove_1"],
        "semantics": {"operation": "turn_on", "relation": "turn_on"},
    },
}

LOADED = CATALOG["turn_on_stove"]
KEYS = ["put_cream_cheese_in_bowl", "put_bowl_on_stove", "turn_on_stove"]
CFG = {
    "composite": {
        "enabled": True,
        "max_steps": 3,
        "supported_pairs": [KEYS],
        "step_max_steps": {"1": 300, "2": 600, "3": 300},
    }
}


def test_three_step_sequence_is_supported_when_listed():
    check = composite_plan.support(KEYS, loaded_task=LOADED, catalog=CATALOG, cfg=CFG)
    assert check["ok"] is True
    assert [row["key"] for row in check["steps"]] == KEYS
    kinds = [row["goal"]["kind"] for row in check["steps"]]
    assert kinds == ["relation", "relation", "native_goal"]


def test_four_steps_are_refused_and_old_two_step_limit_is_gone():
    four = KEYS + ["put_bowl_on_plate"]
    over = composite_plan.support(four, loaded_task=LOADED, catalog=CATALOG, cfg=CFG)
    assert over["reason_code"] == "step_limit_exceeded"
    two = composite_plan.support(KEYS[:2], loaded_task=LOADED, catalog=CATALOG, cfg=CFG)
    assert two["reason_code"] == "unsupported_combo"


def test_build_plan_and_validate_plan_carry_three_steps():
    plan = composite_plan.build_plan(
        KEYS, catalog=CATALOG, loaded_task=LOADED, instruction="prepare the cheese", cfg=CFG
    )
    assert plan["step_limit"] == 3
    assert [step["step_index"] for step in plan["steps"]] == [1, 2, 3]
    assert [step["step_id"] for step in plan["steps"]] == ["c1", "c2", "c3"]
    errors = composite_plan.validate_plan(plan, catalog=CATALOG, loaded_task=LOADED, cfg=CFG)
    assert errors == []
    goals = [composite_plan.step_goal(step) for step in plan["steps"]]
    assert goals[0] == {
        "skill": "pick_place",
        "object_n": "cream_cheese_1",
        "target_n": "akita_black_bowl_1",
        "relation": "in",
    }
    assert goals[2] is None  # the anchor is the loaded scene's native goal


def test_validate_plan_rejects_an_early_native_goal():
    plan = composite_plan.build_plan(
        KEYS, catalog=CATALOG, loaded_task=LOADED, instruction="prepare the cheese", cfg=CFG
    )
    plan["steps"][0]["completion"] = {"kind": "native_goal", "predicate": "native_libero_goal"}
    errors = composite_plan.validate_plan(plan, catalog=CATALOG, loaded_task=LOADED, cfg=CFG)
    assert any(row["code"] == "early_step_native_goal" for row in errors)


def test_step_budget_defaults_cover_a_third_step():
    assert composite_plan.step_budget(3, {}) == composite_plan.DEFAULT_STEP_MAX_STEPS["3"]
    assert composite_plan.step_budget(3, CFG) == 300
    assert composite_plan.step_budget(2, CFG) == 600
