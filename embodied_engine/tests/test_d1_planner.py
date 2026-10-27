#!/usr/bin/env python
"""d1-3B planner: candidate isolation, staged decision, contract and refusals.

These tests use a stub decision client, so they exercise the production
assembly path (``plan_with_d1``) without loading the 3B model or touching GPU.
"""

from __future__ import annotations

import sys
from pathlib import Path

ENGINE_ROOT = Path(__file__).resolve().parents[1]
if str(ENGINE_ROOT) not in sys.path:
    sys.path.insert(0, str(ENGINE_ROOT))

from agent import d1_planner  # noqa: E402

SKILLS = {"skills": [{"name": "whole_task", "summary": "one policy call for the whole task"}]}

SNAPSHOT = {
    "observation_id": "test-observation:0",
    "scene_epoch": 0,
    "sim_action_count": 0,
    "sim": {
        "eef_pos": [0.0, 0.0, 1.0],
        "objects": {
            "akita_black_bowl_1": {"pos": [0.10, 0.20, 0.90]},
            "plate_1": {"pos": [0.20, 0.20, 0.90]},
            "cream_cheese_1": {"pos": [0.30, 0.20, 0.90]},
            "wine_bottle_1": {"pos": [0.40, 0.20, 0.90]},
        },
    },
    "named_poses": {
        "sites": {
            "flat_stove_1_cook_region": {"position": [0.00, 0.00, 0.90]},
            "flat_stove_1_burner": {"position": [0.00, 0.00, 0.95]},
            "wooden_cabinet_1_top_region": {"position": [0.50, 0.00, 0.90]},
            "main_table_plate_region": {"position": [0.20, 0.60, 0.80]},
            "table_top": {"position": [0.00, 0.60, 0.80]},
        }
    },
    "joints": {
        "wooden_cabinet_1_top_level": {"qpos": [0.0]},
        "flat_stove_1_button": {"qpos": [0.0]},
    },
}


class _Stub:
    """Answers the two staged question sets with canned probabilities."""

    def __init__(self, stage1: dict, stage2: dict):
        self.stage1 = stage1
        self.stage2 = stage2
        self.calls: list[list[str]] = []

    def system_one(self, state, questions):
        self.calls.append(sorted(questions))
        if "supported" in questions:
            return {"status": "ok", "answers": self.stage1}
        return {"status": "ok", "answers": self.stage2}


def _stage1(**overrides):
    answers = {
        "supported": {"type": "noul", "noul": 0.95},
        "operation": {
            "type": "choice",
            "choice": "pick_place",
            "confidence": 0.9,
            "probabilities": {
                "pick_place": 0.9,
                "open_drawer": 0.02,
                "push_object": 0.03,
                "turn_on": 0.01,
                "composite": 0.02,
                "other": 0.02,
            },
        },
        "source": {
            "type": "choice",
            "choice": "akita_black_bowl_1",
            "confidence": 0.9,
            "probabilities": {
                "akita_black_bowl_1": 0.9,
                "plate_1": 0.03,
                "cream_cheese_1": 0.04,
                "wine_bottle_1": 0.03,
            },
        },
        "source_specified": {"type": "noul", "noul": 0.95},
        "ambiguous": {"type": "noul", "noul": 0.05},
    }
    answers.update(overrides)
    return answers


def _stage2(**overrides):
    answers = {
        "target": {
            "type": "choice",
            "choice": "flat_stove_1_cook_region",
            "confidence": 0.9,
            "probabilities": {
                "flat_stove_1_cook_region": 0.9,
                "table_top": 0.05,
                "none": 0.05,
            },
        },
        "target_specified": {"type": "noul", "noul": 0.95},
    }
    answers.update(overrides)
    return answers


def _plan(instruction: str = "put the bowl on the stove", **stage1_overrides):
    stub = _Stub(_stage1(**stage1_overrides), _stage2())
    result = d1_planner.plan_with_d1(
        instruction=instruction,
        snapshot=SNAPSHOT,
        skills=SKILLS,
        d1_url="http://unused",
        task_id=1,
        client=stub,
    )
    return result, stub


def test_candidates_only_come_from_scene_metadata():
    candidates = d1_planner.build_candidates(SNAPSHOT)
    object_names = {row["name"] for row in candidates["objects"]}
    target_names = {row["name"] for row in candidates["targets"]}
    assert object_names == set(SNAPSHOT["sim"]["objects"])
    # No instruction, BDDL goal or official task text is accepted as an input.
    assert "instruction" not in candidates
    assert not any("put the bowl" in str(row) for row in candidates["targets"])
    assert "flat_stove_1_cook_region" in target_names
    assert "flat_stove_1_burner" not in target_names  # deduped with the region
    assert [row["name"] for row in candidates["drawers"]] == ["wooden_cabinet_1_top_region"]
    assert [row["name"] for row in candidates["appliances"]] == ["flat_stove_1"]


def test_accepted_decision_assembles_valid_whole_task_plan():
    result, stub = _plan()
    assert result["status"] == "accepted", result
    plan = result["plan"]
    assert plan["steps"] == [
        {
            "skill": "whole_task",
            "object_n": None,
            "target_n": None,
            "relation": "task",
            "prompt": "put the bowl on the stove",
        }
    ]
    assert plan["plan_source"] == "d1-3b"
    assert result["decision"]["source"] == "akita_black_bowl_1"
    assert result["decision"]["target"] == "flat_stove_1_cook_region"
    assert result["decision"]["relation"] == "on"
    assert len(stub.calls) == 2, "one forward pass per decision stage"


def test_unsupported_request_is_refused_before_stage2():
    result, stub = _plan(supported={"type": "noul", "noul": 0.05})
    assert result["status"] == "unsupported"
    assert result["reason_code"] == "request_not_supported"
    assert result["plan"] is None
    assert len(stub.calls) == 1


def test_low_source_confidence_asks_for_clarification():
    result, _ = _plan(
        source={
            "type": "choice",
            "choice": "akita_black_bowl_1",
            "confidence": 0.41,
            "probabilities": {"akita_black_bowl_1": 0.41, "plate_1": 0.35},
        }
    )
    assert result["status"] == "needs_clarification"
    assert result["reason_code"] == "source_uncertain"
    assert result["plan"] is None


def test_contradictory_composite_binding_is_rejected():
    stage1 = _stage1(
        operation={
            "type": "choice",
            "choice": "composite",
            "confidence": 0.9,
            "probabilities": {"composite": 0.9, "pick_place": 0.05},
        }
    )
    stage2 = {
        "open_target": {
            "type": "choice",
            "choice": "wooden_cabinet_1_top_region",
            "confidence": 0.9,
            "probabilities": {"wooden_cabinet_1_top_region": 0.9},
        },
        "object": {
            "type": "choice",
            "choice": "akita_black_bowl_1",
            "confidence": 0.9,
            "probabilities": {"akita_black_bowl_1": 0.9},
        },
        "place_target": {
            "type": "choice",
            "choice": "flat_stove_1_cook_region",
            "confidence": 0.7,
            "probabilities": {"flat_stove_1_cook_region": 0.7},
        },
        "target_specified": {"type": "noul", "noul": 0.95},
    }
    stub = _Stub(stage1, stage2)
    result = d1_planner.plan_with_d1(
        instruction="open the top drawer and put the bowl in it",
        snapshot=SNAPSHOT,
        skills=SKILLS,
        d1_url="http://unused",
        task_id=3,
        client=stub,
    )
    assert result["status"] == "rejected"
    assert result["reason_code"] in {"composite_binding_not_in_scene", "contradictory_binding"}
    assert result["plan"] is None


def test_plan_rejects_a_binding_the_user_did_not_name():
    stage1 = _stage1(
        source={
            "type": "choice",
            "choice": "plate_1",
            "confidence": 0.9,
            "probabilities": {"plate_1": 0.9, "akita_black_bowl_1": 0.05},
        }
    )
    stub = _Stub(stage1, _stage2())
    result = d1_planner.plan_with_d1(
        instruction="put the bowl on the stove",
        snapshot=SNAPSHOT,
        skills=SKILLS,
        d1_url="http://unused",
        task_id=1,
        client=stub,
    )
    assert result["status"] == "rejected"
    assert result["reason_code"] == "user_goal_source_not_preserved"
    assert result["plan"] is None


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"ok  {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL {name}: {exc}")
    raise SystemExit(1 if failures else 0)
