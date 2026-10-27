"""Unit tests for the compact semantic router and its deterministic checks."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from agent import semantic_router  # noqa: E402
from common import task_catalog  # noqa: E402


def _snapshot(*objects: str, sites: tuple[str, ...] = (), joints: dict | None = None) -> dict:
    return {
        "scene_epoch": 1,
        "sim_action_count": 0,
        "observation_id": 1,
        "sim": {
            "objects": {name: {"pos": [0.0, 0.0, 0.0]} for name in objects},
            "eef_pos": [0.0, 0.0, 0.0],
            "gripper_opening_m": 0.08,
        },
        "named_poses": {"sites": {name: {"position": [0.0, 0.0, 0.0]} for name in sites}},
        "joints": joints or {},
    }


def _fake(content: str, calls: list | None = None):
    def chat(messages, max_tokens=None):  # noqa: ANN001
        if calls is not None:
            calls.append({"messages": messages, "max_tokens": max_tokens})
        return content, {
            "elapsed_s": 0.01,
            "prompt_tokens": 100,
            "completion_tokens": 5,
            "ttft_s": 0.005,
            "decode_s": 0.005,
        }

    return chat


def _skills() -> dict:
    return {"skills": [{"name": "whole_task"}]}


def test_prompt_contains_catalog_and_original_instruction_only():
    messages = semantic_router.build_messages("把碗放到炉灶上")
    system = messages[0]["content"]
    for row in task_catalog.tasks():
        assert row["key"] in system
    assert "put the bowl on the stove" not in system
    assert messages[1]["content"] == "把碗放到炉灶上"
    assert "task_id" not in system
    assert "bddl" not in system.lower()


def test_parse_repairs_once_then_accepts_choice():
    calls: list = []
    chat = _fake('{"choice":"put_bowl_on_stove"}', calls)
    decision = semantic_router.decide(
        "把碗放到炉灶上",
        _snapshot("akita_black_bowl_1", "flat_stove_1", sites=("flat_stove_1_cook_region",)),
        chat_fn=chat,
    )
    assert decision["status"] == "ok"
    assert decision["choice"] == "put_bowl_on_stove"
    assert decision["repairs"] == 0
    assert len(calls) == 1
    assert calls[0]["max_tokens"] == semantic_router.MAX_TOKENS


def test_unknown_choice_triggers_one_repair():
    calls: list = []

    def chat(messages, max_tokens=None):  # noqa: ANN001
        calls.append(messages)
        if len(calls) == 1:
            return '{"choice":"task_1"}', {"elapsed_s": 0.01}
        return '{"choice":"put_bowl_on_stove"}', {"elapsed_s": 0.01}

    decision = semantic_router.decide(
        "put the bowl on the stove",
        _snapshot("akita_black_bowl_1", "flat_stove_1", sites=("flat_stove_1_cook_region",)),
        chat_fn=chat,
    )
    assert decision["status"] == "ok"
    assert decision["repairs"] == 1
    assert len(calls) == 2


def test_clarify_and_unsupported_are_distinct():
    clarify = semantic_router.decide(
        "put the bowl on it",
        _snapshot(),
        chat_fn=_fake('{"choice":"clarify"}'),
    )
    unsupported = semantic_router.decide(
        "put the apple in the basket",
        _snapshot(),
        chat_fn=_fake('{"choice":"unsupported"}'),
    )
    assert clarify["status"] == "needs_clarification"
    assert clarify["choice"] == "clarify"
    assert unsupported["status"] == "unsupported"
    assert unsupported["choice"] == "unsupported"


def test_scene_support_and_task_mismatch():
    task = task_catalog.task_for(1)
    decision = semantic_router.decide(
        "put the bowl on the stove",
        _snapshot("akita_black_bowl_1", "flat_stove_1", sites=("flat_stove_1_cook_region",)),
        chat_fn=_fake('{"choice":"put_bowl_on_stove"}'),
    )
    assert decision["status"] == "ok"
    assert decision["scene_support"]["ok"] is True
    assembled = semantic_router.assemble_plan(
        decision,
        loaded_task_id=1,
        instruction="put the bowl on the stove",
        snapshot=_snapshot("akita_black_bowl_1", "flat_stove_1", sites=("flat_stove_1_cook_region",)),
        skills=_skills(),
    )
    assert assembled["status"] == "ok"
    assert assembled["plan"]["steps"][0]["prompt"] == task["official_instruction"]

    mismatch = semantic_router.assemble_plan(
        decision,
        loaded_task_id=4,
        instruction="put the bowl on the stove",
        snapshot=_snapshot("akita_black_bowl_1", "flat_stove_1", sites=("flat_stove_1_cook_region",)),
        skills=_skills(),
    )
    assert mismatch["status"] == "task_mismatch"
    assert mismatch["plan"] is None


def test_missing_scene_entity_is_reported():
    decision = semantic_router.decide(
        "put the bowl on the stove",
        _snapshot(),
        chat_fn=_fake('{"choice":"put_bowl_on_stove"}'),
    )
    assert decision["status"] == "scene_not_supported"
    assert decision["scene_support"]["ok"] is False
    missing = {row["name"] for row in decision["scene_support"]["missing"]}
    assert "akita_black_bowl_1" in missing
    assert "flat_stove_1_cook_region" in missing


def test_composite_keeps_official_instruction():
    snapshot = _snapshot(
        "akita_black_bowl_1",
        "wooden_cabinet_1",
        sites=("wooden_cabinet_1_top_region",),
        joints={"wooden_cabinet_1_top_level": {"qpos": 0.0}},
    )
    decision = semantic_router.decide(
        "打开最上面的抽屉，把碗放进去",
        snapshot,
        chat_fn=_fake(json.dumps({"choice": "open_top_drawer_put_bowl_inside"})),
    )
    assert decision["status"] == "ok"
    assembled = semantic_router.assemble_plan(
        decision,
        loaded_task_id=3,
        instruction="打开最上面的抽屉，把碗放进去",
        snapshot=snapshot,
        skills=_skills(),
    )
    assert assembled["status"] == "ok"
    assert assembled["plan"]["steps"][0]["prompt"] == "open the top drawer and put the bowl inside"
