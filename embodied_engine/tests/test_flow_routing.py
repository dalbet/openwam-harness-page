"""Constrained household flows: the model picks a flow id, code expands it.

The values under test are the two properties the round depends on:

* a flow is expanded to the catalog's ordered keys, then the *same* composite
  contract decides whether it can run — a flow can never introduce a step the
  catalog does not have;
* a single-key answer that happens to be a flow's last step is asked about
  once, so a complete household goal is not silently reduced to "turn the
  stove on".
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from agent import semantic_router  # noqa: E402
from common import task_catalog  # noqa: E402

FLOW_ID = "cream_cheese_heating_prep"
FLOW_KEYS = ["put_cream_cheese_in_bowl", "put_bowl_on_stove", "turn_on_stove"]


def _snapshot(*objects: str) -> dict:
    return {
        "scene_epoch": 1,
        "sim_action_count": 0,
        "observation_id": 1,
        "sim": {
            "objects": {name: {"pos": [0.0, 0.0, 0.0]} for name in objects},
            "eef_pos": [0.0, 0.0, 0.0],
            "gripper_opening_m": 0.08,
        },
        "named_poses": {
            "sites": {
                name: {"position": [0.0, 0.0, 0.0]}
                for name in (
                    "main_table_plate_region",
                    "main_table_stove_front_region",
                    "flat_stove_1_cook_region",
                    "flat_stove_1_button",
                )
            }
        },
        "joints": {},
    }


def _fake_seq(*contents: str, calls: list | None = None):
    queue = list(contents)

    def chat(messages, max_tokens=None):  # noqa: ANN001
        if calls is not None:
            calls.append({"messages": messages, "max_tokens": max_tokens})
        content = queue.pop(0) if len(queue) > 1 else queue[0]
        return content, {
            "elapsed_s": 0.01,
            "prompt_tokens": 100,
            "completion_tokens": 5,
            "ttft_s": 0.005,
            "decode_s": 0.005,
        }

    return chat


FULL_SNAPSHOT = _snapshot("cream_cheese_1", "akita_black_bowl_1", "plate_1", "flat_stove_1")


class TestFlowCatalog:
    def test_flow_exists_with_ordered_keys(self):
        flow = task_catalog.flow_by_id()[FLOW_ID]
        assert [str(key) for key in flow["keys"]] == FLOW_KEYS

    def test_every_flow_key_is_a_catalog_capability(self):
        catalog = task_catalog.task_by_key()
        for flow in task_catalog.flows():
            for key in flow["keys"]:
                assert key in catalog, key

    def test_flow_lines_name_the_steps_and_the_exclusions(self):
        messages = semantic_router.build_messages("帮我把奶酪准备好加热")
        system = messages[0]["content"]
        assert FLOW_ID in system
        for key in FLOW_KEYS:
            assert key in system
        assert "stirring" in system
        # The flow description must not be a keyword list the model can copy.
        assert "奶酪准备好" not in system


class TestFlowParse:
    def test_flow_object_is_parsed_and_expanded_by_code(self):
        choice, errors = semantic_router.parse_choice(json.dumps({"choice": {"flow": FLOW_ID}}))
        assert not errors
        assert choice["kind"] == "flow"
        assert choice["keys"] == FLOW_KEYS

    def test_unknown_flow_is_a_format_error(self):
        choice, errors = semantic_router.parse_choice(json.dumps({"choice": {"flow": "nope"}}))
        assert choice is None
        assert errors[0]["code"] == "not_in_flows"

    def test_flow_with_extra_keys_is_rejected(self):
        choice, errors = semantic_router.parse_choice(
            json.dumps({"choice": {"flow": FLOW_ID, "keys": FLOW_KEYS}})
        )
        assert choice is None
        assert errors[0]["code"] == "unexpected_keys"


class TestFlowDecide:
    def test_flow_choice_becomes_the_ordered_keys(self):
        decision = semantic_router.decide(
            "help me get the cream cheese ready to heat up",
            FULL_SNAPSHOT,
            chat_fn=_fake_seq(json.dumps({"choice": {"flow": FLOW_ID}})),
        )
        assert decision["status"] == "ok"
        assert decision["composite_keys"] == FLOW_KEYS
        assert decision["flow"]["flow_id"] == FLOW_ID
        assert decision["flow"]["source"] == "catalog flows block"

    def test_flow_assembly_records_the_expansion(self):
        decision = semantic_router.decide(
            "help me get the cream cheese ready to heat up",
            FULL_SNAPSHOT,
            chat_fn=_fake_seq(json.dumps({"choice": {"flow": FLOW_ID}})),
        )
        assembled = semantic_router.assemble_plan(
            decision,
            loaded_task_id=7,
            instruction="help me get the cream cheese ready to heat up",
            snapshot=FULL_SNAPSHOT,
            skills={"skills": [{"name": "whole_task"}]},
            loaded_task=task_catalog.task_for(7),
        )
        assert assembled["status"] == "ok"
        plan = assembled["plan"]
        assert plan["flow_expansion"]["flow_id"] == FLOW_ID
        assert plan["flow_expansion"]["keys"] == FLOW_KEYS
        assert plan["composite"]["flow"]["flow_id"] == FLOW_ID
        assert [step["step_task_key"] for step in plan["steps"]] == FLOW_KEYS

    def test_flow_rejected_when_the_scene_lacks_an_entity(self):
        decision = semantic_router.decide(
            "help me get the cream cheese ready to heat up",
            _snapshot("cream_cheese_1"),
            chat_fn=_fake_seq(json.dumps({"choice": {"flow": FLOW_ID}})),
        )
        assert decision["status"] == "scene_not_supported"
        assert decision["reason_code"] == "required_entity_absent"


class TestFlowGuard:
    def test_single_last_step_is_asked_about_and_upgraded(self):
        calls: list = []
        decision = semantic_router.decide(
            "help me get the cream cheese ready to heat up",
            FULL_SNAPSHOT,
            chat_fn=_fake_seq('{"choice":"turn_on_stove"}', '{"scope":"flow"}', calls=calls),
        )
        assert len(calls) == 2, "one routing call plus exactly one guard call"
        assert decision["flow"]["flow_id"] == FLOW_ID
        assert decision["flow"]["source"] == "flow_guard"
        assert decision["composite_keys"] == FLOW_KEYS
        assert decision["flow_guard"]["scope"] == "flow"

    def test_only_the_stove_on_stays_a_single_capability(self):
        calls: list = []
        decision = semantic_router.decide(
            "turn on the stove",
            FULL_SNAPSHOT,
            chat_fn=_fake_seq('{"choice":"turn_on_stove"}', '{"scope":"single"}', calls=calls),
        )
        assert len(calls) == 2
        assert decision["status"] == "ok"
        assert decision.get("composite_keys") in (None, [])
        assert decision["status"] == "ok"
        assert decision["task"]["key"] == "turn_on_stove"
        assert decision["flow_guard"]["scope"] == "single"

    def test_guard_is_skipped_for_a_key_that_is_no_flows_last_step(self):
        calls: list = []
        decision = semantic_router.decide(
            "put the bowl on the plate",
            FULL_SNAPSHOT,
            chat_fn=_fake_seq('{"choice":"put_bowl_on_plate"}', calls=calls),
        )
        assert len(calls) == 1
        assert decision["task"]["key"] == "put_bowl_on_plate"

    def test_unparseable_guard_answer_keeps_the_single_choice(self):
        decision = semantic_router.decide(
            "turn on the stove",
            FULL_SNAPSHOT,
            chat_fn=_fake_seq('{"choice":"turn_on_stove"}', "I think it is the flow"),
        )
        assert decision["task"]["key"] == "turn_on_stove"
        assert decision["flow_guard"]["scope"] is None
        assert decision["flow_guard"]["error"] is not None
