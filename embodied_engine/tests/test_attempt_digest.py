"""Per-attempt telemetry and round-reporting contracts.

A retried attempt used to vanish from the episode record: only the final
attempt stayed in ``steps``, so "task 2 took three attempts and 1800 actions"
could not be broken down after the fact. These tests pin the two pieces that
make that audit possible:

* ``_step_digest`` keeps the numbers that describe what an attempt did,
* the round table never reports a routing mismatch that did not happen, and
  never reports a placement as released when the evidence says otherwise.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from engine.engine_service import _scene_fingerprint, _state_fingerprint, _step_digest

PROJECT = Path(__file__).resolve().parents[1]


def _runner():
    spec = importlib.util.spec_from_file_location(
        "run_language_rounds", PROJECT / "scripts" / "run_language_rounds.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _obs(index: int, *, goal: bool = False, opening: float = 0.01, obj=None, eef=None) -> dict:
    return {
        "sim_action_count": 100 + index,
        "observation_id": f"obs-{index}",
        "native_goal": goal,
        "gripper_opening_m": opening,
        "object_pose": obj or [0.0, 0.0, 0.9],
        "eef_pos": eef or [0.0, 0.0, 1.0],
    }


def test_digest_keeps_the_facts_of_a_failed_attempt() -> None:
    record = {
        "attempt": 2,
        "step": {"skill": "whole_task"},
        "ok": False,
        "sim_actions": 600,
        "chunks_used": 19,
        "failure": {"reason_code": "native_goal_not_reached", "message": "600 actions", "grasp": {"held": False}},
        "observations": [
            _obs(0, opening=0.06),
            _obs(1, opening=0.004),
            _obs(2, goal=True, opening=0.078),
            _obs(3, opening=0.079),
        ],
    }
    digest = _step_digest(record)
    assert digest["attempt"] == 2
    assert digest["skill"] == "whole_task"
    assert digest["ok"] is False
    assert digest["sim_actions"] == 600
    assert digest["chunks_used"] == 19
    assert digest["failure_code"] == "native_goal_not_reached"
    assert digest["held_at_failure"] is False
    assert digest["native_goal_hits"] == 1
    assert digest["native_goal_first_index"] == 2
    assert digest["gripper_open_min_m"] == 0.004
    assert digest["gripper_open_max_m"] == 0.079
    assert digest["gripper_open_first_m"] == 0.06
    assert digest["gripper_open_last_m"] == 0.079
    assert digest["sim_action_count_first"] == 100
    assert digest["sim_action_count_last"] == 103
    assert digest["object_first"] == [0.0, 0.0, 0.9]
    assert digest["eef_last"] == [0.0, 0.0, 1.0]


def test_digest_is_json_safe_and_survives_a_missing_attempt_body() -> None:
    empty = _step_digest({})
    assert empty["sim_actions"] is None
    assert empty["native_goal_hits"] == 0
    assert empty["gripper_open_last_m"] is None
    assert empty["object_last"] is None
    json.dumps(empty)  # lands in the episode record, so it must serialize

    partial = _step_digest({"attempt": 1, "observations": [{"native_goal": False}], "failure": None})
    assert partial["gripper_open_last_m"] is None
    assert partial["failure_code"] is None
    json.dumps(partial)


def _status(tmp_path: Path, record: dict) -> dict:
    path = tmp_path / "episode.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    return {"last_result": {"record": str(path)}}


def _base_record() -> dict:
    return {
        "semantic": {
            "decision_status": "ok",
            "semantic_choice": "put_bowl_on_stove",
            "official_instruction": "put the bowl on the stove",
        },
        "steps": [{"prompt": "put the bowl on the stove", "verification": {"prompt": "put the bowl on the stove"}}],
        "native_task_success": True,
        "user_goal_success": True,
        "plan": {"latency_s": 0.7},
    }


def test_round_table_accepts_the_catalog_key_or_the_official_phrase(tmp_path: Path) -> None:
    runner = _runner()
    task = {"key": "put_bowl_on_stove", "official_instruction": "put the bowl on the stove"}
    status = _status(tmp_path, _base_record())
    for expected in ("put_bowl_on_stove", "put the bowl on the stove"):
        spec = {"round": 1, "task_id": 1, "lang": "en", "instruction": "x", "expected_choice": expected}
        row = runner._round_row(spec, task, status, wall_s=1.0)
        assert row["mapping_ok"] is True, expected

    wrong = {"round": 1, "task_id": 1, "lang": "en", "instruction": "x", "expected_choice": "put_bowl_on_plate"}
    assert runner._round_row(wrong, task, status, wall_s=1.0)["mapping_ok"] is False


def test_round_table_does_not_call_a_held_episode_released(tmp_path: Path) -> None:
    runner = _runner()
    task = {"key": "put_bowl_on_stove", "official_instruction": "put the bowl on the stove", "kind": "pick_place"}
    spec = {"round": 1, "task_id": 1, "lang": "en", "instruction": "x", "expected_choice": "put_bowl_on_stove"}
    base = dict(_base_record(), user_goal_success=False)

    held = dict(base, release_evidence={"state": "held"}, placement=None)
    row = runner._round_row(spec, task, _status(tmp_path, held), wall_s=1.0)
    assert row["full_goal_evidence"] == "held_or_unknown"
    assert row["release_state"] == "held"

    no_evidence = dict(base, release_evidence={"state": "unknown"}, placement=None)
    row = runner._round_row(spec, task, _status(tmp_path, no_evidence), wall_s=1.0)
    assert row["full_goal_evidence"] == "held_or_unknown"

    verified = dict(
        base,
        user_goal_success=True,
        placement={"placement_verified": True, "release_verified": True, "execution_status": "completed"},
    )
    row = runner._round_row(spec, task, _status(tmp_path, verified), wall_s=1.0)
    assert row["release_state"] == "verified"
    assert row["full_goal_evidence"] == "released"


def test_state_fingerprint_catches_a_different_arm_pose() -> None:
    """Two runs can agree on object positions and still see different pixels."""

    def snapshot(dy: float) -> dict:
        return {
            "sim": {
                "objects": {"akita_black_bowl_1": {"pos": [0.0, 0.0, 0.9]}},
                "eef_pos": [0.0, 0.0, 1.0],
                "gripper_opening_m": 0.078,
            },
            "joints": {"robot0_joint1": 0.1, "robot0_joint2": dy},
        }

    same = snapshot(0.2)
    moved = snapshot(0.2004)
    assert _scene_fingerprint(same) == _scene_fingerprint(moved)
    assert _state_fingerprint(same) != _state_fingerprint(moved)
    assert _state_fingerprint(same) == _state_fingerprint(snapshot(0.2))


def test_state_fingerprint_tolerates_a_snapshot_without_joints() -> None:
    empty = {"sim": {"objects": {}, "eef_pos": []}}
    assert len(_state_fingerprint(empty)) == 16
    assert len(_scene_fingerprint(empty)) == 16
