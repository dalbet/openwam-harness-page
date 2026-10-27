#!/usr/bin/env python
"""Whole-task plan contract: what the Agent may emit per selected backend.

Run with the engine interpreter:

    /home/host/page-dmh/openwam_harness/envs/rpent/bin/python tests/test_whole_task_plan.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ENGINE_ROOT = Path(__file__).resolve().parents[1]
if str(ENGINE_ROOT) not in sys.path:
    sys.path.insert(0, str(ENGINE_ROOT))

from adapters.env_client import EnvClient  # noqa: E402
from common import config  # noqa: E402
from common.plan_contract import validate_plan  # noqa: E402


def catalog_for(backend: str) -> dict:
    skills = config.skill_config()
    rows = list(skills.get("skills") or [])
    if backend == "openwam":
        rows = [row for row in rows if row["name"] == "whole_task"]
    else:
        rows = [row for row in rows if row["name"] != "whole_task"]
    return {**skills, "skills": rows}


def main() -> int:
    task = config.task_for(1)
    snapshot = EnvClient("http://127.0.0.1:14561").state_snapshot()
    instruction = task["instruction"]

    whole = catalog_for("openwam")
    assert [row["name"] for row in whole["skills"]] == ["whole_task"], whole["skills"]
    plan = {
        "interpretation": "the user wants the bowl on the stove",
        "steps": [
            {"skill": "whole_task", "object_n": None, "target_n": None, "relation": "task", "prompt": instruction}
        ],
        "success_condition": "native LIBERO goal",
    }
    ok, errors = validate_plan(plan, instruction=instruction, task=task, snapshot=snapshot, skills=whole)
    assert ok is not None and not errors, errors
    assert ok["steps"][0]["prompt"] == instruction
    print("ok  whole_task plan validates; prompt kept verbatim")

    mixed = {
        "interpretation": "x",
        "steps": [
            {"skill": "whole_task", "object_n": None, "target_n": None, "relation": "task", "prompt": instruction},
            {"skill": "whole_task", "object_n": None, "target_n": None, "relation": "task", "prompt": instruction},
        ],
    }
    ok, errors = validate_plan(mixed, instruction=instruction, task=task, snapshot=snapshot, skills=whole)
    assert ok is None and errors[0]["code"] == "whole_task_must_be_alone", errors
    print("ok  two whole_task steps rejected:", errors[0]["code"])

    bound = {
        "interpretation": "x",
        "steps": [
            {"skill": "whole_task", "object_n": "akita_black_bowl_1", "target_n": None, "relation": "task", "prompt": instruction}
        ],
    }
    ok, errors = validate_plan(bound, instruction=instruction, task=task, snapshot=snapshot, skills=whole)
    assert ok is None and errors[0]["code"] == "whole_task_takes_no_binding", errors
    print("ok  object binding on whole_task rejected:", errors[0]["code"])

    # A sub-skill catalog (Pi0.5) must reject whole_task entirely.
    sub = catalog_for("pi05")
    assert all(row["name"] != "whole_task" for row in sub["skills"])
    ok, errors = validate_plan(plan, instruction=instruction, task=task, snapshot=snapshot, skills=sub)
    assert ok is None and errors[0]["code"] == "skill_not_in_catalog", errors
    print("ok  whole_task rejected when the backend is pi05:", errors[0]["code"])

    # ...and the Pi0.5 five-skill catalog still validates its own plan shape.
    pick = {
        "interpretation": "x",
        "steps": [
            {"skill": "pick_place", "object_n": "akita_black_bowl_1", "target_n": "flat_stove_1_cook_region", "relation": "on", "prompt": "put the black bowl on the stove"}
        ],
    }
    ok, errors = validate_plan(pick, instruction=instruction, task=task, snapshot=snapshot, skills=sub)
    assert ok is not None and not errors, errors
    print("ok  Pi0.5 sub-skill plan still validates")
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
