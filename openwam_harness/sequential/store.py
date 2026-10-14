"""Observation identity, localization cache, and task cards.

Cards store semantic refs and skill order. They do not copy episode coordinates.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
CARDS = ROOT / "records" / "task_cards"
FAILURES = ROOT / "records" / "failure_experience"


class LocalizationCache:
    def __init__(self) -> None:
        self.rows: dict[str, dict[str, Any]] = {}

    def put(
        self,
        object_ref: str,
        *,
        scene_epoch: int,
        camera: str,
        observation_id: str,
        xyz: list[float],
        env_step: int | None,
    ) -> None:
        self.rows[object_ref] = {
            "object_ref": object_ref,
            "scene_epoch": scene_epoch,
            "camera": camera,
            "observation_id": observation_id,
            "xyz": [float(value) for value in xyz[:3]],
            "env_step": env_step,
            "valid": True,
        }

    def get(self, object_ref: str, scene_epoch: int, camera: str | None = None) -> dict[str, Any] | None:
        row = self.rows.get(object_ref)
        if not row or not row["valid"] or row["scene_epoch"] != scene_epoch:
            return None
        if camera is not None and row["camera"] != camera:
            return None
        return dict(row)

    def invalidate(self, reason: str, *, object_ref: str | None = None, scene_epoch: int | None = None) -> None:
        for key, row in self.rows.items():
            if object_ref is not None and key != object_ref:
                continue
            if scene_epoch is not None and row["scene_epoch"] != scene_epoch:
                continue
            row["valid"] = False
            row["invalid_reason"] = reason


def new_observation(backend_obs: dict | None, *, scene_epoch: int) -> dict[str, Any]:
    obs = dict(backend_obs or {})
    obs.setdefault("observation_id", uuid.uuid4().hex[:12])
    obs["scene_epoch"] = scene_epoch
    obs.setdefault("timestamp", time.time())
    obs["llm"] = False
    return obs


def candidate_card(plan: dict[str, Any], *, code_version: str, model: str = "Qwen3.5-9B") -> dict[str, Any]:
    return {
        "status": "candidate",
        "scene_id": plan.get("scene_id"),
        "goal": plan.get("goal"),
        "goal_relation": plan.get("goal_relation"),
        "objects": {
            ref: {
                "sim_name": spec.get("sim_name"),
                "role": spec.get("role"),
                "phrases": spec.get("phrases"),
            }
            for ref, spec in (plan.get("objects") or {}).items()
            if isinstance(spec, dict)
        },
        "skill_order": [step.get("skill") for step in plan.get("steps") or []],
        "entry_conditions": ["scene_loaded", "no_reset_between_subtasks"],
        "validators": {
            "grasp": "grasp_held",
            "transport": "object_transported",
            "place": "object_placed",
            "mode": "sim_state_assisted",
        },
        "recovery": {"max_attempts": 2, "events": sorted(["EMPTY_GRASP", "GRASP_LOST", "NO_PROGRESS", "PLACEMENT_UNVERIFIED", "TARGET_AMBIGUOUS", "SERVICE_ERROR"])},
        "model": model,
        "code_version": code_version,
        "source_episode": None,
        "verified_scope": None,
        "plan_source": "manual",
    }


def write_candidate_cards(plans: dict[str, Any], code_version: str) -> None:
    CARDS.mkdir(parents=True, exist_ok=True)
    for scene_id, plan in plans.items():
        card = candidate_card(plan, code_version=code_version)
        (CARDS / f"{scene_id}.json").write_text(
            json.dumps(card, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def save_failure(episode_id: str, payload: dict[str, Any]) -> Path:
    FAILURES.mkdir(parents=True, exist_ok=True)
    path = FAILURES / f"{episode_id}.json"
    body = dict(payload)
    body["status"] = "failure_experience"
    path.write_text(json.dumps(body, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path
