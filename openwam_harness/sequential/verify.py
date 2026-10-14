"""Subtask checks. Tool success, grasp_signal, and LIBERO termination stay separate.

Pose checks are sim_state_assisted. They are not described as vision-only.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
THRESHOLDS = json.loads((ROOT / "thresholds.json").read_text(encoding="utf-8"))
VALIDATOR_VERSION = "sequential-verify-1"


def receipt(
    status: str,
    reason_code: str,
    evidence_source: str,
    observation_ids: list[str],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    if status not in {"success", "failure", "unknown"}:
        raise ValueError(status)
    return {
        "status": status,
        "reason_code": reason_code,
        "evidence_source": evidence_source,
        "observation_ids": observation_ids,
        "checked_at": time.time(),
        "metrics": metrics,
        "validator_version": VALIDATOR_VERSION,
    }


def observation_ok(obs: dict | None, *, scene_epoch: int, min_step: int | None) -> bool:
    if not isinstance(obs, dict):
        return False
    if obs.get("scene_epoch") != scene_epoch:
        return False
    step = obs.get("env_step")
    if min_step is not None and (not isinstance(step, int) or step < min_step):
        return False
    cameras = obs.get("cameras") or {}
    if "agentview" not in cameras or "wrist" not in cameras:
        return False
    return True


def verify_step(
    step: dict[str, Any],
    outcome: Any,
    observations: list[dict],
    *,
    scene_epoch: int,
    min_step: int | None,
    objects: dict[str, Any],
    support_z: dict[str, float],
    goal_relation: str,
    before: dict | None = None,
) -> dict[str, Any]:
    check = step.get("completion_check")
    if not outcome.executed:
        return receipt("failure", "tool_failed", "tool_return", _ids(observations), {"error": outcome.error})
    current = observations[-1] if observations else None
    if not observation_ok(current, scene_epoch=scene_epoch, min_step=min_step):
        return receipt(
            "unknown",
            "STALE_OBSERVATION",
            "sim_state_assisted",
            _ids(observations),
            {"scene_epoch": scene_epoch, "min_step": min_step},
        )
    if check == "observe_ready":
        return receipt("success", "OBSERVE_READY", "sim_state_assisted", _ids(observations), {})
    if check == "object_localized":
        return _localized(step, outcome, observations)
    if check == "grasp_held":
        return _grasp(step, observations, objects, support_z, outcome)
    if check == "object_transported":
        return _transport(step, before, current, objects)
    if check == "object_placed":
        return _place(step, observations, objects, goal_relation)
    if check == "user_confirmed":
        return receipt("unknown", "NEEDS_CONFIRMATION", "manual", _ids(observations), {})
    return receipt("unknown", "NO_CHECKER", "tool_return", _ids(observations), {"check": check})


def _localized(step, outcome, observations) -> dict[str, Any]:
    if outcome.xyz is None:
        return receipt("unknown", "TARGET_NOT_IDENTIFIED", "tool_return", _ids(observations), {})
    return receipt(
        "success",
        "OBJECT_LOCALIZED",
        "tool_return",
        _ids(observations),
        {"xyz": outcome.xyz},
    )


def _grasp(step, observations, objects, support_z, outcome) -> dict[str, Any]:
    spec = objects.get(step.get("object_ref") or "")
    if not isinstance(spec, dict):
        return receipt("unknown", "TARGET_AMBIGUOUS", "sim_state_assisted", _ids(observations), {})
    sim_name = spec.get("sim_name")
    grasp_cfg = THRESHOLDS["grasp"]
    current = _object(observations[-1], sim_name)
    if current is None:
        return receipt(
            "unknown",
            "TARGET_NOT_IDENTIFIED",
            "sim_state_assisted",
            _ids(observations),
            {"sim_name": sim_name, "grasp_signal": outcome.grasp_signal},
        )
    opening = _opening(observations[-1])
    eef = _eef(observations[-1])
    base = support_z.get(sim_name)
    if base is None or opening is None or eef is None:
        return receipt(
            "unknown",
            "INSUFFICIENT_EVIDENCE",
            "sim_state_assisted",
            _ids(observations),
            {"grasp_signal": outcome.grasp_signal},
        )
    lifted = current[2] >= base + float(grasp_cfg["lift_above_support_m"])
    gap = _distance(current, eef)
    empty_closed = opening <= float(grasp_cfg["empty_closed_max_m"])
    left_behind = gap > float(grasp_cfg["left_behind_xy_m"])
    metrics = {
        "grasp_signal": outcome.grasp_signal,
        "lift_m": round(current[2] - base, 4),
        "gripper_opening_m": opening,
        "object_to_eef_m": round(gap, 4),
        "evidence_mode": "sim_state_assisted",
    }
    if (empty_closed or left_behind) and not lifted:
        return receipt("failure", "EMPTY_GRASP", "sim_state_assisted", _ids(observations), metrics)
    held_band = float(grasp_cfg["gripper_held_min_m"]) <= opening <= float(grasp_cfg["gripper_held_max_m"])
    stable = _stable(observations, sim_name, eef, int(grasp_cfg["stable_samples"]), float(grasp_cfg["relative_stable_m"]))
    if lifted and held_band and stable and gap <= float(grasp_cfg["near_eef_m"]):
        return receipt("success", "GRASP_HELD", "sim_state_assisted", _ids(observations), metrics)
    if not stable:
        return receipt("unknown", "NEED_STABILITY_SAMPLE", "sim_state_assisted", _ids(observations), metrics)
    if not lifted:
        return receipt("failure", "EMPTY_GRASP", "sim_state_assisted", _ids(observations), metrics)
    return receipt("unknown", "INSUFFICIENT_EVIDENCE", "sim_state_assisted", _ids(observations), metrics)


def _transport(step, before_obs, after_obs, objects) -> dict[str, Any]:
    spec = objects.get(step.get("object_ref") or "")
    container = _container(objects)
    ids = _ids([obs for obs in (before_obs, after_obs) if isinstance(obs, dict)])
    if not isinstance(spec, dict) or container is None:
        return receipt("unknown", "TARGET_AMBIGUOUS", "sim_state_assisted", ids, {})
    sim_name = spec["sim_name"]
    before = _object(before_obs or {}, sim_name)
    after = _object(after_obs or {}, sim_name)
    target = _object(after_obs or {}, container)
    eef = _eef(after_obs or {})
    if before is None or after is None or target is None or eef is None:
        return receipt("unknown", "INSUFFICIENT_EVIDENCE", "sim_state_assisted", ids, {})
    cfg = THRESHOLDS["transport"]
    progress = _xy(before, target) - _xy(after, target)
    hold = _distance(after, eef)
    metrics = {"progress_m": round(progress, 4), "hold_m": round(hold, 4), "evidence_mode": "sim_state_assisted"}
    if hold > float(cfg["hold_relative_m"]):
        return receipt("failure", "GRASP_LOST", "sim_state_assisted", ids, metrics)
    if progress < float(cfg["min_progress_m"]):
        return receipt("failure", "NO_PROGRESS", "sim_state_assisted", ids, metrics)
    return receipt("success", "TRANSPORTED", "sim_state_assisted", ids, metrics)


def _place(step, observations, objects, goal_relation: str) -> dict[str, Any]:
    relation = step.get("relation")
    if relation != goal_relation:
        return receipt(
            "failure",
            "RELATION_MISMATCH",
            "sim_state_assisted",
            _ids(observations),
            {"relation": relation, "goal_relation": goal_relation},
        )
    spec = objects.get(step.get("object_ref") or "")
    container = _container(objects)
    if not isinstance(spec, dict) or container is None:
        return receipt("unknown", "TARGET_AMBIGUOUS", "sim_state_assisted", _ids(observations), {})
    obj = _object(observations[-1], spec["sim_name"])
    target = _object(observations[-1], container)
    opening = _opening(observations[-1])
    if obj is None or target is None or opening is None:
        return receipt("unknown", "PLACEMENT_UNVERIFIED", "sim_state_assisted", _ids(observations), {})
    cfg = THRESHOLDS["place"]
    radius = {"in": cfg["in_radius_m"], "on": cfg["on_radius_m"], "inside_drawer": cfg["inside_drawer_radius_m"]}[relation]
    xy = _xy(obj, target)
    supported = abs(obj[2] - target[2]) <= float(cfg["support_z_tolerance_m"])
    released = opening >= float(cfg["gripper_open_min_m"])
    stable = _drift(observations, spec["sim_name"], int(cfg["stable_samples"]), float(cfg["drift_m"]))
    metrics = {
        "relation": relation,
        "xy_m": round(xy, 4),
        "z_delta_m": round(obj[2] - target[2], 4),
        "gripper_opening_m": opening,
        "evidence_mode": "sim_state_assisted",
    }
    if not stable:
        return receipt("unknown", "NEED_STABILITY_SAMPLE", "sim_state_assisted", _ids(observations), metrics)
    if xy <= float(radius) and supported and released:
        return receipt("success", "PLACED", "sim_state_assisted", _ids(observations), metrics)
    if not released:
        return receipt("unknown", "PLACEMENT_UNVERIFIED", "sim_state_assisted", _ids(observations), metrics)
    return receipt("failure", "WRONG_PLACEMENT", "sim_state_assisted", _ids(observations), metrics)


def native_success(terminated: bool, truncated: bool) -> bool:
    """LIBERO official success is terminated. Truncation is not success."""
    return bool(terminated) and not bool(truncated)


def _container(objects: dict[str, Any]) -> str | None:
    for spec in objects.values():
        if isinstance(spec, dict) and spec.get("role") == "container":
            return spec.get("sim_name")
    return None


def _object(obs: dict, sim_name: str) -> list[float] | None:
    sim = (obs or {}).get("sim") or {}
    item = (sim.get("objects") or {}).get(sim_name)
    if not isinstance(item, dict) or not item.get("identified"):
        return None
    pos = item.get("pos")
    if not isinstance(pos, list) or len(pos) < 3:
        return None
    return [float(pos[0]), float(pos[1]), float(pos[2])]


def _eef(obs: dict) -> list[float] | None:
    pos = ((obs or {}).get("sim") or {}).get("eef_pos")
    if not isinstance(pos, list) or len(pos) < 3:
        return None
    return [float(pos[0]), float(pos[1]), float(pos[2])]


def _opening(obs: dict) -> float | None:
    value = ((obs or {}).get("sim") or {}).get("gripper_opening_m")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _distance(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


def _xy(a: list[float], b: list[float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _stable(observations, sim_name, eef, samples: int, limit: float) -> bool:
    recent = observations[-samples:]
    if len(recent) < samples:
        return False
    gaps = []
    for obs in recent:
        obj = _object(obs, sim_name)
        center = _eef(obs) or eef
        if obj is None or center is None:
            return False
        gaps.append(_distance(obj, center))
    return max(gaps) - min(gaps) <= limit


def _drift(observations, sim_name, samples: int, limit: float) -> bool:
    recent = observations[-samples:]
    if len(recent) < samples:
        return False
    points = [_object(obs, sim_name) for obs in recent]
    if any(point is None for point in points):
        return False
    return max(_distance(points[0], point) for point in points) <= limit


def _ids(observations: list[dict]) -> list[str]:
    return [str(obs.get("observation_id")) for obs in observations if obs.get("observation_id")]
