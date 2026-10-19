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
    if status not in {"success", "failure", "unknown", "clarify"}:
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
        failed = call_failure_receipt(outcome)
        failed["observation_ids"] = _ids(observations)
        return failed
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
        return _localized(step, outcome, observations, objects)
    if check == "grasp_held":
        return _grasp(step, observations, objects, support_z, outcome)
    if check == "object_transported":
        return _transport(step, before, current, objects)
    if check == "object_placed":
        return _place(step, observations, objects, goal_relation)
    if check == "retreated":
        return receipt("success", "RETREATED", "sim_state", _ids(observations), {"applied_hold_offset": False})
    if check == "user_confirmed":
        return receipt("unknown", "NEEDS_CONFIRMATION", "manual", _ids(observations), {})
    return receipt("unknown", "NO_CHECKER", "tool_return", _ids(observations), {"check": check})


def call_failure_receipt(outcome: Any) -> dict[str, Any]:
    """A raised or refused tool call keeps its own text. It is not renamed tool_failed."""
    category = getattr(outcome, "category", None)
    if category not in {"service", "parameter", "tool_exception"}:
        if getattr(outcome, "error_class", None) == "service":
            category = "service"
        elif getattr(outcome, "error_class", None) == "plan":
            category = "parameter"
        else:
            category = "tool_exception"
    code = {"service": "TOOL_UNAVAILABLE", "parameter": "PARAMETER", "tool_exception": "TOOL_EXCEPTION"}[category]
    return receipt(
        "failure",
        code,
        "tool_return",
        [],
        {"error": outcome.error, "category": category},
    )


_MATCH_M = 0.05
_MARGIN_M = 0.03


def _localized(step, outcome, observations, objects) -> dict[str, Any]:
    ids = _ids(observations)
    if outcome.xyz is None:
        return receipt(
            "unknown",
            "TARGET_NOT_IDENTIFIED",
            "tool_return",
            ids,
            {
                "error": outcome.error,
                "category": "subtask_incomplete",
                "found": (outcome.raw or {}).get("found"),
                "score": (outcome.raw or {}).get("score"),
                "grounding_query": step.get("grounding_query") or (step.get("args") or {}).get("prompt"),
            },
        )
    bound = step.get("object_ref")
    spec = objects.get(bound) if isinstance(bound, str) else None
    queried = _bound_pose(bound, spec, observations[-1] if observations else None, scene_epoch=_epoch(observations))
    if queried["status"] != "ok":
        return receipt(
            "failure",
            queried["reason_code"],
            queried["evidence_source"],
            ids,
            {
                "error": queried["reason"],
                "category": "state_query",
                "object_ref": bound,
                "query_key": queried.get("query_key"),
                "xyz": outcome.xyz,
                "confirm": False,
            },
        )
    if spec.get("place_site"):
        return receipt(
            "success",
            "OBJECT_LOCALIZED",
            "mujoco_site",
            ids,
            {
                "xyz": outcome.xyz,
                "place_xyz": queried["xyz"],
                "object_ref": bound,
                "query_key": queried["query_key"],
                "sampled_at": queried.get("sampled_at"),
                "observation_id": queried.get("observation_id"),
                "evidence_source": "mujoco_site",
            },
        )
    bound_pos = queried["xyz"]
    sim_name = spec.get("sim_name") or bound
    bound_d = _distance(outcome.xyz, bound_pos)
    semantic = spec.get("semantic_type")
    others = []
    for ref, item in objects.items():
        if ref == bound or not isinstance(item, dict):
            continue
        if semantic and item.get("semantic_type") != semantic:
            continue
        other_name = item.get("sim_name") or ref
        pos = _object(observations[-1], other_name)
        if pos is None:
            continue
        others.append({"object_ref": ref, "distance_m": round(_distance(outcome.xyz, pos), 4)})
    nearest_other = min((item["distance_m"] for item in others), default=None)
    metrics = {
        "xyz": outcome.xyz,
        "object_ref": bound,
        "sim_name": sim_name,
        "bound_distance_m": round(bound_d, 4),
        "other_instances": others,
        "match_m": _MATCH_M,
    }
    confirmed = bound_d <= _MATCH_M and (nearest_other is None or bound_d + _MARGIN_M < nearest_other)
    if confirmed:
        return receipt("success", "OBJECT_LOCALIZED", "sim_state", ids, metrics)
    closer = nearest_other is not None and nearest_other + _MARGIN_M < bound_d
    crowded = nearest_other is not None and abs(bound_d - nearest_other) <= _MARGIN_M
    if closer or crowded:
        metrics["error"] = "定位结果无法区分多个同类实例，保持原绑定，不更换"
        metrics["category"] = "clarification"
        metrics["candidates"] = [bound] + [item["object_ref"] for item in others]
        return receipt("clarify", "visual_instance_ambiguous", "sim_state", ids, metrics)
    metrics["error"] = "定位结果不能确认是绑定实例"
    metrics["category"] = "subtask_incomplete"
    metrics["confirm"] = False
    return receipt("failure", "localization_unconfirmed", "sim_state", ids, metrics)


def _grasp(step, observations, objects, support_z, outcome) -> dict[str, Any]:
    spec = objects.get(step.get("object_ref") or "")
    if not isinstance(spec, dict):
        return receipt(
            "unknown",
            "TARGET_AMBIGUOUS",
            "sim_state",
            _ids(observations),
            {"missing": ["object_spec"], "error": "缺少 object_spec"},
        )
    sim_name = spec.get("sim_name")
    grasp_cfg = THRESHOLDS["grasp"]
    current = _object(observations[-1], sim_name)
    opening = _opening(observations[-1])
    eef = _eef(observations[-1])
    base = support_z.get(sim_name)
    contact = ((observations[-1] or {}).get("sim") or {}).get("contact")
    missing = []
    if current is None:
        missing.append("object_pos")
    if opening is None:
        missing.append("gripper_opening_m")
    if eef is None:
        missing.append("eef_pos")
    if base is None:
        missing.append("support_z")
    metrics = _grasp_metrics(observations, outcome, current, opening, eef, base, contact, grasp_cfg)
    if current is None:
        metrics["missing"] = missing
        metrics["error"] = "缺少 " + ", ".join(missing)
        return receipt("unknown", "TARGET_NOT_IDENTIFIED", "sim_state", _ids(observations), metrics)
    if missing:
        metrics["missing"] = missing
        metrics["error"] = "缺少 " + ", ".join(missing)
        return receipt("unknown", "INSUFFICIENT_EVIDENCE", "sim_state", _ids(observations), metrics)
    lifted = current[2] >= base + float(grasp_cfg["lift_above_support_m"])
    gap = _distance(current, eef)
    empty_closed = opening <= float(grasp_cfg["empty_closed_max_m"])
    left_behind = gap > float(grasp_cfg["left_behind_xy_m"])
    held_min = float(grasp_cfg["gripper_held_min_m"])
    held_max = float(grasp_cfg["gripper_held_max_m"])
    held_band = held_min <= opening <= held_max
    near = gap <= float(grasp_cfg["near_eef_m"])
    need = int(grasp_cfg["stable_samples"])
    limit = float(grasp_cfg["relative_stable_m"])
    lift_line = base + float(grasp_cfg["lift_above_support_m"])
    lifted_obs = []
    for obs in observations:
        pos = _object(obs, sim_name)
        if pos is not None and pos[2] >= lift_line:
            lifted_obs.append(obs)
    span = _relative_span(lifted_obs[-need:], sim_name, eef) if len(lifted_obs) >= need else None
    stable = span is not None and span <= limit
    unmet = []
    if not lifted:
        unmet.append(
            f"lift_m {metrics['lift_m']} < lift_above_support_m {grasp_cfg['lift_above_support_m']}"
        )
    metrics["gripper_band_note"] = (
        f"gripper_opening_m {opening} is |qpos[0]|+|qpos[1]| in metres, not a contact gap. "
        f"The recorded band is [{held_min}, {held_max}]. A thin pinch can sit below {held_min}."
    )
    metrics["origin_note"] = (
        "object_to_eef_m is the bowl origin to the eef origin. "
        "The akita bowl horizontal_radius_site is 0.035 m from the origin, so this is not contact distance."
    )
    if len(lifted_obs) < need:
        unmet.append(f"lifted samples {len(lifted_obs)} < stable_samples {need}")
    elif span is not None and span > limit:
        unmet.append(f"relative_span_m {round(span, 4)} > relative_stable_m {limit}")
    metrics["unmet"] = unmet
    metrics["lifted"] = lifted
    metrics["held_band"] = held_band
    metrics["stable"] = stable
    metrics["relative_span_m"] = None if span is None else round(span, 4)
    metrics["support_baseline"] = "same_object_initial_z"
    metrics["support_object"] = sim_name
    metrics["samples"] = [_sample_row(obs, sim_name) for obs in observations]
    supports = _nearby_supports(observations[-1], sim_name, current)
    metrics["nearby_supports"] = supports
    metrics["possible_external_support"] = bool(supports)
    if unmet:
        metrics["error"] = "；".join(unmet)
    if (empty_closed or left_behind or not near) and not lifted:
        return receipt("failure", "EMPTY_GRASP", "sim_state", _ids(observations), metrics)
    if lifted and stable and _contact_ok(contact) and not supports:
        return receipt("success", "GRASP_HELD", "sim_state", _ids(observations), metrics)
    if lifted and stable and supports:
        metrics["error"] = "对象高度接近其他表面，接触字段未提供，不能把支撑当成夹爪持有"
        return receipt("unknown", "INSUFFICIENT_EVIDENCE", "sim_state", _ids(observations), metrics)
    if len(lifted_obs) < need:
        return receipt("unknown", "NEED_STABILITY_SAMPLE", "sim_state", _ids(observations), metrics)
    if span is not None and span > limit:
        return receipt("failure", "GRASP_SLIP", "sim_state", _ids(observations), metrics)
    if not lifted:
        return receipt("failure", "EMPTY_GRASP", "sim_state", _ids(observations), metrics)
    return receipt("unknown", "INSUFFICIENT_EVIDENCE", "sim_state", _ids(observations), metrics)


def _contact_ok(contact) -> bool:
    """Missing contact is not a failed check. An explicit held=false is."""
    if not isinstance(contact, dict):
        return True
    if contact.get("held") is False:
        return False
    return True


def _grasp_metrics(observations, outcome, current, opening, eef, base, contact, grasp_cfg) -> dict[str, Any]:
    lift = None if current is None or base is None else round(current[2] - base, 4)
    gap = None if current is None or eef is None else round(_distance(current, eef), 4)
    return {
        "grasp_signal": outcome.grasp_signal,
        "object_pos": current,
        "eef_pos": eef,
        "gripper_opening_m": opening,
        "support_z": base,
        "lift_m": lift,
        "object_to_eef_m": gap,
        "contact": contact,
        "contact_status": "provided" if isinstance(contact, dict) else "not_provided",
        "contact_note": "环境提供了接触字段" if isinstance(contact, dict) else "环境没有提供接触字段，不能解释成已测到接触",
        "observation_ids": _ids(observations),
        "timestamps": [obs.get("timestamp") for obs in observations if isinstance(obs, dict)],
        "evidence_source": "sim_state",
        "lift_above_support_m": grasp_cfg["lift_above_support_m"],
        "window_samples": len(observations),
    }


def _transport(step, before_obs, after_obs, objects) -> dict[str, Any]:
    spec = objects.get(step.get("object_ref") or "")
    container = _target_sim(step, objects)
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
    origin = _distance(after, eef)
    rel_before = _relative(before_obs or {}, sim_name)
    rel_after = _relative(after_obs or {}, sim_name)
    delta = None if rel_before is None or rel_after is None else _distance(rel_before, rel_after)
    metrics = {
        "progress_m": round(progress, 4),
        "origin_to_eef_m": round(origin, 4),
        "relative_before": None if rel_before is None else [round(v, 4) for v in rel_before],
        "relative_after": None if rel_after is None else [round(v, 4) for v in rel_after],
        "relative_delta_m": None if delta is None else round(delta, 4),
        "hold_relative_m": cfg["hold_relative_m"],
        "evidence_mode": "sim_state_assisted",
        "hold_note": "hold_relative_m is the change in object-minus-eef, not the origin distance",
    }
    if delta is None:
        return receipt("unknown", "INSUFFICIENT_EVIDENCE", "sim_state_assisted", ids, metrics)
    if delta > float(cfg["hold_relative_m"]):
        return receipt("failure", "GRASP_LOST", "sim_state_assisted", ids, metrics)
    if (step.get("args") or {}).get("motion") != "vertical" and progress < float(cfg["min_progress_m"]):
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
    container = _target_sim(step, objects)
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


def _target_sim(step: dict, objects: dict[str, Any]) -> str | None:
    ref = step.get("target_ref")
    spec = objects.get(ref) if isinstance(ref, str) else None
    if isinstance(spec, dict) and spec.get("sim_name"):
        return spec["sim_name"]
    return _container(objects)


def _container(objects: dict[str, Any]) -> str | None:
    for spec in objects.values():
        if isinstance(spec, dict) and spec.get("role") == "container":
            return spec.get("sim_name")
    return None


def _epoch(observations: list) -> int | None:
    if not observations or not isinstance(observations[-1], dict):
        return None
    return observations[-1].get("scene_epoch")


def _bound_pose(instance_ref, spec, observation, *, scene_epoch) -> dict[str, Any]:
    from object_state import query_bound_state

    return query_bound_state(
        instance_ref if isinstance(instance_ref, str) else "",
        spec if isinstance(spec, dict) else None,
        observation if isinstance(observation, dict) else None,
        scene_epoch=scene_epoch,
        role="target" if isinstance(spec, dict) and spec.get("place_site") else "object",
    )


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


def _relative(obs: dict, sim_name: str) -> list[float] | None:
    obj = _object(obs, sim_name)
    center = _eef(obs)
    if obj is None or center is None:
        return None
    return [obj[i] - center[i] for i in range(3)]


def _sample_row(obs: dict, sim_name: str) -> dict[str, Any]:
    sim = (obs or {}).get("sim") or {}
    contact = sim.get("contact")
    rel = _relative(obs, sim_name)
    return {
        "timestamp": obs.get("timestamp"),
        "env_action_count": obs.get("env_step"),
        "object_pos": _object(obs, sim_name),
        "eef_pos": _eef(obs),
        "relative_pos": None if rel is None else [round(v, 4) for v in rel],
        "gripper_opening_m": _opening(obs),
        "contact": contact,
        "contact_status": "provided" if isinstance(contact, dict) else "not_provided",
        "evidence_source": sim.get("evidence_source") or "sim_state",
        "hold_step": obs.get("hold_step"),
    }


def _nearby_supports(obs: dict, sim_name: str, pos: list[float] | None) -> list[str]:
    """Other objects close in xy and z. Their height is not this object's baseline."""
    if pos is None:
        return []
    found = []
    for name, item in (((obs or {}).get("sim") or {}).get("objects") or {}).items():
        if name == sim_name or not isinstance(item, dict):
            continue
        other = item.get("pos")
        if not isinstance(other, list) or len(other) < 3:
            continue
        other = [float(other[0]), float(other[1]), float(other[2])]
        if abs(other[2] - pos[2]) <= 0.04 and _xy(other, pos) <= 0.08:
            found.append(name)
    return found


def _relative_span(observations, sim_name, eef) -> float | None:
    gaps = []
    for obs in observations:
        obj = _object(obs, sim_name)
        center = _eef(obs) or eef
        if obj is None or center is None:
            return None
        gaps.append(_distance(obj, center))
    if not gaps:
        return None
    return max(gaps) - min(gaps)


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
