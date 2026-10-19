"""World-frame placement and retreat.

support_origin and object_origin are object origins in the world frame.
eef_origin is the end-effector origin in the world frame.
hold_offset is object_origin minus eef_origin while the object is grasped.
A retreat starts at the release eef_origin and never subtracts hold_offset.
"""

from __future__ import annotations

import math
from typing import Any

from schema import THRESHOLDS, _in_workspace

KEEP_OUT_M = 0.05
RECORDED_RETREAT = {
    "episode_id": "2b2a7e4b1d0d",
    "step_id": "retreat_step2",
    "support_origin": [0.0466, -0.0276, 0.9097],
    "fixed_xy_offset": [0.04, 0.0],
    "z_offset_m": 0.12,
    "hold_offset": [0.0073086268597994525, 0.004951531002340609, -0.12636852116610497],
    "rejected_eef_origin": [0.07929137314020056, -0.03255153100234061, 1.156068521166105],
    "release_eef_origin": [0.08353841602408565, -0.06333128019510717, 1.0495447505881603],
    "released_object_origin": [0.0908470428838851, -0.05837974919276656, 0.9231762294220553],
}


def legacy_retreat_eef(support_origin, xy_offset, z_offset, hold_offset) -> list[float]:
    """Old path: object target minus hold_offset, including after the gripper opened."""
    object_target = [
        float(support_origin[0]) + float(xy_offset[0]),
        float(support_origin[1]) + float(xy_offset[1]),
        float(support_origin[2]) + float(z_offset),
    ]
    return [object_target[index] - float(hold_offset[index]) for index in range(3)]


def choose_object_target(support_origin, occupied, *, radius: float, keepout: float = KEEP_OUT_M) -> dict[str, Any]:
    """Pick an object_origin inside the support disc that clears occupied origins."""
    origin = [float(value) for value in support_origin[:3]]
    blockers = [[float(value) for value in item[:3]] for item in occupied if isinstance(item, (list, tuple)) and len(item) >= 2]
    candidates = [(0.0, 0.0)]
    for distance in (0.04, min(float(radius), 0.055)):
        if distance <= 0 or distance > float(radius) + 1e-9:
            continue
        for index in range(8):
            angle = index * math.pi / 4
            candidates.append((distance * math.cos(angle), distance * math.sin(angle)))
    rejected = []
    for dx, dy in candidates:
        if math.hypot(dx, dy) > float(radius) + 1e-9:
            rejected.append({"dx": dx, "dy": dy, "reason": "outside_support"})
            continue
        point = [origin[0] + dx, origin[1] + dy, origin[2]]
        blocked = _nearest(point, blockers)
        if blocked is not None and blocked < keepout:
            rejected.append({"object_origin": _round3(point), "reason": "occupied", "distance_m": round(blocked, 4)})
            continue
        return {"ok": True, "object_origin": _round3(point), "frame": "object_origin", "rejected": rejected}
    return {
        "ok": False,
        "code": "support_occupied",
        "reason": "支持区域没有通过占用检查的落点",
        "frame": "object_origin",
        "rejected": rejected,
    }


def plan_retreat(release_eef, released_object, others, workspace=None) -> dict[str, Any]:
    """A feasible eef_origin. An out-of-workspace candidate is rejected, not clamped."""
    start = [float(value) for value in release_eef[:3]]
    obj = [float(value) for value in released_object[:3]]
    blockers = [[float(value) for value in item[:3]] for item in others if isinstance(item, (list, tuple)) and len(item) >= 3]
    away = _unit_away(start, obj)
    proposals = [(0.0, 0.0, dz, "up") for dz in (0.04, 0.08, 0.12)]
    for distance in (0.04, 0.08):
        proposals.append((away[0] * distance, away[1] * distance, 0.0, "aside"))
        proposals.append((away[0] * distance, away[1] * distance, 0.04, "aside_up"))
    rejected = []
    start_gap = _distance(start, obj)
    for dx, dy, dz, kind in proposals:
        target = [start[0] + dx, start[1] + dy, start[2] + dz]
        if workspace is not None and not _inside(target, workspace):
            rejected.append({"eef_origin": target, "kind": kind, "reason": "outside_workspace"})
            continue
        if not _in_workspace(target):
            rejected.append({"eef_origin": target, "kind": kind, "reason": "outside_workspace"})
            continue
        gap = _distance(target, obj)
        if gap <= start_gap + 1e-4:
            rejected.append({"eef_origin": _round3(target), "kind": kind, "reason": "no_clearance"})
            continue
        crowded = _nearest(target, blockers)
        if crowded is not None and crowded < KEEP_OUT_M:
            rejected.append({"eef_origin": _round3(target), "kind": kind, "reason": "other_object"})
            continue
        return {
            "ok": True,
            "eef_origin": _round3(target),
            "frame": "eef_origin",
            "kind": kind,
            "applied_hold_offset": False,
            "rejected": rejected,
        }
    return {
        "ok": False,
        "code": "retreat_no_feasible_path",
        "reason": "退离没有留在工作空间内并躲开物体的路径",
        "frame": "eef_origin",
        "applied_hold_offset": False,
        "rejected": rejected,
    }


def resolve_motion_step(step: dict[str, Any], bindings: dict[str, list[float]], obs: dict | None, plan: dict[str, Any]) -> dict[str, Any]:
    """Resolve one move from the latest observation. Does not clamp into the workspace."""
    from schema import PlanError, resolve_args

    args = dict(step.get("args") or {})
    motion = args.get("motion") or "travel"
    if motion == "retreat":
        return _resolve_retreat(step, obs, plan)
    if args.get("placement") == "support_region":
        return _resolve_support(step, bindings, obs, plan)
    try:
        resolved = resolve_args(args, bindings)
    except PlanError as exc:
        return {"ok": False, "code": "plan", "reason": str(exc), "stage": motion}
    return {"ok": True, "args": resolved, "stage": motion, "applied_hold_offset": "hold_offset" in args or "carry_offset" in args}


def _resolve_retreat(step, obs, plan) -> dict[str, Any]:
    eef = _eef(obs)
    obj = _object_origin(obs, plan, step.get("object_ref") or (step.get("args") or {}).get("object_ref"))
    if eef is None or obj is None:
        return {"ok": False, "code": "retreat_state_missing", "reason": "退离前没有释放末端或物体的世界坐标", "stage": "retreat"}
    others = _other_origins(obs, plan, step.get("object_ref") or (step.get("args") or {}).get("object_ref"))
    decision = plan_retreat(eef, obj, others, THRESHOLDS["workspace_m"])
    decision["stage"] = "retreat"
    if not decision["ok"]:
        return decision
    gripper = (step.get("args") or {}).get("gripper", -1)
    return {
        "ok": True,
        "stage": "retreat",
        "applied_hold_offset": False,
        "retreat": decision,
        "args": {
            "xyz": decision["eef_origin"],
            "gripper": gripper,
            "object_ref": (step.get("args") or {}).get("object_ref"),
            "max_steps": int((step.get("args") or {}).get("max_steps") or 40),
        },
    }


def arrival(requested, actual, tolerance_m: float | None = None) -> dict[str, Any]:
    """The commanded xyz is not evidence that the arm arrived."""
    limit = float(THRESHOLDS["place"]["drift_m"] if tolerance_m is None else tolerance_m)
    if not isinstance(requested, list) or len(requested) < 3:
        return {"ok": False, "code": "pose_not_provided", "reason": "没有要求到达的末端位置", "requested": requested, "actual": actual}
    if not isinstance(actual, list) or len(actual) < 3:
        return {"ok": False, "code": "pose_not_provided", "reason": "动作后没有读到末端位姿", "requested": requested, "actual": actual, "tolerance_m": limit}
    gap = _distance(requested, actual)
    return {
        "ok": gap <= limit,
        "code": "" if gap <= limit else "MOTION_NOT_ARRIVED",
        "reason": "" if gap <= limit else f"末端距离目标 {round(gap, 4)} m，大于 {limit} m",
        "requested": [round(float(value), 4) for value in requested[:3]],
        "actual": [round(float(value), 4) for value in actual[:3]],
        "gap_m": round(gap, 4),
        "tolerance_m": limit,
        "evidence_source": "sim_state",
    }


def preplace_gate(step, obs, plan, bindings) -> dict[str, Any]:
    """Conditions before descent or release. Does not require the object to be released."""
    args = dict(step.get("args") or {})
    phase = "release" if step.get("skill") == "set_gripper" else "descend"
    ref = step.get("object_ref") or args.get("object_ref")
    unmet = []
    sources = {}
    obj = _object_origin(obs, plan, ref)
    eef = _eef(obs)
    sources["object"] = "sim_state" if obj is not None else "pose_not_provided"
    sources["eef"] = "sim_state" if eef is not None else "pose_not_provided"
    if obj is None:
        unmet.append("object_pos 缺失")
    if eef is None:
        unmet.append("eef_pos 缺失")
    surface, surface_source, surface_key = _support_surface(step, bindings, obs, plan)
    sources["target_region"] = surface_source or "pose_not_provided"
    if surface is None:
        unmet.append("target_region_pose 缺失")
    baseline = bindings.get(f"hold:{ref}") if isinstance(ref, str) else None
    if obj is not None and eef is not None and isinstance(baseline, list) and len(baseline) >= 3:
        delta = _distance([obj[i] - eef[i] for i in range(3)], baseline)
        if delta > float(THRESHOLDS["transport"]["hold_relative_m"]):
            unmet.append(f"relative_delta_m {round(delta, 4)} > hold_relative_m {THRESHOLDS['transport']['hold_relative_m']}")
    elif not isinstance(baseline, list):
        unmet.append("grasp_baseline 缺失")
    if surface is not None and obj is not None and phase == "release":
        release_z = float(surface[2]) + 0.03
        if abs(float(obj[2]) - release_z) > float(THRESHOLDS["place"]["support_z_tolerance_m"]) and abs(float(obj[2]) - float(surface[2])) > float(THRESHOLDS["place"]["support_z_tolerance_m"]):
            unmet.append(f"object_z {round(float(obj[2]), 4)} 仍不在释放高度 {round(release_z, 4)} 或区域表面 {round(float(surface[2]), 4)}")
    geometry = None
    if not unmet and phase == "descend":
        geometry = _resolve_support(step, bindings, obs, plan)
        if not geometry.get("ok"):
            unmet.append(geometry.get("reason") or geometry.get("code") or "落点不可用")
    if unmet:
        summary = "放置前暂停：" + unmet[0]
        return {
            "ok": False,
            "code": "preplace_blocked",
            "reason": summary,
            "summary": summary,
            "unmet": unmet,
            "evidence_sources": sources,
            "surface_key": surface_key,
            "released_required": False,
            "placed_required": False,
            "stage": "preplace_ready",
        }
    return {
        "ok": True,
        "stage": "preplace_ready",
        "evidence_sources": sources,
        "surface": [round(float(value), 4) for value in surface[:3]],
        "geometry": geometry,
        "released_required": False,
        "placed_required": False,
    }


def region_pose_ready(step, obs, plan) -> bool:
    """True when this move does not need a site, or the current observation already has it."""
    spec = (plan.get("objects") or {}).get(step.get("target_ref") or "")
    if not isinstance(spec, dict) or not spec.get("place_site"):
        return True
    surface, _source, _key = _support_surface(step, {}, obs, plan)
    return surface is not None


def _support_surface(step, bindings, obs, plan):
    args = dict(step.get("args") or {})
    source = str(args.get("xyz_from") or "")
    target_ref = step.get("target_ref")
    spec = (plan.get("objects") or {}).get(target_ref) if isinstance(target_ref, str) else None
    site_name = spec.get("place_site") if isinstance(spec, dict) else None
    sites = ((obs or {}).get("named_poses") or {}).get("sites") if isinstance(obs, dict) else None
    if isinstance(site_name, str) and site_name:
        named = (obs or {}).get("named_poses") if isinstance(obs, dict) else None
        if isinstance(named, dict) and named.get("error"):
            return None, "state_service_error", site_name
        site = sites.get(site_name) if isinstance(sites, dict) else None
        if isinstance(site, list) and len(site) >= 3:
            return [float(site[0]), float(site[1]), float(site[2])], "mujoco_site", site_name
        return None, "pose_not_provided", site_name
    surface = bindings.get(source) if source else None
    if isinstance(surface, list) and len(surface) >= 3:
        return [float(surface[0]), float(surface[1]), float(surface[2])], "segment_binding", source
    sim_name = spec.get("sim_name") if isinstance(spec, dict) else None
    item = (((obs or {}).get("sim") or {}).get("objects") or {}).get(sim_name) if sim_name else None
    pos = item.get("pos") if isinstance(item, dict) else None
    if isinstance(pos, list) and len(pos) >= 3:
        return [float(pos[0]), float(pos[1]), float(pos[2])], "sim_state", sim_name
    return None, "pose_not_provided", source or sim_name


def _resolve_support(step, bindings, obs, plan) -> dict[str, Any]:
    from schema import PlanError

    args = dict(step.get("args") or {})
    source = str(args.get("xyz_from") or "")
    surface, surface_source, surface_key = _support_surface(step, bindings, obs, plan)
    if surface is None:
        code = "state_service_error" if surface_source == "state_service_error" else "pose_not_provided"
        return {
            "ok": False,
            "code": code,
            "reason": f"没有目标区域的世界坐标 {surface_key or source}",
            "stage": "place_motion",
            "evidence_source": surface_source,
            "region_key": surface_key,
        }
    z_offset = float(args.get("z_offset") or 0.0)
    low, high = THRESHOLDS["approach_z_offset_m"]
    if not low <= z_offset <= high:
        return {"ok": False, "code": "range", "reason": f"z_offset {z_offset} 不在 [{low}, {high}]", "stage": "place_motion"}
    ref = args.get("hold_offset") or args.get("carry_offset") or step.get("object_ref")
    obj = _object_origin(obs, plan, ref)
    eef = _eef(obs)
    if obj is None or eef is None:
        return {"ok": False, "code": "missing_hold_offset", "reason": "搬运前没有物体原点和末端的世界坐标", "stage": "place_motion"}
    hold = [obj[index] - eef[index] for index in range(3)]
    relation = step.get("relation") or args.get("relation") or "on"
    radius = float(THRESHOLDS["place"]["in_radius_m"] if relation == "in" else THRESHOLDS["place"]["on_radius_m"])
    chosen = choose_object_target(surface, _other_origins(obs, plan, ref), radius=radius)
    if not chosen["ok"]:
        chosen["stage"] = "place_motion"
        return chosen
    object_target = [
        chosen["object_origin"][0],
        chosen["object_origin"][1],
        float(surface[2]) + z_offset,
    ]
    eef_target = [object_target[index] - hold[index] for index in range(3)]
    if not _in_workspace(eef_target):
        return {
            "ok": False,
            "code": "workspace",
            "reason": f"末端目标不在工作空间内：{eef_target}",
            "stage": "place_motion",
            "eef_origin": eef_target,
            "object_origin": object_target,
            "hold_offset": hold,
            "applied_hold_offset": True,
        }
    try:
        resolved = {
            "xyz": [round(value, 4) for value in eef_target],
            "gripper": args.get("gripper"),
            "object_ref": args.get("object_ref") or ref,
            "max_steps": args.get("max_steps", 80),
        }
        if args.get("max_steps") is None:
            resolved.pop("max_steps")
    except PlanError as exc:
        return {"ok": False, "code": "plan", "reason": str(exc), "stage": "place_motion"}
    return {
        "ok": True,
        "stage": "place_motion",
        "applied_hold_offset": True,
        "object_origin": [round(value, 4) for value in object_target],
        "hold_offset": [round(value, 4) for value in hold],
        "support_origin": [round(float(value), 4) for value in surface[:3]],
        "region_source": surface_source,
        "region_key": surface_key,
        "object_z_rule": "region_z + z_offset once; hold_offset is applied once from the object origin to the eef",
        "args": {key: value for key, value in resolved.items() if value is not None or key == "gripper"},
    }


def _object_origin(obs, plan, ref) -> list[float] | None:
    spec = ((plan or {}).get("objects") or {}).get(ref) or {}
    sim_name = spec.get("sim_name") or ref
    item = ((((obs or {}).get("sim") or {}).get("objects") or {}).get(sim_name) or {})
    pos = item.get("pos") if isinstance(item, dict) else None
    if isinstance(pos, list) and len(pos) >= 3:
        return [float(pos[0]), float(pos[1]), float(pos[2])]
    return None


def _other_origins(obs, plan, ref) -> list[list[float]]:
    spec = ((plan or {}).get("objects") or {}).get(ref) or {}
    skip = spec.get("sim_name") or ref
    found = []
    for name, item in ((((obs or {}).get("sim") or {}).get("objects") or {}).items()):
        if name == skip or not isinstance(item, dict):
            continue
        pos = item.get("pos")
        if isinstance(pos, list) and len(pos) >= 3:
            found.append([float(pos[0]), float(pos[1]), float(pos[2])])
    return found


def _eef(obs) -> list[float] | None:
    eef = ((obs or {}).get("sim") or {}).get("eef_pos")
    if isinstance(eef, list) and len(eef) >= 3:
        return [float(eef[0]), float(eef[1]), float(eef[2])]
    return None


def _inside(point, workspace) -> bool:
    names = ("x", "y", "z")
    return all(workspace[name][0] <= float(value) <= workspace[name][1] for name, value in zip(names, point))


def _distance(left, right) -> float:
    return math.sqrt(sum((float(left[index]) - float(right[index])) ** 2 for index in range(3)))


def _nearest(point, others) -> float | None:
    if not others:
        return None
    return min(_distance(point, item) for item in others)


def _unit_away(start, obj) -> list[float]:
    dx = start[0] - obj[0]
    dy = start[1] - obj[1]
    norm = math.hypot(dx, dy)
    if norm < 1e-6:
        return [1.0, 0.0]
    return [dx / norm, dy / norm]


def _round3(point) -> list[float]:
    return [round(float(value), 4) for value in point[:3]]
