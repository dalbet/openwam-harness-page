"""Task-specific completion and release evidence.

Native LIBERO goal is recorded as an event. It is not the user goal by
itself: a pick/place episode is complete only after the object relation still
holds in a newer observation after an explicit release. Drawer, push, and
turn-on tasks do not use release evidence.
"""

from __future__ import annotations

import math
from typing import Any

from common import geometry

DRAWER_OPEN_QPOS = -0.14
STOVE_ON_QPOS = 0.5
MIN_RELEASE_OPENING_M = 0.05
RELEASE_HEIGHT_TOL_M = 0.04


def drawer_level(name: str | None) -> str | None:
    text = str(name or "")
    for level in ("top", "middle", "bottom"):
        if text.endswith(f"_{level}_region"):
            return level
    return None


def drawer_owner(name: str | None) -> str | None:
    text = str(name or "")
    for level in ("top", "middle", "bottom"):
        suffix = f"_{level}_region"
        if text.endswith(suffix):
            return text[: -len(suffix)] or None
    return None


def drawer_joint(
    snapshot: dict[str, Any], level: str, target: str | None = None
) -> tuple[str | None, float | None]:
    """Resolve the joint owned by the drawer, not any joint with that level.

    The wine rack's top region is a fixed support site, while the cabinet owns
    ``wooden_cabinet_1_top_level``. A suffix-only lookup would read the
    cabinet joint for the rack; the owner prefix prevents that.
    """
    joints = snapshot.get("joints") or {}
    owner = drawer_owner(target)
    if owner:
        name = f"{owner}_{level}_level"
        if name in joints:
            try:
                return name, float(joints[name])
            except (TypeError, ValueError):
                return name, None
        return None, None
    for name, value in joints.items():
        if str(name).endswith(f"_{level}_level"):
            try:
                return str(name), float(value)
            except (TypeError, ValueError):
                return str(name), None
    return None, None


def stove_button_joint(snapshot: dict[str, Any]) -> tuple[str | None, float | None]:
    joints = snapshot.get("joints") or {}
    for name, value in joints.items():
        if "stove" in str(name) and str(name).endswith("_button"):
            try:
                return str(name), float(value)
            except (TypeError, ValueError):
                return str(name), None
    return None, None


def _obj_pos(snapshot: dict[str, Any], name: str) -> list[float] | None:
    row = ((snapshot.get("sim") or {}).get("objects") or {}).get(name) or {}
    pos = row.get("pos") if isinstance(row, dict) else None
    return list(pos) if isinstance(pos, list) and len(pos) >= 3 else None


def _site_pose(snapshot: dict[str, Any], name: str) -> tuple[list[float] | None, dict[str, Any]]:
    named = snapshot.get("named_poses") or {}
    xyz = (named.get("sites") or {}).get(name)
    shape = (named.get("site_geometry") or {}).get(name) or {}
    return (list(xyz) if isinstance(xyz, list) and len(xyz) >= 3 else None), shape


def relation(snapshot: dict[str, Any], object_name: str, target_name: str, kind: str) -> dict[str, Any]:
    return geometry.relation_check(snapshot, object_name, target_name, kind)


def placement_height(snapshot: dict[str, Any], object_name: str, target_name: str, relation_kind: str) -> dict[str, Any]:
    """Distance between the object bottom and the target support surface."""
    object_pos = _obj_pos(snapshot, object_name)
    site_pos, shape = _site_pose(snapshot, target_name)
    object_bottom_offset = geometry.object_bottom_z(object_name)
    if site_pos is not None and shape.get("size"):
        surface = float(site_pos[2]) + abs(float(shape["size"][2]))
        if relation_kind == "in":
            # Inside a drawer the relevant floor is the box lower bound,
            # which is not a single body surface. in_box is the gate.
            return {"ready": True, "kind": "inside_box", "surface_z": surface}
        if object_pos is None:
            return {"ready": False, "kind": "site", "reason": "object_pose_missing"}
        bottom_z = float(object_pos[2]) + object_bottom_offset
        delta = bottom_z - surface
        return {
            "ready": -0.02 <= delta <= RELEASE_HEIGHT_TOL_M,
            "kind": "site",
            "object_bottom_z": round(bottom_z, 4),
            "surface_z": round(surface, 4),
            "delta_m": round(delta, 4),
        }
    target_pos = _obj_pos(snapshot, target_name)
    if target_pos is None:
        return {"ready": False, "kind": "object", "reason": "target_pose_missing"}
    asset = geometry.asset_of(target_name) or {}
    if asset.get("kind_when_target") == "interior":
        plane = float(target_pos[2]) + float(asset["floor_z"])
        kind = "interior_floor"
    elif asset.get("kind_when_target") == "support_body":
        plane = float(target_pos[2]) + float(asset["support_z"])
        kind = "support_plane"
    else:
        return {"ready": True, "kind": "object_distance"}
    if object_pos is None:
        return {"ready": False, "kind": kind, "reason": "object_pose_missing"}
    delta = float(object_pos[2]) + object_bottom_offset - plane
    return {
        "ready": -0.02 <= delta <= RELEASE_HEIGHT_TOL_M,
        "kind": kind,
        "object_bottom_z": round(float(object_pos[2]) + object_bottom_offset, 4),
        "surface_z": round(plane, 4),
        "delta_m": round(delta, 4),
    }


def goal_state(snapshot: dict[str, Any], step: dict[str, Any], task: dict[str, Any], baseline: dict[str, Any] | None = None) -> dict[str, Any]:
    """One observation's state for the skill in ``step``.

    ``ready`` means the executor may stop the closed-loop phase. ``ok`` is
    the relation/state itself, independent of release.
    """
    skill = step.get("skill")
    native = bool(snapshot.get("native_goal"))
    out: dict[str, Any] = {
        "skill": skill,
        "native_goal": native,
        "observation_id": snapshot.get("observation_id"),
        "sim_action_count": snapshot.get("sim_action_count"),
        "scene_epoch": snapshot.get("scene_epoch"),
        "release_required": skill == "pick_place",
    }
    if skill == "open_drawer":
        target = step.get("target_n")
        level = drawer_level(target) or task.get("drawer_level")
        joint_name, qpos = drawer_joint(snapshot, str(level), str(target) if target else None)
        ok = qpos is not None and qpos < DRAWER_OPEN_QPOS
        out.update(
            {
                "ok": ok,
                "ready": ok,
                "predicate": "drawer_open",
                "joint": joint_name,
                "qpos": qpos,
                "threshold": DRAWER_OPEN_QPOS,
                "evidence_source": "sim_joint",
            }
        )
        return out
    if skill == "turn_on":
        joint_name, qpos = stove_button_joint(snapshot)
        ok = qpos is not None and qpos >= STOVE_ON_QPOS
        out.update(
            {
                "ok": ok,
                "ready": ok,
                "predicate": "stove_button",
                "joint": joint_name,
                "qpos": qpos,
                "threshold": STOVE_ON_QPOS,
                "evidence_source": "sim_joint",
            }
        )
        return out
    object_name = step.get("object_n")
    target_name = step.get("target_n")
    relation_kind = step.get("relation")
    if not isinstance(object_name, str) or not isinstance(target_name, str):
        out.update({"ok": False, "ready": False, "reason_code": "missing_binding"})
        return out
    check = relation(snapshot, object_name, target_name, str(relation_kind))
    height = placement_height(snapshot, object_name, target_name, str(relation_kind))
    pos = _obj_pos(snapshot, object_name)
    # ``baseline`` is a full state snapshot, so objects live under ``sim``.
    # Reading ``baseline["objects"]`` silently produced displacement=None and
    # made every push fail its completion check.
    baseline_pos = _obj_pos(baseline or {}, object_name)
    displacement = None
    if isinstance(pos, list) and isinstance(baseline_pos, list):
        displacement = math.dist(pos[:3], baseline_pos[:3])
    if skill == "push_object":
        ok = bool(check.get("ok")) and displacement is not None and displacement > 0.03
        out.update(
            {
                "ok": ok,
                "ready": bool(check.get("ok")),
                "predicate": "push_displacement+relation",
                "relation": check,
                "displacement_m": None if displacement is None else round(displacement, 4),
                "displacement_threshold_m": 0.03,
                "evidence_source": check.get("evidence_source"),
            }
        )
        return out
    ok = bool(check.get("ok")) and bool(height.get("ready"))
    # A pick/place into an open container may be completed by releasing just
    # above the opening: the policy positions the bowl over the drawer but does
    # not descend the last few centimetres. The release trigger is separate
    # from success: final verification (three fresh samples after release)
    # still requires the real in_box relation to hold.
    drop = _drop_release(snapshot, pos, check, str(relation_kind), str(skill))
    ready = ok or bool(drop.get("trigger"))
    out.update(
        {
            "ok": ok,
            "ready": ready,
            "predicate": "relation+release_height" if not drop.get("trigger") else "relation+drop_release",
            "relation": check,
            "height": height,
            "object_pose": pos,
            "drop_release": drop,
            "evidence_source": check.get("evidence_source"),
        }
    )
    return out


DROP_RELEASE_Z_TOL_M = 0.05
DROP_RELEASE_XY_TOL_M = 0.01


def _drop_release(
    snapshot: dict[str, Any],
    pos: list[float] | None,
    check: dict[str, Any],
    relation_kind: str,
    skill: str,
) -> dict[str, Any]:
    """Whether releasing now is a legitimate attempt to finish an ``in`` place."""
    if skill != "pick_place" or relation_kind != "in":
        return {"trigger": False, "reason": "not_in_place"}
    lower = check.get("lower")
    upper = check.get("upper")
    if not isinstance(lower, list) or not isinstance(upper, list) or not isinstance(pos, list):
        return {"trigger": False, "reason": "no_box_bounds"}
    horizontal = (
        float(lower[0]) - DROP_RELEASE_XY_TOL_M <= float(pos[0]) <= float(upper[0]) + DROP_RELEASE_XY_TOL_M
        and float(lower[1]) - DROP_RELEASE_XY_TOL_M <= float(pos[1]) <= float(upper[1]) + DROP_RELEASE_XY_TOL_M
    )
    above = float(lower[2]) - 0.02 <= float(pos[2]) <= float(upper[2]) + DROP_RELEASE_Z_TOL_M
    return {
        "trigger": bool(horizontal and above and not check.get("ok")),
        "horizontal_inside": bool(horizontal),
        "height_above_box": bool(above),
        "z_above_upper_m": round(float(pos[2]) - float(upper[2]), 4) if upper else None,
    }


def verification(
    snapshots: list[dict[str, Any]],
    step: dict[str, Any],
    task: dict[str, Any],
    baseline: dict[str, Any] | None,
    *,
    release: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Final user-goal verification from at least three fresh observations.

    A pick/place needs a release record and all samples after it: gripper
    open, relation true, new observation id, larger action count.
    """
    states = [goal_state(s, step, task, baseline) for s in snapshots]
    stable = len(states) >= 3 and all(s.get("ok") for s in states)
    required = 3
    out: dict[str, Any] = {
        "skill": step.get("skill"),
        "samples": len(states),
        "required_samples": required,
        "stable": stable,
        "per_sample_ok": [bool(s.get("ok")) for s in states],
        "last": states[-1] if states else None,
        "evidence_source": "sim_state",
        "release_required": step.get("skill") == "pick_place",
        "release": release,
    }
    if step.get("skill") != "pick_place":
        out["ok"] = stable
        out["reason_code"] = None if stable else "insufficient_stable_samples"
        return out
    release = release or {}
    opening = release.get("gripper_opening_m")
    release_count = release.get("sim_action_count")
    release_obs = release.get("observation_id")
    final = states[-1] if states else {}
    final_count = final.get("sim_action_count")
    final_obs = final.get("observation_id")
    final_opening = None
    final_snapshot = snapshots[-1] if snapshots else {}
    sim = final_snapshot.get("sim") or {}
    if sim.get("gripper_opening_m") is not None:
        final_opening = float(sim["gripper_opening_m"])
    release_ok = (
        bool(release.get("commanded"))
        and isinstance(opening, (int, float))
        and float(opening) >= MIN_RELEASE_OPENING_M
        and isinstance(release_count, int)
        and isinstance(final_count, int)
        and final_count > release_count
        and isinstance(release_obs, str)
        and isinstance(final_obs, str)
        and final_obs != release_obs
        and final_opening is not None
        and final_opening >= MIN_RELEASE_OPENING_M
    )
    out["release_evidence"] = {
        "status": "verified" if release_ok else "failed",
        "release_opening_m": opening,
        "final_opening_m": final_opening,
        "release_sim_action_count": release_count,
        "final_sim_action_count": final_count,
        "release_observation_id": release_obs,
        "final_observation_id": final_obs,
        "min_opening_m": MIN_RELEASE_OPENING_M,
    }
    out["ok"] = stable and release_ok
    out["reason_code"] = None if out["ok"] else (
        "release_evidence_failed" if not release_ok else "insufficient_stable_samples"
    )
    return out
