"""Geometry for the official LIBERO-Goal relations.

Copied from ``openwam_harness/sequential/transfer_geometry.py`` and
``sequential/regions.py`` (site ``under`` / ``in_box``). Values come from the
installed LIBERO asset XML and from ``ObjectState.check_ontop`` /
``SiteObject.in_box``; they are not tuned per run.
"""

from __future__ import annotations

import math
import re
from typing import Any


ASSETS = {
    "akita_black_bowl": {
        "source": "stable_scanned_objects/akita_black_bowl/akita_black_bowl.xml",
        "kind_when_target": "interior",
        "bottom_z": -0.012,
        "floor_z": -0.012,
        "rim_z": 0.04,
        "inner_xy_m": 0.025,
    },
    "plate": {
        "source": "stable_scanned_objects/plate/plate.xml",
        "kind_when_target": "support_body",
        "bottom_z": -0.02744,
        "support_z": 0.00535,
        "xy_limit_m": 0.03,
    },
    "cream_cheese": {
        "source": "stable_hope_objects/cream_cheese/cream_cheese.xml",
        "bottom_z": -0.040608,
    },
    "wine_bottle": {
        "source": "turbosquid_objects/wine_bottle/wine_bottle.xml",
        "bottom_z": 0.00111,
    },
}

OFFICIAL_ON_XY_M = 0.03
Z_TOLERANCE_M = 0.02


def asset_key(name: str | None) -> str:
    return re.sub(r"_\d+$", "", name or "")


def asset_of(name: str | None) -> dict[str, Any] | None:
    row = ASSETS.get(asset_key(name))
    return dict(row) if row else None


def object_bottom_z(name: str | None) -> float:
    row = asset_of(name) or {}
    return float(row.get("bottom_z", 0.0))


def _matvec(mat: list[float], vec: list[float]) -> list[float]:
    return [sum(float(mat[row * 3 + col]) * float(vec[col]) for col in range(3)) for row in range(3)]


def site_under(object_pos: list[float], site_pos: list[float], site_mat: list[float], site_size: list[float], *, other_height: float = 0.10) -> dict[str, Any]:
    delta = _matvec(site_mat, [object_pos[i] - site_pos[i] for i in range(3)])
    size = [float(site_size[0]), float(site_size[1]), float(site_size[2])]
    inside_xy = abs(delta[0]) < size[0] and abs(delta[1]) < size[1]
    supported = size[2] - 0.005 < delta[2] < size[2] + other_height
    return {
        "ok": inside_xy and supported,
        "predicate": "SiteObject.under",
        "delta_site": [round(v, 4) for v in delta],
        "inside_xy": inside_xy,
        "supported": supported,
        "site_size": size,
    }


def site_in_box(object_pos: list[float], site_pos: list[float], site_mat: list[float], site_size: list[float]) -> dict[str, Any]:
    """``SiteObject.in_box``: axis-aligned bounds from ``|mat @ size|``, z lower -= 0.01."""
    total = [abs(sum(float(site_mat[row * 3 + col]) * float(site_size[col]) for col in range(3))) for row in range(3)]
    upper = [float(site_pos[i]) + total[i] for i in range(3)]
    lower = [float(site_pos[i]) - total[i] for i in range(3)]
    lower[2] -= 0.01
    inside = all(lower[i] < float(object_pos[i]) < upper[i] for i in range(3))
    return {
        "ok": inside,
        "predicate": "SiteObject.in_box",
        "lower": [round(v, 4) for v in lower],
        "upper": [round(v, 4) for v in upper],
        "delta_site": [round(float(object_pos[i]) - float(site_pos[i]), 4) for i in range(3)],
    }


def object_relation(object_pos: list[float] | None, target_pos: list[float] | None, object_name: str, target_name: str, relation: str) -> dict[str, Any] | None:
    """Bowl-interior and plate-support checks. A site target is handled elsewhere."""
    target = asset_of(target_name)
    carried = asset_of(object_name) or {}
    if not target or target.get("kind_when_target") not in {"interior", "support_body"}:
        return None
    if not isinstance(object_pos, list) or not isinstance(target_pos, list) or len(object_pos) < 3 or len(target_pos) < 3:
        return {"ok": False, "predicate": "pose_not_provided"}
    xy = math.hypot(float(object_pos[0]) - float(target_pos[0]), float(object_pos[1]) - float(target_pos[1]))
    bottom = float(object_pos[2]) + float(carried.get("bottom_z", 0.0))
    if target["kind_when_target"] == "interior":
        floor = float(target_pos[2]) + float(target["floor_z"])
        rim = float(target_pos[2]) + float(target["rim_z"])
        limit = float(target["inner_xy_m"])
        ok = xy <= limit and (floor - Z_TOLERANCE_M <= bottom <= rim)
        return {
            "ok": ok,
            "predicate": "bowl_interior",
            "xy_m": round(xy, 4),
            "xy_limit_m": limit,
            "object_bottom_z": round(bottom, 4),
            "floor_z": round(floor, 4),
            "rim_z": round(rim, 4),
        }
    plane = float(target_pos[2]) + float(target["support_z"])
    limit = float(target["xy_limit_m"])
    ok = xy <= limit and abs(bottom - plane) <= Z_TOLERANCE_M
    return {
        "ok": ok,
        "predicate": "plate_support",
        "xy_m": round(xy, 4),
        "xy_limit_m": limit,
        "object_bottom_z": round(bottom, 4),
        "support_z": round(plane, 4),
    }


def desired_object_z(target_name: str, object_name: str, target_pos: list[float], target_size: list[float] | None, relation: str) -> float:
    """Object-origin height used as the descent target.

    For a site the site's own half thickness plus the object bottom offset is
    used. For a bowl/plate target the published interior/support planes are
    used. The executor still stops on observed object height, not on this
    number alone.
    """
    bottom = object_bottom_z(object_name)
    target = asset_of(target_name)
    if target and target.get("kind_when_target") == "interior":
        return float(target_pos[2]) + float(target["floor_z"]) - bottom
    if target and target.get("kind_when_target") == "support_body":
        return float(target_pos[2]) + float(target["support_z"]) - bottom
    half_z = abs(float(target_size[2])) if isinstance(target_size, list) and len(target_size) >= 3 else 0.0
    if relation == "in":
        return float(target_pos[2])
    return float(target_pos[2]) + half_z - bottom


def relation_check(
    snapshot: dict[str, Any] | None,
    object_name: str,
    target_name: str,
    relation: str,
) -> dict[str, Any]:
    """Resolve the target from the snapshot and apply the official relation."""
    snapshot = snapshot if isinstance(snapshot, dict) else {}
    sim = snapshot.get("sim") or {}
    objects = sim.get("objects") or {}
    named = snapshot.get("named_poses") or {}
    sites = named.get("sites") or {}
    geometry = named.get("site_geometry") or {}
    obj_row = objects.get(object_name) or {}
    object_pos = obj_row.get("pos") if isinstance(obj_row, dict) else None
    if not isinstance(object_pos, list) or len(object_pos) < 3:
        return {"ok": False, "reason_code": "object_pose_missing", "predicate": "relation", "evidence_source": "sim_state"}
    site_xyz = sites.get(target_name)
    if isinstance(site_xyz, list) and len(site_xyz) >= 3:
        shape = geometry.get(target_name) or {}
        mat = shape.get("mat")
        size = shape.get("size")
        if not mat or not size:
            return {"ok": False, "reason_code": "site_geometry_missing", "predicate": "relation", "evidence_source": "mujoco_site"}
        check = site_in_box(object_pos, site_xyz, mat, size) if relation == "in" else site_under(object_pos, site_xyz, mat, size)
        check["target_pose"] = list(site_xyz)
        check["evidence_source"] = "mujoco_site"
        return check
    target_row = objects.get(target_name) or {}
    target_pos = target_row.get("pos") if isinstance(target_row, dict) else None
    if not isinstance(target_pos, list) or len(target_pos) < 3:
        return {"ok": False, "reason_code": "target_pose_missing", "predicate": "relation", "evidence_source": "sim_state"}
    shaped = object_relation(object_pos, target_pos, object_name, target_name, relation)
    if shaped is not None:
        shaped["target_pose"] = list(target_pos)
        shaped["evidence_source"] = "asset_geometry"
        return shaped
    xy = math.hypot(float(object_pos[0]) - float(target_pos[0]), float(object_pos[1]) - float(target_pos[1]))
    z_ok = abs(float(object_pos[2]) - float(target_pos[2])) <= 0.06
    return {
        "ok": xy <= 0.08 and z_ok,
        "predicate": "center_distance",
        "xy_m": round(xy, 4),
        "z_delta_m": round(float(object_pos[2]) - float(target_pos[2]), 4),
        "evidence_source": "sim_state",
        "target_pose": list(target_pos),
    }
