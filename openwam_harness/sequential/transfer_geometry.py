"""Placement geometry for the official Goal transport tasks.

Numbers come from the installed LIBERO assets and from ObjectState.check_ontop.
A target body origin plus one z_offset is not a placement rule.
Travel clearance stays on the region reference so a stove approach is not the landing height.
"""

from __future__ import annotations

import math
import re
from typing import Any


# Collision and site measurements read from the asset XML. bottom_site is not used:
# several objects copy bottom_site z=-0.06 even when the collision mesh does not.
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
        "footprint_xy_m": 0.021336,
    },
    "wine_bottle": {
        "source": "turbosquid_objects/wine_bottle/wine_bottle.xml",
        "bottom_z": 0.00111,
        "footprint_xy_m": 0.02844,
        "top_z": 0.15804,
    },
    "flat_stove": {
        "source": "articulated_objects/flat_stove.xml",
        "kind_when_target": "support_site",
        "site_name": "cook_region",
        "site_half_z": 0.0025,
        "site_half_xy_m": 0.075,
    },
    "wooden_cabinet": {
        "source": "articulated_objects/wooden_cabinet.xml",
        "kind_when_target": "support_site",
        "site_name": "top_side",
        "site_half_z": 0.00147,
        "site_half_xy_m": 0.09438,
    },
}

OFFICIAL_ON_XY_M = 0.03
Z_TOLERANCE_M = 0.02


def asset_key(name: str | None) -> str:
    return re.sub(r"_\d+$", "", name or "")


def asset_of(name: str | None) -> dict[str, Any] | None:
    row = ASSETS.get(asset_key(name))
    return dict(row) if row else None


def motion_target(
    object_name: str | None,
    target_name: str | None,
    surface: list[float],
    *,
    motion: str,
    clearance: float,
    site_size: list[float] | None = None,
) -> dict[str, Any]:
    """Object-origin height and the xy disc the object is allowed to land in."""
    carried = asset_of(object_name) or {}
    target = asset_of(target_name) or {}
    kind = _kind(target_name, target, site_size)
    bottom = float(carried.get("bottom_z", 0.0))
    half_z = _half_z(target, site_size)
    if motion != "vertical":
        return {
            "ok": True,
            "kind": kind,
            "object_z": float(surface[2]) + float(clearance),
            "xy_radius_m": _xy_radius(kind, target, site_size, carried),
            "z_rule": "travel_clearance_above_region_reference",
            "object_bottom_z": bottom,
            "source": target.get("source") or carried.get("source"),
        }
    if kind == "interior":
        floor = float(target.get("floor_z", 0.0))
        rim = float(target.get("rim_z", 0.04))
        origin_z = float(surface[2]) + floor - bottom
        if origin_z > float(surface[2]) + rim:
            return {"ok": False, "code": "object_taller_than_cavity", "reason": "物体高度超过碗的碗沿，不能按内部放置"}
        z_rule = "bowl_floor_minus_object_bottom"
    elif kind == "support_body":
        origin_z = float(surface[2]) + float(target.get("support_z", 0.0)) - bottom
        z_rule = "support_plane_minus_object_bottom"
    else:
        origin_z = float(surface[2]) + half_z - bottom
        z_rule = "site_top_minus_object_bottom"
    return {
        "ok": True,
        "kind": kind,
        "object_z": origin_z,
        "xy_radius_m": _xy_radius(kind, target, site_size, carried),
        "z_rule": z_rule,
        "object_bottom_z": bottom,
        "source": target.get("source"),
    }


def relation_check(obj, target_pos, object_name: str | None, target_name: str | None) -> dict[str, Any] | None:
    """Geometric result for a plate or a bowl. A site target stays with SiteObject.under."""
    target = asset_of(target_name)
    carried = asset_of(object_name) or {}
    if not target or target.get("kind_when_target") not in {"interior", "support_body"}:
        return None
    if not isinstance(obj, list) or not isinstance(target_pos, list) or len(obj) < 3 or len(target_pos) < 3:
        return {"ok": False, "reason_code": "pose_not_provided", "predicate": target["kind_when_target"]}
    xy = math.hypot(float(obj[0]) - float(target_pos[0]), float(obj[1]) - float(target_pos[1]))
    bottom = float(obj[2]) + float(carried.get("bottom_z", 0.0))
    if target["kind_when_target"] == "interior":
        floor = float(target_pos[2]) + float(target["floor_z"])
        rim = float(target_pos[2]) + float(target["rim_z"])
        limit = float(target["inner_xy_m"])
        inside_xy = xy <= limit
        inserted = floor - Z_TOLERANCE_M <= bottom <= rim
        return {
            "ok": inside_xy and inserted,
            "predicate": "bowl_interior",
            "xy_m": round(xy, 4),
            "xy_limit_m": limit,
            "object_bottom_z": round(bottom, 4),
            "floor_z": round(floor, 4),
            "rim_z": round(rim, 4),
            "contact_status": "not_provided",
            "official_on_xy_m": OFFICIAL_ON_XY_M,
            "reason": "" if inside_xy and inserted else "物体不在碗的内部",
        }
    plane = float(target_pos[2]) + float(target["support_z"])
    limit = float(target["xy_limit_m"])
    on_plane = abs(bottom - plane) <= Z_TOLERANCE_M
    inside = xy <= limit
    return {
        "ok": inside and on_plane,
        "predicate": "plate_support",
        "xy_m": round(xy, 4),
        "xy_limit_m": limit,
        "object_bottom_z": round(bottom, 4),
        "support_z": round(plane, 4),
        "contact_status": "not_provided",
        "official_on_xy_m": OFFICIAL_ON_XY_M,
        "reason": "" if inside and on_plane else "物体没有落在盘子的支撑面上",
    }


def _kind(target_name: str | None, target: dict, site_size) -> str:
    declared = target.get("kind_when_target")
    if declared:
        return declared
    if site_size:
        return "support_site"
    return "support_site"


def _half_z(target: dict, site_size) -> float:
    if isinstance(site_size, list) and len(site_size) >= 3:
        return abs(float(site_size[2]))
    if "site_half_z" in target:
        return float(target["site_half_z"])
    return 0.0


def _xy_radius(kind: str, target: dict, site_size, carried: dict) -> float:
    footprint = float(carried.get("footprint_xy_m") or 0.0)
    if kind == "interior":
        return float(target.get("inner_xy_m", OFFICIAL_ON_XY_M))
    if kind == "support_body":
        return float(target.get("xy_limit_m", OFFICIAL_ON_XY_M))
    half = None
    if isinstance(site_size, list) and len(site_size) >= 2:
        half = min(abs(float(site_size[0])), abs(float(site_size[1])))
    elif "site_half_xy_m" in target:
        half = float(target["site_half_xy_m"])
    if half is None:
        return OFFICIAL_ON_XY_M
    return max(0.0, half - footprint)
