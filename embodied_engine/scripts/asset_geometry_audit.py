#!/usr/bin/env python
"""What the bowl and cheese assets actually say, before any threshold is frozen.

Three numbers decide whether "the cheese is seated" means anything:

``seat``
    measured as ``object_bottom_z - floor_z``. The reference plane in
    ``common/geometry.py`` is the bowl's *exterior* bottom (``floor_z = -0.012``
    relative to the bowl body), not the surface the cheese rests on. This
    script measures the interior floor from the collision geoms so the two
    cannot be confused again.

``xy``
    compared against ``horizontal_radius_site`` (0.025 m). That site is a
    naming convention robosuite reads as ``horizontal_radius``; it is not the
    wall. The wall is the first ring of collision boxes; its inner face is what
    a 4 cm block can actually touch.

``rim``
    the top of the asset (``top_site``) versus the top of the collision ring.

Everything is computed from the installed XML, with box corners rotated by each
geom's own quaternion. Nothing is read from a run, so the numbers cannot drift
with the scene.

    python scripts/asset_geometry_audit.py --out records/acceptance/asset_geometry_20261105.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
ASSETS = Path(
    "/home/host/page-dmh/openwam_harness/envs/rpent/lib/python3.10/site-packages/libero/libero/assets"
)
BOWL_XML = ASSETS / "stable_scanned_objects/akita_black_bowl/akita_black_bowl.xml"
CHEESE_XML = ASSETS / "stable_hope_objects/cream_cheese/cream_cheese.xml"


def geom_corners(pos: list[float], quat: list[float], size: list[float]) -> np.ndarray:
    """The 8 world corners of a box geom.

    ``size`` is a half-extent per axis, so rotating the *vector* of half
    extents is wrong (it mixes axes); the rotation matrix is applied to the
    corner signs and each world extent is read off the result.
    """
    mat = quat_wxyz_to_mat(quat)
    signs = np.array(
        [
            [sx, sy, sz]
            for sx in (-1.0, 1.0)
            for sy in (-1.0, 1.0)
            for sz in (-1.0, 1.0)
        ]
    )
    local = signs * np.array(size)
    return (mat @ local.T).T + np.array(pos)


def quat_wxyz_to_mat(quat: list[float]) -> np.ndarray:
    w, x, y, z = [float(v) for v in quat]
    norm = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def parse(path: Path) -> dict[str, Any]:
    import xml.etree.ElementTree as ET

    root = ET.parse(path).getroot()
    geoms = []
    for geom in root.iter("geom"):
        if (geom.get("group") or "0") != "0":
            continue  # group 1 is the visual mesh
        if geom.get("type") != "box":
            continue
        pos = [float(v) for v in (geom.get("pos") or "0 0 0").split()]
        quat = [float(v) for v in (geom.get("quat") or "1 0 0 0").split()]
        size = [float(v) for v in (geom.get("size") or "0 0 0").split()]
        corners = geom_corners(pos, quat, size)
        radii = np.hypot(corners[:, 0], corners[:, 1])
        geoms.append(
            {
                "pos": pos,
                "size": size,
                "z_min": float(corners[:, 2].min()),
                "z_max": float(corners[:, 2].max()),
                "r_centre": math.hypot(pos[0], pos[1]),
                "r_min": float(radii.min()),
                "r_max": float(radii.max()),
            }
        )
    sites = {
        site.get("name"): [float(v) for v in (site.get("pos") or "0 0 0").split()]
        for site in root.iter("site")
    }
    return {"geoms": geoms, "sites": sites, "file": str(path)}


def describe(asset: dict[str, Any], label: str) -> dict[str, Any]:
    geoms = asset["geoms"]
    bottom = min(g["z_min"] for g in geoms)
    top = max(g["z_max"] for g in geoms)
    # Cluster geoms by height band. A ring is many boxes sharing a z band at
    # the same radius; the interior wall's inner face is what a block touches.
    bands: dict[str, list[dict[str, Any]]] = {}
    for geom in geoms:
        key = f"{round(geom['z_min'], 3)}|{round(geom['z_max'], 3)}"
        bands.setdefault(key, []).append(geom)
    ring_rows = []
    for key, items in sorted(bands.items(), key=lambda pair: float(pair[0].split("|")[0])):
        ring_rows.append(
            {
                "count": len(items),
                "z_min": round(min(g["z_min"] for g in items), 5),
                "z_max": round(max(g["z_max"] for g in items), 5),
                "r_centre_median": round(float(np.median([g["r_centre"] for g in items])), 5),
                "inner_radius_m": round(min(g["r_min"] for g in items), 5),
                "outer_radius_m": round(max(g["r_max"] for g in items), 5),
            }
        )
    return {
        "label": label,
        "file": asset["file"],
        "geoms": len(geoms),
        "bottom_z_m": round(bottom, 6),
        "top_z_m": round(top, 6),
        "height_m": round(top - bottom, 6),
        "sites": {name: pos for name, pos in asset["sites"].items()},
        "bands": ring_rows,
        "max_radius_m": round(max(g["r_max"] for g in geoms), 5),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(PROJECT / "records/acceptance/asset_geometry_20261105.json"))
    args = parser.parse_args()

    bowl = describe(parse(BOWL_XML), "akita_black_bowl")
    cheese = describe(parse(CHEESE_XML), "cream_cheese")

    bands = bowl["bands"]
    # The interior wall is the band whose z range sits above the floor plate
    # and whose radius is below the rim ring.
    wall = max(
        (row for row in bands if row["z_max"] < 0.03 and row["r_centre_median"] < 0.04),
        key=lambda row: row["r_centre_median"],
        default=None,
    )
    rim = max((row for row in bands if row["z_max"] >= 0.03), key=lambda row: row["z_max"], default=None)
    low = [row for row in bands if row["z_max"] < 0.02]
    interior_floor = round(max((row["z_max"] for row in low), default=0.0), 6)

    cheese_half = cheese["height_m"] / 2.0
    cheese_bottom_offset = cheese["bottom_z_m"]
    cheese_half_width = round(cheese["max_radius_m"], 6)
    bowl_floor_reference = -0.012  # common/geometry.py ASSETS['akita_black_bowl']['floor_z']
    seated_seat = round(float(interior_floor) - bowl_floor_reference, 6)
    seated_cheese_origin = round(float(interior_floor) - cheese_bottom_offset, 6)

    payload = {
        "schema": "embodied_engine.asset_geometry.v1",
        "bowl": bowl,
        "cheese": cheese,
        "measured": {
            "bowl_exterior_bottom_m": bowl["bottom_z_m"],
            "bowl_interior_floor_m": interior_floor,
            "bowl_top_site_m": (bowl["sites"].get("top_site") or [0, 0, None])[2],
            "bowl_wall_inner_radius_m": (wall or {}).get("inner_radius_m"),
            "bowl_wall_centre_radius_m": (wall or {}).get("r_centre_median"),
            "bowl_rim_inner_radius_m": (rim or {}).get("inner_radius_m"),
            "bowl_rim_top_m": (rim or {}).get("z_max"),
            "horizontal_radius_site_m": (bowl["sites"].get("horizontal_radius_site") or [None])[0],
        },
        "candidate_conditions": {
            "seat_range_m": [0.006, 0.029],
            "xy_limit_m": 0.015,
            "lift_above_rim_m": 0.02,
            "seat_reference": "object_bottom_z - floor_z, where floor_z = bowl origin + (-0.012), the bowl's exterior bottom",
            "seated_seat_value_m": seated_seat,
            "seated_cheese_origin_above_bowl_m": seated_cheese_origin,
            "cheese_bottom_offset_m": cheese_bottom_offset,
            "cheese_height_m": cheese["height_m"],
            "cheese_half_height_m": round(cheese_half, 6),
            "cheese_max_radius_m": cheese_half_width,
            "wall_clearance_for_cheese_m": (
                round(float(wall["inner_radius_m"]) - cheese_half_width, 6) if wall else None
            ),
            "rim_inner_clearance_for_cheese_m": (
                round(float(rim["inner_radius_m"]) - cheese_half_width, 6) if rim else None
            ),
            "lift_target_cheese_origin_z_m": (
                round(float((rim or {}).get("z_max", 0.04)) + 0.02 - cheese_bottom_offset, 6)
            ),
            "lift_target_bowl_origin_z_m": round(float((rim or {}).get("z_max", 0.04)) + 0.02, 6),
        },
        "reading": {
            "seat": (
                "seat is measured from the bowl's EXTERIOR bottom (-0.012), so a cheese "
                f"resting on the interior floor ({interior_floor} m) reads {seated_seat} m. "
                "The 0.006 lower bound is therefore below the floor by "
                f"{round(float(interior_floor) + 0.012 - 0.006, 6)} m: it is a tolerance floor, not a seated depth."
            ),
            "xy": (
                "xy is a centre distance. The asset's horizontal_radius_site (0.025 m) is a "
                "convention robosuite reads as horizontal_radius, the official In predicate "
                "uses it, and it is NOT the wall: the collision wall's inner face is "
                f"{(wall or {}).get('inner_radius_m')} m, and the cheese's own max radius is "
                f"{cheese_half_width} m, which leaves "
                f"{round(float(wall['inner_radius_m']) - cheese_half_width, 6) if wall else None} m. "
                "The candidate 0.015 m sits between those two, i.e. stricter than the goal "
                "predicate and looser than 'fully clear of the wall'."
            ),
            "rim": (
                f"the asset's top_site is at {(bowl['sites'].get('top_site') or [0,0,None])[2]} m and the "
                f"collision rim reaches {(rim or {}).get('z_max')} m; lifting 2 cm above the rim "
                "means the cheese bottom is at or above "
                f"{round(float((rim or {}).get('z_max', 0.04)) + 0.02, 5)} m relative to the bowl."
            ),
        },
        "status": "candidate values measured against the installed asset; none of them is a validated success predictor",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(payload["measured"], ensure_ascii=False, indent=1))
    print(json.dumps(payload["candidate_conditions"], ensure_ascii=False, indent=1))
    print(f"written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
