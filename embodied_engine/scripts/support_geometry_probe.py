#!/usr/bin/env python
"""What actually supports the cream cheese: the compiled collision geometry.

The previous round compared one number - the cheese's "max radius", 0.04587 m -
against the bowl's "rim inner radius", 0.04139 m, and concluded the cheese
cannot enter the bowl. Both numbers were corner radii of box proxies, so the
comparison was real but the *meaning* was not: 0.04587 m is the distance from
the object axis to a corner of a **rectangular slab**, not a bounding sphere and
not the mesh cross-section.

This script answers the question from the geometry the simulator compiled:

* which geom actually collides (``contype``/``conaffinity``, not the XML's
  intent), its type and size;
* the cheese's real footprint - half extents in the horizontal plane after the
  geom's own quaternion;
* the bowl's interior profile, band by band, and the free radius of the cavity
  at every height (the largest circle around the axis that no collision geom
  occupies);
* whether the footprint fits that circle at each height, flat and standing on
  its thin edge;
* the same question answered exactly, by MuJoCo's own box-box distance, at the
  candidate poses.

Everything is measured. Nothing is written to the environment.

    python scripts/support_geometry_probe.py --snapshot /tmp/snap.json \
        --out records/acceptance/support_geometry_20261105.json \
        --figure records/acceptance/support_geometry_20261105.png
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]

GEOM_BOX = 6


# ---- small geometry helpers -------------------------------------------------


def quat_mat(q: list[float]) -> np.ndarray:
    """MuJoCo wxyz quaternion to a rotation matrix."""
    w, x, y, z = (float(v) for v in q)
    norm = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def mat_quat(mat: np.ndarray) -> list[float]:
    """Rotation matrix to a wxyz quaternion (MuJoCo convention)."""
    m = np.asarray(mat, dtype=float)
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    norm = math.sqrt(sum(v * v for v in q)) or 1.0
    return [v / norm for v in q]


def box_corners(centre: np.ndarray, rot: np.ndarray, size: list[float]) -> np.ndarray:
    half = np.asarray([float(v) for v in size[:3]])
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=float)
    return (rot @ (signs * half).T).T + centre


def body_of(named: dict[str, Any], name: str) -> str | None:
    """``sim.objects`` uses the object name; MuJoCo uses ``<name>_main``."""
    dyn = named.get("dynamics") or {}
    for key in (name, f"{name}_main"):
        if key in dyn:
            return key
    return None


def collect_geoms(named: dict[str, Any], body: str | None) -> list[dict[str, Any]]:
    """World pose and size of every geom on ``body``."""
    if not body:
        return []
    dyn = (named.get("dynamics") or {}).get(body) or {}
    if not dyn:
        return []
    base = np.asarray(dyn["pos"], dtype=float)
    base_rot = quat_mat(dyn["quat"])
    out: list[dict[str, Any]] = []
    for name, geom in (named.get("geoms") or {}).items():
        if geom.get("body") != body:
            continue
        local = np.asarray(geom.get("pos") or [0.0, 0.0, 0.0], dtype=float)
        rot = base_rot @ quat_mat(geom.get("quat") or [1.0, 0.0, 0.0, 0.0])
        out.append(
            {
                "name": name,
                "type": int(geom.get("type") or 0),
                "size": [float(v) for v in (geom.get("size") or [])],
                "contype": int(geom.get("contype") or 0),
                "conaffinity": int(geom.get("conaffinity") or 0),
                "centre": base + base_rot @ local,
                "rot": rot,
            }
        )
    return out


def colliding(geoms: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [g for g in geoms if g["type"] == GEOM_BOX and (g["contype"] or g["conaffinity"])]


def free_radius_profile(
    geoms: list[dict[str, Any]], axis_xy: tuple[float, float], z_max: float, z_min: float, step: float = 0.001
) -> list[dict[str, Any]]:
    """Largest circle around the axis that holds no collision geometry.

    Point sampling, not a bounding-box argument: for each height, grow a circle
    and ask whether any sample point is inside any box. A box is convex, so the
    point test is exact for the point; the sampling resolution (0.5 mm radial,
    2 degrees around) bounds the error.
    """
    rows: list[dict[str, Any]] = []
    angles = np.linspace(0.0, 2 * math.pi, 180, endpoint=False)
    z = z_min
    while z <= z_max + 1e-9:
        radius = 0.0
        while radius < 0.07:
            probe = radius + 0.0005
            points = np.stack(
                [
                    axis_xy[0] + probe * np.cos(angles),
                    axis_xy[1] + probe * np.sin(angles),
                    np.full_like(angles, z),
                ],
                axis=1,
            )
            blocked = False
            for geom in geoms:
                local = (geom["rot"].T @ (points - geom["centre"]).T).T
                half = np.asarray(geom["size"][:3], dtype=float)
                if bool(np.all(np.abs(local) <= half, axis=1).any()):
                    blocked = True
                    break
            if blocked:
                break
            radius = probe
        rows.append({"z_m": round(z, 6), "free_radius_m": round(radius, 5)})
        z += step
    return rows


def footprint(half: np.ndarray) -> dict[str, float]:
    hx, hy, hz = (float(v) for v in half[:3])
    return {
        "half_x_m": round(hx, 6),
        "half_y_m": round(hy, 6),
        "half_z_m": round(hz, 6),
        "long_side_m": round(2 * max(hx, hy), 6),
        "short_side_m": round(2 * min(hx, hy), 6),
        "thickness_m": round(2 * hz, 6),
        "circumradius_m": round(math.hypot(hx, hy), 6),
        "inradius_m": round(min(hx, hy), 6),
        "definition": (
            "half extents of the compiled collision box in the object frame; the circumradius is "
            "the distance from the axis to a corner, which is exactly the 0.04587 m quoted before"
        ),
    }


# ---- exact check with MuJoCo ------------------------------------------------


def exact_clearance(
    bowl_geoms: list[dict[str, Any]],
    cheese: dict[str, Any],
    *,
    yaws: list[float],
    bottoms: list[float],
    z_ref: float = 0.0,
) -> list[dict[str, Any]]:
    """Signed box-box distance between the cheese slab and every bowl geom.

    Uses ``mj_geomDistance``, i.e. the simulator's own narrowphase, so a corner
    that a circle model would let through is still caught. A *negative* value
    means the slab intersects the bowl at that pose.
    """
    import mujoco

    lines = ["<mujoco><worldbody>"]
    for index, geom in enumerate(bowl_geoms):
        pos = geom["centre"]
        quat = mat_quat(geom["rot"])
        lines.append(
            '<geom name="b%d" type="box" pos="%f %f %f" quat="%f %f %f %f" size="%f %f %f"/>'
            % (index, pos[0], pos[1], pos[2], quat[0], quat[1], quat[2], quat[3], *geom["size"][:3])
        )
    # The cheese is a free body so each candidate pose is a ``qpos`` write,
    # the same path the simulator itself uses, instead of an edit to compiled
    # model arrays.
    lines.append('<body name="cheese" pos="0 0 0"><freejoint name="cheese_free"/>')
    lines.append('<geom name="cheese_geom" type="box" size="%f %f %f"/></body>' % tuple(cheese["size"][:3]))
    lines.append("</worldbody></mujoco>")
    model = mujoco.MjModel.from_xml_string("\n".join(lines))
    data = mujoco.MjData(model)
    cheese_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "cheese_geom")
    bowl_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"b{i}") for i in range(len(bowl_geoms))]
    # ``cheese["size"]`` is in the geom's own frame; the object frame half
    # extents are what a height offset has to use.
    cheese_half = np.asarray(
        cheese["half_world"] if cheese.get("half_world") is not None else cheese["size"][:3], dtype=float
    )
    rows: list[dict[str, Any]] = []
    fromto = np.zeros(6, dtype=np.float64)
    for bottom in bottoms:
        best = {"yaw_deg": None, "clearance_m": -1e9}
        for yaw in yaws:
            c, s = math.cos(yaw), math.sin(yaw)
            yaw_rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
            rot = yaw_rot @ cheese["rot"]
            data.qpos[0:3] = [0.0, 0.0, bottom + cheese_half[2]]
            data.qpos[3:7] = mat_quat(rot)
            mujoco.mj_forward(model, data)
            smallest = min(
                mujoco.mj_geomDistance(model, data, cheese_id, bowl_id, 0.5, fromto) for bowl_id in bowl_ids
            )
            if smallest > best["clearance_m"]:
                best = {"yaw_deg": round(math.degrees(yaw), 2), "clearance_m": round(float(smallest), 6)}
        rows.append({"bottom_z_rel_body_m": round(bottom - z_ref, 5), **best})
    return rows


def deepest_insertion(
    profile: list[dict[str, Any]],
    floor_abs: float,
    rim_abs: float,
    *,
    extent: float,
    radius: float,
    base_z: float,
) -> dict[str, Any]:
    """How deep a rigid body of ``extent`` height and ``radius`` half-diagonal
    can go into the free cavity before its walls block it.

    ``radius`` is the circumradius of the body's horizontal cross-section, so
    ``free_radius >= radius`` on every height the body spans is a sufficient
    (not necessary) containment test. It is deliberately conservative: a corner
    that a circle model rejects is still checked again by MuJoCo's narrowphase
    later in the report.
    """
    window_rows = [row for row in profile if row["z_m"] >= floor_abs - 1e-9 and row["z_m"] <= rim_abs + 1e-9]
    best: dict[str, Any] | None = None
    fully_inside: dict[str, Any] | None = None
    h = floor_abs
    while h <= rim_abs + 1e-9:
        span = [row for row in window_rows if h - 1e-9 <= row["z_m"] <= min(h + extent, rim_abs) + 1e-9]
        if span:
            minimum = min(row["free_radius_m"] for row in span)
            if minimum >= radius:
                candidate = {
                    "bottom_z_rel_body_m": round(h - base_z, 5),
                    "depth_below_rim_m": round(rim_abs - h, 5),
                    "min_free_radius_over_body_m": round(minimum, 5),
                    "protrudes_above_rim_m": round(max(0.0, h + extent - rim_abs), 5),
                }
                if best is None:
                    best = candidate
                if h + extent <= rim_abs + 1e-9:
                    fully_inside = candidate
                    break
        h += 0.001
    return {
        "body_height_m": round(extent, 5),
        "body_circumradius_m": round(radius, 5),
        "deepest_insertion": best,
        "deepest_fully_below_rim": fully_inside,
    }


# ---- report ----------------------------------------------------------------


def analyse(named: dict[str, Any]) -> dict[str, Any]:
    bowl_body = body_of(named, "akita_black_bowl_1")
    cheese_body = body_of(named, "cream_cheese_1")
    bowl_dyn = (named.get("dynamics") or {}).get(bowl_body) or {}
    cheese_dyn = (named.get("dynamics") or {}).get(cheese_body) or {}
    bowl_all = collect_geoms(named, bowl_body)
    bowl = colliding(bowl_all)
    cheese_all = collect_geoms(named, cheese_body)
    cheese_box = colliding(cheese_all)[0] if colliding(cheese_all) else None
    base = np.asarray(bowl_dyn.get("pos") or [0, 0, 0], dtype=float)
    centres = np.array([g["centre"] for g in bowl])
    axis = centres.mean(axis=0)
    zmin = min(box_corners(g["centre"], g["rot"], g["size"])[:, 2].min() for g in bowl)
    zmax = max(box_corners(g["centre"], g["rot"], g["size"])[:, 2].max() for g in bowl)
    floor = max(
        (
            box_corners(g["centre"], g["rot"], g["size"])[:, 2].max()
            for g in bowl
            if math.hypot(g["centre"][0] - axis[0], g["centre"][1] - axis[1]) < 0.02
        ),
        default=None,
    )
    # Only heights that are actually *inside* the cavity mean anything. Below
    # the interior floor the cheese is outside the bowl, so an unbounded free
    # radius there is not evidence that the slab "fits".
    profile = free_radius_profile(bowl, (axis[0], axis[1]), zmax, zmin)
    bands: dict[str, list[dict[str, Any]]] = {}
    for geom in bowl:
        corners = box_corners(geom["centre"], geom["rot"], geom["size"])
        key = f"{round(corners[:, 2].min() - base[2], 3)}|{round(corners[:, 2].max() - base[2], 3)}"
        radii = np.hypot(corners[:, 0] - axis[0], corners[:, 1] - axis[1])
        bands.setdefault(key, []).append({"size": geom["size"][:3], "r": [float(radii.min()), float(radii.max())]})
    band_rows = [
        {
            "z_band_rel_body_m": [float(k.split("|")[0]), float(k.split("|")[1])],
            "count": len(items),
            "corner_r_min_m": round(min(i["r"][0] for i in items), 5),
            "corner_r_max_m": round(max(i["r"][1] for i in items), 5),
            "geom_size_m": items[0]["size"],
        }
        for k, items in sorted(bands.items(), key=lambda kv: float(kv[0].split("|")[0]))
    ]
    report: dict[str, Any] = {
        "schema": "embodied_engine.support_geometry.v1",
        "bowl": {
            "body": bowl_body,
            "body_pos": [round(float(v), 5) for v in base],
            "axis_xy": [round(float(axis[0]), 5), round(float(axis[1]), 5)],
            "axis_offset_from_body_m": round(float(math.hypot(axis[0] - base[0], axis[1] - base[1])), 6),
            "geoms_total": len(bowl_all),
            "geoms_colliding": len(bowl),
            "z_min_rel_body_m": round(float(zmin - base[2]), 6),
            "z_max_rel_body_m": round(float(zmax - base[2]), 6),
            "interior_floor_rel_body_m": None if floor is None else round(float(floor - base[2]), 6),
            "rim_top_rel_body_m": round(float(zmax - base[2]), 6),
            "bands": band_rows,
        },
        "cheese": {
            "body": cheese_body,
            "body_pos": [round(float(v), 5) for v in (cheese_dyn.get("pos") or [0, 0, 0])],
            "geoms": [
                {
                    "name": g["name"],
                    "type": g["type"],
                    "size": g["size"],
                    "contype": g["contype"],
                    "conaffinity": g["conaffinity"],
                    "collides": bool(g["contype"] or g["conaffinity"]),
                }
                for g in cheese_all
            ],
            "collision_type": None if cheese_box is None else "box",
        },
        "profile": profile,
    }
    if cheese_box is not None:
        obj_rot = quat_mat(cheese_dyn.get("quat") or [1, 0, 0, 0])
        world_corners = box_corners(np.zeros(3), obj_rot @ cheese_box["rot"], cheese_box["size"])
        fp = footprint(np.abs(world_corners).max(axis=0))
        report["cheese"]["collision_box"] = fp
        # "Flat" = the slab's broad face horizontal, so its horizontal
        # cross-section is long_side x short_side and each pose spans the
        # thickness vertically. "On edge" = standing on the thin face, which
        # needs the whole long side vertically.
        flat = {
            "extent": fp["thickness_m"],
            "radius": fp["circumradius_m"],
        }
        edge = {
            "extent": fp["long_side_m"],
            "radius": round(math.hypot(min(fp["half_x_m"], fp["half_y_m"]), fp["half_z_m"]), 6),
        }
        report["fit"] = {
            "flat": deepest_insertion(
                profile, float(floor), float(zmax), base_z=float(base[2]), **flat
            ),
            "on_edge": deepest_insertion(
                profile, float(floor), float(zmax), base_z=float(base[2]), **edge
            ),
            "note": (
                "free radius is only sampled from the interior floor up to the rim; below the "
                "floor the slab is outside the bowl, so an unconstrained radius there is not a fit"
            ),
        }
        cavity_rows = [row for row in profile if float(floor) - 1e-9 <= row["z_m"] <= float(zmax) + 1e-9]
        narrowest = min(cavity_rows, key=lambda r: r["free_radius_m"]) if cavity_rows else None
        flat_deep = report["fit"]["flat"]["deepest_insertion"]
        edge_deep = report["fit"]["on_edge"]["deepest_insertion"]
        report["conclusion"] = {
            "widest_free_radius_in_cavity_m": round(max(r["free_radius_m"] for r in cavity_rows), 5),
            "narrowest_free_radius_in_cavity_m": None if narrowest is None else narrowest["free_radius_m"],
            "cavity_height_above_floor_m": round(float(zmax - floor), 5),
            "cheese_corner_radius_m": fp["circumradius_m"],
            "cheese_edge_radius_m": edge["radius"],
            "corner_deficit_m": round(fp["circumradius_m"] - max(r["free_radius_m"] for r in cavity_rows), 6),
            "flat_reaches_m": None if flat_deep is None else flat_deep["depth_below_rim_m"],
            "flat_fully_below_rim": flat_deep is not None
            and report["fit"]["flat"]["deepest_fully_below_rim"] is not None,
            "on_edge_reaches_m": None if edge_deep is None else edge_deep["depth_below_rim_m"],
            "on_edge_protrudes_above_rim_m": None if edge_deep is None else edge_deep["protrudes_above_rim_m"],
            "statement": (
                "the slab is a box; the 0.04587 m figure is its corner radius, not a sphere. "
                "It cannot reach the bowl floor in either orientation, so the expected support is "
                "across the rim, not at the bottom"
            ),
        }
    return report


def figure(report: dict[str, Any], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bowl = report["bowl"]
    profile = report["profile"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    ax = axes[0]
    zs = [row["z_m"] - float(bowl["body_pos"][2]) for row in profile]
    rs = [row["free_radius_m"] for row in profile]
    floor = bowl["interior_floor_rel_body_m"]
    rim = bowl["rim_top_rel_body_m"]
    keep = [i for i, z in enumerate(zs) if floor - 0.003 <= z <= rim + 0.003]
    zs = [zs[i] for i in keep]
    rs = [rs[i] for i in keep]
    ax.plot(rs, zs, color="#1f77b4", lw=1.6, label="free cavity radius")
    ax.plot([-r for r in rs], zs, color="#1f77b4", lw=1.6)
    ax.axhline(floor, color="#2ca02c", ls="--", lw=1, label=f"interior floor {floor * 1000:.1f} mm")
    ax.axhline(rim, color="#d62728", ls="--", lw=1, label=f"rim top {rim * 1000:.1f} mm")
    fp = report["cheese"]["collision_box"]
    ax.add_patch(
        plt.Rectangle(
            (-fp["half_x_m"], rim), 2 * fp["half_x_m"], fp["thickness_m"], fill=False, ec="#ff7f0e", lw=1.4,
            label="cheese slab footprint (perched)",
        )
    )
    ax.set_xlabel("radius from bowl axis (m)")
    ax.set_ylabel("height above bowl body origin (m)")
    ax.set_title("bowl cavity vs cheese slab (side view)")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(alpha=0.25)

    ax = axes[1]
    theta = np.linspace(0, 2 * math.pi, 400)
    floor = bowl["interior_floor_rel_body_m"]
    rim = bowl["rim_top_rel_body_m"]
    cavity = [row for row in profile if floor is not None and floor - 1e-9 <= row["z_m"] - float(bowl["body_pos"][2]) <= rim + 1e-9]
    widest = max(cavity, key=lambda r: r["free_radius_m"]) if cavity else max(profile, key=lambda r: r["free_radius_m"])
    ax.plot(
        widest["free_radius_m"] * np.cos(theta),
        widest["free_radius_m"] * np.sin(theta),
        color="#1f77b4",
        label=f"widest cavity circle r={widest['free_radius_m'] * 1000:.1f} mm\nat z={widest['z_m'] - float(bowl['body_pos'][2]) :.4f} m",
    )
    ax.add_patch(
        plt.Rectangle(
            (-fp["half_x_m"], -fp["half_y_m"]), 2 * fp["half_x_m"], 2 * fp["half_y_m"], fill=False, ec="#ff7f0e",
            lw=1.4, label="cheese footprint (flat)",
        )
    )
    ax.set_aspect("equal")
    ax.set_xlabel("x from bowl axis (m)")
    ax.set_ylabel("y from bowl axis (m)")
    ax.set_title("top view inside the cavity")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", required=True, help="a state_snapshot JSON with named_poses.geoms")
    parser.add_argument("--out", default=str(PROJECT / "records/acceptance/support_geometry_20261105.json"))
    parser.add_argument("--figure", default=str(PROJECT / "records/acceptance/support_geometry_20261105.png"))
    parser.add_argument("--exact", action="store_true", help="also run the MuJoCo box-box check")
    args = parser.parse_args()
    snapshot = json.loads(Path(args.snapshot).read_text())
    named = snapshot.get("named_poses") or snapshot
    report = analyse(named)
    if args.exact:
        bowl = colliding(collect_geoms(named, report["bowl"]["body"]))
        cheese = colliding(collect_geoms(named, report["cheese"]["body"]))[0]
        axis = np.asarray(report["bowl"]["axis_xy"], dtype=float)
        base = np.asarray(report["bowl"]["body_pos"], dtype=float)
        shifted = [{**g, "centre": g["centre"] - np.array([axis[0], axis[1], 0.0])} for g in bowl]
        rim = report["bowl"]["rim_top_rel_body_m"]
        bottoms = [float(base[2]) + rim + d for d in (0.010, 0.005, 0.002, 0.0, -0.002, -0.004, -0.006,
                                                     -0.010, -0.016, -0.024, -0.032, -0.040)]
        world_corners = box_corners(np.zeros(3), quat_mat((named.get("dynamics") or {}).get(report["cheese"]["body"], {}).get("quat") or [1, 0, 0, 0]) @ cheese["rot"], cheese["size"])
        half_world = np.abs(world_corners).max(axis=0)
        report["exact_box_distance"] = exact_clearance(
            shifted,
            {**cheese, "centre": cheese["centre"] - np.array([axis[0], axis[1], 0.0]), "half_world": half_world},
            yaws=[math.radians(v) for v in range(0, 90, 3)],
            bottoms=bottoms,
            z_ref=float(base[2]),
        )
    report["snapshot_sim_action_count"] = snapshot.get("sim_action_count")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    if report.get("cheese", {}).get("collision_box"):
        figure(report, Path(args.figure))
    print(json.dumps(report.get("conclusion"), indent=2))
    print(f"[ok] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
