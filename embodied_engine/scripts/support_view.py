#!/usr/bin/env python
"""A checkable view of how the bowl actually holds the object at the end of a run.

Two panels, both drawn in the **bowl's own frame**, because "how is it supported"
is a question about the bowl, not about the world:

    left   the measured free-radius profile of the bowl cavity (from
           ``support_geometry_probe.py``, which compiles the asset and casts
           rays from inside), the bowl axis, and the object's footprint at the
           height it actually occupies
    right  the same profile seen from the side, with the interior floor, the
           rim, the object drawn at its measured tilt and height, and the
           collision geoms that were in contact marked

Every number on the plot is read from the episode record or from the geometry
probe; nothing is drawn from an assumed shape. The drawing is a schematic in the
sense that the object is drawn as its collision box (which it is) and the bowl
wall is drawn as the ray-cast profile axis-revolved (which is what it measured).

    python scripts/support_view.py --record ep.json --geometry geometry.json \
        --out figure.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

BOWL = "akita_black_bowl_1"
CHEESE = "cream_cheese_1"


def quat_to_matrix(quat: Any) -> np.ndarray:
    """Rotation matrix from a (w, x, y, z) quaternion, as the simulator stores it."""
    if not isinstance(quat, (list, tuple)) or len(quat) != 4:
        return np.eye(3)
    w, x, y, z = (float(v) for v in quat)
    norm = (w * w + x * x + y * y + z * z) ** 0.5
    if norm <= 1e-9:
        return np.eye(3)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def convex_hull(points: np.ndarray) -> np.ndarray:
    """2-D convex hull (monotone chain); the plot needs an outline, not a package."""
    pts = sorted({(round(float(p[0]), 9), round(float(p[1]), 9)) for p in points})
    if len(pts) <= 2:
        return np.array(pts)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for point in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    for point in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return np.array(lower[:-1] + upper[:-1] + [lower[0]])


def final_row(record: dict[str, Any], step_index: int) -> dict[str, Any]:
    timeline = ((record.get("steps") or [])[step_index - 1].get("goal_watch") or {}).get("timeline") or []
    timeline = [row for row in timeline if isinstance(row, dict)]
    timeline.sort(key=lambda row: row.get("sim_action_count") or -1)
    return timeline[-1]


def body(row: dict[str, Any], name: str) -> dict[str, Any]:
    bodies = ((row.get("dynamics") or {}).get("bodies") or {})
    entry = bodies.get(name)
    if isinstance(entry, dict):
        return entry
    for key, value in bodies.items():
        if isinstance(value, dict) and (key.startswith(f"{name}_") or value.get("body") == name):
            return value
    return {}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", required=True)
    parser.add_argument("--geometry", required=True)
    parser.add_argument("--step", type=int, default=2)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    record = json.loads(Path(args.record).read_text())
    geometry = json.loads(Path(args.geometry).read_text())
    row = final_row(record, args.step)
    bowl = body(row, BOWL)
    cheese = body(row, CHEESE)
    relation = (row.get("relative") or {}).get("c1") or {}

    rim_rel = float(geometry["bowl"]["rim_top_rel_body_m"])
    floor_rel = float(geometry["bowl"]["interior_floor_rel_body_m"])
    box = geometry["cheese"]["collision_box"]
    half_len, half_thick = box["half_x_m"], box["half_z_m"]
    half_width = box["half_y_m"]
    axis_xy = np.asarray(geometry["bowl"]["axis_xy"], dtype=float) - np.asarray(geometry["bowl"]["body_pos"], dtype=float)[:2]

    # observed numbers
    bowl_pos = np.asarray(bowl.get("pos") or [0, 0, 0], dtype=float)
    cheese_pos = np.asarray(cheese.get("pos") or [0, 0, 0], dtype=float)
    rel_xy = float(relation.get("xy_m") or 0.0)
    bowl_tilt = float(bowl.get("tilt_deg") or 0.0)
    cheese_tilt = float(cheese.get("tilt_deg") or 0.0)

    # The collision box is a real box, so its lowest point comes from all eight
    # corners of the recorded orientation, not from the centre minus half the
    # thickness. ``tilt_deg`` is only the angle of the body's own +z, and the
    # long axis can be near horizontal while the body still reads 44 deg; the
    # quaternion is what carries the rest of the orientation.
    rot = quat_to_matrix(cheese.get("quat"))
    bowl_rot = quat_to_matrix(bowl.get("quat"))
    corners_world = np.array(
        [cheese_pos + rot @ np.array([sx * half_len, sy * half_width, sz * half_thick])
         for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
    )
    # express everything in the bowl's frame: relative position, bowl-inverse rotation
    corners_bowl = (corners_world - bowl_pos) @ bowl_rot
    bottom_z_rel = float(corners_bowl[:, 2].min())
    radial = np.hypot(corners_bowl[:, 0] - axis_xy[0], corners_bowl[:, 1] - axis_xy[1])

    profile = geometry["profile"]
    zs = np.array([row["z_m"] for row in profile]) - bowl_pos[2]
    rs = np.array([row["free_radius_m"] for row in profile])

    contacts = [
        pair
        for pair in ((row.get("dynamics") or {}).get("contacts") or [])
        if "cream_cheese" in f"{pair.get('a')}{pair.get('b')}" and "bowl" in f"{pair.get('a')}{pair.get('b')}"
    ]
    geoms = sorted({str(pair.get("geom_b") if "cream" in str(pair.get("geom_a")) else pair.get("geom_a")) for pair in contacts})

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6))
    fig.suptitle(
        f"{Path(args.record).name}  step {args.step} final: bowl tilt {bowl_tilt:.1f} deg, "
        f"object tilt {cheese_tilt:.1f} deg, xy offset {1000 * rel_xy:.1f} mm",
        fontsize=10,
    )

    # ---- left: cross-section through the measured offset -------------------
    ax = axes[0]
    ax.plot(rs, zs, color="#1f6feb", lw=2, label="measured free radius (ray-cast)")
    ax.fill_betweenx(zs, 0, rs, color="#1f6feb", alpha=0.10)
    ax.axhline(rim_rel, color="#666", ls="--", lw=1, label=f"rim top {1000 * rim_rel:.1f} mm")
    ax.axhline(floor_rel, color="#999", ls=":", lw=1, label=f"interior floor {1000 * floor_rel:.1f} mm")
    # object as its collision box, projected into the (radius, height) plane
    hull = convex_hull(np.column_stack([radial, corners_bowl[:, 2]]))
    ax.plot(hull[:, 0], hull[:, 1], color="#c2410c", lw=2, label="object collision box (8 corners)")
    ax.axvline(rel_xy, color="#c2410c", ls=":", lw=1)
    ax.set_xlabel("radius from bowl axis [m]")
    ax.set_ylabel("height above bowl origin [m]")
    ax.set_title("cross-section through the measured offset", fontsize=9)
    ax.set_aspect("equal")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.2)

    # ---- right: top view (footprint) --------------------------------------
    ax = axes[1]
    theta = np.linspace(0, 2 * np.pi, 200)
    for z_level, style in ((rim_rel, "-"), (bottom_z_rel, "--")):
        idx = int(np.argmin(np.abs(zs - z_level)))
        ax.plot(rs[idx] * np.cos(theta), rs[idx] * np.sin(theta), style, lw=1.2,
                label=f"free radius at {1000 * z_level:.1f} mm = {1000 * rs[idx]:.1f} mm")
    footprint = (corners_bowl[:, :2] - axis_xy)
    fp_hull = convex_hull(footprint)
    ax.plot(fp_hull[:, 0], fp_hull[:, 1], color="#c2410c", lw=2)
    ax.plot(0.0, 0.0, "+", color="#1f6feb", ms=8, label="bowl axis")
    ax.annotate(f"{1000 * rel_xy:.1f} mm centre offset", (0, 0), textcoords="offset points", xytext=(6, 6), fontsize=8, color="#1f6feb")
    ax.set_xlim(-0.07, 0.07)
    ax.set_ylim(-0.07, 0.07)
    ax.set_aspect("equal")
    ax.set_title(f"top view; object touches bowl geoms {', '.join(g.split('_')[-1] for g in geoms) or 'none'}", fontsize=9)
    ax.set_xlabel("x in bowl frame [m]")
    ax.set_ylabel("y in bowl frame [m]")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(alpha=0.2)

    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"[ok] {out}")
    print(
        json.dumps(
            {
                "bowl_tilt_deg": bowl_tilt,
                "object_tilt_deg": cheese_tilt,
                "xy_offset_m": rel_xy,
                "object_bottom_above_floor_m": bottom_z_rel - floor_rel,
                "object_bottom_below_rim_m": rim_rel - bottom_z_rel,
                "contact_geoms": geoms,
                "contacts_listed": len(contacts),
            },
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
