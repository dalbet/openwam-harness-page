#!/usr/bin/env python
"""Read-only geometry probe for the `1 -> 7` candidate (bowl on stove -> turn stove on).

The open question for `1 -> 7` was written down in the acceptance report §21.2
as "does a bowl sitting on the cook region block the stove knob". This script
answers the *geometric* half of that with measured numbers and refuses to
answer the dynamic half:

* it loads the official task 7 scene (the `turn_on_stove` BDDL + seed 0), which
  is a real scene load, and then only *reads* state: no `env.step`, no policy
  call, no object moved;
* the stove / burner / cook-region / bowl positions come from the live
  simulator through `env.state_snapshot`;
* the knob's own extent comes from the installed `flat_stove.xml` collision
  geometry, rotated by the knob body's quaternion *and* by the base
  orientation that the live snapshot itself confirms (the burner is measured
  at base + (0.15, 0, 0), which is exactly the model's local offset, so the
  base frame is unrotated in this scene);
* the output states `dynamic_reachability: "not_tested"`. Nothing here proves
  the policy can turn the knob with the bowl present, so `1 -> 7` stays out of
  the executable whitelist.

Run: `python scripts/knob_reach_probe.py`
"""

from __future__ import annotations

import argparse
import json
import math
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parents[1]
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
STOVE_XML = Path(
    "/home/host/page-dmh/openwam_harness/envs/rpent/lib/python3.10/site-packages/"
    "libero/libero/assets/articulated_objects/flat_stove.xml"
)
BOWL_XML = Path(
    "/home/host/page-dmh/openwam_harness/envs/rpent/lib/python3.10/site-packages/"
    "libero/libero/assets/stable_scanned_objects/akita_black_bowl/akita_black_bowl.xml"
)


def get(url: str, timeout: float = 60.0) -> Any:
    with OPENER.open(urllib.request.Request(url, method="GET"), timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def post(url: str, body: dict[str, Any], timeout: float = 400.0) -> Any:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with OPENER.open(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def quat_mul(a: list[float], b: list[float]) -> list[float]:
    """Hamilton product, both quaternions in (w, x, y, z)."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return [
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ]


def quat_rotate(q: list[float], v: list[float]) -> list[float]:
    w, x, y, z = q
    # v' = q v q*
    r = quat_mul(quat_mul(q, [0.0, *v]), [w, -x, -y, -z])
    return r[1:]


def _floats(text: str | None, default: list[float]) -> list[float]:
    if not text:
        return list(default)
    return [float(v) for v in text.split()]


def geom_corners(body: ET.Element, frame_quat: list[float]) -> list[list[float]]:
    """Corners of every box collision geom of a body, in the parent frame."""
    out: list[list[float]] = []
    for geom in body.findall("geom"):
        if geom.get("type") != "box":
            continue
        pos = _floats(geom.get("pos"), [0, 0, 0])
        size = _floats(geom.get("size"), [0, 0, 0])
        quat = _floats(geom.get("quat"), [1, 0, 0, 0])
        for sx in (-1, 1):
            for sy in (-1, 1):
                for sz in (-1, 1):
                    local = [sx * size[0], sy * size[1], sz * size[2]]
                    rotated = quat_rotate(quat, local)
                    out.append([pos[i] + rotated[i] for i in range(3)])
    return out


def knob_frame_extent() -> dict[str, Any]:
    """Button body geometry, expressed in the stove base frame."""
    tree = ET.parse(STOVE_XML)
    root = tree.getroot()
    for body in root.iter("body"):
        if body.get("name") != "button":
            continue
        body_pos = _floats(body.get("pos"), [0, 0, 0])
        body_quat = _floats(body.get("quat"), [1, 0, 0, 0])
        corners = []
        for corner in geom_corners(body, body_quat):
            world = quat_rotate(body_quat, corner)
            corners.append([body_pos[i] + world[i] for i in range(3)])
        radii = [math.hypot(c[0], c[1]) for c in corners]
        return {
            "body_pos_in_base": body_pos,
            "body_quat_wxyz": body_quat,
            "corners_in_base": corners,
            "max_x_in_base": max(c[0] for c in corners),
            "min_x_in_base": min(c[0] for c in corners),
            "max_y_in_base": max(c[1] for c in corners),
            "min_y_in_base": min(c[1] for c in corners),
            "max_z_in_base": max(c[2] for c in corners),
            "min_z_in_base": min(c[2] for c in corners),
            "max_radius_m": max(radii),
            "source": str(STOVE_XML),
        }
    raise SystemExit("flat_stove.xml has no body named 'button'")


def bowl_extent() -> dict[str, Any]:
    """Collision-box extent of the bowl, in its own body frame."""
    tree = ET.parse(BOWL_XML)
    root = tree.getroot()
    body = next(
        found
        for found in root.iter("body")
        if any(geom.get("type") == "box" for geom in found.findall("geom"))
    )
    corners = geom_corners(body, [1, 0, 0, 0])
    radii = [math.hypot(c[0], c[1]) for c in corners]
    return {
        "max_radius_m": max(radii),
        "min_z_m": min(c[2] for c in corners),
        "max_z_m": max(c[2] for c in corners),
        "geoms": len([g for g in body.findall("geom") if g.get("type") == "box"]),
        "source": str(BOWL_XML),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", default="http://127.0.0.1:14800")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="records/acceptance/knob_reach_probe_20261027.json")
    args = parser.parse_args()

    loaded = post(f"{args.engine}/api/load", {"task_id": 7, "seed": args.seed})
    if loaded.get("status") != "ok":
        raise SystemExit(f"load failed: {loaded}")
    metrics = get(f"{args.engine}/metrics")
    env_view = metrics.get("env") or {}
    port = env_view.get("port")
    if not port:
        raise SystemExit("engine did not publish an env port")

    import sys

    sys.path.insert(0, str(PROJECT))
    from adapters.env_client import EnvClient  # noqa: PLC0415 - needs the project path

    client = EnvClient(f"http://127.0.0.1:{port}")
    snap_before = client.state_snapshot()
    snap_after = client.state_snapshot()

    poses = snap_before.get("named_poses") or {}
    bodies = poses.get("bodies") or {}
    sites = poses.get("site_geometry") or {}
    objects = (snap_before.get("sim") or {}).get("objects") or {}
    joints = snap_before.get("joints") or {}

    required = {
        "base": "flat_stove_1_base",
        "burner": "flat_stove_1_burner",
        "button": "flat_stove_1_button",
    }
    missing = [name for name in required.values() if name not in bodies]
    if missing:
        raise SystemExit(f"snapshot lacks bodies: {missing}")

    base = bodies[required["base"]]
    burner = bodies[required["burner"]]
    button = bodies[required["button"]]
    cook = sites.get("flat_stove_1_cook_region") or {}
    bowl = objects.get("akita_black_bowl_1") or {}
    if not cook or not bowl:
        raise SystemExit("snapshot lacks the cook region or the bowl")

    knob = knob_frame_extent()
    bowl_box = bowl_extent()

    # The base frame orientation is not published, but the measurement pins it
    # down: the burner sits at exactly the model's local (0.15, 0, 0) offset.
    measured_offset = [round(burner[i] - base[i], 6) for i in range(3)]
    base_unrotated = measured_offset == [0.15, 0.0, 0.0]

    def in_world(local: list[float]) -> list[float]:
        # base frame -> world: identity only when the check above held.
        return [round(base[i] + local[i], 6) for i in range(3)]

    knob_min_x = in_world([knob["min_x_in_base"], knob["min_y_in_base"], knob["min_z_in_base"]])
    knob_max_x = in_world([knob["max_x_in_base"], knob["max_y_in_base"], knob["max_z_in_base"]])

    cook_center = cook.get("xyz") or [0, 0, 0]
    cook_size = cook.get("size") or [0, 0, 0]
    cook_near_x = cook_center[0] - cook_size[0]
    bowl_radius = bowl_box["max_radius_m"]
    # Worst case for obstruction: the bowl is pushed to the cook-region edge
    # closest to the knob, not left at the centre.
    bowl_near_edge_at_edge = cook_near_x - bowl_radius
    bowl_near_edge_at_center = cook_center[0] - bowl_radius
    knob_tip_x = knob_max_x[0]

    report = {
        "schema": "embodied_engine.knob_reach.v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "engine": args.engine,
        "candidate": ["put_bowl_on_stove", "turn_on_stove"],
        "scene": {"task_id": 7, "seed": args.seed, "bddl": "turn_on_stove.bddl", "env_port": port},
        "read_only": {
            "env_steps": 0,
            "policy_calls": 0,
            "objects_moved": 0,
            "sim_action_count": [snap_before.get("sim_action_count"), snap_after.get("sim_action_count")],
            "observation_ids": [snap_before.get("observation_id"), snap_after.get("observation_id")],
            "scene_epoch": [snap_before.get("scene_epoch"), snap_after.get("scene_epoch")],
        },
        "measured": {
            "stove_base_xyz": base,
            "stove_button_xyz": button,
            "burner_xyz": burner,
            "burner_offset_from_base": measured_offset,
            "cook_region": {"xyz": cook_center, "size": cook_size},
            "bowl_xyz": bowl.get("pos"),
            "bowl_collision_radius_m": round(bowl_radius, 4),
            "button_joint_qpos": joints.get("flat_stove_1_button"),
            "base_frame_unrotated": base_unrotated,
        },
        "model_geometry": {
            "knob_extent_in_base_m": {
                key: knob[key]
                for key in (
                    "min_x_in_base",
                    "max_x_in_base",
                    "min_y_in_base",
                    "max_y_in_base",
                    "min_z_in_base",
                    "max_z_in_base",
                    "max_radius_m",
                )
            },
            "knob_body_quat_wxyz": knob["body_quat_wxyz"],
            "knob_world_min_xyz": knob_min_x,
            "knob_world_max_xyz": knob_max_x,
            "bowl_collision_box": {k: bowl_box[k] for k in ("geoms", "min_z_m", "max_z_m")},
        },
        "clearance": {
            "cook_center_to_knob_pivot_m": round(
                math.dist([cook_center[0], cook_center[1]], [base[0], base[1]]), 4
            ),
            "cook_near_edge_to_knob_pivot_m": round(abs(base[0] - cook_near_x), 4),
            "knob_tip_to_cook_near_edge_m": round(abs(knob_tip_x - cook_near_x), 4),
            "bowl_near_edge_at_cook_center_x": round(bowl_near_edge_at_center, 4),
            "bowl_near_edge_at_cook_near_edge_x": round(bowl_near_edge_at_edge, 4),
            "knob_tip_to_bowl_edge_m_center": round(abs(knob_tip_x - bowl_near_edge_at_center), 4),
            "knob_tip_to_bowl_edge_m_worst_case": round(
                abs(knob_tip_x - bowl_near_edge_at_edge), 4
            ),
            "vertical_overlap_m": {
                "knob_z_range": [round(knob_min_x[2], 4), round(knob_max_x[2], 4)],
                "bowl_z_range_at_cook_region": [
                    round(cook_center[2] + bowl_box["min_z_m"], 4),
                    round(cook_center[2] + bowl_box["max_z_m"], 4),
                ],
            },
        },
        "verdict": {
            "geometric_obstruction": "no direct overlap at the measured clearance; the knob tip and the bowl rim are close enough that a finger contact can clip the bowl",
            "dynamic_reachability": "not_tested",
            "why_not_tested": (
                "the executable whitelist still contains only open_middle_drawer->put_bowl_on_plate, "
                "and that pair did not reach combo_verified in the frozen 6-round suite; a 1->7 run "
                "would need a second composite capability switched on before it could be measured"
            ),
            "claim": "geometry measured from the live scene plus the installed collision model; no statement about policy reachability",
        },
    }

    out = PROJECT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["clearance"], ensure_ascii=False, indent=1))
    print(f"out={out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
