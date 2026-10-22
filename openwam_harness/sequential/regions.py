"""Scene region registry.

A region id is owner_instance + region_name. The bare name top_region is not
unique: the cabinet and the wine rack both declare one. Table ranges initialize
fixtures. They are not the fixture's placement site.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Any


def parse_bddl_regions(text: str) -> list[dict[str, Any]]:
    """Every region declaration. Owners missing from this problem stay marked."""
    fixtures = _names(text, "fixtures")
    objects = _names(text, "objects")
    present = fixtures | objects
    goals = set(_goal_atoms(text)) | set(_names(text, "obj_of_interest"))
    rows = []
    for item in _declarations(text):
        owner = item["owner_instance"]
        region_id = f"{owner}_{item['region_name']}"
        relation = "init_support" if item["has_ranges"] else "fixture_site"
        if owner not in present:
            status = "owner_not_in_scene"
        else:
            status = "declared"
        rows.append(
            {
                "region_id": region_id,
                "owner_instance": owner,
                "region_name": item["region_name"],
                "query_key": region_id,
                "semantic_relation": relation,
                "geometry_type": "table_range" if item["has_ranges"] else "mujoco_site",
                "status": status,
                "native_goal": region_id in goals,
                "motion_class": _motion_class(item["region_name"], owner),
                "source": "bddl_regions",
            }
        )
    return rows


def registry_from_text(text: str) -> dict[str, Any]:
    rows = parse_bddl_regions(text)
    version = hashlib.sha256(
        "\n".join(f"{row['region_id']}:{row['semantic_relation']}:{row['status']}" for row in rows).encode("utf-8")
    ).hexdigest()[:16]
    return {"regions": rows, "version": version}


def place_regions(text: str) -> list[dict[str, Any]]:
    """Fixture sites that exist in this problem. Init ranges are not included."""
    rows = []
    for item in parse_bddl_regions(text):
        if item["status"] != "declared" or item["semantic_relation"] != "fixture_site":
            continue
        rows.append(
            {
                "region": item["region_name"],
                "target": item["owner_instance"],
                "site_name": item["query_key"],
                "region_id": item["region_id"],
                "semantic_relation": item["semantic_relation"],
                "native_goal": item["native_goal"],
            }
        )
    return rows


def attach_place_site(spec: dict[str, Any], regions: list[dict[str, Any]]) -> None:
    """One operational site can be bound. Several stay listed and unresolved."""
    owned = [item for item in regions if item.get("target") == spec.get("name") or item.get("target") == spec.get("sim_name")]
    if len(owned) == 1:
        spec["place_site"] = owned[0]["site_name"]
        spec["place_region"] = owned[0]["region"]
        spec["place_region_id"] = owned[0]["region_id"]
        spec["place_site_source"] = "scene_fixture_region"
        spec["native_goal_region"] = bool(owned[0].get("native_goal"))
    elif len(owned) > 1:
        spec["place_sites"] = [item["site_name"] for item in owned]
        spec["place_site_source"] = "scene_fixture_region"


_RIGID_POSES: dict[tuple, dict[str, Any]] = {}


def remember_pose(scene_epoch, region_id: str, owner_xyz, pose: dict[str, Any]) -> None:
    """Rigid site pose for one scene epoch and one fixture transform. Definitions stay in the registry."""
    _RIGID_POSES[(scene_epoch, region_id)] = {
        "owner_xyz": None if owner_xyz is None else [round(float(v), 6) for v in owner_xyz[:3]],
        "pose": dict(pose),
    }


def recall_pose(scene_epoch, region_id: str, owner_xyz) -> dict[str, Any] | None:
    row = _RIGID_POSES.get((scene_epoch, region_id))
    if row is None:
        return None
    current = None if owner_xyz is None else [round(float(v), 6) for v in owner_xyz[:3]]
    if row["owner_xyz"] != current:
        return None
    return dict(row["pose"])


def clear_pose_cache() -> None:
    _RIGID_POSES.clear()


def query_region(region_id: str, observation: dict[str, Any] | None, registry: dict[str, Any] | None) -> dict[str, Any]:
    """One entry for transport, placement, and the final check."""
    rows = {item["region_id"]: item for item in (registry or {}).get("regions") or []}
    declared = rows.get(region_id)
    if declared is None:
        return _error("region_not_declared", region_id, "场景注册表没有这个区域")
    if declared["status"] != "declared":
        return _error("region_not_declared", region_id, declared["status"])
    if declared["semantic_relation"] != "fixture_site":
        return _error("region_not_placeable", region_id, "这是设施初始化区域，不是落点")
    if not isinstance(observation, dict):
        return _error("state_service_error", region_id, "没有当前观测")
    named = observation.get("named_poses")
    if isinstance(named, dict) and named.get("error"):
        return _error("state_service_error", region_id, str(named.get("error")))
    sites = named.get("sites") if isinstance(named, dict) else None
    xyz = _xyz(sites.get(region_id) if isinstance(sites, dict) else None)
    geometry = {}
    if isinstance(named, dict):
        geometry = (named.get("site_geometry") or {}).get(region_id) or {}
    quat = geometry.get("quat_wxyz")
    size = geometry.get("size")
    owner_xyz = None
    bodies = named.get("bodies") if isinstance(named, dict) else None
    if isinstance(bodies, dict):
        owner_xyz = _xyz(bodies.get(declared["owner_instance"]))
    epoch = observation.get("scene_epoch")
    if xyz is None:
        cached = recall_pose(epoch, region_id, owner_xyz)
        if cached is None or declared.get("motion_class") == "articulated":
            return _error("region_site_missing", region_id, f"当前状态没有 site {region_id}")
        cached = dict(cached)
        cached["source"] = "cached_rigid_site"
        cached["registry_version"] = (registry or {}).get("version")
        return cached
    if isinstance(quat, list) and declared.get("motion_class") != "articulated":
        remember_pose(epoch, region_id, owner_xyz, {
            "status": "ok",
            "reason_code": "",
            "region_id": region_id,
            "owner_instance": declared["owner_instance"],
            "query_key": region_id,
            "semantic_relation": declared["semantic_relation"],
            "geometry_type": declared["geometry_type"],
            "motion_class": declared.get("motion_class"),
            "xyz": xyz,
            "quat_wxyz": quat,
            "size": size if isinstance(size, list) else None,
            "native_goal": declared["native_goal"],
            "source": "mujoco_site",
            "observation_id": observation.get("observation_id"),
            "scene_epoch": epoch,
            "step_clock": observation.get("step_clock"),
            "env_step": observation.get("env_step"),
        })
    return {
        "status": "ok",
        "reason_code": "",
        "region_id": region_id,
        "owner_instance": declared["owner_instance"],
        "query_key": region_id,
        "semantic_relation": declared["semantic_relation"],
        "geometry_type": declared["geometry_type"],
        "motion_class": declared.get("motion_class"),
        "xyz": xyz,
        "quat_wxyz": quat if isinstance(quat, list) else None,
        "size": size if isinstance(size, list) else None,
        "native_goal": declared["native_goal"],
        "source": "mujoco_site",
        "registry_version": (registry or {}).get("version"),
        "observation_id": observation.get("observation_id"),
        "scene_epoch": observation.get("scene_epoch"),
        "step_clock": observation.get("step_clock"),
        "env_step": observation.get("env_step"),
    }


def placement_mode(region: dict[str, Any]) -> dict[str, Any]:
    """Position servo is enough when the support normal stays vertical.

    A yaw about world z, including the cabinet's init yaw of pi, does not tilt
    ``top_side``. Tilt is ``acos(|n · z|)`` with ``n = R @ [0, 0, 1]`` from the
    runtime site matrix. A non-unit quaternion is not itself a tilt.
    """
    normal = _support_normal(region.get("mat"), region.get("quat_wxyz"))
    quat = region.get("quat_wxyz")
    if normal is None:
        return {
            "ok": True,
            "mode": "translation_unmeasured",
            "reason": "这次观测没有 site 朝向，只记录位置",
        }
    tilt = math.acos(max(-1.0, min(1.0, abs(normal[2]))))
    measured = {
        "normal_world": [round(value, 6) for value in normal],
        "tilt_rad": round(tilt, 6),
        "quat_wxyz": [round(float(value), 6) for value in quat[:4]] if isinstance(quat, list) and len(quat) >= 4 else None,
    }
    if tilt <= math.radians(5.0):
        return {"ok": True, "mode": "translation", **measured}
    return {
        "ok": False,
        "mode": "tilted_site",
        "code": "orientation_not_commandable",
        "reason": (
            f"支撑面法向相对竖直轴倾斜 {round(math.degrees(tilt), 2)}°。"
            "绕竖直轴的偏航不在此列。当前 move_to 只做位置伺服，不能放下倾斜面。"
        ),
        "size": region.get("size"),
        "xyz": region.get("xyz"),
        "missing_implementation": "把倾斜支撑面接到带完成检查和取消的末端姿态命令",
        **measured,
    }


def _support_normal(mat, quat) -> list[float] | None:
    """World direction of the site's local +z. Row-major ``site_xmat`` wins."""
    if isinstance(mat, list) and len(mat) >= 9:
        try:
            normal = [float(mat[2]), float(mat[5]), float(mat[8])]
        except (TypeError, ValueError):
            normal = []
        if len(normal) == 3 and math.sqrt(sum(value * value for value in normal)) > 1e-6:
            return normal
    if not isinstance(quat, list) or len(quat) < 4:
        return None
    try:
        w, x, y, z = (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))
    except (TypeError, ValueError):
        return None
    return [
        2.0 * (x * z + w * y),
        2.0 * (y * z - w * x),
        1.0 - 2.0 * (x * x + y * y),
    ]


def on_site(object_pos, site_pos, site_mat, site_size, *, other_height: float = 0.10) -> dict[str, Any]:
    """LIBERO SiteObject.under. site_mat is the 3x3 site_xmat, row-major."""
    delta = _matvec(site_mat, [object_pos[i] - site_pos[i] for i in range(3)])
    size = [float(site_size[0]), float(site_size[1]), float(site_size[2])]
    inside_xy = abs(delta[0]) < size[0] and abs(delta[1]) < size[1]
    supported = size[2] - 0.005 < delta[2] < size[2] + other_height
    return {
        "ok": inside_xy and supported,
        "delta_site": [round(v, 4) for v in delta],
        "inside_xy": inside_xy,
        "supported": supported,
        "predicate": "SiteObject.under",
    }


def _motion_class(region_name: str, owner: str) -> str:
    text = f"{owner} {region_name}"
    if any(token in text for token in ("drawer", "door", "knob", "handle", "joint")):
        return "articulated"
    if "cabinet" in owner and region_name in {"top_region", "middle_region", "bottom_region"}:
        return "articulated"
    return "rigid"


def _declarations(text: str) -> list[dict[str, Any]]:
    start = text.find("(:regions")
    end = text.find("(:fixtures")
    if start < 0 or end < 0 or end <= start:
        return []
    rows = []
    current = None
    has_ranges = False
    target_name = None

    def flush() -> None:
        nonlocal current, has_ranges, target_name
        if current and target_name:
            rows.append({"region_name": current, "owner_instance": target_name, "has_ranges": has_ranges})
        current = None
        has_ranges = False
        target_name = None

    for line in text[start:end].splitlines():
        header = re.match(r"\s*\(([A-Za-z0-9_]+)\s*$", line)
        if header and header.group(1) != "regions":
            flush()
            current = header.group(1)
            continue
        if current and "(:ranges" in line:
            has_ranges = True
        target = re.search(r"\(:target\s+([A-Za-z0-9_]+)\)", line)
        if current and target:
            target_name = target.group(1)
    flush()
    return rows


def _names(text: str, section: str) -> set[str]:
    match = re.search(rf"\(:{section}\b(.*?)\)\s*\n", text, re.S)
    if not match:
        return set()
    found = set()
    for line in match.group(1).splitlines():
        if " - " in line:
            raw, _semantic = line.split(" - ", 1)
            found.update(raw.split())
        else:
            found.update(re.findall(r"[A-Za-z0-9_]+", line))
    return found


def _goal_atoms(text: str) -> list[str]:
    match = re.search(r"\(:goal\b(.*)\)\s*$", text, re.S)
    if not match:
        return []
    body = match.group(1)
    return re.findall(r"\((?:On|In)\s+\S+\s+([A-Za-z0-9_]+)\)", body) + re.findall(r"\(Open\s+([A-Za-z0-9_]+)\)", body)


def _xyz(value) -> list[float] | None:
    if isinstance(value, dict):
        value = value.get("xyz")
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None
    try:
        return [float(value[0]), float(value[1]), float(value[2])]
    except (TypeError, ValueError):
        return None


def _matvec(mat, vec) -> list[float]:
    if not isinstance(mat, list) or len(mat) < 9:
        return [float(vec[0]), float(vec[1]), float(vec[2])]
    return [
        mat[0] * vec[0] + mat[1] * vec[1] + mat[2] * vec[2],
        mat[3] * vec[0] + mat[4] * vec[1] + mat[5] * vec[2],
        mat[6] * vec[0] + mat[7] * vec[1] + mat[8] * vec[2],
    ]


def _error(code: str, region_id: str, reason: str) -> dict[str, Any]:
    return {
        "status": "error",
        "reason_code": code,
        "reason": reason,
        "region_id": region_id,
        "xyz": None,
        "quat_wxyz": None,
        "size": None,
        "source": code,
    }


def mat_to_quat_wxyz(mat: list[float]) -> list[float]:
    """Row-major 3x3 to MuJoCo wxyz. Shepperd's method."""
    m = [float(v) for v in mat[:9]]
    trace = m[0] + m[4] + m[8]
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2
        return [0.25 * s, (m[7] - m[5]) / s, (m[2] - m[6]) / s, (m[3] - m[1]) / s]
    if m[0] > m[4] and m[0] > m[8]:
        s = math.sqrt(1.0 + m[0] - m[4] - m[8]) * 2
        return [(m[7] - m[5]) / s, 0.25 * s, (m[1] + m[3]) / s, (m[2] + m[6]) / s]
    if m[4] > m[8]:
        s = math.sqrt(1.0 + m[4] - m[0] - m[8]) * 2
        return [(m[2] - m[6]) / s, (m[1] + m[3]) / s, 0.25 * s, (m[5] + m[7]) / s]
    s = math.sqrt(1.0 + m[8] - m[0] - m[4]) * 2
    return [(m[3] - m[1]) / s, (m[2] + m[6]) / s, (m[5] + m[7]) / s, 0.25 * s]
