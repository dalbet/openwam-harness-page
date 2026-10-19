"""Read one bound instance from the current observation.

Movable objects use the exact sim object key. A fixture target uses the BDDL
goal-atom site, never the facility body origin and never a trimmed name.
"""

from __future__ import annotations

from typing import Any


QUERY_CODES = frozenset({
    "instance_not_found",
    "pose_not_provided",
    "stale_observation",
    "state_service_error",
    "visual_instance_ambiguous",
})


def query_bound_state(
    instance_ref: str,
    spec: dict[str, Any] | None,
    observation: dict[str, Any] | None,
    *,
    scene_epoch: int | None,
    role: str,
) -> dict[str, Any]:
    """One instance, one query key. A catalog name is not a pose."""
    if not isinstance(observation, dict):
        return _fail("state_service_error", instance_ref, role, "观测不是对象")
    if scene_epoch is not None and observation.get("scene_epoch") not in (None, scene_epoch):
        return _fail(
            "stale_observation",
            instance_ref,
            role,
            f"观测 scene_epoch={observation.get('scene_epoch')}，计划 scene_epoch={scene_epoch}",
            observation=observation,
        )
    if not isinstance(instance_ref, str) or not instance_ref:
        return _fail("instance_not_found", instance_ref, role, "没有 instance_ref")
    objects = ((observation.get("sim") or {}).get("objects") or {})
    if not isinstance(spec, dict):
        if instance_ref not in objects:
            return _fail("instance_not_found", instance_ref, role, "场景里没有这个实例规格")
        spec = {"sim_name": instance_ref, "movable": True}
    if spec.get("semantic_type") == instance_ref and spec.get("sim_name") not in (None, instance_ref):
        return _fail("instance_not_found", instance_ref, role, "类型名不能当作实例")
    named = observation.get("named_poses")
    if isinstance(named, dict) and named.get("error") and spec.get("place_site"):
        return _fail("state_service_error", instance_ref, role, str(named.get("error")), query_key=spec.get("place_site"), observation=observation)
    place_site = spec.get("place_site")
    if isinstance(place_site, str) and place_site:
        sites = named.get("sites") if isinstance(named, dict) else None
        xyz = _xyz(sites.get(place_site) if isinstance(sites, dict) else None)
        if xyz is None:
            return _fail(
                "pose_not_provided",
                instance_ref,
                role,
                f"放置区域 {place_site} 没有位姿。不会改用设施 body 原点。",
                query_key=place_site,
                observation=observation,
            )
        return _ok(instance_ref, role, place_site, xyz, "mujoco_site", observation)
    query_key = spec.get("sim_name") if isinstance(spec.get("sim_name"), str) and spec.get("sim_name") else instance_ref
    item = objects.get(query_key)
    if not isinstance(item, dict):
        return _fail("pose_not_provided", instance_ref, role, "仿真状态没有这个实例的位姿", query_key=query_key, observation=observation)
    if item.get("identified") is False:
        return _fail("pose_not_provided", instance_ref, role, "实例未标识", query_key=query_key, observation=observation)
    xyz = _xyz(item.get("pos"))
    if xyz is None:
        return _fail("pose_not_provided", instance_ref, role, "位姿字段缺失", query_key=query_key, observation=observation)
    return _ok(instance_ref, role, query_key, xyz, "sim_state", observation)


def goal_evidence(
    predicates: list[dict[str, Any]],
    observation: dict[str, Any] | None,
    objects: dict[str, Any] | None,
    *,
    scene_epoch: int | None,
) -> dict[str, Any]:
    """Current poses for the final conditions. A placed step is not reused."""
    rows = []
    sim_objects: dict[str, Any] = {}
    missing = []
    for predicate in predicates or []:
        if not isinstance(predicate, dict):
            continue
        for role, ref in (("object", predicate.get("object_ref")), ("target", predicate.get("target_ref"))):
            if not isinstance(ref, str) or ref in sim_objects or ref in missing:
                continue
            spec = (objects or {}).get(ref)
            found = query_bound_state(ref, spec if isinstance(spec, dict) else None, observation, scene_epoch=scene_epoch, role=role)
            rows.append(found)
            if found["status"] == "ok":
                sim_objects[ref] = {
                    "pos": found["xyz"],
                    "identified": True,
                    "query_key": found["query_key"],
                    "evidence_source": found["evidence_source"],
                }
            else:
                missing.append(found)
    if missing:
        return {
            "status": "unknown",
            "reason_code": missing[0]["reason_code"],
            "reason": missing[0]["reason"],
            "evidence_source": missing[0]["reason_code"],
            "queries": rows,
            "sim": None,
        }
    return {
        "status": "ready",
        "reason_code": "",
        "reason": "",
        "evidence_source": "sim_state" if not any(item["evidence_source"] == "mujoco_site" for item in rows) else "mujoco_site",
        "queries": rows,
        "sim": {"objects": sim_objects},
    }


def _ok(instance_ref, role, query_key, xyz, evidence, observation) -> dict[str, Any]:
    return {
        "status": "ok",
        "reason_code": "",
        "reason": "",
        "instance_ref": instance_ref,
        "role": role,
        "query_key": query_key,
        "xyz": xyz,
        "evidence_source": evidence,
        "sampled_at": (observation or {}).get("timestamp"),
        "observation_id": (observation or {}).get("observation_id"),
        "scene_epoch": (observation or {}).get("scene_epoch"),
    }


def _fail(code, instance_ref, role, reason, *, query_key=None, observation=None) -> dict[str, Any]:
    return {
        "status": "error",
        "reason_code": code,
        "reason": reason,
        "instance_ref": instance_ref,
        "role": role,
        "query_key": query_key,
        "xyz": None,
        "evidence_source": code,
        "sampled_at": (observation or {}).get("timestamp") if isinstance(observation, dict) else None,
        "observation_id": (observation or {}).get("observation_id") if isinstance(observation, dict) else None,
        "scene_epoch": (observation or {}).get("scene_epoch") if isinstance(observation, dict) else None,
    }


def _xyz(value) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None
    try:
        return [float(value[0]), float(value[1]), float(value[2])]
    except (TypeError, ValueError):
        return None
