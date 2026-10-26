"""Authoritative snapshot identity.

``sim_action_count`` is the env server's count of applied simulation actions.
``observation_id`` identifies one read. It is not a physical version.
A tool-record index and a display frame number are never that count.
"""

from __future__ import annotations

from typing import Any


OBSERVATION_ID_MEANING = "readonly_or_action_query"
PHYSICAL_VERSION_FIELD = "sim_action_count"


def query_id(service_instance_id: str, scene_epoch: int, sim_action_count: int, query_seq: int) -> str:
    return f"query-{service_instance_id}:{scene_epoch}:{sim_action_count}:{query_seq}"


def stamp_snapshot(
    payload: dict[str, Any],
    *,
    service_instance_id: str,
    scene_epoch: int,
    scene_generation: int,
    sim_action_count: int,
    query_seq: int,
    terminated: bool,
    truncated: bool,
    source: str,
) -> dict[str, Any]:
    """One read. ``source`` says whether the caller is the action return or a later query."""
    row = dict(payload)
    row["service_instance_id"] = service_instance_id
    row["scene_epoch"] = scene_epoch
    row["scene_generation"] = scene_generation
    row["sim_action_count"] = int(sim_action_count)
    row["query_seq"] = int(query_seq)
    row["observation_id"] = query_id(service_instance_id, scene_epoch, sim_action_count, query_seq)
    row["observation_id_meaning"] = OBSERVATION_ID_MEANING
    row["step_clock"] = "sim_action"
    row["terminated"] = bool(terminated)
    row["truncated"] = bool(truncated)
    row["state_source"] = source
    return row


def _count(obs: dict | None) -> int | None:
    if not isinstance(obs, dict):
        return None
    value = obs.get("sim_action_count")
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def post_action_freshness(before: dict | None, after: dict | None, *, completed_action_count: int | None) -> dict[str, Any]:
    """Decide whether ``after`` covers the action that just finished.

    A repeated read of the same ``sim_action_count`` is current state, not a stale
    observation. It still cannot pretend to be a second physical sample.
    """
    if not isinstance(before, dict):
        return _bad("before_observation_missing", "before_observation", None, None, "动作前快照缺失")
    if not isinstance(after, dict):
        return _bad("after_observation_missing", "after_observation", None, None, "动作后快照缺失")
    left_service = before.get("service_instance_id")
    right_service = after.get("service_instance_id")
    if left_service and right_service and left_service != right_service:
        return _bad(
            "cross_service_instance",
            "service_instance_id",
            left_service,
            right_service,
            "快照属于另一个服务实例",
        )
    if before.get("scene_epoch") != after.get("scene_epoch"):
        return _bad(
            "cross_scene",
            "scene_epoch",
            before.get("scene_epoch"),
            after.get("scene_epoch"),
            "快照属于另一个场景代次",
        )
    left_gen = before.get("scene_generation")
    right_gen = after.get("scene_generation")
    if left_gen is not None and right_gen is not None and left_gen != right_gen:
        return _bad("cross_reset", "scene_generation", left_gen, right_gen, "快照属于另一次 reset")
    left = _count(before)
    right = _count(after)
    if left is None:
        return _bad("sim_action_count_missing", "before.sim_action_count", None, right, "动作前快照没有仿真动作计数")
    if right is None:
        return _bad(
            "sim_action_count_missing",
            "after.sim_action_count",
            left,
            after.get("env_step"),
            "动作后快照没有仿真动作计数。工具记录序号不能代替它",
        )
    need = left if completed_action_count is None else completed_action_count
    if not isinstance(need, int):
        return _bad("sim_action_count_missing", "completed_action_count", left, right, "动作完成计数缺失")
    if right < need:
        return _bad(
            "pre_action_state",
            "sim_action_count",
            need,
            right,
            "检查用的状态早于该动作实际完成的仿真计数",
        )
    return {
        "ok": True,
        "reason_code": "",
        "field": "sim_action_count",
        "expected": need,
        "received": right,
        "same_physical_state": right == left,
        "summary": "",
    }


def same_physical_state(left: dict | None, right: dict | None) -> bool:
    if _count(left) is None or _count(right) is None:
        return False
    if left.get("service_instance_id") and right.get("service_instance_id"):
        if left.get("service_instance_id") != right.get("service_instance_id"):
            return False
    if left.get("scene_epoch") != right.get("scene_epoch"):
        return False
    if left.get("scene_generation") is not None and right.get("scene_generation") is not None:
        if left.get("scene_generation") != right.get("scene_generation"):
            return False
    return _count(left) == _count(right)


def distinct_physical_samples(samples: list[dict]) -> list[dict]:
    """Drop a second row that only reread the same simulation count."""
    kept = []
    seen = set()
    for sample in samples:
        count = _count(sample)
        if count is None:
            continue
        key = (sample.get("service_instance_id"), sample.get("scene_epoch"), sample.get("scene_generation"), count)
        if key in seen:
            continue
        seen.add(key)
        kept.append(sample)
    return kept


def static_region_valid(cached: dict | None, current: dict | None) -> dict[str, Any]:
    """A rigid site stays valid when only the action count moves.

    Drawers, knobs, and regions that ride on them are not fixed caches.
    """
    if not isinstance(cached, dict) or not isinstance(current, dict):
        return {"ok": False, "reason_code": "region_cache_missing", "reason": "没有可比较的区域缓存"}
    if cached.get("motion_class") == "articulated" or current.get("motion_class") == "articulated":
        return {"ok": False, "reason_code": "articulated_region", "reason": "抽屉、旋钮及其附着区域不能当固定区域缓存"}
    if cached.get("service_instance_id") and current.get("service_instance_id"):
        if cached.get("service_instance_id") != current.get("service_instance_id"):
            return {"ok": False, "reason_code": "cross_service_instance", "reason": "区域缓存属于另一个服务实例"}
    if cached.get("scene_epoch") != current.get("scene_epoch"):
        return {"ok": False, "reason_code": "cross_scene", "reason": "区域缓存属于另一个场景代次"}
    if cached.get("owner_xyz") != current.get("owner_xyz"):
        return {"ok": False, "reason_code": "facility_moved", "reason": "设施位姿变了，静态区域失效"}
    return {"ok": True, "reason_code": "", "reason": ""}


def _bad(code: str, field: str, expected, received, summary: str) -> dict[str, Any]:
    return {
        "ok": False,
        "reason_code": code,
        "field": field,
        "expected": expected,
        "received": received,
        "same_physical_state": False,
        "summary": summary,
    }
