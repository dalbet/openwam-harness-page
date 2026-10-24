"""Middle-drawer opener. It reuses ``_vlm_chunk`` and does not call ``env.step`` itself.

Success is ``WoodenCabinet.is_open`` on the joint attached to the current
instance's middle region. ``pi0_doubled`` is not called, and its termination
flag is not this skill's success label.
"""

from __future__ import annotations

import math
import re
import time
from pathlib import Path
from typing import Any

from site_paths import site

ASSET = site("libero", "libero", "assets", "articulated_objects", "wooden_cabinet.xml")
BDDL = site(
    "liberopro",
    "liberopro",
    "bddl_files",
    "libero_goal",
    "open_the_middle_drawer_of_the_cabinet.bddl",
)
MAX_CHUNKS = 8
UPSTREAM_CHUNK_BUDGET = 20
MAX_S = 45.0
PROGRESS_EPS_M = 0.002


def official_instruction() -> str:
    from task_catalog import cached_catalog

    for row in cached_catalog():
        if row.get("task_key") == "libero_goal:t0":
            return str(row["instruction_en"])
    raise RuntimeError("任务注册表没有 libero_goal:t0")


def official_region() -> str:
    text = BDDL.read_text(encoding="utf-8")
    match = re.search(r"\(Open\s+([A-Za-z0-9_]+)\)", text)
    if match is None:
        raise RuntimeError("中层抽屉 BDDL 没有 Open 目标")
    return match.group(1)


def asset_joint() -> dict[str, Any]:
    """Local joint tag on the middle body. The sim name is this name after the instance prefix."""
    text = ASSET.read_text(encoding="utf-8")
    match = re.search(
        r'<joint name="middle_level"[^>]*axis="([^"]+)"[^>]*range="([^"]+)"',
        text,
        re.S,
    )
    if match is None:
        raise RuntimeError("wooden_cabinet.xml 没有 middle_level 关节")
    return {
        "local_name": "middle_level",
        "axis": match.group(1),
        "range": match.group(2),
        "source": str(ASSET),
    }


def read_joint(env: Any, region: str, instance: str | None = None) -> dict[str, Any]:
    """Same joint lookup as ``SiteObjectState.is_open``. The name comes from the site, not a guess.

    The live backend's env is an RPC client without ``object_sites_dict``;
    there the joint qpos rides the existing readonly ``env.state_snapshot``.
    """
    sites = getattr(env, "object_sites_dict", None)
    if isinstance(sites, dict):
        return _read_joint_local(env, region, sites)
    return _read_joint_rpc(env, region, instance)


def _read_joint_local(env: Any, region: str, sites: dict) -> dict[str, Any]:
    site = sites.get(region)
    if site is None:
        return {"ok": False, "reason": f"当前场景没有区域 {region}", "code": "region_missing"}
    joints = [item for item in (getattr(site, "joints", None) or []) if isinstance(item, str) and item]
    if len(joints) != 1:
        return {
            "ok": False,
            "reason": f"{region} 的关节不是一个：{joints}",
            "code": "joint_unresolved",
            "joints": joints,
        }
    parent_name = getattr(site, "parent_name", None)
    parent = env.get_object(parent_name) if parent_name else None
    if parent is None or not hasattr(parent, "is_open"):
        return {"ok": False, "reason": f"{region} 没有可判断开合的父物体", "code": "joint_unresolved"}
    joint = joints[0]
    try:
        address = env.sim.model.get_joint_qpos_addr(joint)
        qpos = float(env.sim.data.qpos[address])
    except Exception as exc:
        return {"ok": False, "reason": f"读不到关节 {joint}：{exc}", "code": "joint_unreadable"}
    return _joint_result(region, parent_name, joint, qpos, parent)


def _region_instance(region: str) -> str | None:
    text = str(region or "")
    suffix = "_middle_region"
    if text.endswith(suffix) and len(text) > len(suffix):
        return text[: -len(suffix)]
    return None


def _parent_object(instance: str) -> Any:
    """Build the articulated object class locally. ``is_open`` needs no sim."""
    from libero.libero.envs.objects import get_object_fn

    base = re.sub(r"_\d+$", "", str(instance or ""))
    if not base:
        return None
    try:
        factory = get_object_fn(base)
    except Exception:
        return None
    try:
        return factory(str(instance))
    except Exception:
        return None


def _read_joint_rpc(env: Any, region: str, instance: str | None) -> dict[str, Any]:
    name = instance if isinstance(instance, str) and instance else _region_instance(region)
    if not name:
        return {"ok": False, "reason": f"无法从区域 {region} 推出实例", "code": "region_missing"}
    client = getattr(env, "_client", None)
    if client is None or not callable(getattr(client, "call", None)):
        return {"ok": False, "reason": "env 既不是进程内环境，也没有状态快照客户端", "code": "state_service_error"}
    joint = f"{name}_{asset_joint()['local_name']}"
    try:
        snapshot = client.call("env.state_snapshot", timeout_s=10.0)
    except Exception as exc:
        return {"ok": False, "reason": f"状态快照读取失败：{type(exc).__name__}: {exc}", "code": "state_service_error"}
    joints = snapshot.get("joints") if isinstance(snapshot, dict) else None
    if not isinstance(joints, dict) or joint not in joints:
        return {"ok": False, "reason": f"状态快照没有关节 {joint}", "code": "joint_unreadable"}
    try:
        qpos = float(joints[joint])
    except (TypeError, ValueError):
        return {"ok": False, "reason": f"关节 {joint} 的读数不是数值", "code": "joint_unreadable"}
    parent = _parent_object(name)
    if parent is None or not hasattr(parent, "is_open"):
        return {"ok": False, "reason": f"{name} 没有可判断开合的对象类", "code": "joint_unresolved"}
    row = _joint_result(region, name, joint, qpos, parent)
    row["evidence_source"] = f"{type(parent).__name__}.is_open via env.state_snapshot"
    return row


def _joint_result(region: str, parent_name: str, joint: str, qpos: float, parent: Any) -> dict[str, Any]:
    ranges = ((getattr(parent, "object_properties", {}) or {}).get("articulation") or {}).get("default_open_ranges") or []
    threshold = max(float(item) for item in ranges) if ranges else None
    opened = bool(parent.is_open(qpos))
    comparison = "qpos < max(default_open_ranges)" if type(parent).__name__ == "WoodenCabinet" else "parent.is_open"
    return {
        "ok": True,
        "region": region,
        "parent": parent_name,
        "joint": joint,
        "qpos": qpos,
        "threshold": threshold,
        "open_ranges": [float(item) for item in ranges],
        "comparison": comparison,
        "opened": opened,
        "predicate": "Open",
        "evidence_source": f"{type(parent).__name__}.is_open",
        "asset": asset_joint(),
    }


def run_open_drawer(primitives: Any, args: dict[str, Any] | None = None) -> dict[str, Any]:
    """Finite chunk loop. Cancel and termination are checked before every chunk."""
    payload = dict(args or {})
    instruction = official_instruction()
    region = str(payload.get("region") or official_region())
    if not region.endswith("_middle_region"):
        return {
            "opened": False,
            "instruction": instruction,
            "region": region,
            "stop_reason": "region_not_middle",
            "reason": "本轮只打开当前柜子的中层抽屉",
            "chunks_used": 0,
            "max_chunks": MAX_CHUNKS,
            "pi0_success_used": False,
        }
    max_chunks = MAX_CHUNKS
    max_s = MAX_S
    env = primitives.env
    instance = payload.get("instance") if isinstance(payload.get("instance"), str) else None
    initial = read_joint(env, region, instance=instance)
    if not initial.get("ok"):
        return {
            "opened": False,
            "instruction": instruction,
            "region": region,
            "stop_reason": initial.get("code") or "joint_unresolved",
            "reason": initial.get("reason"),
            "chunks_used": 0,
            "max_chunks": max_chunks,
            "pi0_success_used": False,
        }
    if initial.get("opened"):
        return _evidence(instruction, region, initial, initial, 0, max_chunks, max_s, "already_open", trace=[])
    started = time.monotonic()
    current = initial
    chunks = 0
    stop = "chunk_budget"
    stage = None
    trace = []
    previous = initial.get("qpos")
    toward = _toward_open(initial.get("comparison"))
    saw_progress = False
    while chunks < UPSTREAM_CHUNK_BUDGET:
        primitives._check_cancelled()
        if getattr(env, "terminated", False) or getattr(env, "truncated", False):
            stop = "terminated" if env.terminated else "truncated"
            break
        now = time.monotonic()
        if now - started > max_s:
            stop = "time_budget"
            break
        if chunks >= max_chunks and not saw_progress:
            stop = "chunk_budget"
            break
        before_actions = getattr(primitives, "_env_action_count", None)
        returned = primitives._vlm_chunk(instruction)
        chunks += 1
        current = read_progress(env, region, instance=instance)
        progress = _qpos_progress(previous, current.get("qpos"), toward)
        row = _chunk_row(
            chunks,
            current,
            progress,
            _actions_applied(primitives, before_actions, returned),
            now - started,
            returned,
            env,
        )
        trace.append(row)
        if not current.get("ok"):
            stop = current.get("code") or "joint_unreadable"
            break
        if isinstance(returned, dict) and returned.get("cancelled"):
            stop = "cancelled"
            break
        if getattr(env, "terminated", False) or getattr(env, "truncated", False):
            stop = "terminated" if env.terminated else "truncated"
            break
        if current.get("opened"):
            stop = "predicate_open"
            break
        if progress is None:
            previous = current.get("qpos")
            continue
        if progress > PROGRESS_EPS_M:
            saw_progress = True
            previous = current.get("qpos")
            continue
        primitives._check_cancelled()
        stop = "zero_progress"
        stage = _stall_stage(trace)
        break
    else:
        if saw_progress and not current.get("opened"):
            stop = "upstream_budget"
            stage = "pulling_not_open"
    current = dict(current)
    current["stage"] = stage
    if stop == "chunk_budget":
        current["reason"] = f"动作块预算耗尽（{chunks}/{max_chunks}），关节没有向 Open 移动"
    elif stop == "upstream_budget":
        current["reason"] = f"关节仍在移动，但总预算耗尽（{chunks}/{UPSTREAM_CHUNK_BUDGET}），尚未满足 Open"
    elif stop == "zero_progress":
        current["reason"] = f"关节没有继续向 Open 移动（{chunks} 块后停止）：{stage}"
    elif stop == "time_budget":
        current["reason"] = f"时间预算耗尽（{max_s}s），关节仍未满足 Open"
    elif stop == "cancelled":
        current["reason"] = "动作块返回取消，已停止，不继续同一指令"
    return _evidence(instruction, region, initial, current, chunks, max_chunks, max_s, stop, trace=trace)


def read_progress(env: Any, region: str, instance: str | None = None) -> dict[str, Any]:
    """Joint predicate plus fields the snapshot already has. No invented handle point."""
    sites = getattr(env, "object_sites_dict", None)
    if isinstance(sites, dict):
        row = _read_joint_local(env, region, sites)
        row.update(_empty_context())
        return row
    name = instance if isinstance(instance, str) and instance else _region_instance(region)
    if not name:
        return {"ok": False, "reason": f"无法从区域 {region} 推出实例", "code": "region_missing"}
    client = getattr(env, "_client", None)
    if client is None or not callable(getattr(client, "call", None)):
        return {"ok": False, "reason": "env 既不是进程内环境，也没有状态快照客户端", "code": "state_service_error"}
    joint = f"{name}_{asset_joint()['local_name']}"
    try:
        snapshot = client.call("env.state_snapshot", timeout_s=10.0)
    except Exception as exc:
        return {"ok": False, "reason": f"状态快照读取失败：{type(exc).__name__}: {exc}", "code": "state_service_error"}
    if not isinstance(snapshot, dict):
        snapshot = {}
    joints = snapshot.get("joints")
    if not isinstance(joints, dict) or joint not in joints:
        failed = {"ok": False, "reason": f"状态快照没有关节 {joint}", "code": "joint_unreadable"}
        failed.update(_snapshot_context(snapshot))
        return failed
    try:
        qpos = float(joints[joint])
    except (TypeError, ValueError):
        return {"ok": False, "reason": f"关节 {joint} 的读数不是数值", "code": "joint_unreadable"}
    parent = _parent_object(name)
    if parent is None or not hasattr(parent, "is_open"):
        return {"ok": False, "reason": f"{name} 没有可判断开合的对象类", "code": "joint_unresolved"}
    row = _joint_result(region, name, joint, qpos, parent)
    row["evidence_source"] = f"{type(parent).__name__}.is_open via env.state_snapshot"
    row.update(_snapshot_context(snapshot))
    return row


def _toward_open(comparison: str | None) -> float | None:
    """Sign of ``current - previous`` that counts as opening, from the predicate text."""
    text = str(comparison or "")
    if text.startswith("qpos <"):
        return -1.0
    if text.startswith("qpos >"):
        return 1.0
    return None


def _qpos_progress(previous, current, toward: float | None) -> float | None:
    if toward is None or isinstance(previous, bool) or isinstance(current, bool):
        return None
    if not isinstance(previous, (int, float)) or not isinstance(current, (int, float)):
        return None
    return toward * (float(current) - float(previous))


def _stall_stage(trace: list[dict[str, Any]]) -> str:
    earlier = [row.get("progress_m") for row in trace[:-1]]
    if any(isinstance(value, (int, float)) and value > PROGRESS_EPS_M for value in earlier):
        return "pull_stalled"
    if any(row.get("contact_pull") is True for row in trace):
        return "contacted_no_pull"
    return "no_joint_progress"


def _actions_applied(primitives: Any, before, returned) -> dict[str, Any]:
    after = getattr(primitives, "_env_action_count", None)
    if isinstance(before, int) and isinstance(after, int) and not isinstance(before, bool) and not isinstance(after, bool):
        return {
            "actions_applied": after - before,
            "action_count_status": "env_action_count",
            "env_actions_before": before,
            "env_actions_after": after,
        }
    if isinstance(returned, dict):
        for key in ("actions_applied", "actions_sent", "sim_action_count"):
            value = returned.get(key)
            if isinstance(value, int) and not isinstance(value, bool):
                return {"actions_applied": value, "action_count_status": key}
    return {"actions_applied": None, "action_count_status": "missing"}


def _empty_context() -> dict[str, Any]:
    return {
        "handle_status": "missing_handle_site",
        "handle_site": None,
        "handle_distance_m": None,
        "handle_xyz": None,
        "eef_pos": None,
        "eef_status": "not_in_local_site_table",
        "contact": None,
        "contact_status": "not_provided",
        "contact_pull": None,
        "sim_time_s": None,
        "sim_time_status": "not_provided",
    }


def _snapshot_context(snapshot: dict) -> dict[str, Any]:
    sim = snapshot.get("sim") if isinstance(snapshot.get("sim"), dict) else {}
    eef = sim.get("eef_pos")
    eef_pos = [float(eef[0]), float(eef[1]), float(eef[2])] if isinstance(eef, list) and len(eef) >= 3 else None
    contact = sim.get("contact") if isinstance(sim.get("contact"), dict) else None
    named = snapshot.get("named_poses") if isinstance(snapshot.get("named_poses"), dict) else {}
    handle = _handle_from_sites(named.get("sites"), eef_pos)
    sim_time = snapshot.get("sim_time")
    timed = isinstance(sim_time, (int, float)) and not isinstance(sim_time, bool)
    return {
        "eef_pos": eef_pos,
        "eef_status": "sim_state" if eef_pos is not None else "not_provided",
        "contact": contact,
        "contact_status": "provided" if contact is not None else "not_provided",
        "contact_pull": bool(contact.get("contact")) if isinstance(contact, dict) and "contact" in contact else None,
        "sim_time_s": float(sim_time) if timed else None,
        "sim_time_status": "provided" if timed else "not_provided",
        **handle,
    }


def _handle_from_sites(sites, eef_pos) -> dict[str, Any]:
    if not isinstance(sites, dict):
        return {"handle_status": "missing_handle_site", "handle_site": None, "handle_distance_m": None, "handle_xyz": None}
    for key, value in sites.items():
        if "handle" not in str(key):
            continue
        pos = value if isinstance(value, list) else value.get("pos") if isinstance(value, dict) else None
        if not isinstance(pos, list) or len(pos) < 3:
            return {"handle_status": "handle_without_pose", "handle_site": str(key), "handle_distance_m": None, "handle_xyz": None}
        xyz = [float(pos[0]), float(pos[1]), float(pos[2])]
        if not isinstance(eef_pos, list):
            return {"handle_status": "measured_without_eef", "handle_site": str(key), "handle_distance_m": None, "handle_xyz": xyz}
        distance = math.sqrt(sum((xyz[index] - float(eef_pos[index])) ** 2 for index in range(3)))
        return {
            "handle_status": "measured",
            "handle_site": str(key),
            "handle_distance_m": round(distance, 4),
            "handle_xyz": xyz,
        }
    return {"handle_status": "missing_handle_site", "handle_site": None, "handle_distance_m": None, "handle_xyz": None}


def _chunk_row(index, current, progress, actions, wall_s, returned, env) -> dict[str, Any]:
    return {
        "chunk_index": index,
        "joint": current.get("joint"),
        "qpos": current.get("qpos"),
        "threshold": current.get("threshold"),
        "comparison": current.get("comparison"),
        "opened": current.get("opened"),
        "progress_m": None if progress is None else round(float(progress), 4),
        "progress_status": "unmeasured" if progress is None else "measured",
        "actions_applied": actions.get("actions_applied"),
        "action_count_status": actions.get("action_count_status"),
        "env_actions_before": actions.get("env_actions_before"),
        "env_actions_after": actions.get("env_actions_after"),
        "wall_s": round(float(wall_s), 4),
        "sim_time_s": current.get("sim_time_s"),
        "sim_time_status": current.get("sim_time_status"),
        "eef_pos": current.get("eef_pos"),
        "eef_status": current.get("eef_status"),
        "handle_site": current.get("handle_site"),
        "handle_status": current.get("handle_status"),
        "handle_distance_m": current.get("handle_distance_m"),
        "handle_xyz": current.get("handle_xyz"),
        "contact_status": current.get("contact_status"),
        "contact_pull": current.get("contact_pull"),
        "cancelled": bool(isinstance(returned, dict) and returned.get("cancelled")),
        "terminated": bool(getattr(env, "terminated", False)),
        "truncated": bool(getattr(env, "truncated", False)),
    }


def _evidence(instruction, region, initial, current, chunks, max_chunks, max_s, stop, trace=None) -> dict[str, Any]:
    rows = list(trace or [])
    actions = [row.get("actions_applied") for row in rows]
    measured_actions = bool(rows) and all(isinstance(item, int) for item in actions)
    return {
        "opened": bool(current.get("opened")),
        "instruction": instruction,
        "region": region,
        "joint": current.get("joint"),
        "initial_qpos": initial.get("qpos"),
        "current_qpos": current.get("qpos"),
        "threshold": current.get("threshold"),
        "comparison": current.get("comparison"),
        "open_ranges": current.get("open_ranges"),
        "predicate": "Open",
        "evidence_source": current.get("evidence_source"),
        "asset": current.get("asset") or initial.get("asset"),
        "chunks_used": chunks,
        "max_chunks": max_chunks,
        "upstream_chunk_budget": UPSTREAM_CHUNK_BUDGET,
        "continued_past_wrapper": chunks > max_chunks,
        "max_s": max_s,
        "stop_reason": stop,
        "stage": current.get("stage"),
        "reason": current.get("reason"),
        "chunk_trace": rows,
        "actions_applied": sum(actions) if measured_actions else None,
        "action_count_status": "measured" if measured_actions else "missing",
        "wall_s": None if not rows else rows[-1].get("wall_s"),
        "sim_time_s": rows[-1].get("sim_time_s") if rows else None,
        "sim_time_status": (rows[-1].get("sim_time_status") if rows else None) or "not_provided",
        "handle_status": (rows[-1].get("handle_status") if rows else None) or "missing_handle_site",
        "handle_site": rows[-1].get("handle_site") if rows else None,
        "handle_distance_m": rows[-1].get("handle_distance_m") if rows else None,
        "handle_xyz": rows[-1].get("handle_xyz") if rows else None,
        "eef_pos": rows[-1].get("eef_pos") if rows else None,
        "pi0_success_used": False,
    }
