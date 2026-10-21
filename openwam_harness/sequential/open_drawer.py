"""Middle-drawer opener. It reuses ``_vlm_chunk`` and does not call ``env.step`` itself.

Success is ``WoodenCabinet.is_open`` on the joint attached to the current
instance's middle region. ``pi0_doubled`` is not called, and its termination
flag is not this skill's success label.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

ASSET = Path(
    "/root/Page_dmh/openwam_harness/envs/rpent/lib/python3.10/site-packages/"
    "libero/libero/assets/articulated_objects/wooden_cabinet.xml"
)
BDDL = Path(
    "/root/Page_dmh/openwam_harness/envs/rpent/lib/python3.10/site-packages/"
    "liberopro/liberopro/bddl_files/libero_goal/open_the_middle_drawer_of_the_cabinet.bddl"
)
MAX_CHUNKS = 8
MAX_S = 45.0


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
        return _evidence(instruction, region, initial, initial, 0, max_chunks, max_s, "already_open")
    started = time.monotonic()
    current = initial
    chunks = 0
    stop = "chunk_budget"
    for _index in range(max_chunks):
        primitives._check_cancelled()
        if getattr(env, "terminated", False) or getattr(env, "truncated", False):
            stop = "terminated" if env.terminated else "truncated"
            break
        if time.monotonic() - started > max_s:
            stop = "time_budget"
            break
        primitives._vlm_chunk(instruction)
        chunks += 1
        current = read_joint(env, region, instance=instance)
        if not current.get("ok"):
            stop = current.get("code") or "joint_unreadable"
            break
        if getattr(env, "terminated", False) or getattr(env, "truncated", False):
            stop = "terminated" if env.terminated else "truncated"
            break
        if current.get("opened"):
            stop = "predicate_open"
            break
    if stop == "chunk_budget":
        current = dict(current)
        current["reason"] = f"动作块预算耗尽（{chunks}/{max_chunks}），关节仍未满足 Open"
    if stop == "time_budget":
        current = dict(current)
        current["reason"] = f"时间预算耗尽（{max_s}s），关节仍未满足 Open"
    return _evidence(instruction, region, initial, current, chunks, max_chunks, max_s, stop)


def _evidence(instruction, region, initial, current, chunks, max_chunks, max_s, stop) -> dict[str, Any]:
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
        "max_s": max_s,
        "stop_reason": stop,
        "reason": current.get("reason"),
        "pi0_success_used": False,
    }
