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


def read_joint(env: Any, region: str) -> dict[str, Any]:
    """Same joint lookup as ``SiteObjectState.is_open``. The name comes from the site, not a guess."""
    sites = getattr(env, "object_sites_dict", None)
    site = sites.get(region) if isinstance(sites, dict) else None
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
    initial = read_joint(env, region)
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
        current = read_joint(env, region)
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
