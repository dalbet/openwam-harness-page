"""Bind a task description to one instance. Never default to the first name."""

from __future__ import annotations

import math
import re
from typing import Any


def actions_allowed(binding: dict | None) -> bool:
    return bool(binding) and binding.get("status") == "resolved" and bool(binding.get("object_ref"))


def positions_moved(previous: dict[str, list[float]] | None, current: dict[str, list[float]], tol: float = 0.01) -> bool:
    previous = previous or {}
    names = set(previous) | set(current)
    for name in names:
        old = previous.get(name)
        new = current.get(name)
        if old is None or new is None or len(old) < 2 or len(new) < 2:
            return True
        if math.dist(old[:3], new[:3]) > tol:
            return True
    return False


def bind_task(
    language: str,
    objects: dict[str, Any],
    poses: dict[str, list[float]] | None,
    *,
    observation_id: str | None,
    scene_epoch: int | None = None,
) -> dict[str, Any]:
    """Return an independent binding. sim poses are not a visual judgment."""
    text = language.strip().lower()
    description, candidates, anchors, kind = _query(text, objects)
    base = {
        "description": description,
        "object_ref": None,
        "evidence_source": None,
        "observation_id": observation_id,
        "scene_epoch": scene_epoch,
        "status": "ambiguous",
        "region_seen": False,
        "bound_to_object_ref": False,
        "candidates": candidates,
        "anchors": anchors,
        "relation": kind,
        "scores": [],
        "note": "",
    }
    if not candidates:
        base["reason"] = "no_candidate"
        return base
    if kind is None and len(candidates) == 1:
        ref = candidates[0]
        base.update(
            status="resolved",
            object_ref=ref,
            evidence_source="unique_instance",
            bound_to_object_ref=True,
            capability_source=objects[ref].get("capability_source"),
            note="只有一个实例，不需要空间消歧",
        )
        return base
    if not observation_id:
        base["reason"] = "missing_observation"
        return base
    poses = poses or {}
    if kind is None or not anchors:
        base["reason"] = "ambiguous_instances"
        base["note"] = "同名实例没有唯一空间关系，不默认第一个"
        return base
    missing = [name for name in candidates + anchors if name not in poses]
    if missing:
        base["reason"] = "missing_pose"
        base["missing"] = missing
        base["note"] = "仿真位姿不完整，不能绑定"
        return base
    scores = _scores(kind, candidates, anchors, poses)
    base["scores"] = scores
    winners = [item for item in scores if item["qualified"]]
    base["evidence_source"] = "sim_state"
    base["note"] = "仿真位姿辅助绑定，不是视觉判断。看到目标区域与绑定 object_ref 是两件事。"
    if len(winners) == 1:
        ref = winners[0]["object_ref"]
        base.update(
            status="resolved",
            object_ref=ref,
            bound_to_object_ref=True,
            capability_source=objects[ref].get("capability_source"),
        )
        return base
    base["reason"] = "ambiguous_spatial"
    return base


def _query(text: str, objects: dict[str, Any]) -> tuple[str, list[str], list[str], str | None]:
    match = re.search(
        r"pick up the (.+?) (?:between the (.+?) and the (.+)|next to the (.+)|on the (.+)|in the (.+)|from table center) and place it",
        text,
    )
    if match:
        noun = match.group(1).strip()
        candidates = _refs(noun, objects, movable=True)
        if match.group(2) and match.group(3):
            anchors = _refs(match.group(2), objects) + _refs(match.group(3), objects)
            return f"{noun} between {match.group(2)} and {match.group(3)}", candidates, anchors, "between"
        if match.group(4):
            return f"{noun} next to {match.group(4)}", candidates, _refs(match.group(4), objects), "next_to"
        if match.group(5):
            return f"{noun} on {match.group(5)}", candidates, _refs(match.group(5), objects), "on"
        if match.group(6):
            return f"{noun} in {match.group(6)}", candidates, _refs(match.group(6), objects), "in"
        return f"{noun} at table center", candidates, [], "center"
    noun = _leading_object(text)
    candidates = _refs(noun, objects, movable=True) if noun else []
    return noun or text, candidates, [], None


def _leading_object(text: str) -> str:
    for pattern in (
        r"pick up the ([a-z0-9 ]+?) and ",
        r"put both the ([a-z0-9 ]+?) and the ",
        r"put the ([a-z0-9 ]+?) (?:on|in) ",
    ):
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
    return ""


def _refs(text: str, objects: dict[str, Any], movable: bool | None = None) -> list[str]:
    found = []
    needle = text.strip().lower()
    for ref, spec in objects.items():
        if movable is not None and bool(spec.get("movable")) != movable:
            continue
        phrases = [str(item).lower() for item in spec.get("phrases") or []]
        semantic = str(spec.get("semantic_type") or "").replace("_", " ")
        if any(phrase and (phrase in needle or needle in phrase) for phrase in phrases) or (semantic and (semantic in needle or needle in semantic)):
            found.append(ref)
    return found


def _scores(kind: str, candidates: list[str], anchors: list[str], poses: dict[str, list[float]]) -> list[dict[str, Any]]:
    if kind == "between" and len(anchors) >= 2:
        return [_between(ref, poses[anchors[0]], poses[anchors[1]], poses[ref]) for ref in candidates]
    if kind == "center":
        ranked = sorted(candidates, key=lambda ref: math.hypot(poses[ref][0], poses[ref][1]))
        return _margin(ranked, poses, lambda ref: math.hypot(poses[ref][0], poses[ref][1]), limit=0.04)
    anchor = poses[anchors[0]]
    if kind == "on":
        return _on(candidates, anchor, poses)
    ranked = sorted(candidates, key=lambda ref: math.dist(poses[ref][:2], anchor[:2]))
    return _margin(ranked, poses, lambda ref: math.dist(poses[ref][:2], anchor[:2]), limit=0.04)


def _between(ref: str, start: list[float], end: list[float], point: list[float]) -> dict[str, Any]:
    abx, aby = end[0] - start[0], end[1] - start[1]
    denom = abx * abx + aby * aby
    if denom <= 1e-8:
        return {"object_ref": ref, "qualified": False, "reason": "anchors_coincide"}
    tee = ((point[0] - start[0]) * abx + (point[1] - start[1]) * aby) / denom
    clamped = min(1.0, max(0.0, tee))
    closest = (start[0] + clamped * abx, start[1] + clamped * aby)
    dist = math.hypot(point[0] - closest[0], point[1] - closest[1])
    return {
        "object_ref": ref,
        "t": round(tee, 4),
        "distance_m": round(dist, 4),
        "qualified": 0.2 <= tee <= 0.8 and dist <= 0.06,
    }


def _on(candidates: list[str], anchor: list[float], poses: dict[str, list[float]]) -> list[dict[str, Any]]:
    rows = []
    for ref in candidates:
        pos = poses[ref]
        rows.append(
            {
                "object_ref": ref,
                "xy_m": round(math.dist(pos[:2], anchor[:2]), 4),
                "z": pos[2],
                "qualified": False,
            }
        )
    if len(rows) == 1:
        rows[0]["qualified"] = rows[0]["xy_m"] <= 0.1
        return rows
    ordered = sorted(rows, key=lambda item: item["z"], reverse=True)
    best, other = ordered[0], ordered[1]
    if best["z"] - other["z"] >= 0.02 and best["xy_m"] <= 0.1:
        best["qualified"] = True
    return rows


def _margin(ranked: list[str], poses: dict[str, list[float]], distance, limit: float) -> list[dict[str, Any]]:
    rows = [{"object_ref": ref, "distance_m": round(float(distance(ref)), 4), "qualified": False} for ref in ranked]
    if len(rows) == 1:
        rows[0]["qualified"] = True
        return rows
    if rows[1]["distance_m"] - rows[0]["distance_m"] >= limit:
        rows[0]["qualified"] = True
    return rows
