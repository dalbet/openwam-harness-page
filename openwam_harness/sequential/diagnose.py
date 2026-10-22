"""Grasp and drawer labels, plus a closed-loop contrast that does not call the policy.

One official-versus-hybrid pair is not a success rate. Origin-to-eef distance is
recorded and is not treated as contact.
"""

from __future__ import annotations

import math
from typing import Any


COMPARE_KEYS = (
    "checkpoint",
    "normalization",
    "initial_state_id",
    "camera_preprocessing",
    "language",
)


def annotate_grasp(metrics, step, observations, outcome, **flags) -> None:
    stage = grasp_stage(metrics, observations, **flags)
    metrics["grasp_stage"] = stage
    metrics["attempt"] = attempt_evidence(step, observations, outcome, metrics, stage)


def grasp_stage(metrics, observations, *, lifted, stable, held_band, empty_closed, near, contact) -> str:
    """Name the failure. This does not change the checker that already returned."""
    contact_true = isinstance(contact, dict) and (contact.get("contact") is True or contact.get("held") is True)
    approached = _approached(observations, metrics)
    if lifted and stable and held_band and not near and not contact_true:
        return "origin_far_hold_unconfirmed"
    if lifted and not stable:
        return "slipped_after_lift"
    if (held_band or contact_true) and not lifted:
        return "grasped_not_lifted"
    if empty_closed and not lifted:
        return "empty_grasp"
    if contact_true and not empty_closed and not lifted:
        return "touched_not_closed"
    if approached is False:
        return "not_approached"
    if not empty_closed and not lifted:
        return "not_closed"
    return "unclassified"


def attempt_evidence(step, observations, outcome, metrics, stage: str) -> dict[str, Any]:
    raw = getattr(outcome, "raw", None)
    payload = raw if isinstance(raw, dict) else {}
    chunks = payload.get("chunks") if isinstance(payload.get("chunks"), list) else None
    samples = metrics.get("samples") if isinstance(metrics.get("samples"), list) else []
    start = samples[0] if samples and isinstance(samples[0], dict) else {}
    relative = [row.get("relative_pos") if isinstance(row, dict) else None for row in samples]
    missing = []
    if not (step.get("args") or {}).get("prompt"):
        missing.append("command")
    if chunks is None:
        missing.append("chunks")
    if not start.get("eef_pos"):
        missing.append("start_eef")
    if not start.get("object_pos"):
        missing.append("start_object")
    if not relative or any(item is None for item in relative):
        missing.append("relative_trajectory")
    contact_status = metrics.get("contact_status") or "not_provided"
    if contact_status != "provided":
        missing.append("contact")
    actions = _chunk_actions(chunks)
    if actions is None:
        missing.append("actions_applied")
    return {
        "command": (step.get("args") or {}).get("prompt"),
        "camera": (step.get("args") or {}).get("camera"),
        "object_ref": step.get("object_ref"),
        "grounding_query": step.get("grounding_query"),
        "start_eef": start.get("eef_pos"),
        "start_object": start.get("object_pos"),
        "relative_trajectory": relative,
        "gripper_opening_m": metrics.get("gripper_opening_m"),
        "object_z": None if not isinstance(metrics.get("object_pos"), list) else metrics["object_pos"][2],
        "object_to_eef_m": metrics.get("object_to_eef_m"),
        "origin_to_eef_is_contact": False,
        "contact_status": contact_status,
        "chunks": chunks,
        "actions_applied": actions,
        "cancelled": payload.get("cancelled"),
        "terminated": getattr(outcome, "native_terminated", None),
        "grasp_stage": stage,
        "missing": missing,
    }


def closed_loop_contrast(official: dict | None, hybrid: dict | None) -> dict[str, Any]:
    """Compare two already recorded runs. Does not call Pi0.5 or the planner."""
    if not isinstance(official, dict) or not isinstance(hybrid, dict):
        missing = [name for name, value in (("official", official), ("hybrid", hybrid)) if not isinstance(value, dict)]
        return {"ok": False, "conclusion": "missing_arm", "missing": missing, "success_rate": None, "rounds": 1}
    mismatched = [
        key for key in COMPARE_KEYS
        if official.get(key) != hybrid.get(key) or official.get(key) in {None, ""}
    ]
    if mismatched:
        return {
            "ok": False,
            "conclusion": "not_comparable",
            "mismatched": mismatched,
            "success_rate": None,
            "rounds": 1,
            "note": "对照要求同一 checkpoint、归一化、初始状态、相机预处理和官方任务语言",
        }
    official_ok = official.get("native_success") is True
    hybrid_ok = hybrid.get("status") == "success" or hybrid.get("native_success") is True
    if official_ok and hybrid_ok:
        conclusion = "both_succeeded_not_a_rate"
    elif official_ok and not hybrid_ok:
        conclusion = "subskill_prompt_entry_or_checker"
    elif not official_ok and hybrid_ok:
        conclusion = "hybrid_only_not_a_rate"
    else:
        conclusion = "policy_or_preprocessing"
    return {
        "ok": True,
        "conclusion": conclusion,
        "official_native_success": official_ok,
        "hybrid_success": hybrid_ok,
        "success_rate": None,
        "rounds": 1,
        "note": "一轮差异不能当作成功率",
    }


def _approached(observations, metrics) -> bool | None:
    gaps = []
    for obs in observations or []:
        if not isinstance(obs, dict):
            continue
        obj = None
        eef = ((obs.get("sim") or {}).get("eef_pos"))
        objects = (obs.get("sim") or {}).get("objects") or {}
        target = metrics.get("support_object")
        item = objects.get(target) if isinstance(target, str) else None
        if isinstance(item, dict) and isinstance(item.get("pos"), list):
            obj = item["pos"]
        if isinstance(obj, list) and isinstance(eef, list) and len(obj) >= 2 and len(eef) >= 2:
            gaps.append(math.hypot(float(obj[0]) - float(eef[0]), float(obj[1]) - float(eef[1])))
    if len(gaps) < 2:
        return None
    return gaps[-1] < gaps[0] - 0.02


def _chunk_actions(chunks) -> int | None:
    if not isinstance(chunks, list) or not chunks:
        return None
    total = 0
    seen = False
    for row in chunks:
        if not isinstance(row, dict):
            continue
        value = row.get("actions_sent")
        if value is None:
            value = row.get("env_actions_after")
            before = row.get("env_actions_before")
            if isinstance(value, int) and isinstance(before, int):
                total += value - before
                seen = True
                continue
        if isinstance(value, int) and not isinstance(value, bool):
            total += value
            seen = True
    return total if seen else None
