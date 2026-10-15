"""Numbered scene candidates and the SAM3 top-score limit.

The planner chooses a candidate number. The backend checks that the number
exists, names a legal instance, and agrees with any instance id. A visual
mask is not an instance id unless the link records its own evidence.
"""

from __future__ import annotations

import math
from typing import Any


SAM3_CONTRACT = {
    "text_segment": "top_score_only",
    "selector": "Sam3Facade._select_top keeps np.argmax(scores) and drops the other masks",
    "point_prompt": "segment(point=[row, col]) exists; _segment_point still returns one top mask",
    "multi_candidate_rpc": None,
    "same_type_policy": "scene candidates keep every instance; a text mask is not assigned to the first one",
}


def sam3_contract_from_source(source: str) -> dict[str, Any]:
    """Confirm the installed server source. A missing argmax does not invent multi-candidate support."""
    contract = dict(SAM3_CONTRACT)
    contract["source_has_argmax"] = "np.argmax" in source
    contract["source_has_point"] = "point_coords" in source and "predict_inst" in source
    contract["text_segment"] = "top_score_only" if contract["source_has_argmax"] else "unknown"
    if not contract["source_has_argmax"]:
        contract["multi_candidate_rpc"] = None
    return contract


def number_candidates(objects: dict[str, Any]) -> list[dict[str, Any]]:
    """Stable numbers for this scene. Duplicate semantic types all stay in the list."""
    rows = []
    for index, (ref, spec) in enumerate(objects.items()):
        rows.append(
            {
                "n": index + 1,
                "object_ref": ref,
                "semantic_type": spec.get("semantic_type"),
                "phrases": list(spec.get("phrases") or []),
                "movable": bool(spec.get("movable")),
                "can_be_target": bool(spec.get("can_be_target")),
                "target_relations": list(spec.get("target_relations") or []),
                "capability_source": spec.get("capability_source"),
            }
        )
    return rows


def scene_graph(objects: dict[str, Any], poses: dict[str, list[float]] | None) -> dict[str, Any]:
    """Instances plus near edges from sim poses. Poses are not a visual judgment."""
    nodes = number_candidates(objects)
    poses = poses or {}
    edges = []
    for left in nodes:
        origin = poses.get(left["object_ref"])
        if not origin:
            continue
        for right in nodes:
            if right["n"] <= left["n"]:
                continue
            other = poses.get(right["object_ref"])
            if not other:
                continue
            distance = math.dist(origin[:3], other[:3])
            if distance <= 0.15:
                edges.append(
                    {
                        "from": left["n"],
                        "to": right["n"],
                        "relation": "near",
                        "distance_m": round(distance, 4),
                        "evidence_source": "sim_state",
                    }
                )
    return {"nodes": nodes, "edges": edges, "evidence_source": "sim_state", "vision": False}


def resolve_numbers(plan: dict[str, Any], numbered: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Replace candidate numbers with instance ids and record the evidence."""
    by_n = {int(item["n"]): item for item in numbered}
    by_ref = {item["object_ref"]: item for item in numbered}
    copied = dict(plan)
    steps = []
    evidence = []
    for index, step in enumerate(plan.get("steps") or []):
        if not isinstance(step, dict):
            steps.append(step)
            continue
        item = dict(step)
        for role, number_key, ref_key in (
            ("object", "object_n", "object_ref"),
            ("target", "target_n", "target_ref"),
        ):
            number = item.get(number_key)
            ref = item.get(ref_key)
            if number is not None:
                try:
                    chosen = by_n[int(number)]
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f"steps[{index}].{number_key} 不在候选编号里") from exc
                if isinstance(ref, str) and ref != chosen["object_ref"]:
                    raise ValueError(
                        f"steps[{index}] 的 {number_key}={number} 对应 {chosen['object_ref']}，与 {ref_key}={ref} 不一致"
                    )
                item[ref_key] = chosen["object_ref"]
                evidence.append(
                    {
                        "step_index": index,
                        "role": role,
                        "candidate_n": int(number),
                        "object_ref": chosen["object_ref"],
                        "evidence_source": "planner_candidate_number",
                        "candidate_source": "scene_graph",
                    }
                )
            elif isinstance(ref, str) and ref in by_ref:
                evidence.append(
                    {
                        "step_index": index,
                        "role": role,
                        "candidate_n": by_ref[ref]["n"],
                        "object_ref": ref,
                        "evidence_source": "planner_instance_id",
                        "candidate_source": "scene_graph",
                    }
                )
            item.pop(number_key, None)
        steps.append(item)
    copied["steps"] = steps
    return copied, evidence


def link_visual_candidate(mask: dict[str, Any] | None, numbered: list[dict[str, Any]]) -> dict[str, Any]:
    """A top-scoring mask stays unlinked when more than one instance could match."""
    if not mask or not mask.get("found"):
        return {
            "status": "unlinked",
            "object_ref": None,
            "candidate_n": None,
            "evidence_source": "sam3_top1",
            "note": "SAM3 没有给出可关联的掩码",
        }
    same = [
        item for item in numbered
        if mask.get("semantic_type") and item.get("semantic_type") == mask.get("semantic_type")
    ]
    if mask.get("evidence_source") == "point_prompt" and len(same) == 1:
        return {
            "status": "linked",
            "object_ref": same[0]["object_ref"],
            "candidate_n": same[0]["n"],
            "evidence_source": "point_prompt",
            "score": mask.get("score"),
            "note": "点提示只指向这一只实例",
        }
    return {
        "status": "ambiguous" if len(same) != 1 else "unlinked",
        "object_ref": None,
        "candidate_n": None,
        "evidence_source": "sam3_top1_unlinked",
        "score": mask.get("score"),
        "same_type_count": len(same),
        "note": "文本分割只保留最高分，没有实例证据时不默认第一只",
    }


def binding_question(binding: dict[str, Any], numbered: list[dict[str, Any]]) -> str:
    by_ref = {item["object_ref"]: item for item in numbered}
    labels = []
    for ref in binding.get("candidates") or []:
        item = by_ref.get(ref)
        if item is None:
            labels.append(str(ref))
        else:
            labels.append(f"{item['n']} 号 {item['object_ref']}")
    if not labels:
        return "没有可绑定的候选，不会默认第一只物体。"
    return "要抓哪一只？请指定编号：" + "、".join(labels) + "。不会默认第 1 只。"
