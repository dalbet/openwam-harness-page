"""Numbered scene candidates and the SAM3 top-score limit.

The planner chooses a candidate number. The backend checks that the number
exists, names a legal instance, and agrees with any instance id. A visual
mask is not an instance id unless the link records its own evidence.
"""

from __future__ import annotations

import hashlib
import json
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
                "grounding_query": spec.get("grounding_query"),
                "place_site": spec.get("place_site"),
                "place_sites": list(spec.get("place_sites") or []),
            }
        )
    return rows


def scene_graph(
    objects: dict[str, Any],
    poses: dict[str, list[float]] | None = None,
    *,
    scene_epoch: int | None = None,
    observation_id: str | None = None,
) -> dict[str, Any]:
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
    return {
        "nodes": nodes,
        "edges": edges,
        "evidence_source": "sim_state",
        "vision": False,
        "scene_epoch": scene_epoch,
        "observation_id": observation_id,
        "candidate_set_version": candidate_set_version(scene_epoch, observation_id, nodes),
    }


def candidate_set_version(scene_epoch: int | None, observation_id: str | None, nodes: list[dict[str, Any]]) -> str:
    payload = json.dumps(
        {
            "scene_epoch": scene_epoch,
            "observation_id": observation_id,
            "refs": [item.get("object_ref") for item in nodes],
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _catalog_region(plan: dict[str, Any], names: list[str]) -> str | None:
    """The official row's region when the sentence is that task. A generic cabinet request stays unresolved."""
    text = plan.get("canonical_instruction") or plan.get("goal") or ""
    if not isinstance(text, str) or not text.strip():
        return None
    from task_catalog import normalize_phrase, tasks_for_scene

    wanted = normalize_phrase(text)
    for row in tasks_for_scene("libero_goal", names):
        phrases = [row.get("instruction_en"), row.get("instruction_zh"), *(row.get("paraphrases_zh") or [])]
        if any(isinstance(item, str) and normalize_phrase(item) == wanted for item in phrases):
            region = row.get("target_region")
            return region if isinstance(region, str) else None
    return None


def resolve_plan_bindings(plan: dict[str, Any], graph: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Derive instance ids from candidate numbers on this scene's candidate table."""
    from schema import PlanError, _err

    if not isinstance(graph, dict) or not isinstance(graph.get("nodes"), list):
        raise PlanError([_err("candidate_set", "candidate_graph", graph, "scene_graph with nodes")])
    nodes = graph["nodes"]
    version = graph.get("candidate_set_version") or candidate_set_version(graph.get("scene_epoch"), graph.get("observation_id"), nodes)
    claimed = plan.get("candidate_set_version")
    if isinstance(claimed, str) and claimed != version:
        raise PlanError([_err("plan.candidate_set_version", "stale_candidate_set", claimed, version)])
    if plan.get("scene_epoch") not in (None, graph.get("scene_epoch")) and graph.get("scene_epoch") is not None:
        raise PlanError([_err("plan.scene_epoch", "stale_candidate_set", plan.get("scene_epoch"), graph.get("scene_epoch"))])
    by_n = {}
    by_ref = {}
    types = {}
    for item in nodes:
        by_n[int(item["n"])] = item
        by_ref[item["object_ref"]] = item
        if item.get("semantic_type"):
            types.setdefault(item["semantic_type"], []).append(item)
    copied = dict(plan)
    steps = []
    bindings = []
    errors = []
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
            chosen = None
            if number is not None:
                try:
                    chosen = by_n[int(number)]
                except (KeyError, TypeError, ValueError):
                    errors.append(_err(f"steps[{index}].{number_key}", "candidate_number", number, "当前场景候选编号"))
                    continue
                if isinstance(ref, str) and ref != chosen["object_ref"]:
                    code = "type_name_not_instance" if ref in types else "binding_mismatch"
                    errors.append(_err(f"steps[{index}].{ref_key}", code, ref, chosen["object_ref"]))
                    continue
                item[ref_key] = chosen["object_ref"]
            elif isinstance(ref, str) and ref in by_ref:
                chosen = by_ref[ref]
            elif isinstance(ref, str) and ref in types:
                errors.append(_err(f"steps[{index}].{ref_key}", "type_name_not_instance", ref, "候选实例编号"))
                continue
            else:
                errors.append(_err(f"steps[{index}].{number_key}", "binding_contract", ref, "候选编号"))
                continue
            if chosen["object_ref"] not in by_ref:
                errors.append(_err(f"steps[{index}].{ref_key}", "binding_instance", chosen["object_ref"], "当前场景实例"))
                continue
            spec_bits = chosen
            if role == "target" and not spec_bits.get("can_be_target"):
                errors.append(_err(f"steps[{index}].target_n", "target_capability", spec_bits["object_ref"], "可放置目标"))
            query_key = spec_bits["object_ref"]
            if role == "target" and not spec_bits.get("movable"):
                sites = list(spec_bits.get("place_sites") or [])
                requested = item.get("target_region") or copied.get("target_region") or _catalog_region(copied, list(by_ref))
                if spec_bits.get("place_site"):
                    query_key = spec_bits["place_site"]
                elif isinstance(requested, str) and requested in sites:
                    query_key = requested
                    item["target_region"] = requested
                elif len(sites) > 1:
                    errors.append(_err(f"steps[{index}].target_n", "target_region_ambiguous", sites, "一个 BDDL 目标区域"))
                    continue
                else:
                    errors.append(_err(f"steps[{index}].target_n", "target_region_missing", spec_bits["object_ref"], "场景设施放置区域"))
                    continue
            canonical = spec_bits.get("semantic_type") or spec_bits["object_ref"]
            phrase = item.get("grounding_query") if role == "object" else None
            bindings.append(
                {
                    "step_id": item.get("step_id"),
                    "step_index": index,
                    "role": role,
                    "candidate_id": int(spec_bits["n"]),
                    "instance_ref": spec_bits["object_ref"],
                    "canonical_name": canonical,
                    "grounding_query": phrase if isinstance(phrase, str) and phrase.strip() else None,
                    "vla_prompt": item.get("vla_prompt") if role == "object" else None,
                    "evidence_source": "planner_candidate_number" if number is not None else "planner_instance_id",
                    "query_key": query_key,
                    "movable": bool(spec_bits.get("movable")),
                    "pose": {
                        "status": "not_sampled",
                        "xyz": None,
                        "sampled_at": None,
                        "observation_id": graph.get("observation_id"),
                        "scene_epoch": graph.get("scene_epoch"),
                        "candidate_set_version": version,
                    },
                }
            )
            item.pop(number_key, None)
        steps.append(item)
    if errors:
        raise PlanError(errors)
    _complete_binding_text(bindings, by_ref)
    copied["steps"] = steps
    copied["plan_bindings"] = bindings
    copied["scene_epoch"] = graph.get("scene_epoch")
    copied["observation_id"] = graph.get("observation_id")
    copied["candidate_set_version"] = version
    copied["bindings_checked"] = True
    return copied, bindings


def _complete_binding_text(bindings: list[dict[str, Any]], by_ref: dict[str, dict[str, Any]]) -> None:
    from schema import _grounding_query, _short_pick

    for item in bindings:
        spec = {
            "semantic_type": item.get("canonical_name"),
            "phrases": by_ref.get(item["instance_ref"], {}).get("phrases") or [],
            "sim_name": item["instance_ref"],
            "grounding_query": by_ref.get(item["instance_ref"], {}).get("grounding_query"),
        }
        if not item.get("grounding_query"):
            item["grounding_query"] = _grounding_query(spec)
        if item["role"] == "object" and not (isinstance(item.get("vla_prompt"), str) and item["vla_prompt"].strip()):
            item["vla_prompt"] = _short_pick("", item["grounding_query"])
        if item["role"] == "target":
            item["vla_prompt"] = None


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
