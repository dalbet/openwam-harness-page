#!/usr/bin/env python3
"""Offline: which kind of failure is a composite step, and how far off target?

Scope B of the 2026-11-05 round asks for a targeted diagnosis of a failed step
instead of a whole new audit, and for the landing error to be measured against
the *official* target region rather than against the coordinate range of rounds
that happened to succeed.

Everything here is read from the archived episode record: the step's own
verification geometry (``delta_site`` is already expressed in the site's local
frame by ``common.geometry.site_under``), its gripper separation trace, the
release events, the carried-object watch, and the policy's action budget.
Nothing is re-simulated and no new per-action RPC is introduced.

Six failure classes, each decided by a measurement:

``grasp_failed``          the fingers closed on nothing (min separation at the
                          empty-close level) and no release was ever observed
``lifted_not_transferred``grasped, but the end effector never travelled far
                          enough in XY to be carrying the object anywhere
``not_reached_target``    the object ended outside the official site bounds
``drifted_after_release`` released inside the bounds, ended outside them
``carried_object_lost``   the carried-object watch recorded the object leaving
                          its container and the final check confirms it
``judgement_mismatch``    the step's own geometry verification passed, yet the
                          step was recorded as failed (or the reverse)

A step can also lose a *different* step's goal while it runs (the arm knocks the
already-placed cheese off while grasping the bowl). That is reported as
``collateral`` on the step that ran, so the loss is never hidden inside the
other step's row.

Usage:
    python tools/step_diagnosis.py --out /tmp/step_diagnosis.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

RAW = Path("/home/host/flashrt-exp/pi05_round/raw/composite")
# A parallel-jaw gripper closing on nothing reads as a few millimetres; holding
# a 60-80 mm bowl reads as 60 mm and above. This only separates "closed on
# nothing" from "closed on the object", never a success claim.
EMPTY_CLOSE_M = 0.02
LIFT_TRAVEL_M = 0.05


def _num(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def _relation_of(row: Any) -> dict[str, Any]:
    if not isinstance(row, dict):
        return {}
    inner = row.get("relation")
    return inner if isinstance(inner, dict) else {}


def _best_evidence(step: dict[str, Any]) -> dict[str, Any]:
    """The richest evidence row: one whose relation actually carries a predicate."""
    candidates = [step.get("final_evidence"), step.get("evidence"), step.get("verification")]
    with_predicate = [row for row in candidates if _relation_of(row).get("predicate")]
    if with_predicate:
        return with_predicate[0]
    for row in candidates:
        if isinstance(row, dict) and row:
            return row
    return {}


def target_error(relation: dict[str, Any]) -> dict[str, Any]:
    """Distance from the object to the *official* target region.

    ``delta_site`` is ``mat @ (object - site)``, i.e. already in the site's own
    axes, so the box test uses the published half-extents directly and the
    radial distance is frame independent.
    """
    delta = relation.get("delta_site")
    size = relation.get("site_size")
    if not (isinstance(delta, list) and isinstance(size, list) and len(delta) >= 2 and len(size) >= 2):
        return {}
    dx, dy = abs(float(delta[0])), abs(float(delta[1]))
    sx, sy = float(size[0]), float(size[1])
    outside = max(dx - sx, dy - sy)
    return {
        "predicate": relation.get("predicate"),
        "delta_site_m": [round(float(value), 4) for value in delta],
        "site_half_extents_m": [sx, sy],
        "inside_xy": bool(relation.get("inside_xy")),
        "outside_xy_m": round(outside, 4),
        "radial_from_site_centre_m": round(math.hypot(float(delta[0]), float(delta[1])), 4),
        "near_miss": bool(0.0 < outside <= 0.03),
        "source": "official site bounds (SiteObject.under), not a success range",
    }


def interior_error(relation: dict[str, Any]) -> dict[str, Any]:
    """The bowl-interior counterpart: how far the object is from the container."""
    xy, limit = _num(relation.get("xy_m")), _num(relation.get("xy_limit_m"))
    bottom, floor = _num(relation.get("object_bottom_z")), _num(relation.get("floor_z"))
    if xy is None:
        return {}
    return {
        "predicate": relation.get("predicate"),
        "xy_m": xy,
        "xy_limit_m": limit,
        "xy_over_m": round(xy - limit, 4) if limit is not None else None,
        "seat_delta_m": round(bottom - floor, 4) if None not in (bottom, floor) else None,
        "inside": bool(relation.get("ok")),
    }


def eef_travel(step: dict[str, Any]) -> dict[str, Any]:
    rows = [row for row in (step.get("observations") or []) if row.get("eef_pos")]
    if len(rows) < 2:
        return {}
    first, last = rows[0]["eef_pos"], rows[-1]["eef_pos"]
    return {
        "xy_m": round(math.hypot(last[0] - first[0], last[1] - first[1]), 4),
        "z_m": round(float(last[2]) - float(first[2]), 4),
        "samples": len(rows),
    }


def object_travel(step: dict[str, Any]) -> dict[str, Any]:
    rows = [row for row in (step.get("observations") or []) if row.get("object_pose")]
    if len(rows) < 2:
        return {}
    first, last = rows[0]["object_pose"], rows[-1]["object_pose"]
    return {
        "xy_m": round(math.hypot(last[0] - first[0], last[1] - first[1]), 4),
        "z_m": round(float(last[2]) - float(first[2]), 4),
        "samples": len(rows),
    }


def carry_losses(steps: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Every watched goal that left its container, keyed by the goal's task key.

    The watch runs inside whichever step was executing, so the record of a loss
    lives on a step other than the one whose goal broke. ``per_goal`` is keyed
    by step id (``c1``), which this maps back to the catalog task key so a row
    can be matched to the step whose goal it describes.
    """
    by_id = {str(step.get("step_id")): step for step in steps}
    out: dict[str, dict[str, Any]] = {}
    for step in steps:
        watch = step.get("carry_watch") or {}
        for goal_id, row in (watch.get("per_goal") or {}).items():
            first = row.get("first_lost")
            if not first:
                continue
            state = first.get("state") or {}
            owner = by_id.get(str(goal_id)) or {}
            key = str(
                owner.get("step_task_key") or owner.get("task_key") or row.get("key") or goal_id
            )
            out[key] = {
                "key": key,
                "goal_step_id": goal_id,
                "lost_during_step": step.get("step_id"),
                "sim_action_count": first.get("sim_action_count"),
                "skill_sim_actions": first.get("skill_sim_actions"),
                "from": first.get("from"),
                "to": first.get("to"),
                "relation": interior_error(_relation_of(state)),
                "inside_at_start": row.get("ok_first"),
                "inside_at_end": row.get("ok_last"),
            }
    return out


def classify(step: dict[str, Any], losses: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """One failure class plus the numbers that decided it."""
    relation = _relation_of(_best_evidence(step))
    evidence = _best_evidence(step)
    grasp = step.get("grasp") or {}
    release = step.get("release") or {}
    min_open = _num(grasp.get("min_opening_m"))
    final_open = _num(grasp.get("final_opening_m"))
    releases = int(release.get("count") or 0)
    error = target_error(relation)
    interior = interior_error(relation)
    travel = eef_travel(step)
    verification_ok = (step.get("verification") or {}).get("ok")
    key = str(step.get("step_task_key") or step.get("task_key") or "")
    lost = losses.get(key)
    budget = _num(step.get("sim_actions")) or 0.0
    max_steps = _num(step.get("max_steps")) or 0.0
    facts = {
        "sim_actions": step.get("sim_actions"),
        "max_steps": step.get("max_steps"),
        "budget_exhausted": bool(max_steps and budget >= max_steps),
        "stop_reason": step.get("stop_reason"),
        "phase_reached": step.get("phase_reached"),
        "min_gripper_opening_m": min_open,
        "final_gripper_opening_m": final_open,
        "release_events": releases,
        "relation_ok": relation.get("ok"),
        "verification_ok": verification_ok,
        "height": evidence.get("height"),
        "eef_travel": travel,
        "object_travel": object_travel(step),
        "interior": interior,
        "target": error,
        "native_events": len(step.get("native_events") or []),
        "carry_loss": lost,
    }
    if verification_ok is True and step.get("ok") is False:
        code = "judgement_mismatch"
        why = "the step's own geometry verification passed but the step was recorded as failed"
    elif verification_ok is False and step.get("ok") is True:
        code = "judgement_mismatch"
        why = "the step was recorded as complete but its own verification reads false"
    elif lost:
        code = "carried_object_lost"
        why = (
            f"the carried-object watch saw {lost['key']} leave its container at sim action "
            f"{lost['sim_action_count']} while {lost['lost_during_step']} was executing"
            + (
                f" (xy {lost['relation'].get('xy_m')} m vs limit {lost['relation'].get('xy_limit_m')} m)"
                if lost.get("relation")
                else ""
            )
        )
    elif (
        releases == 0
        and step.get("ok") is False
        and min_open is not None
        and min_open < EMPTY_CLOSE_M
    ):
        code = "grasp_failed"
        why = (
            f"fingers closed to {min_open * 1000:.1f} mm (empty close), no release was observed, "
            f"and the step used {int(budget)} of {int(max_steps)} actions"
        )
    elif releases == 0 and step.get("ok") is False and travel.get("xy_m") is not None and travel["xy_m"] < LIFT_TRAVEL_M:
        code = "lifted_not_transferred"
        why = f"the end effector travelled {travel['xy_m']} m in XY and the object was never released"
    elif step.get("ok") is False and error and error.get("inside_xy") is False:
        code = "not_reached_target"
        why = (
            f"the object ended {error['outside_xy_m']} m outside the official site bounds "
            f"({error['radial_from_site_centre_m']} m from its centre)"
        )
    elif step.get("ok") is False and interior.get("xy_over_m") is not None and interior["xy_over_m"] > 0:
        code = "not_reached_target"
        why = f"the object ended {interior['xy_over_m']} m outside the container's official XY limit"
    elif releases and step.get("ok") is False:
        code = "drifted_after_release"
        why = "the object was released and the final relation no longer holds"
    elif not relation and not grasp:
        code = "insufficient_evidence"
        why = (
            "this archived record carries no relation geometry and no gripper trace for the "
            "step; it predates the evidence columns, so it is not classified"
        )
    else:
        code = "unclassified"
        why = "no measurement in this record separates the failure; reported as unclassified"
    return {"code": code, "why": why, "facts": facts}


def iter_records(roots: Iterable[Path]):
    seen: set[str] = set()
    for root in roots:
        for path in sorted(root.rglob("*.json")):
            if path.name.startswith("."):
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(data, dict) or not data.get("composite"):
                continue
            episode = str(data.get("episode_id") or path.name)
            if episode in seen:
                continue
            seen.add(episode)
            yield path, data


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("/tmp/step_diagnosis.json"))
    parser.add_argument("--root", type=Path, action="append")
    args = parser.parse_args()
    roots = args.root or [RAW]
    rows = []
    for path, record in iter_records(roots):
        composite = record.get("composite") or {}
        steps = composite.get("steps") or []
        losses = carry_losses(steps)
        failed = [step for step in steps if step.get("ok") is False or step.get("ok_final") is False]
        if not failed and record.get("user_goal_success"):
            continue
        # The composite row is a compact projection; the per-step trace
        # (grasp separation, release events, observations, budget) lives on the
        # episode's own ``steps`` list. Joining them keeps the classification on
        # measurements the record already contains.
        traced = {str(item.get("step_id")): item for item in (record.get("steps") or [])}
        if not failed and record.get("failure_code") == "composite_transport_not_ready":
            evidence = (record.get("failure_evidence") or {}).get("transport_ready") or {}
            rows.append(
                {
                    "record": str(path),
                    "episode_id": record.get("episode_id"),
                    "instruction": record.get("instruction"),
                    "round_ok": False,
                    "round_failure_code": record.get("failure_code"),
                    "step_id": evidence.get("step_id"),
                    "step_task_key": evidence.get("step_task_key"),
                    "prompt": None,
                    "code": "transport_not_ready",
                    "why": (
                        f"the round stopped before carrying: {evidence.get('reason_code')} "
                        f"(seat {evidence.get('seat_delta_m')} m, XY margin "
                        f"{evidence.get('xy_margin_m')} m)"
                    ),
                    "facts": {"transport_ready": evidence},
                    "collateral": {},
                }
            )
            continue
        for step in failed or [{}]:
            trace = traced.get(str(step.get("step_id"))) or {}
            step = {**trace, **{k: v for k, v in step.items() if v is not None}}
            row = {
                "record": str(path),
                "episode_id": record.get("episode_id"),
                "instruction": record.get("instruction"),
                "round_ok": bool(record.get("user_goal_success")),
                "round_failure_code": record.get("failure_code"),
                "step_id": step.get("step_id"),
                "step_task_key": step.get("step_task_key"),
                "prompt": step.get("prompt"),
            }
            row.update(classify(step, losses) if step else {"code": "unclassified", "why": "no step failed", "facts": {}})
            others = {key: value for key, value in losses.items() if key != row.get("step_task_key")}
            row["collateral"] = others
            rows.append(row)
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["code"]] = counts.get(row["code"], 0) + 1
    report = {
        "schema": "embodied_engine.step_diagnosis.v1",
        "roots": [str(root) for root in roots],
        "counts": counts,
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    for row in rows:
        facts = row.get("facts") or {}
        target = facts.get("target") or {}
        interior = facts.get("interior") or {}
        print(
            f"{row['code']:<24} {row.get('episode_id')} {row.get('step_id')} "
            f"outside={target.get('outside_xy_m')} interior_over={interior.get('xy_over_m')} :: {row['why']}"
        )
    print(json.dumps(counts, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
