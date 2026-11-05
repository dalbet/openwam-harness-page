"""Transport readiness: is the placed object safe to *carry* in the next step?

A composite step like "put the cream cheese in the bowl" ends on an official
``In``/``On`` relation. That relation says the object is inside the container;
it does not say the object is seated deep enough to survive the arm picking the
container up, accelerating, and rotating. This module evaluates that second,
*independent* question and never changes the step's own goal.

Calibration (offline, 47 archived rounds in ``raw/composite`` whose first step
is ``put_cream_cheese_in_bowl``, measured on each step's own evidence at the
moment it completed; see ``tools/ready_features.py``):

* seat depth (cheese bottom minus the bowl's published interior floor) splits
  into ``0.0102 m .. 0.0263 m`` (35 rounds) and ``0.0329 m .. 0.0392 m``
  (12 rounds, cheese resting on the rim);
* of the 12 perched rounds, **11 transported the cheese without losing it**;
  the single archived loss (``step3_en_delivery``, ``ep-20261104-121734-febbf1``,
  seat ``0.0357 m``) sits exactly where two rounds that kept the cheese also
  sat (``0.0357 m`` twice), and its XY (``0.0194 m``) is inside the passing
  range, so no archived feature separates it from them;
* the losing round's mechanism is visible in its step-2 record (the bowl grasp
  pushed a rim-perched cheese off), which is a post-grasp fact, not a pre-grasp
  measurement.

So this module keeps three different verdicts apart, and none is a success
claim:

* **not measured** (``ok = None``): no interior geometry, or no samples at all.
  Nothing is executed on this answer;
* **not ready** (``ok = False``): a hard failure -- relation not met, cheese
  below the interior floor, at the edge of the opening, or (when two or more
  samples exist) still sliding. This may trigger **one** bounded re-seat;
* **ready with a warning** (``ok = True``, ``warning = "perched"``): the cheese
  rests high on the rim. 11 of 12 archived perched rounds still travelled, so
  the warning is recorded and neither stops nor re-seats anything.

No threshold here is presented as a predictor of loss; the archived numbers do
not support one.

The recorded numbers stay attached to the round, so a later acceptance run can
show whether the re-seat actually removed a loss instead of assuming it did.
"""

from __future__ import annotations

import math
from typing import Any

SCHEMA = "embodied_engine.composite.transport_ready.v1"

DEFAULTS: dict[str, Any] = {
    "enabled": True,
    # Seat depth = object bottom minus the container's published interior
    # floor. Archived rounds fill 0.0102..0.0392 with no gap, so this is only
    # the boundary above which a seat is *called* perched (a warning, not a
    # rejection): 11 of the 12 archived perched rounds still transported fine.
    "max_seat_delta_m": 0.029,
    # Below the interior floor means the object is not in the container at all
    # (it fell beside it); the official relation's own tolerance is 0.02, so
    # this is deliberately tighter than the goal predicate.
    "min_seat_delta_m": -0.002,
    # The official ``bowl_interior`` XY limit is 0.025 m. Archived rounds reach
    # 0.0224 m and still transport; 0.002 of margin only flags the edge.
    "min_xy_margin_m": 0.002,
    # Relative displacement between two fresh samples, i.e. the object is
    # still sliding while the arm is holding still. Archived samples move
    # < 0.1 mm, so this is a safety net, not a discriminator.
    "max_relative_drift_m": 0.015,
    "samples_required": 3,
    # A rim seat is a warning by default: 11 of the 12 archived perched rounds
    # carried the cheese fine, so rejecting them would be a false stop. Set
    # this to false to make perched a hard failure again (recorded as such).
    "perched_is_warning": True,
    "reseat": {
        "enabled": True,
        "max_calls": 1,
        "max_steps": 240,
        "timeout_s": 120.0,
        # ``stop``: the episode ends with an explicit not-ready failure instead
        # of carrying an object the check just judged unsafe. ``proceed``: keep
        # the plan and let the round show what happens.
        "on_failure": "stop",
    },
}

REASON_LABELS = {
    "not_applicable": "no later step carries the target container",
    "not_measured": "the geometry could not be measured on this evidence",
    "relation_not_met": "the step's own relation is not met on fresh samples",
    "samples_insufficient": "fewer than the required post-release samples; not measured",
    "fell_beside_bowl": "the object dropped below the container's interior floor",
    "perched_above_floor": "the object rests high on the container's rim (warning only)",
    "edge_of_bowl": "the object is at the edge of the container's opening",
    "still_moving": "the object is still sliding relative to the container",
}


def options(cfg: dict[str, Any] | None) -> dict[str, Any]:
    """Effective readiness limits; config overrides defaults, never the reverse."""
    block = dict(((cfg or {}).get("composite") or {}).get("transport_ready") or {})
    out = {
        key: block.get(key, value)
        for key, value in DEFAULTS.items()
        if key != "reseat"
    }
    reseat = dict(DEFAULTS["reseat"])
    reseat.update({k: v for k, v in (block.get("reseat") or {}).items() if k in reseat})
    out["reseat"] = reseat
    return out


def applies(
    steps: list[dict[str, Any]],
    index: int,
    catalog_by_key: dict[str, dict[str, Any]],
) -> bool:
    """True when a later step will physically carry this step's target.

    The check is about transport, so it is only meaningful when some *later*
    step's catalog row lists the earlier step's target among its objects. No
    keyword is involved: the same catalog rows the plan was built from are
    read here.
    """
    if index + 1 >= len(steps):
        return False
    completion = steps[index].get("completion") or {}
    if str(completion.get("kind") or "") != "relation":
        return False
    target = str(completion.get("target_n") or "")
    if not target:
        return False
    for later in steps[index + 1:]:
        row = catalog_by_key.get(str(later.get("step_task_key"))) or {}
        names = [str(name) for name in (row.get("objects") or [])]
        if target in names:
            return True
    return False


def _relation_of(sample: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Pull the interior relation and the height row out of one sample row."""
    row = sample.get("relation") or {}
    inner = row.get("relation") if isinstance(row.get("relation"), dict) else {}
    if not isinstance(inner, dict):
        inner = {}
    if not inner and "xy_m" in row:
        inner = row
    height = row.get("height") if isinstance(row.get("height"), dict) else {}
    return inner, height


GEOMETRY_KEYS = ("predicate", "floor_z", "rim_z", "xy_limit_m", "target_pose")


def _geometry_defaults(evidence: dict[str, Any] | None) -> dict[str, Any]:
    """The container geometry a step-level evidence row carries once.

    ``final_evidence.relation_samples`` keeps only what changes per sample
    (``xy_m``, ``object_bottom_z``, ``target_pose``); the floor, rim and the
    official XY limit are properties of the asset and appear once on the
    relation row. Merging them here is what lets the same rule run on either
    sample shape, instead of silently reporting "not measured".
    """
    if not isinstance(evidence, dict):
        return {}
    inner = evidence.get("relation") if isinstance(evidence.get("relation"), dict) else {}
    return {key: inner[key] for key in GEOMETRY_KEYS if inner.get(key) is not None}


def views(
    samples: list[dict[str, Any]] | None,
    evidence: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], str | None]:
    """Normalise the two sample shapes into one view list.

    Source order is explicit: placement samples (fresh observation ids, rich
    rows) first, then the step's own ``relation_samples``. The returned source
    string is recorded, so a reader can tell which evidence the verdict used.
    """
    defaults = _geometry_defaults(evidence)
    rows = [row for row in (samples or []) if isinstance(row, dict)]
    source = "placement.samples" if rows else None
    if not rows and isinstance(evidence, dict):
        raw = [row for row in (evidence.get("relation_samples") or []) if isinstance(row, dict)]
        if raw:
            rows = [{"relation": row} for row in raw]
            source = "evidence.relation_samples"
    out = []
    for row in rows:
        inner, height = _relation_of(row)
        merged = dict(defaults)
        merged.update({key: value for key, value in inner.items() if value is not None})
        seat = None
        if isinstance(merged.get("object_bottom_z"), (int, float)) and isinstance(
            merged.get("floor_z"), (int, float)
        ):
            seat = round(float(merged["object_bottom_z"]) - float(merged["floor_z"]), 5)
        out.append(
            {
                "relation": merged,
                "height": height,
                "seat_delta_m": seat,
                "object_pos": row.get("object_pos"),
                "observation_id": row.get("observation_id"),
                "sim_action_count": row.get("sim_action_count"),
            }
        )
    return out, source


def _relative_xy(object_pos: Any, target_pos: Any) -> list[float] | None:
    if not (isinstance(object_pos, (list, tuple)) and isinstance(target_pos, (list, tuple))):
        return None
    if len(object_pos) < 3 or len(target_pos) < 3:
        return None
    return [
        float(object_pos[0]) - float(target_pos[0]),
        float(object_pos[1]) - float(target_pos[1]),
        float(object_pos[2]) - float(target_pos[2]),
    ]


def drift(views: list[dict[str, Any]]) -> dict[str, Any]:
    """Largest change of the object relative to the container between samples.

    Both poses come from the same snapshot, so the difference is a *relative*
    motion: the arm is holding still during the post-release samples, and any
    change here is the object settling or sliding inside/against the target.

    When every sample carries full poses the vector is 3-D. ``relation_samples``
    in ``final_evidence`` carry no object pose, so the fallback vector is the
    pair the check actually reasons about: radial offset ``xy_m`` and seat
    depth. The mode is reported; the two are never mixed in one number.
    """
    full = [
        _relative_xy(view.get("object_pos"), (view.get("relation") or {}).get("target_pose"))
        for view in views
    ]
    mode = "object_minus_target_3d" if views and all(vec is not None for vec in full) else "xy_and_seat"
    vectors: list[list[float]] = []
    for index, view in enumerate(views):
        if mode == "object_minus_target_3d":
            vectors.append([round(value, 6) for value in full[index]])  # type: ignore[arg-type]
        else:
            inner = view.get("relation") or {}
            xy = inner.get("xy_m")
            seat = view.get("seat_delta_m")
            if not isinstance(xy, (int, float)) or not isinstance(seat, (int, float)):
                continue
            vectors.append([float(xy), 0.0, float(seat)])
    steps: list[dict[str, Any]] = []
    for before, after in zip(vectors, vectors[1:]):
        delta = [round(after[i] - before[i], 5) for i in range(3)]
        steps.append(
            {
                "delta_m": delta,
                "xy_m": round(math.hypot(delta[0], delta[1]), 5),
                "distance_m": round(math.sqrt(sum(v * v for v in delta)), 5),
            }
        )
    peak = max(steps, key=lambda row: row["distance_m"]) if steps else None
    return {
        "mode": mode,
        "samples": len(vectors),
        "pairs": steps,
        "max_xy_m": peak["xy_m"] if peak else None,
        "max_distance_m": peak["distance_m"] if peak else None,
    }


def evaluate(
    *,
    step: dict[str, Any],
    relation: dict[str, Any] | None,
    samples: list[dict[str, Any]] | None,
    cfg: dict[str, Any] | None,
    evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Judge transport readiness from the step's own post-release samples."""
    opts = options(cfg)
    evidence = evidence if isinstance(evidence, dict) else relation
    rows, source = views(samples, evidence)
    single_read = False
    if not rows and isinstance(evidence, dict):
        # A step completed by the policy carries one geometry read in its own
        # evidence and no per-sample rows (its placement finalizer did not run).
        # That is enough to judge seat depth and edge margin; it is *not*
        # enough to measure drift, which is then reported as not measured
        # instead of as zero.
        fallback, fallback_source = views([{"relation": evidence.get("relation") or {}}], evidence)
        if fallback:
            rows, source, single_read = fallback, "evidence.relation", True
    block: dict[str, Any] = {
        "schema": SCHEMA,
        "step_id": step.get("step_id"),
        "step_task_key": step.get("step_task_key"),
        "enabled": bool(opts["enabled"]),
        "checked": False,
        "measurable": False,
        "single_read": False,
        "drift_measured": False,
        "ok": None,
        "ready": None,
        "risk": None,
        "reason_code": None,
        "reason": None,
        "reasons": [],
        "features": {},
        "thresholds": {
            key: opts[key]
            for key in (
                "max_seat_delta_m",
                "min_seat_delta_m",
                "min_xy_margin_m",
                "max_relative_drift_m",
                "samples_required",
            )
        },
        "evidence": {
            "method": "geometry on post-release samples",
            "source": source,
            "samples": len(rows),
            "single_read": bool(single_read),
            "observation_ids": [row.get("observation_id") for row in rows],
            "sim_action_counts": [row.get("sim_action_count") for row in rows],
        },
        "note": (
            "Independent of the step goal: the BDDL relation says the object is "
            "in the container, this says whether it is seated deep enough to "
            "survive being carried. Never a success judgement."
        ),
    }
    if not opts["enabled"]:
        block["reason_code"] = "disabled"
        block["reason"] = "transport readiness is disabled in config"
        return block

    inner, height = ({}, {})
    if rows:
        inner, height = rows[-1]["relation"], rows[-1]["height"]
    if not inner and isinstance(relation, dict):
        inner = relation.get("relation") if isinstance(relation.get("relation"), dict) else relation
        if isinstance(relation.get("height"), dict):
            height = relation["height"]
    if not isinstance(inner, dict) or inner.get("predicate") != "bowl_interior":
        block["reason_code"] = "not_applicable"
        block["reason"] = REASON_LABELS["not_applicable"]
        block["checked"] = False
        block["ok"] = None
        return block

    block["checked"] = True
    xy = inner.get("xy_m")
    limit = inner.get("xy_limit_m")
    bottom = inner.get("object_bottom_z")
    floor = inner.get("floor_z")
    rim = inner.get("rim_z")
    seat = None
    if isinstance(bottom, (int, float)) and isinstance(floor, (int, float)):
        seat = round(float(bottom) - float(floor), 5)
    margin = None
    if isinstance(xy, (int, float)) and isinstance(limit, (int, float)):
        margin = round(float(limit) - float(xy), 5)
    motion = drift(rows)
    block["features"] = {
        "xy_m": xy,
        "xy_limit_m": limit,
        "xy_margin_m": margin,
        "seat_delta_m": seat,
        "object_bottom_z": bottom,
        "floor_z": floor,
        "rim_z": rim,
        "rim_clearance_m": (
            round(float(rim) - float(bottom), 5)
            if isinstance(rim, (int, float)) and isinstance(bottom, (int, float))
            else None
        ),
        "relation_ok": bool(inner.get("ok")),
        "height_ready": height.get("ready"),
        "height_kind": height.get("kind"),
        "relative_drift": motion,
    }

    # Measured, and how much: geometry needs one read, drift needs two. A
    # missing drift is reported as unmeasured, never as "no motion".
    block["measurable"] = bool(rows)
    block["single_read"] = single_read
    block["drift_measured"] = len(rows) >= 2 and motion.get("samples", 0) >= 2
    if not block["measurable"]:
        block["reasons"] = ["samples_insufficient"]
        block["warning_reasons"] = []
        block["reason_code"] = "samples_insufficient"
        block["reason"] = REASON_LABELS["samples_insufficient"]
        block["risk"] = "not_measured"
        block["ok"] = None
        block["ready"] = None
        return block

    hard: list[str] = []
    if inner.get("ok") is not True:
        hard.append("relation_not_met")
    if seat is None:
        hard.append("relation_not_met")
    elif seat < float(opts["min_seat_delta_m"]):
        hard.append("fell_beside_bowl")
    if margin is not None and margin < float(opts["min_xy_margin_m"]):
        hard.append("edge_of_bowl")
    peak = motion.get("max_distance_m")
    if (
        block["drift_measured"]
        and isinstance(peak, (int, float))
        and float(peak) > float(opts["max_relative_drift_m"])
    ):
        hard.append("still_moving")

    # A rim seat is a warning, not a rejection: 11 of the 12 archived perched
    # rounds carried the cheese without losing it.
    warning: list[str] = []
    if seat is not None and seat > float(opts["max_seat_delta_m"]):
        if opts.get("perched_is_warning", True):
            warning.append("perched_above_floor")
        else:
            hard.append("perched_above_floor")

    risk = "nominal"
    if "fell_beside_bowl" in hard:
        risk = "fell"
    elif "edge_of_bowl" in hard:
        risk = "edge"
    elif "still_moving" in hard:
        risk = "moving"
    elif "relation_not_met" in hard:
        risk = "broken"
    elif warning:
        risk = "perched"

    block["reasons"] = hard + warning
    block["hard_reasons"] = hard
    block["warning_reasons"] = warning
    block["warning"] = warning[0] if warning else None
    block["reason_code"] = hard[0] if hard else None
    block["reason"] = REASON_LABELS.get(block["reason_code"] or (warning[0] if warning else ""))
    block["risk"] = risk
    block["ok"] = not hard
    block["ready"] = not hard
    return block


def change(
    before: dict[str, Any] | None,
    after: dict[str, Any] | None,
    *,
    tolerance_m: float = 0.002,
) -> dict[str, Any]:
    """How much the re-seat moved the object, in the same units as the check.

    A re-seat re-runs the catalog capability from live state, so the policy may
    decide the placement goal is already satisfied and emit almost no motion.
    Recording the before/after geometry keeps that case from being read as a
    repair: ``changed = False`` says the attempt executed and nothing moved.
    """
    left = before or {}
    right = after or {}

    def _step(low: Any, high: Any) -> float | None:
        if not isinstance(low, (int, float)) or not isinstance(high, (int, float)):
            return None
        return round(float(high) - float(low), 5)

    seat = _step(left.get("seat_delta_m"), right.get("seat_delta_m"))
    xy = _step(left.get("xy_m"), right.get("xy_m"))
    moved = [abs(value) for value in (seat, xy) if value is not None]
    changed = bool(moved) and max(moved) >= tolerance_m
    return {
        "seat_delta_before_m": left.get("seat_delta_m"),
        "seat_delta_after_m": right.get("seat_delta_m"),
        "seat_delta_change_m": seat,
        "xy_before_m": left.get("xy_m"),
        "xy_after_m": right.get("xy_m"),
        "xy_change_m": xy,
        "tolerance_m": tolerance_m,
        "changed": changed,
        "note": (
            "the re-seat moved the object"
            if changed
            else (
                "the re-seat ran but the geometry did not change: the whole-task "
                "policy treats the placement goal as already satisfied"
            )
        ),
    }


def summarise(block: dict[str, Any] | None) -> dict[str, Any]:
    """The short row the composite record and the round summary carry."""
    block = block or {}
    return {
        "checked": block.get("checked"),
        "ok": block.get("ok"),
        "risk": block.get("risk"),
        "warning": block.get("warning"),
        "reason_code": block.get("reason_code"),
        "seat_delta_m": (block.get("features") or {}).get("seat_delta_m"),
        "xy_margin_m": (block.get("features") or {}).get("xy_margin_m"),
        "relative_drift_m": ((block.get("features") or {}).get("relative_drift") or {}).get(
            "max_distance_m"
        ),
    }


__all__ = [
    "DEFAULTS",
    "REASON_LABELS",
    "SCHEMA",
    "applies",
    "change",
    "drift",
    "evaluate",
    "options",
    "summarise",
]
