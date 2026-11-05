#!/usr/bin/env python
"""Where a carried object first starts to slide, charged to a phase, not to a guess.

The stored episode already has one snapshot per executed action, so the whole
comparison below is read-only: nothing here steps the simulator or issues an
inference request.

Why phases and not action numbers
    Two runs of the same composite spend a different number of actions on the
    approach, so "action 39" of one run is not the same moment as "action 39"
    of another. Every row is therefore charged to a phase derived from the
    measured state, and only then compared:

    approach   gripper still open, the container has not left its start height
    grasp      the fingers are closing and the container still rests on the table
    lift       the container has left the table and is still rising
    carry      the container is off the table and roughly level
    place      the container is off the table and descending
    retreat    the fingers have opened again after the carry

    A phase is a label on evidence (gripper opening, container height above its
    step-start height, smoothed vertical speed of the container), never a
    constant read off one run.

Why a measured noise floor
    The drift threshold is not "5 mm because 5 mm sounds small". It is taken
    from the segment of the same run in which the container is provably
    untouched: gripper open, container still resting on its start height. In
    that segment the object cannot move, so whatever the reported relative
    position does there is measurement jitter. The threshold is the larger of
    three times that jitter and 5 mm, and both numbers are printed.

    python scripts/transport_phase_compare.py --record a.json --record b.json \
        --step 2 --out compare.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT / "scripts"))

import transport_trace_report as trace  # noqa: E402

BOWL = "akita_black_bowl_1"
CHEESE = "cream_cheese_1"

OPEN_GRIP_M = 0.065          # above this the fingers are not holding anything
LIFT_EPS_M = 0.003           # container height above its start that counts as lifted
VZ_EPS_MPS = 0.004           # smoothed vertical speed that counts as rising/falling
MIN_DRIFT_M = 0.005
NOISE_K = 3.0


def median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def smooth(values: list[float | None], window: int = 3) -> list[float | None]:
    out: list[float | None] = []
    half = window // 2
    for index in range(len(values)):
        span = [v for v in values[max(0, index - half): index + half + 1] if isinstance(v, (int, float))]
        out.append(median(span))
    return out


def root_of(name: Any) -> str:
    """Simulator body name -> the catalog object it belongs to.

    ``akita_black_bowl_1_main`` and ``akita_black_bowl_1_g13`` (a collision
    geom) are both the bowl; the fingers are ``gripper0_*``. Matching raw
    strings would silently miss every pair whose partner happens to be a
    sub-body, which is most of them.
    """
    text = str(name or "")
    if text.startswith("gripper") or text.startswith("robot"):
        return "<gripper>"
    for suffix in ("_main",):
        if text.endswith(suffix):
            text = text[: -len(suffix)]
    head, _, tail = text.rpartition("_g")
    if head and tail.isdigit():
        text = head
    return text


def touches(contacts: list[dict[str, Any]] | None, a: str, b: str) -> bool:
    for item in contacts or []:
        if {root_of(item.get("a")), root_of(item.get("b"))} == {a, b}:
            return True
    return False


def gripper_touch(contacts: list[dict[str, Any]] | None, body: str) -> bool:
    for item in contacts or []:
        names = {root_of(item.get("a")), root_of(item.get("b"))}
        if body in names and "<gripper>" in names:
            return True
    return False


def contact_state(row: dict[str, Any]) -> str:
    """One of ``none``, ``present``, ``truncated``, ``unavailable``.

    ``none`` and ``unavailable`` are different claims: an empty list from a
    snapshot that never carried contacts proves nothing about contact, whereas
    an empty list from a full snapshot does. A list that was cut by the engine's
    cap is reported as truncated so a missing pair is not read as "no contact".
    """
    if not row.get("contacts_available"):
        return "unavailable"
    total = row.get("contacts_total")
    listed = len(row.get("contacts") or [])
    if isinstance(total, int) and total > listed:
        return "truncated"
    return "present" if listed else "none"


def series_of(record: dict[str, Any], step_index: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = trace.trace_rows(record, step_index)
    series: list[dict[str, Any]] = []
    for row in rows:
        timeline_row = row.get("dyn") or {}
        dyn = timeline_row.get("dynamics") or {}
        bowl = trace.body(dyn, BOWL)
        cheese = trace.body(dyn, CHEESE)
        geometry = timeline_row.get("relative") or {}
        geometry_rows = [v for v in geometry.values() if isinstance(v, dict)]
        rel_vel = dyn.get("relative_vel") or {}
        rel_speeds = [v.get("speed") for v in rel_vel.values() if isinstance(v, dict)]
        series.append(
            {
                "action": row.get("action"),
                "chunk_id": dyn.get("chunk_id"),
                "grip": (row.get("trace") or {}).get("grip"),
                "eef": (row.get("trace") or {}).get("eef"),
                "bowl_pos": (bowl or {}).get("pos"),
                "bowl_vel": (bowl or {}).get("vel"),
                "bowl_tilt": (bowl or {}).get("tilt_deg"),
                "cheese_pos": (cheese or {}).get("pos"),
                "cheese_tilt": (cheese or {}).get("tilt_deg"),
                "rel_xy": geometry_rows[0].get("xy_m") if geometry_rows else None,
                "rel_speed": max([s for s in rel_speeds if isinstance(s, (int, float))], default=None),
                "contacts": dyn.get("contacts") or [],
                "contacts_total": dyn.get("contacts_total"),
                "contacts_available": dyn.get("contacts_available"),
            }
        )
    for row in series:
        row["contact_state"] = contact_state(row)
    # height of the container above the height it had when the step started
    start_z = next(
        (float(row["bowl_pos"][2]) for row in series if isinstance(row.get("bowl_pos"), (list, tuple)) and len(row["bowl_pos"]) == 3),
        None,
    )
    for row in series:
        pos = row.get("bowl_pos")
        row["lift_m"] = (
            float(pos[2]) - start_z if isinstance(pos, (list, tuple)) and len(pos) == 3 and start_z is not None else None
        )
    return series, {"rows": len(rows)}


def phases_of(series: list[dict[str, Any]]) -> list[str]:
    vz = smooth([row["bowl_vel"][2] if isinstance(row.get("bowl_vel"), (list, tuple)) else None for row in series])
    labels: list[str] = []
    seen_carry = False
    released = False
    for index, row in enumerate(series):
        lift = row.get("lift_m")
        grip = row.get("grip")
        if released or (seen_carry and isinstance(grip, float) and grip >= OPEN_GRIP_M and (lift or 0) < LIFT_EPS_M):
            released = seen_carry
            labels.append("retreat")
            continue
        if lift is None or lift < LIFT_EPS_M:
            labels.append("approach" if not isinstance(grip, float) or grip >= OPEN_GRIP_M else "grasp")
            continue
        speed_z = vz[index]
        if isinstance(speed_z, (int, float)) and speed_z > VZ_EPS_MPS:
            labels.append("lift")
        elif isinstance(speed_z, (int, float)) and speed_z < -VZ_EPS_MPS:
            labels.append("place")
            seen_carry = True
        else:
            labels.append("carry")
            seen_carry = True
    return labels


def noise_floor(series: list[dict[str, Any]], labels: list[str]) -> dict[str, Any]:
    still = [row for row, label in zip(series, labels) if label == "approach"]
    deltas: list[float] = []
    base = None
    for row in still:
        value = row.get("rel_xy")
        if not isinstance(value, (int, float)):
            continue
        if base is None:
            base = value
        deltas.append(abs(value - base))
    sigma = median(deltas) if deltas else None
    threshold = max(MIN_DRIFT_M, NOISE_K * sigma) if sigma is not None else MIN_DRIFT_M
    return {
        "window": "gripper open and container still on its start height",
        "samples": len(still),
        "median_abs_deviation_m": sigma,
        "sigma_rule": f"threshold = max({MIN_DRIFT_M}, {NOISE_K} * median_abs_deviation)",
        "drift_threshold_m": threshold,
    }


def analyse(record: dict[str, Any], step_index: int) -> dict[str, Any]:
    series, meta = series_of(record, step_index)
    labels = phases_of(series)
    for row, label in zip(series, labels):
        row["phase"] = label
    floor = noise_floor(series, labels)
    threshold = floor["drift_threshold_m"]

    start_cheese = next((row.get("cheese_pos") for row in series if isinstance(row.get("cheese_pos"), list)), None)
    start_bowl = next((row.get("bowl_pos") for row in series if isinstance(row.get("bowl_pos"), list)), None)
    outward = None
    if start_cheese and start_bowl:
        dx, dy = start_cheese[0] - start_bowl[0], start_cheese[1] - start_bowl[1]
        norm = (dx * dx + dy * dy) ** 0.5
        if norm > 1e-6:
            outward = (dx / norm, dy / norm)

    # per-action drift, signed along the initial bowl->object direction so that
    # "sliding further out" and "sliding back in" do not look identical. The
    # vector is taken between the two bodies, not from the world origin, so the
    # transport motion of the container cancels out.
    prev_cheese = None
    base_vec = None
    for row in series:
        cheese, bowl = row.get("cheese_pos"), row.get("bowl_pos")
        drift_radial = None
        drift_vector = None
        if isinstance(cheese, list) and isinstance(bowl, list):
            vec = (cheese[0] - bowl[0], cheese[1] - bowl[1])
            if base_vec is None:
                base_vec = vec
            delta = (vec[0] - base_vec[0], vec[1] - base_vec[1])
            drift_vector = (delta[0] * delta[0] + delta[1] * delta[1]) ** 0.5
            if outward:
                drift_radial = delta[0] * outward[0] + delta[1] * outward[1]
        row["drift_radial_m"] = drift_radial
        row["drift_vector_m"] = drift_vector
        row["delta_xy_from_step_start_m"] = drift_vector
        step_speed = None
        if isinstance(cheese, list) and isinstance(prev_cheese, list):
            step_speed = sum((cheese[i] - prev_cheese[i]) ** 2 for i in range(2)) ** 0.5
        row["cheese_step_xy_m"] = step_speed
        if isinstance(cheese, list):
            prev_cheese = cheese

    first = None
    for row in series:
        if isinstance(row.get("delta_xy_from_step_start_m"), float) and row["delta_xy_from_step_start_m"] > threshold:
            first = {
                "action": row["action"],
                "phase": row["phase"],
                "chunk_id": row.get("chunk_id"),
                "delta_xy_m": row.get("delta_xy_from_step_start_m"),
                "rel_xy_m": row.get("rel_xy"),
                "drift_radial_m": row.get("drift_radial_m"),
                "bowl_tilt_deg": row.get("bowl_tilt"),
                "rel_speed_mps": row.get("rel_speed"),
                "threshold_m": threshold,
            }
            break

    chunks: list[dict[str, Any]] = []
    for row in series:
        cid = row.get("chunk_id")
        if chunks and chunks[-1]["chunk_id"] == cid:
            chunks[-1]["rows"].append(row)
        else:
            chunks.append({"chunk_id": cid, "rows": [row]})
    chunk_rows: list[dict[str, Any]] = []
    for chunk in chunks:
        rows = chunk["rows"]
        phases = sorted({r["phase"] for r in rows})
        chunk_rows.append(
            {
                "chunk_id": chunk["chunk_id"],
                "actions": [rows[0]["action"], rows[-1]["action"]],
                "phases": phases,
                "grip_min": min([r["grip"] for r in rows if isinstance(r.get("grip"), float)], default=None),
                "lift_max_m": max(
                    [
                        (r["bowl_pos"][2] - series[0]["bowl_pos"][2])
                        for r in rows
                        if isinstance(r.get("bowl_pos"), list) and isinstance(series[0].get("bowl_pos"), list)
                    ],
                    default=None,
                ),
                "bowl_tilt_max_deg": max([r["bowl_tilt"] for r in rows if isinstance(r.get("bowl_tilt"), float)], default=None),
                "rel_xy_start_m": next((r["rel_xy"] for r in rows if isinstance(r.get("rel_xy"), float)), None),
                "rel_xy_end_m": next((r["rel_xy"] for r in reversed(rows) if isinstance(r.get("rel_xy"), float)), None),
                "drift_radial_end_m": next((r["drift_radial_m"] for r in reversed(rows) if isinstance(r.get("drift_radial_m"), float)), None),
                "rel_speed_max_mps": max([r["rel_speed"] for r in rows if isinstance(r.get("rel_speed"), float)], default=None),
                "object_container_touch": sum(1 for r in rows if touches(r.get("contacts"), BOWL, CHEESE)),
                "gripper_object_touch": sum(1 for r in rows if gripper_touch(r.get("contacts"), CHEESE)),
                "gripper_container_touch": sum(1 for r in rows if gripper_touch(r.get("contacts"), BOWL)),
                "contact_states": sorted({r["contact_state"] for r in rows}),
                "contacts_available": rows[-1].get("contacts_available"),
                "contacts_total_max": max(
                    [r["contacts_total"] for r in rows if isinstance(r.get("contacts_total"), int)], default=None
                ),
            }
        )

    tail = series[-1]
    return {
        "record": str(record.get("__path__") or ""),
        "episode_id": record.get("episode_id"),
        "seed": (record.get("env") or {}).get("seed"),
        "policy_seed": (record.get("sampling") or {}).get("policy_seed"),
        "user_goal_success": record.get("user_goal_success"),
        "failure_code": record.get("failure_code"),
        "step": step_index,
        "samples": meta["rows"],
        "noise": floor,
        "first_drift": first,
        "final": {
            "rel_xy_m": tail.get("rel_xy"),
            "drift_radial_m": tail.get("drift_radial_m"),
            "bowl_tilt_deg": tail.get("bowl_tilt"),
            "phase": tail.get("phase"),
        },
        "phase_spans": _spans(labels, series),
        "chunks": chunk_rows,
        "series": series,
    }


def _spans(labels: list[str], series: list[dict[str, Any]]) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    for label, row in zip(labels, series):
        if spans and spans[-1]["phase"] == label:
            spans[-1]["actions"][1] = row["action"]
        else:
            spans.append({"phase": label, "actions": [row["action"], row["action"]]})
    return spans


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", action="append", required=True)
    parser.add_argument("--step", type=int, default=2)
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    reports = []
    for path_text in args.record:
        path = Path(path_text)
        record = json.loads(path.read_text())
        record["__path__"] = str(path)
        reports.append(analyse(record, args.step))
    payload = {
        "schema": "embodied_engine.transport_phase_compare.v1",
        "note": (
            "read-only: every number comes from the per-action snapshots the engine already stored. "
            "Phases are derived from measured gripper opening and container height/vertical speed; "
            "the drift threshold comes from the untouched segment of the same run"
        ),
        "records": reports,
    }
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2))
        print(f"[ok] {out}")
    def mm(value: Any, width: int) -> str:
        return f"{1000 * value:{width}.2f}" if isinstance(value, (int, float)) else " " * (width - 2) + "--"

    print(f"{'episode':24} {'seed':>4} {'ok':>5} {'thr':>6} {'first':>5} {'phase':>8} {'dxy':>6} {'radial':>6} {'final':>6}")
    for report in reports:
        name = str(report.get("episode_id") or Path(report["record"]).stem)[-24:]
        first = report.get("first_drift") or {}
        tail = report.get("final") or {}
        print(
            f"{name:24} {str(report.get('seed')):>4} {str(report.get('user_goal_success')):>5} "
            f"{mm(report['noise']['drift_threshold_m'], 6)} "
            f"{str(first.get('action')):>5} {str(first.get('phase')):>8} "
            f"{mm(first.get('delta_xy_m'), 6)} {mm(tail.get('drift_radial_m'), 6)} {mm(tail.get('rel_xy_m'), 6)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
