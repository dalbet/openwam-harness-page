#!/usr/bin/env python
"""Where exactly does step 2 move the drawer step 1 opened?

Reads archived composite rounds and reports, per round:

* the drawer joint's value at the handoff (the state step 2 was handed),
* the first action (whole-episode and step-local) after which the joint moved
  by more than 0.1 mm, with the contact that was measured at that instant,
* the first action after which it moved by more than 5 mm (the point where the
  regression stops being contact chatter),
* the drawer value at the end of the episode and the regression in mm,
* the bowl's trajectory (start / min z / end) while step 2 ran.

The joint name is read from the record itself; nothing here is specific to the
middle drawer.

    python scripts/summarize_drawer_regression.py \
        --rounds records/acceptance/composite_restore_trace_6_20261028.json \
        --out records/acceptance/composite_restore_trace_6_20261028_drawer.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parents[1]


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _joint_from_goal(record: dict[str, Any]) -> str | None:
    handoff = ((record.get("composite") or {}).get("handoff") or {})
    goal = ((handoff.get("pre") or {}).get("goal") or {})
    return goal.get("joint")


def _baseline_qpos(record: dict[str, Any], joint: str | None) -> float | None:
    handoff = ((record.get("composite") or {}).get("handoff") or {})
    goal = ((handoff.get("pre") or {}).get("goal") or {})
    if goal.get("qpos") is not None:
        return float(goal["qpos"])
    for step in (record.get("composite") or {}).get("steps") or []:
        evidence = step.get("evidence") or {}
        if evidence.get("joint") == joint and evidence.get("qpos") is not None:
            return float(evidence["qpos"])
    return None


def _end_qpos(record: dict[str, Any], joint: str | None) -> tuple[float | None, str | None]:
    evidence = record.get("failure_evidence") or {}
    for row in evidence.get("steps") or []:
        verification = row.get("verification") or {}
        if joint and verification.get("qpos") is not None:
            value = verification["qpos"]
            if isinstance(value, list) and value:
                return float(value[0]), "final_verification_samples"
            if isinstance(value, (int, float)):
                return float(value), "final_verification_samples"
    for row in evidence.get("prior_steps_at_failure") or []:
        state = row.get("state") or {}
        if state.get("joint") == joint and state.get("qpos") is not None:
            return float(state["qpos"]), "prior_step_state_at_failure"
    return None, None


def _walk(record: dict[str, Any]) -> dict[str, Any] | None:
    """The step that ran after the drawer was opened (raw runner result)."""
    composite_steps = (record.get("composite") or {}).get("steps") or []
    raw_steps = record.get("steps") or []
    if len(composite_steps) < 2:
        return None
    wanted = composite_steps[1].get("step_id")
    for row in raw_steps:
        if row.get("step_id") == wanted:
            return row
    return raw_steps[1] if len(raw_steps) > 1 else None


def trace(record: dict[str, Any]) -> dict[str, Any]:
    joint = _joint_from_goal(record)
    out: dict[str, Any] = {
        "episode_id": record.get("episode_id"),
        "handoff_mode": ((record.get("composite") or {}).get("handoff_mode")),
        "seed": (record.get("env") or {}).get("seed"),
        "drawer_joint": joint,
        "handoff_qpos": _baseline_qpos(record, joint),
        "first_move": None,
        "first_move_5mm": None,
        "end_qpos": None,
        "end_source": None,
        "regression_mm": None,
        "step_2": None,
    }
    end, source = _end_qpos(record, joint)
    out["end_qpos"] = end
    out["end_source"] = source
    if end is not None and out["handoff_qpos"] is not None:
        out["regression_mm"] = round((end - float(out["handoff_qpos"])) * 1000.0, 1)

    step2 = _walk(record)
    if step2 is None:
        return out
    out["step_2"] = {
        "sim_actions": step2.get("sim_actions"),
        "duration_s": step2.get("duration_s"),
        "ok": step2.get("ok"),
        "stop_reason": step2.get("stop_reason"),
    }
    baseline_step2 = (step2.get("watch_baseline") or {}).get(joint) if joint else None
    out["step2_baseline_qpos"] = None if baseline_step2 is None else round(float(baseline_step2), 6)
    if baseline_step2 is not None:
        # What the handoff itself moved (detach + settle + retreat) is separate
        # from what step 2 moved; both are reported so neither is attributed to
        # the other.
        if out["handoff_qpos"] is not None:
            out["handoff_to_step2_mm"] = round(
                (float(baseline_step2) - float(out["handoff_qpos"])) * 1000.0, 1
            )
        if end is not None:
            out["regression_from_baseline_mm"] = round(
                (float(end) - float(baseline_step2)) * 1000.0, 1
            )
    events = step2.get("watch_events") or []
    if events:
        event = dict(events[0])
        event["contacts"] = {
            key: value
            for key, value in (event.get("contacts") or {}).items()
            if key in {"count", "penetration_max_m"}
        }
        pairs = (events[0].get("contacts") or {}).get("gripper_vs_nonrobot") or []
        event["contact_pairs"] = [
            {"geom1": pair.get("geom1"), "geom2": pair.get("geom2"), "dist": pair.get("dist")}
            for pair in pairs[:4]
        ]
        out["first_move"] = event

    trace_rows = step2.get("action_trace") or []
    baseline = ((step2.get("watch_baseline") or {}).get(joint)) if joint else None
    if baseline is not None:
        for row in trace_rows:
            watched = (row.get("watched") or {}).get(joint)
            if watched is None:
                continue
            if abs(float(watched) - float(baseline)) >= 0.005:
                out["first_move_5mm"] = {
                    "skill_sim_actions": row.get("n"),
                    "qpos": round(float(watched), 6),
                    "delta_mm": round((float(watched) - float(baseline)) * 1000.0, 1),
                }
                break
        # the whole trace, thinned to every 25th action, for the report
        thin = []
        for row in trace_rows:
            watched = (row.get("watched") or {}).get(joint)
            if watched is None:
                continue
            if int(row.get("n") or 0) % 25 == 0 or row is trace_rows[0]:
                thin.append(
                    {
                        "n": row.get("n"),
                        "qpos": round(float(watched), 5),
                        "eef": [round(float(v), 4) for v in (row.get("eef") or [])],
                    }
                )
        out["drawer_trace"] = thin

    observations = step2.get("observations") or []
    poses = [row.get("object_pose") for row in observations if row.get("object_pose")]
    if poses:
        zs = [float(pose[2]) for pose in poses]
        out["object_trajectory"] = {
            "samples": len(poses),
            "start": [round(float(v), 4) for v in poses[0]],
            "min_z": round(min(zs), 4),
            "end": [round(float(v), 4) for v in poses[-1]],
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    manifest = load(PROJECT / args.rounds)
    rows = []
    for row in manifest.get("rows") or []:
        path = row.get("record")
        if not path:
            continue
        candidate = Path(path)
        if not candidate.exists():
            continue
        record = load(candidate)
        traced = trace(record)
        traced["round"] = row.get("round")
        rows.append(traced)

    payload = {
        "schema": "embodied_engine.drawer_regression.v1",
        "source": args.rounds,
        "note": (
            "First-movement is the first action after which the earlier step's drawer joint "
            "moved by more than 0.1 mm from the value the handoff handed over; the contact "
            "read at that instant names the geom pair. first_move_5mm is the first action "
            "after which it moved by at least 5 mm."
        ),
        "rounds": rows,
    }
    out = PROJECT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for row in rows:
        first = row.get("first_move") or {}
        print(
            f"{row['round']:<16} {str(row['handoff_mode']):<20} s{row['seed']} "
            f"handoff={row['handoff_qpos']} s2base={row.get('step2_baseline_qpos')} end={row['end_qpos']} "
            f"d_pre={row['regression_mm']}mm d_s2={row.get('regression_from_baseline_mm')}mm "
            f"first=n{first.get('sim_action_count')}/step{first.get('skill_sim_actions')} "
            f"d={first.get('delta')} 5mm={row.get('first_move_5mm')}"
        )
    print(f"out={out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
