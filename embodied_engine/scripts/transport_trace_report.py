#!/usr/bin/env python
"""First point of instability in a carried object, read from a stored episode.

The episode record already carries one row per executed action: the engine
takes a snapshot after every action, so pose/quaternion/velocity of every
watched body, the relative velocity between the carried object and its
container, and the live contact pairs cost no extra round trip and no extra
simulator step. This script only reads that trace.

It deliberately does **not** decide that a drop happened. It reports, in
action order, the first action at which each signal leaves the range the
*stable* segment of the same run occupies, and it prints the measured range so
a reader can see how wide the band was instead of trusting a constant.

    python scripts/transport_trace_report.py --record ep.json --out report.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parents[1]

BOWL = "akita_black_bowl_1"
CHEESE = "cream_cheese_1"


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def step_spans(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Action-count span of every step, in the order the engine ran them."""
    spans: list[dict[str, Any]] = []
    cursor = 0
    for step in record.get("steps") or []:
        actions = int(step.get("sim_actions") or 0)
        spans.append(
            {
                "index": step.get("index"),
                "step_id": step.get("step_id"),
                "task_key": (step.get("step") or {}).get("task_key"),
                "start": cursor,
                "end": cursor + actions,
                "actions": actions,
                "ok": step.get("ok"),
                "ok_final": step.get("ok_final"),
            }
        )
        cursor += actions
    return spans


def trace_rows(record: dict[str, Any], step_index: int) -> list[dict[str, Any]]:
    """Actions of one step, each paired with the snapshot taken *after* it.

    ``action_trace`` numbers actions within the step (1..N); ``goal_watch.timeline``
    numbers snapshots by the episode-wide ``sim_action_count`` and includes one
    row for the state the step started from. The two are therefore joined by
    order, not by value: ``timeline[0]`` is the pre-action state and
    ``timeline[i + 1]`` is the state after the i-th action.
    """
    steps = record.get("steps") or []
    if step_index < 1 or step_index > len(steps):
        return []
    step = steps[step_index - 1]
    timeline = [
        row
        for row in (((step.get("goal_watch") or {}).get("timeline")) or [])
        if isinstance(row, dict)
    ]
    timeline.sort(key=lambda row: row.get("sim_action_count") if isinstance(row.get("sim_action_count"), int) else -1)
    out: list[dict[str, Any]] = []
    for index, row in enumerate(step.get("action_trace") or []):
        if not isinstance(row, dict):
            continue
        after = timeline[index + 1] if index + 1 < len(timeline) else (timeline[-1] if timeline else None)
        out.append({"action": row.get("n"), "trace": row, "dyn": after, "start": timeline[0] if timeline else None})
    return out


def body(dyn: dict[str, Any] | None, name: str) -> dict[str, Any] | None:
    """One body from a timeline row's ``dynamics`` block, by catalog or body name."""
    if not isinstance(dyn, dict):
        return None
    rows = dyn.get("bodies") if "bodies" in dyn else None
    if not isinstance(rows, dict):
        return None
    entry = rows.get(name)
    if isinstance(entry, dict):
        return entry
    for key, value in rows.items():
        if isinstance(value, dict) and (key.startswith(f"{name}_") or value.get("body") == name):
            return value
    return None


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = q * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def analyse(record: dict[str, Any], step_index: int) -> dict[str, Any]:
    spans = step_spans(record)
    rows = trace_rows(record, step_index)
    series: list[dict[str, Any]] = []
    for row in rows:
        timeline_row = row.get("dyn") or {}
        dyn = timeline_row.get("dynamics") or {}
        bowl = body(dyn, BOWL)
        cheese = body(dyn, CHEESE)
        rel = dyn.get("relative_vel") or {}
        rel_speeds = [v for v in rel.values() if isinstance(v, dict)]
        geometry = timeline_row.get("relative") or {}
        geometry_rows = [v for v in geometry.values() if isinstance(v, dict)]
        entries = [
            {
                "a": item.get("a"),
                "b": item.get("b"),
                "dist": item.get("dist"),
                "robot": item.get("robot"),
            }
            for item in (dyn.get("contacts") or [])
            if isinstance(item, dict)
        ]
        series.append(
            {
                "action": row.get("action"),
                "sim_action_count": timeline_row.get("sim_action_count"),
                "chunk_id": dyn.get("chunk_id"),
                "tilt_deg": None if not bowl else bowl.get("tilt_deg"),
                "cheese_tilt_deg": None if not cheese else cheese.get("tilt_deg"),
                "bowl_pos": None if not bowl else bowl.get("pos"),
                "bowl_vel": None if not bowl else bowl.get("vel"),
                "cheese_pos": None if not cheese else cheese.get("pos"),
                "cheese_vel": None if not cheese else cheese.get("vel"),
                "relative_speed": (
                    max(float(v.get("speed")) for v in rel_speeds if isinstance(v.get("speed"), (int, float)))
                    if rel_speeds
                    else None
                ),
                "relative_xy_m": (
                    geometry_rows[0].get("xy_m")
                    if geometry_rows
                    else None
                ),
                "relative_distance_m": (
                    geometry_rows[0].get("distance_m")
                    if geometry_rows
                    else None
                ),
                "grip": (row.get("trace") or {}).get("grip"),
                "eef": (row.get("trace") or {}).get("eef"),
                "contacts": entries,
                "contacts_available": dyn.get("contacts_available"),
            }
        )
    observed = [row for row in series if row.get("tilt_deg") is not None]
    speeds = [row["relative_speed"] for row in observed if isinstance(row.get("relative_speed"), (int, float))]
    tilts = [row["tilt_deg"] for row in observed if isinstance(row.get("tilt_deg"), (int, float))]
    # The stable band is measured on the first third of the carry: the arm is
    # already moving, but the object has not been disturbed yet. A constant
    # threshold would be a guess; this is the same run's own quiet segment.
    quiet_n = max(1, len(observed) // 3)
    quiet = observed[:quiet_n]
    quiet_speeds = [row["relative_speed"] for row in quiet if isinstance(row.get("relative_speed"), (int, float))]
    quiet_tilts = [row["tilt_deg"] for row in quiet if isinstance(row.get("tilt_deg"), (int, float))]

    def first_above(key: str, limit: float | None) -> dict[str, Any] | None:
        if limit is None:
            return None
        for row in observed:
            value = row.get(key)
            if isinstance(value, (int, float)) and value > limit:
                return {"action": row["action"], "value": round(float(value), 5), "limit": round(limit, 5),
                        "chunk_id": row.get("chunk_id")}
        return None

    speed_limit = None
    if quiet_speeds:
        speed_limit = max(percentile(quiet_speeds, 0.99) or 0.0, 0.02)
    tilt_limit = None
    if quiet_tilts:
        tilt_limit = max(percentile(quiet_tilts, 0.99) or 0.0, max(quiet_tilts) + 5.0)
    return {
        "record": str(record.get("__path__") or ""),
        "episode_id": record.get("episode_id"),
        "seed": (record.get("env") or {}).get("seed"),
        "policy_seed": (record.get("sampling") or {}).get("policy_seed"),
        "failure_code": record.get("failure_code"),
        "user_goal_success": record.get("user_goal_success"),
        "step_spans": spans,
        "step": step_index,
        "samples": len(series),
        "measured_band": {
            "relative_speed_quiet_p99": speed_limit,
            "relative_speed_whole_p99": percentile(speeds, 0.99),
            "relative_speed_whole_max": max(speeds) if speeds else None,
            "tilt_quiet_max_deg": max(quiet_tilts) if quiet_tilts else None,
            "tilt_whole_max_deg": max(tilts) if tilts else None,
            "tilt_limit_deg": tilt_limit,
            "quiet_samples": quiet_n,
        },
        "first_exceed": {
            "relative_speed": first_above("relative_speed", speed_limit),
            "tilt": first_above("tilt_deg", tilt_limit),
        },
        "series": series,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", required=True)
    parser.add_argument("--step", type=int, default=2, help="1-based step index to trace")
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    path = Path(args.record)
    record = load(path)
    record["__path__"] = str(path)
    report = analyse(record, args.step)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2))
        print(f"[ok] {out}")
    summary = {k: report[k] for k in ("episode_id", "seed", "policy_seed", "failure_code",
                                      "user_goal_success", "samples", "measured_band", "first_exceed")}
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
