#!/usr/bin/env python3
"""Offline: which measured features separate a carried object from a lost one?

Reads archived composite rounds (read-only) and, for the step that places an
object into a container, pulls the geometry that ``common.transport_ready``
uses. The label is *not* the step's own goal: it is the step's fresh re-check
after the whole episode (``final_evidence.ok``), i.e. "was the object still in
the bowl once the episode ended". A round whose step goal was met but whose
fresh re-check failed is exactly the case this rule is about.

Output is a JSON report plus a table, so a threshold can be argued from
archived rounds instead of from one failing number.

Usage:
    python tools/ready_features.py --out /tmp/ready_features.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from common import transport_ready as tr  # noqa: E402

RAW = Path("/home/host/flashrt-exp/pi05_round/raw/composite")


def _iter_records(roots: list[Path]):
    for root in roots:
        for path in sorted(root.rglob("*.json")):
            if path.name.startswith("."):
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001 - unrelated json is skipped
                continue
            if not isinstance(data, dict) or "composite" not in data:
                continue
            yield path, data


def _step_row(data: dict, step_key: str) -> dict | None:
    steps = (data.get("composite") or {}).get("steps") or []
    for step in steps:
        if str(step.get("task_key")) == step_key:
            return step
    return None


def features_for(data: dict, step_key: str, source: str = "evidence") -> dict | None:
    step = _step_row(data, step_key)
    if step is None:
        return None
    row = step.get(source) or {}
    inner = row.get("relation") or {}
    if inner.get("predicate") != "bowl_interior":
        return None
    block = tr.evaluate(
        step={"step_id": step.get("step_id"), "step_task_key": step_key},
        relation=row,
        samples=None,
        evidence=row,
        cfg=None,
    )
    feature = block.get("features") or {}
    return {
        "seat_delta_m": feature.get("seat_delta_m"),
        "xy_m": feature.get("xy_m"),
        "xy_margin_m": feature.get("xy_margin_m"),
        "rim_clearance_m": feature.get("rim_clearance_m"),
        "relative_drift_m": (feature.get("relative_drift") or {}).get("max_distance_m"),
        "drift_mode": (feature.get("relative_drift") or {}).get("mode"),
        "samples": (block.get("evidence") or {}).get("samples"),
        "samples_source": (block.get("evidence") or {}).get("source"),
        "verdict": block.get("ok"),
        "risk": block.get("risk"),
        "reason_code": block.get("reason_code"),
        "reasons": block.get("reasons"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default=str(RAW))
    parser.add_argument("--step-key", default="put_cream_cheese_in_bowl")
    parser.add_argument("--out", default="")
    parser.add_argument("--source", default="evidence", choices=("evidence", "final_evidence"))
    parser.add_argument("--threshold", type=float, default=tr.options(None)["max_seat_delta_m"])
    parser.add_argument("--min-seat", type=float, default=tr.options(None)["min_seat_delta_m"])
    args = parser.parse_args()

    roots = [Path(args.raw)]
    rows = []
    for path, data in _iter_records(roots):
        step = _step_row(data, args.step_key)
        if step is None:
            continue
        feature = features_for(data, args.step_key, args.source)
        if feature is None:
            continue
        survived = (step.get("final_evidence") or {}).get("ok")
        rows.append(
            {
                "record": str(path),
                "episode_id": data.get("episode_id"),
                "tag": Path(path).parent.name,
                "step_ok": step.get("ok"),
                "carried_ok_final": survived,
                "user_goal_success": data.get("user_goal_success"),
                "execution_status": data.get("execution_status"),
                **feature,
            }
        )

    nominal = [r for r in rows if isinstance(r["seat_delta_m"], float) and r["seat_delta_m"] <= args.threshold]
    perched = [r for r in rows if isinstance(r["seat_delta_m"], float) and r["seat_delta_m"] > args.threshold]

    def band(group, key):
        values = [r[key] for r in group if isinstance(r.get(key), float)]
        return [round(min(values), 5), round(max(values), 5)] if values else None

    false_reject = [r["episode_id"] for r in nominal if r["carried_ok_final"] is False]
    missed = [r["episode_id"] for r in perched if r["carried_ok_final"] is not False]
    report = {
        "schema": "embodied_engine.ready_features.v1",
        "raw": args.raw,
        "step_key": args.step_key,
        "features_from": args.source,
        "thresholds": {
            "max_seat_delta_m": args.threshold,
            "min_seat_delta_m": args.min_seat,
        },
        "rounds": len(rows),
        "labelled_lost": sum(1 for r in rows if r["carried_ok_final"] is False),
        "labelled_kept": sum(1 for r in rows if r["carried_ok_final"] is True),
        "nominal": {
            "count": len(nominal),
            "seat_m": band(nominal, "seat_delta_m"),
            "xy_m": band(nominal, "xy_m"),
            "lost_despite_nominal": false_reject,
        },
        "perched": {
            "count": len(perched),
            "seat_m": band(perched, "seat_delta_m"),
            "xy_m": band(perched, "xy_m"),
            "kept_despite_perched": missed,
        },
        "rows": rows,
        "note": (
            "Labels come from each step's fresh final re-check (was the object still in the "
            "container at episode end), never from the step goal. 'kept_despite_perched' is "
            "the false-reject side of the rule and is reported, not hidden."
        ),
    }
    print(f"{'tag':44} {'seat_mm':>8} {'xy_mm':>7} {'rim_mm':>7} {'kept':>5} {'verdict':>8}")
    for row in sorted(rows, key=lambda r: (r["seat_delta_m"] is None, r["seat_delta_m"])):
        seat = row["seat_delta_m"]
        print(
            f"{row['tag'][:44]:44} "
            f"{(seat * 1000 if isinstance(seat, float) else float('nan')):8.1f} "
            f"{(row['xy_m'] * 1000 if isinstance(row['xy_m'], float) else float('nan')):7.1f} "
            f"{(row['rim_clearance_m'] * 1000 if isinstance(row['rim_clearance_m'], float) else float('nan')):7.1f} "
            f"{str(row['carried_ok_final']):>5} "
            f"{str(row['risk']):>8}"
        )
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, ensure_ascii=False, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"written: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
