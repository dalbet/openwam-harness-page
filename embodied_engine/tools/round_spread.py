#!/usr/bin/env python3
"""Side-by-side view of the same frozen manifest run several times.

One batch of six rounds is not a success rate. This tool reads any number of
archived batches of the *same* manifest and prints, per round, what each batch
did, plus the pooled counts. Nothing is averaged into a single "score": the
spread is the finding.

    python tools/round_spread.py \
        --batch A=/path/formal_3step_20261105 \
        --batch B=/path/formal_3step_final_20261105 \
        --out /path/round_spread_20261105.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(batch: Path) -> list[dict]:
    rows = []
    for path in sorted(batch.glob("rounds_runs/*.json")):
        run = json.loads(path.read_text(encoding="utf-8"))
        composite = run.get("composite") or {}
        steps = composite.get("steps") or []
        first = steps[0] if steps else {}
        ready = first.get("transport_ready") or {}
        relation = ((run.get("failure_evidence") or {}).get("steps") or [{}])[0]
        verification = (relation.get("verification") or {}).get("relation") or {}
        rows.append(
            {
                "episode": run.get("episode_id"),
                "instruction": run.get("instruction"),
                "lang": (composite.get("instruction") or {}).get("lang")
                if isinstance(composite.get("instruction"), dict)
                else None,
                "user_goal_success": run.get("user_goal_success"),
                "final_all_goals_met": composite.get("final_all_goals_met"),
                "failure_code": run.get("failure_code"),
                "sim_actions": run.get("sim_action_count"),
                "duration_s": run.get("duration_s"),
                "step_1_met": composite.get("step_1_met"),
                "ready_to_transport": composite.get("ready_to_transport"),
                "ready_step": composite.get("ready_to_transport_step"),
                "reseat_attempted": composite.get("reseat_attempted"),
                "reseat_succeeded": composite.get("reseat_succeeded"),
                "repair_actions": composite.get("repair_actions"),
                "seat_delta_m": ready.get("seat_delta_m"),
                "xy_m": ready.get("xy_m"),
                "xy_margin_m": ready.get("xy_margin_m"),
                "rim_clearance_m": ready.get("rim_clearance_m"),
                "relative_drift_m": ready.get("relative_drift_m"),
                "ready_risk": ready.get("risk"),
                "ready_checked": ready.get("checked"),
                "final_xy_m": verification.get("xy_m"),
                "final_xy_limit_m": verification.get("xy_limit_m"),
                "first_pass": composite.get("first_pass_success"),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", action="append", default=[], help="label=path")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    batches = {}
    for spec in args.batch:
        label, _, path = spec.partition("=")
        batches[label] = load(Path(path))

    order = [row["instruction"] for row in next(iter(batches.values()), [])]
    for instruction in order:
        print(f"\n{instruction}")
        for label, rows in batches.items():
            row = next((r for r in rows if r["instruction"] == instruction), None)
            if row is None:
                print(f"  {label}: (not run)")
                continue
            verdict = "PASS" if row["user_goal_success"] else "FAIL"
            detail = row["failure_code"] or ""
            metric = ""
            if row["final_xy_m"] is not None:
                metric = f" final_xy={row['final_xy_m'] * 1000:.1f}/{row['final_xy_limit_m'] * 1000:.0f}mm"
            seat = "" if row["seat_delta_m"] is None else f" seat={row['seat_delta_m'] * 1000:.1f}mm"
            margin = "" if row["xy_margin_m"] is None else f" margin={row['xy_margin_m'] * 1000:.1f}mm"
            print(
                f"  {label}: {verdict:4s} actions={row['sim_actions']:>4} "
                f"dur={row['duration_s']}s ready={row['ready_to_transport']}"
                f"{seat}{margin}{metric} {detail}"
            )

    print("\npooled")
    for label, rows in batches.items():
        passed = sum(1 for r in rows if r["user_goal_success"])
        mapped = len(rows)
        print(f"  {label}: {passed}/{mapped} goals met")
    per_round = {}
    for instruction in order:
        outcomes = [
            bool(next((r for r in rows if r["instruction"] == instruction), {}).get("user_goal_success"))
            for rows in batches.values()
        ]
        per_round[instruction] = outcomes
        print(f"  [{sum(outcomes)}/{len(outcomes)}] {instruction}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(
                {
                    "schema": "embodied_engine.round_spread.v1",
                    "batches": {label: rows for label, rows in batches.items()},
                    "per_round": per_round,
                },
                ensure_ascii=False,
                indent=1,
            ),
            encoding="utf-8",
        )
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
