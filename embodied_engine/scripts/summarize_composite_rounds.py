#!/usr/bin/env python
"""Compact one-row-per-round table for composite acceptance.

Reads the frozen round manifest plus the archived raw records and prints the
columns that must not be merged: language mapping, step-1 goal (own and final),
step-2 native goal (own and final), release source, action counts, handoff
evidence, and the failure stage. Never fills a missing value with a success.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parents[1]


def record_of(row: dict[str, Any]) -> dict[str, Any]:
    for key in ("archived_record", "record"):
        path = row.get(key)
        if path and Path(path).exists():
            try:
                return json.loads(Path(path).read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
    return {}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", default="records/acceptance/composite_rounds_20261027.json")
    parser.add_argument("--out", default="records/acceptance/composite_rounds_20261027_summary.json")
    args = parser.parse_args()

    data = json.loads((PROJECT / args.rounds).read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for row in data.get("rows") or []:
        record = record_of(row)
        composite = record.get("composite") or {}
        handoff = composite.get("handoff") or {}
        retreat = handoff.get("retreat") or {}
        steps = composite.get("steps") or []
        placement = record.get("placement") or {}
        timing = record.get("timing") or {}
        rows.append(
            {
                "round": row.get("round"),
                "seed": row.get("seed"),
                "lang": row.get("lang"),
                "instruction": row.get("user_instruction"),
                "mapping_ok": row.get("mapping_ok"),
                "router_keys": row.get("composite_keys"),
                "router_latency_s": row.get("router_latency_s"),
                "router_ttft_s": row.get("router_ttft_s"),
                "execution_status": record.get("execution_status", row.get("execution_status")),
                "failure_code": record.get("failure_code", row.get("failure_code")),
                "failure_reason": record.get("failure_reason", row.get("failure_reason")),
                "step_1_ok": composite.get("step_1_ok"),
                "step_1_final_ok": composite.get("step_1_final_ok"),
                "step_1_actions": (steps[0] or {}).get("sim_actions") if steps else None,
                "step_2_native": composite.get("step_2_native"),
                "step_2_final_native": composite.get("step_2_final_native"),
                "step_2_actions": (steps[1] or {}).get("sim_actions") if len(steps) > 1 else None,
                "final_verification_read": composite.get("final_verification_read"),
                "combo_verified": composite.get("combo_verified"),
                "native_goal_current": composite.get("native_goal_current"),
                "native_goal_ever": composite.get("native_goal_ever"),
                "step_1_goal_ever": composite.get("step_1_goal_ever"),
                "scene_epoch_continuous": composite.get("scene_epoch_continuous"),
                "scene_epoch_start": composite.get("scene_epoch_start"),
                "scene_epoch_end": composite.get("scene_epoch_end"),
                "env_reset_between_steps": composite.get("env_reset_between_steps"),
                "env_reload_between_steps": composite.get("env_reload_between_steps"),
                "bddl_mutation": composite.get("bddl_mutation"),
                "object_teleport": composite.get("object_teleport"),
                "handoff_status": handoff.get("status"),
                "handoff_mode": composite.get("handoff_mode") or handoff.get("mode") or row.get("handoff_mode"),
                "robot_state_restore": composite.get("robot_state_restore"),
                "robot_teleport": composite.get("robot_teleport"),
                "physical_continuity": composite.get("physical_continuity"),
                "handoff_reason_code": handoff.get("reason_code") or row.get("handoff_reason_code"),
                "restore_applied": (handoff.get("restore") or {}).get("applied"),
                "restore_verified": (handoff.get("restore") or {}).get("verified"),
                "restore_candidate_checked": ((handoff.get("restore") or {}).get("write") or {}).get("candidate", {}).get("checked")
                if isinstance(((handoff.get("restore") or {}).get("write") or {}).get("candidate"), dict)
                else None,
                "restore_penetration_m": ((handoff.get("restore") or {}).get("write") or {}).get("candidate", {}).get("penetration_max_m")
                if isinstance(((handoff.get("restore") or {}).get("write") or {}).get("candidate"), dict)
                else None,
                "restore_eef_drift_m": (handoff.get("post_restore") or {}).get("eef_drift_m"),
                "detach_actions": (handoff.get("detach") or {}).get("actions"),
                "action_accounting": composite.get("action_accounting"),
                "watch_events": row.get("watch_events"),
                "handoff_actions": handoff.get("sim_actions"),
                "handoff_elapsed_s": handoff.get("elapsed_s"),
                "handoff_start_angle_deg": retreat.get("start_angle_deg"),
                "handoff_end_angle_deg": retreat.get("end_angle_deg"),
                "handoff_retreat_reason": retreat.get("reason"),
                "handoff_buffer_reset": bool((handoff.get("buffer_reset") or {}).get("token")),
                "release_source": placement.get("evidence_source") or "none",
                "release_verified": placement.get("release_verified"),
                "placement_verified": placement.get("placement_verified"),
                "episode_actions": record.get("episode_sim_actions"),
                "duration_s": record.get("duration_s"),
                "timing": {
                    key: timing.get(key)
                    for key in ("planner_s", "execution_s", "handoff_s", "release_s", "verification_s")
                },
                "recovery_count": record.get("recovery_count"),
                "episode_id": record.get("episode_id"),
                "record": row.get("archived_record") or row.get("record"),
            }
        )

    summary = {
        "schema": "embodied_engine.composite_rounds_summary.v1",
        "rounds_file": args.rounds,
        "rows": rows,
        "counts": {
            "rounds": len(rows),
            "mapping_ok": sum(1 for row in rows if row["mapping_ok"]),
            "step_1_ok": sum(1 for row in rows if row["step_1_ok"]),
            "step_2_native": sum(1 for row in rows if row["step_2_native"]),
            "combo_verified": sum(1 for row in rows if row["combo_verified"]),
            "scene_epoch_continuous": sum(1 for row in rows if row["scene_epoch_continuous"]),
            "handoff_ready": sum(1 for row in rows if row["handoff_status"] == "ready"),
        },
        # Grouped, never pooled: a mode's success rate is not the suite's.
        "by_mode": {},
    }
    for mode in sorted({str(row.get("handoff_mode") or "none") for row in rows}):
        subset = [row for row in rows if str(row.get("handoff_mode") or "none") == mode]
        summary["by_mode"][mode] = {
            "rounds": len(subset),
            "step_1_ok": sum(1 for row in subset if row["step_1_ok"]),
            "step_1_final_ok": sum(1 for row in subset if row["step_1_final_ok"]),
            "step_2_native": sum(1 for row in subset if row["step_2_native"]),
            "step_2_final_native": sum(1 for row in subset if row["step_2_final_native"]),
            "combo_verified": sum(1 for row in subset if row["combo_verified"]),
            "handoff_ready": sum(1 for row in subset if row["handoff_status"] == "ready"),
            "handoff_not_ready": sum(1 for row in subset if row["handoff_status"] not in {"ready", None}),
            "episode_actions": [row["episode_actions"] for row in subset],
        }
    out = PROJECT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary["counts"], ensure_ascii=False))
    print(json.dumps(summary["by_mode"], ensure_ascii=False))
    for row in rows:
        print(
            f"{row['round']:<20} seed={row['seed']} mode={row['handoff_mode']} map={row['mapping_ok']} "
            f"s1={row['step_1_ok']}/{row['step_1_final_ok']} s2={row['step_2_native']}/{row['step_2_final_native']} "
            f"combo={row['combo_verified']} handoff={row['handoff_status']} "
            f"({row['handoff_start_angle_deg']}->{row['handoff_end_angle_deg']} deg) "
            f"release={row['release_source']} actions={row['episode_actions']} "
            f"fail={row['failure_code']}"
        )
    print(f"out={out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
