#!/usr/bin/env python3
"""Summarise an OpenWAM backend acceptance suite from its archived records.

Usage:
    python3 scripts/summarize_openwam.py records/acceptance/openwam_libero_goal_all10.json
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path


def load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else "records/acceptance/openwam_libero_goal_all10.json"
    suite = load(path)
    rows = []
    for row in suite["rows"]:
        record_path = row.get("archived_record") or row.get("record")
        record = load(record_path) if record_path and Path(record_path).exists() else {}
        step = (record.get("steps") or [{}])[0]
        timing = step.get("timing") or {}
        verification = (record.get("final_verification") or {}).get("steps") or [{}]
        final_check = (verification[0] or {}).get("verification") or {}
        rows.append(
            {
                "task": row["task_id"],
                "instruction": record.get("instruction"),
                "prompt": step.get("prompt"),
                "ok": bool(record.get("user_goal_success")),
                "native": bool(record.get("native_task_success")),
                "duration_s": record.get("duration_s"),
                "planner_s": round(((record.get("plan") or {}).get("latency_s") or 0), 3),
                "sim_actions": step.get("sim_actions"),
                "first_inference_s": timing.get("first_inference_s"),
                "warm_inference_median_s": timing.get("warm_inference_median_s"),
                "inference_sum_s": timing.get("inference_s_sum"),
                "env_step_sum_s": timing.get("env_step_s_sum"),
                "observe_sum_s": timing.get("observe_s_sum"),
                "snapshot_sum_s": timing.get("snapshot_s_sum"),
                "peak_gpu_mib": timing.get("peak_gpu_mib"),
                "release_observed": final_check.get("release_observed"),
                "gripper_at_goal_m": final_check.get("gripper_opening_m"),
                "release_events": (step.get("release") or {}).get("count"),
                "backend": (record.get("model") or {}).get("backend"),
                "mode": record.get("inference_mode"),
                "recovery": record.get("recovery_count"),
            }
        )

    ok = [r for r in rows if r["ok"]]
    print(f"suite: {path}")
    print(f"success: {len(ok)}/{len(rows)}")
    durations = [r["duration_s"] for r in rows if r["duration_s"]]
    if durations:
        print(f"duration_s: min={min(durations):.1f} median={statistics.median(durations):.1f} max={max(durations):.1f}")
    actions = [r["sim_actions"] for r in rows if r["sim_actions"]]
    if actions:
        print(f"sim_actions: min={min(actions)} median={statistics.median(actions)} max={max(actions)}")
    planners = [r["planner_s"] for r in rows]
    print(f"planner_s: median={statistics.median(planners):.2f}")
    firsts = [r["first_inference_s"] for r in rows if r["first_inference_s"]]
    warms = [r["warm_inference_median_s"] for r in rows if r["warm_inference_median_s"]]
    if firsts:
        print(f"first_inference_s: median={statistics.median(firsts):.2f} max={max(firsts):.2f}")
    if warms:
        print(f"warm_inference_median_s: median={statistics.median(warms):.3f} max={max(warms):.3f}")
    peaks = [r["peak_gpu_mib"] for r in rows if r["peak_gpu_mib"]]
    if peaks:
        print(f"peak_gpu_mib: max={max(peaks)}")
    print(f"release_observed at goal: {sum(1 for r in rows if r['release_observed'])}/{len(rows)}")
    print(f"recoveries: {sum(int(r['recovery'] or 0) for r in rows)}")
    print()
    header = (
        f"{'task':>4} {'ok':>3} {'native':>6} {'dur_s':>7} {'plan_s':>6} {'acts':>5} "
        f"{'1st_inf':>7} {'warm_inf':>8} {'inf_sum':>7} {'env_sum':>7} {'gpu':>6} {'rel':>4} {'grip@goal':>10}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['task']:>4} {str(r['ok']):>5} {str(r['native']):>6} {r['duration_s'] or 0:7.1f} "
            f"{r['planner_s']:6.2f} {r['sim_actions'] or 0:5d} {r['first_inference_s'] or 0:7.3f} "
            f"{r['warm_inference_median_s'] or 0:8.3f} {r['inference_sum_s'] or 0:7.2f} "
            f"{r['env_step_sum_s'] or 0:7.2f} {r['peak_gpu_mib'] or 0:6d} "
            f"{str(r['release_observed']):>5} {str(r['gripper_at_goal_m']):>22}"
        )
    print()
    for r in rows:
        print(f"task {r['task']}: {r['instruction']} -> prompt={r['prompt']!r} mode={r['mode']} backend={r['backend']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
