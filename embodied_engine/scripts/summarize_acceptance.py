#!/usr/bin/env python3
"""Summarize an acceptance run (suite JSON) without loading frames.

Usage: python3 scripts/summarize_acceptance.py records/acceptance/final_3x.json
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path


def load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _record_path(row: dict) -> str:
    """Prefer the copy archived next to the suite JSON.

    ``records/runs`` is pruned to the newest ``MAX_RUNS`` episodes, so older
    suites would otherwise lose the records their rows point at.
    """
    for key in ("archived_record", "record"):
        value = str(row.get(key) or "")
        if value and Path(value).exists():
            return value
    return ""


def step_timing(record: dict, key: str) -> float:
    total = 0.0
    for step in record.get("steps") or []:
        timing = step.get("timing") or {}
        value = timing.get(key)
        if isinstance(value, (int, float)):
            total += float(value)
    return total


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    suite = load(sys.argv[1])
    meta = suite.get("meta") or {}
    rows = suite.get("rows") or suite.get("results") or []
    by_task: dict[int, list[dict]] = {}
    for row in rows:
        by_task.setdefault(int(row["task_id"]), []).append(row)
    print(f"suite: {sys.argv[1]}")
    print(f"runs_per_task: {meta.get('runs_per_task')}  agent: {meta.get('agent_model')}  "
          f"action: {meta.get('action_model')}  inference: {meta.get('inference_mode')}")
    print()
    print(f"{'task':>4} {'succ':>5} {'median_s':>8} {'actions':>7} {'recov':>5} {'inf_s':>7} "
          f"{'env_s':>7} {'snap_s':>7} {'gap_s':>6} {'pf_used':>7} {'pf_saved':>8}  failures")
    total_ok = 0
    for task_id in sorted(by_task):
        group = by_task[task_id]
        ok = sum(1 for row in group if row.get("user_goal_success"))
        total_ok += ok
        durs, acts, inf, env, snap, gap, rec, used, saved = [], [], [], [], [], [], [], [], []
        failures = []
        for row in group:
            record = load(_record_path(row)) if _record_path(row) else {}
            durs.append(float(row.get("duration_s") or record.get("duration_s") or 0.0))
            acts.append(row.get("episode_sim_actions") or record.get("episode_sim_actions") or 0)
            rec.append(row.get("recovery_count") or 0)
            inf.append(step_timing(record, "inference_s_sum"))
            env.append(step_timing(record, "env_step_s_sum"))
            snap.append(step_timing(record, "snapshot_s_sum"))
            gap.append(step_timing(record, "chunk_gap_s_sum"))
            for step in record.get("steps") or []:
                prefetch = step.get("prefetch") or {}
                used.append(prefetch.get("used") or 0)
                saved.append(step.get("timing", {}).get("prefetch_inference_overlap_s_est") or 0.0)
            if not row.get("user_goal_success"):
                failures.append(str(row.get("failure_code") or row.get("reason") or "?")[:28])
        print(f"{task_id:>4} {ok}/{len(group):<3} {statistics.median(durs):>8.1f} {int(statistics.median(acts)):>7} "
              f"{sum(rec):>5} {statistics.median(inf):>7.2f} {statistics.median(env):>7.2f} "
              f"{statistics.median(snap):>7.2f} {statistics.median(gap):>6.2f} "
              f"{int(sum(used)):>7} {sum(saved):>8.2f}  {','.join(failures)}")
    print()
    print(f"total: {total_ok}/{len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
