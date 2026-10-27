#!/usr/bin/env python3
"""Turn an A/B run JSON into compact, reviewable evidence.

The full ``ab_full.json`` keeps every episode row (including per-request chunk
stats).  That is useful locally but too heavy to version alongside 60 full
episode replays.  This script emits:

- ``<stem>_slim.json``: meta + summary + one flat row per episode;
- ``<stem>_episodes.csv``: the same rows as CSV;
- ``failure_evidence/<task>_<seed>_<mode>.json``: only the plan, decision,
  failure reason and timing for failed/aborted episodes.

It never re-runs an episode and never drops a failure.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parents[1]


def _num(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def _stats(values: list[Any]) -> dict[str, Any]:
    clean = sorted(v for v in (_num(x) for x in values) if v is not None)
    if not clean:
        return {"n": 0, "median": None, "p95": None, "max": None}
    index = min(len(clean) - 1, max(0, int(round(0.95 * (len(clean) - 1)))))
    return {
        "n": len(clean),
        "median": round(statistics.median(clean), 4),
        "p95": round(clean[index], 4),
        "max": round(clean[-1], 4),
    }


def _rate(rows: list[dict[str, Any]], key: str) -> float | None:
    if not rows:
        return None
    return round(sum(1 for row in rows if row.get(key)) / len(rows), 4)


def _timing(row: dict[str, Any], key: str) -> Any:
    return (row.get("timing") or {}).get(key)


def _chunk(row: dict[str, Any], section: str, key: str) -> Any:
    openwam = row.get("openwam") or {}
    stats = openwam.get("chunk_stats") or {}
    return (stats.get(section) or {}).get(key)


def slim_row(row: dict[str, Any]) -> dict[str, Any]:
    decision = row.get("decision") or {}
    openwam = row.get("openwam") or {}
    chunk = openwam.get("chunk_stats") or {}
    wall = openwam.get("wall") or {}
    return {
        "task_id": row.get("task_id"),
        "seed": row.get("seed"),
        "init_index": row.get("init_index"),
        "order": row.get("order"),
        "mode": row.get("mode"),
        "agent_mode": row.get("agent_mode"),
        "plan_source": row.get("plan_source"),
        "decision_status": row.get("decision_status"),
        "decision_reason": row.get("decision_reason"),
        "plan_semantic": row.get("plan_semantic"),
        "binding_checks": row.get("binding_checks"),
        "execution_status": row.get("execution_status"),
        "native_task_success": row.get("native_task_success"),
        "native_goal_final": row.get("native_goal_final"),
        "user_goal_success": row.get("user_goal_success"),
        "failure_code": row.get("failure_code"),
        "reason": row.get("reason"),
        "instruction": row.get("instruction"),
        "openwam_prompt": row.get("openwam_prompt"),
        "openwam_prompt_actual": row.get("openwam_prompt_actual"),
        "prompt_matches_user": row.get("prompt_matches_user"),
        "duration_s": row.get("duration_s"),
        "wall_s": row.get("wall_s"),
        "planner_s": _timing(row, "planner_s"),
        "plan_to_first_action_s": _timing(row, "plan_to_first_action_s"),
        "execution_s": _timing(row, "execution_s"),
        "verification_s": _timing(row, "verification_s"),
        "inference_wall_s": (wall.get("phases_sum_s") or {}).get("inference_wall"),
        "env_step_s": (wall.get("phases_sum_s") or {}).get("env_step"),
        "observe_s": (wall.get("phases_sum_s") or {}).get("observe"),
        "snapshot_s": (wall.get("phases_sum_s") or {}).get("snapshot"),
        "episode_sim_actions": row.get("episode_sim_actions"),
        "sim_time_s": wall.get("sim_time_s"),
        "control_hz_wall": wall.get("control_hz_wall"),
        "sim_real_time_factor": wall.get("sim_real_time_factor"),
        "action_interval_median_s": (wall.get("action_interval_s") or {}).get("median"),
        "chunk_generations": chunk.get("chunk_generations"),
        "cache_hits": chunk.get("cache_hits"),
        "chunk_ids": chunk.get("chunk_ids"),
        "generated_actions_total": chunk.get("generated_actions_total"),
        "consumed_actions_total": chunk.get("consumed_actions_total"),
        "chunk_gen_median_s": (chunk.get("generation") or {}).get("client_inference_s", {}).get("median"),
        "cache_fetch_median_s": (chunk.get("cache") or {}).get("client_inference_s", {}).get("median"),
        "buffer_clear_ok": row.get("buffer_clear_ok"),
        "initial_scene_fingerprint": row.get("initial_scene_fingerprint"),
        "episode_id": row.get("episode_id"),
        "record": row.get("record"),
        "archived_record": row.get("archived_record"),
        "timed_out": row.get("timed_out", False),
        "decision": {
            key: decision.get(key)
            for key in (
                "status",
                "reason_code",
                "operation",
                "source",
                "target",
                "relation",
                "open_target",
                "source_confidence",
                "target_confidence",
                "operation_confidence",
            )
            if decision.get(key) is not None
        }
        or None,
    }


def aggregate(slim: list[dict[str, Any]], seeds: list[int]) -> dict[str, Any]:
    out: dict[str, Any] = {"overall": {}, "per_task": {}, "pairs": {}}
    modes = sorted({row["mode"] for row in slim})
    for mode in modes:
        rows = [row for row in slim if row["mode"] == mode]
        success = [row for row in rows if row.get("user_goal_success")]
        out["overall"][mode] = {
            "attempts": len(rows),
            "completed": sum(1 for row in rows if row.get("execution_status") == "completed"),
            "timeout": sum(1 for row in rows if row.get("timed_out")),
            "plan_semantic": sum(1 for row in rows if row.get("plan_semantic")),
            "native_task_success": sum(1 for row in rows if row.get("native_task_success")),
            "native_goal_final": sum(1 for row in rows if row.get("native_goal_final")),
            "user_goal_success": len(success),
            "plan_semantic_rate": _rate(rows, "plan_semantic"),
            "native_rate": _rate(rows, "native_task_success"),
            "user_goal_rate": _rate(rows, "user_goal_success"),
            "failure_codes": sorted(
                {row["failure_code"] for row in rows if row.get("failure_code")}
            ),
            "planner_s": _stats([row.get("planner_s") for row in rows]),
            "plan_to_first_action_s": _stats([row.get("plan_to_first_action_s") for row in rows]),
            "execution_s": _stats([row.get("execution_s") for row in rows]),
            "verification_s": _stats([row.get("verification_s") for row in rows]),
            "duration_all_s": _stats([row.get("duration_s") for row in rows]),
            "duration_success_s": _stats([row.get("duration_s") for row in success]),
            "chunk_generation_s": _stats([row.get("chunk_gen_median_s") for row in rows]),
            "cache_fetch_s": _stats([row.get("cache_fetch_median_s") for row in rows]),
            "control_hz_wall": _stats([row.get("control_hz_wall") for row in rows]),
            "sim_real_time_factor": _stats([row.get("sim_real_time_factor") for row in rows]),
            "sim_actions": _stats([row.get("episode_sim_actions") for row in rows]),
        }
        for task_id in sorted({int(row["task_id"]) for row in rows}):
            task_rows = [row for row in rows if int(row["task_id"]) == task_id]
            task_success = [row for row in task_rows if row.get("user_goal_success")]
            out["per_task"].setdefault(str(task_id), {})[mode] = {
                "attempts": len(task_rows),
                "plan_semantic": sum(1 for row in task_rows if row.get("plan_semantic")),
                "native": sum(1 for row in task_rows if row.get("native_task_success")),
                "user_goal": len(task_success),
                "duration_all_s": _stats([row.get("duration_s") for row in task_rows]),
                "duration_success_s": _stats([row.get("duration_s") for row in task_success]),
            }
    for task_id in sorted({int(row["task_id"]) for row in slim}):
        for seed in seeds:
            pair = [row for row in slim if int(row["task_id"]) == task_id and int(row["seed"]) == seed]
            if not pair:
                continue
            prints = {row["mode"]: row.get("initial_scene_fingerprint") for row in pair}
            out["pairs"][f"{task_id}/{seed}"] = {
                "same_init": len(set(prints.values())) == 1 and all(prints.values()),
                "fingerprint": next(iter(set(prints.values()))),
                "modes": {
                    row["mode"]: {
                        "user_goal": row.get("user_goal_success"),
                        "native": row.get("native_task_success"),
                        "plan_semantic": row.get("plan_semantic"),
                        "duration_s": row.get("duration_s"),
                    }
                    for row in pair
                },
            }
    return out


def _rejection_evidence(record_path: Any) -> dict[str, Any] | None:
    """Pull the staged decision evidence for a refused plan.

    The episode row only carries the reason code; the confidence that caused a
    refusal lives in the full episode record. Copy just the stage-1 fields so a
    reviewer can see whether the model was wrong or merely over-cautious.
    """
    if not record_path:
        return None
    try:
        record = json.loads(Path(str(record_path)).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    response = (record.get("plan") or {}).get("agent_response") or {}
    evidence = response.get("evidence") or {}
    stage1 = evidence.get("stage1") or {}
    keys = (
        "supported",
        "operation",
        "operation_confidence",
        "source",
        "source_confidence",
        "source_margin",
        "source_specified",
        "ambiguous",
    )
    compact = {key: stage1.get(key) for key in keys if stage1.get(key) is not None}
    if compact:
        return compact
    # Rejections keep the binding that failed in the structured error value,
    # even when the full stage-1 evidence block is not serialized.
    errors = response.get("errors") or []
    return {"errors": errors} if errors else None


def failure_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": row.get("task_id"),
        "seed": row.get("seed"),
        "mode": row.get("mode"),
        "instruction": row.get("instruction"),
        "openwam_prompt": row.get("openwam_prompt"),
        "plan_source": row.get("plan_source"),
        "decision": row.get("decision"),
        "decision_status": row.get("decision_status"),
        "decision_reason": row.get("decision_reason"),
        "decision_evidence": _rejection_evidence(row.get("record")),
        "binding_checks": row.get("binding_checks"),
        "plan_semantic": row.get("plan_semantic"),
        "execution_status": row.get("execution_status"),
        "failure_code": row.get("failure_code"),
        "reason": row.get("reason"),
        "native_task_success": row.get("native_task_success"),
        "native_goal_final": row.get("native_goal_final"),
        "user_goal_success": row.get("user_goal_success"),
        "duration_s": row.get("duration_s"),
        "planner_s": row.get("planner_s"),
        "plan_to_first_action_s": row.get("plan_to_first_action_s"),
        "execution_s": row.get("execution_s"),
        "verification_s": row.get("verification_s"),
        "episode_sim_actions": row.get("episode_sim_actions"),
        "chunk_generations": row.get("chunk_generations"),
        "cache_hits": row.get("cache_hits"),
        "initial_scene_fingerprint": row.get("initial_scene_fingerprint"),
        "episode_id": row.get("episode_id"),
        "timed_out": row.get("timed_out", False),
    }


CSV_FIELDS = [
    "task_id", "seed", "init_index", "order", "mode", "agent_mode", "plan_source",
    "decision_status", "decision_reason", "plan_semantic", "execution_status",
    "native_task_success", "native_goal_final", "user_goal_success", "failure_code",
    "duration_s", "planner_s", "plan_to_first_action_s", "execution_s",
    "verification_s", "episode_sim_actions", "sim_time_s", "control_hz_wall",
    "sim_real_time_factor", "action_interval_median_s", "chunk_generations",
    "cache_hits", "chunk_gen_median_s", "cache_fetch_median_s",
    "openwam_prompt", "openwam_prompt_actual", "initial_scene_fingerprint",
]


def markdown(payload: dict[str, Any]) -> str:
    agg = payload["aggregate"]
    modes = list(agg["overall"].keys())
    lines = ["## A/B 汇总", ""]
    lines.append("| 指标 | " + " | ".join(modes) + " |")
    lines.append("| --- | " + " | ".join("---" for _ in modes) + " |")

    def row(label: str, getter) -> None:
        lines.append(f"| {label} | " + " | ".join(str(getter(agg['overall'][m])) for m in modes) + " |")

    row("尝试回合", lambda o: o["attempts"])
    row("完整执行", lambda o: o["completed"])
    row("计划语义正确", lambda o: f"{o['plan_semantic']}/{o['attempts']} ({o['plan_semantic_rate']})")
    row("原生目标成立", lambda o: f"{o['native_task_success']}/{o['attempts']} ({o['native_rate']})")
    row("用户目标满足", lambda o: f"{o['user_goal_success']}/{o['attempts']} ({o['user_goal_rate']})")
    row("失败码", lambda o: ", ".join(o["failure_codes"]) or "-")
    row("Planner 中位 (s)", lambda o: o["planner_s"]["median"])
    row("规划完成→首动作中位 (s)", lambda o: o["plan_to_first_action_s"]["median"])
    row("执行阶段中位 (s)", lambda o: o["execution_s"]["median"])
    row("校验中位 (s)", lambda o: o["verification_s"]["median"])
    row("整回合中位（全部）(s)", lambda o: o["duration_all_s"]["median"])
    row("整回合中位（成功）(s)", lambda o: o["duration_success_s"]["median"])
    row("整回合 p95（全部）(s)", lambda o: o["duration_all_s"]["p95"])
    row("完整 chunk 推理中位 (s)", lambda o: o["chunk_generation_s"]["median"])
    row("缓存取动作中位 (s)", lambda o: o["cache_fetch_s"]["median"])
    row("墙钟控制 Hz 中位", lambda o: o["control_hz_wall"]["median"])
    row("仿真实时因子中位", lambda o: o["sim_real_time_factor"]["median"])
    row("仿真动作中位", lambda o: o["sim_actions"]["median"])

    lines += ["", "## 逐任务", ""]
    lines.append("| task | mode | 语义 | 原生 | 用户 | 尝试 | 全部耗时中位 (s) | 成功耗时中位 (s) |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for task_id, per_mode in sorted(agg["per_task"].items(), key=lambda item: int(item[0])):
        for mode in modes:
            row_data = per_mode.get(mode)
            if not row_data:
                continue
            lines.append(
                f"| {task_id} | {mode} | {row_data['plan_semantic']}/{row_data['attempts']} | "
                f"{row_data['native']}/{row_data['attempts']} | {row_data['user_goal']}/{row_data['attempts']} | "
                f"{row_data['attempts']} | {row_data['duration_all_s']['median']} | "
                f"{row_data['duration_success_s']['median']} |"
            )

    same = sum(1 for pair in agg["pairs"].values() if pair["same_init"])
    lines += ["", f"同一初始化指纹配对：{same}/{len(agg['pairs'])}。"]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", help="ab_full.json")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--markdown", action="store_true", help="also write <stem>_tables.md")
    args = parser.parse_args()

    source = Path(args.source)
    data = json.loads(source.read_text(encoding="utf-8"))
    out_dir = Path(args.out_dir) if args.out_dir else source.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = data.get("rows") or []
    if not rows:
        raise SystemExit(f"{source} has no rows")
    seeds = [int(seed) for seed in (data.get("meta") or {}).get("seeds") or []]
    slim = [slim_row(row) for row in rows]
    payload = {
        "schema": "embodied_engine.ab_slim.v1",
        "source": str(source),
        "meta": data.get("meta"),
        "summary": data.get("summary"),
        "aggregate": aggregate(slim, seeds),
        "episodes": slim,
    }
    slim_path = out_dir / f"{source.stem}_slim.json"
    slim_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_path = out_dir / f"{source.stem}_episodes.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in slim:
            writer.writerow({key: row.get(key) for key in CSV_FIELDS})

    failures_dir = out_dir / "failure_evidence"
    failures_dir.mkdir(exist_ok=True)
    failures = [
        row for row in slim
        if not row.get("user_goal_success") or row.get("execution_status") != "completed"
    ]
    for row in failures:
        name = f"task{row['task_id']}_seed{row['seed']}_{row['mode']}.json"
        (failures_dir / name).write_text(
            json.dumps(failure_payload(row), ensure_ascii=False, indent=2), encoding="utf-8"
        )
    if args.markdown:
        table_path = out_dir / f"{source.stem}_tables.md"
        table_path.write_text(markdown(payload), encoding="utf-8")
        print(f"wrote {table_path}")
    print(
        f"wrote {slim_path} ({len(slim)} episodes), {csv_path}, "
        f"{len(failures)} failure evidence files in {failures_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
