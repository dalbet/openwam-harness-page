#!/usr/bin/env python
"""Rebuild records/acceptance/progress_probe_20261027.json from recorded episodes.

Offline only: reads observation logs already on disk, steps nothing, and prints
one row per episode. Thresholds are the frozen ones in engine.progress; this
script cannot change them, which keeps the "would a stall rule have fired"
question answerable after the fact.

    python scripts/progress_probe.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine import progress  # noqa: E402

ROUND_MANIFESTS = [
    ROOT / "records" / "acceptance" / "placement_rounds_20261027.json",
    ROOT / "records" / "acceptance" / "placement_rounds_final_20261027.json",
    ROOT / "records" / "acceptance" / "placement_rounds_final2_20261027.json",
]
FAILED_ATTEMPT = ROOT / "records" / "runs" / "ep-20261027-110254-b1e664.json"
OUT = ROOT / "records" / "acceptance" / "progress_probe_20261027.json"


def observations(path: Path) -> list[dict]:
    record = json.loads(path.read_text())
    steps = record.get("steps") or []
    return steps[0].get("observations") or [] if steps else []


def row(label: str, task: int, lang: str, native: bool, user: bool, path: Path) -> dict:
    summary = progress.summarize(observations(path))
    return {
        "label": label,
        "task": task,
        "lang": lang,
        "native": native,
        "user": user,
        "observations": summary["observations"],
        "object_total_motion_m": summary["object_total_motion_m"],
        "first_object_motion_n": summary["first_object_motion_n"],
        "never_touched": summary["never_touched"],
        "stall_windows": len(summary["stall_windows"]),
        "stall_actions_after_contact": summary["stall_actions_after_contact"],
        "first_window": summary["stall_windows"][0] if summary["stall_windows"] else None,
    }


def sensitivity(rows: list[dict]) -> dict:
    """How many successful episodes a blind "no object motion" abort would cut."""
    out = {}
    for window in (60, 90, 100, 110, 115, 119, 120, 150):
        cut = [
            item["label"]
            for item in rows
            if item["native"] and item["first_object_motion_n"] is not None
            and item["first_object_motion_n"] >= window
        ]
        out[str(window)] = {"would_cut": len(cut), "episodes": cut}
    return out


def main() -> int:
    rows = []
    for manifest_path in ROUND_MANIFESTS:
        if not manifest_path.exists():
            continue
        tag = manifest_path.stem.replace("placement_rounds_", "").replace("_20261027", "")
        manifest = json.loads(manifest_path.read_text())
        for item in manifest["rows"]:
            record = ROOT / item["record"]
            if not record.exists():
                continue
            rows.append(
                row(
                    f"{tag}/round-{item['round']}",
                    item["task_id"],
                    item["lang"],
                    item["native_task_success"],
                    item["user_goal_success"],
                    record,
                )
            )
    if FAILED_ATTEMPT.exists():
        rows.append(row("legacy-failed-attempt-3", 2, "en", False, False, FAILED_ATTEMPT))
    payload = {
        "schema": "embodied_engine.progress_probe.v1",
        "frozen": {
            "window_actions": progress.DEFAULT_WINDOW_ACTIONS,
            "object_motion_threshold_m": progress.DEFAULT_OBJECT_MOTION_M,
            "eef_motion_threshold_m": progress.DEFAULT_EEF_MOTION_M,
        },
        "enforced": False,
        "rows": rows,
        "sensitivity": sensitivity(rows),
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=1))
    for item in rows:
        print(
            f"{item['label']:>28} task{item['task']} native={item['native']!s:5} "
            f"user={item['user']!s:5} obj_motion={item['object_total_motion_m']} "
            f"first_motion_n={item['first_object_motion_n']} stalls={item['stall_windows']}"
        )
    failing = [item for item in rows if not item["native"]]
    print(f"\nnon-native rows: {len(failing)}; every one has never_touched={[i['never_touched'] for i in failing]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
