#!/usr/bin/env python
"""Stop and reset during the step whose execution window is narrowed.

The stop mechanics themselves are already covered by
``scripts/composite_stop_tests.py``; what is new here is *where* the stop lands:
inside the composite step that the diagnostic arm runs with a shorter executed
window (``composite.transport_execute_horizon``). A stop that only ever lands in
the default window would not say anything about the new branch.

The case is a thin spec on top of that script's ``run_case``: same journal
trigger, same "did the action counter freeze" check, same ``/api/load`` leak
check. Nothing about the stop path is re-implemented.

    python scripts/transport_stop_test.py --engine http://127.0.0.1:14821 \
        --journal $EMBODIED_RECORDS_DIR/run_journal.jsonl --out stop.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))

import composite_stop_tests as base  # noqa: E402

COMPOSITE = "put the cream cheese in the bowl, then put the bowl on the stove, then turn on the stove"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", default="http://127.0.0.1:14821")
    parser.add_argument(
        "--journal",
        default=os.environ.get("EMBODIED_RECORDS_DIR", str(PROJECT / "records")) + "/run_journal.jsonl",
    )
    parser.add_argument("--out", default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--policy-seed", type=int, default=101)
    parser.add_argument("--min-actions", type=int, default=12)
    parser.add_argument("--timeout", type=float, default=420.0)
    args = parser.parse_args()

    case = {
        "id": "stop-transport-window",
        "kind": "step-2",
        "task_id": 7,
        "load_task_id": 7,
        "seed": args.seed,
        "policy_seed": args.policy_seed,
        "min_actions": args.min_actions,
        "instruction": COMPOSITE,
    }
    result = base.run_case(args, case)
    result["intent"] = (
        "stop while step 2 (the container transport) is executing under the narrowed window; "
        "then prove a fresh load does not inherit the stopped episode"
    )
    payload = {"schema": "embodied_engine.transport_stop.v1", "cases": [result]}
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2))
        print(f"[ok] {out}")
    print(
        f"[{result['case']}] ok={result.get('ok')} phase_at_trigger={result.get('trigger_phase')} "
        f"actions_at_trigger={result.get('trigger_sim_actions')} final={result.get('final_phase')} "
        f"record_status={result.get('record_execution_status')} "
        f"after_stop_advanced={result.get('after_stop', {}).get('advanced')}"
    )
    leak = result.get("leak_check") or {}
    print(
        "  reset: "
        + json.dumps(
            {
                k: leak.get(k)
                for k in ("load_status", "no_actions_left", "new_episode", "sim_action_count_after_load", "scene_epoch_after_load")
            }
        )
    )
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
