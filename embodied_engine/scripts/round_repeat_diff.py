#!/usr/bin/env python
"""Two episode records of the same seeded round: what is identical, what is not.

The engine stores a fingerprint of the scene, of the initial state and of the
fixtures it seeded, and it stores the action vector it executed at every
action. So "did the same configuration reproduce?" is answered by comparing
stored values, not by re-running and eyeballing two videos.

    identical scene/state/fixture fingerprints -> the round started from the
        same numbers, so the comparison downstream is about execution
    action streams equal within a tolerance -> the policy consumed the same
        noise and produced the same behaviour
    first divergence at action k -> everything after k is not a repeat; report
        it instead of averaging over it

    python scripts/round_repeat_diff.py --a first.json --b repeat.json --out diff.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

FINGERPRINTS = ("initial_scene_fingerprint", "initial_state_fingerprint", "initial_fixture_fingerprint")


def actions_of(record: dict[str, Any]) -> list[list[float]]:
    """Executed action vectors, step by step, in execution order."""
    out: list[list[float]] = []
    for step in record.get("steps") or []:
        for row in step.get("action_trace") or []:
            vector = row.get("a7")
            if isinstance(vector, list):
                out.append([float(v) for v in vector])
    return out


def compare(a: dict[str, Any], b: dict[str, Any], tol: float) -> dict[str, Any]:
    fingerprints = {}
    for name in FINGERPRINTS:
        left, right = a.get(name), b.get(name)
        fingerprints[name] = {"a": left, "b": right, "same": left == right}
    first = actions_of(a)
    second = actions_of(b)
    first_divergence = None
    worst = 0.0
    for index in range(min(len(first), len(second))):
        delta = max(abs(x - y) for x, y in zip(first[index], second[index]))
        worst = max(worst, delta)
        if delta > tol and first_divergence is None:
            first_divergence = {
                "action": index + 1,
                "max_abs_delta": delta,
                "a": first[index],
                "b": second[index],
            }
    return {
        "a": {"record": a.get("__path__"), "episode_id": a.get("episode_id"),
              "success": a.get("user_goal_success"), "failure_code": a.get("failure_code"),
              "actions": a.get("episode_sim_actions"), "duration_s": a.get("duration_s")},
        "b": {"record": b.get("__path__"), "episode_id": b.get("episode_id"),
              "success": b.get("user_goal_success"), "failure_code": b.get("failure_code"),
              "actions": b.get("episode_sim_actions"), "duration_s": b.get("duration_s")},
        "fingerprints": fingerprints,
        "same_source": all(item["same"] for item in fingerprints.values()),
        "actions": {"a": len(first), "b": len(second), "tolerance": tol,
                    "max_abs_delta": worst, "first_divergence": first_divergence},
        "outcome": {
            "success_same": a.get("user_goal_success") == b.get("user_goal_success"),
            "failure_code_same": a.get("failure_code") == b.get("failure_code"),
        },
        "final_verification": {
            "a": (a.get("final_verification") or {}).get("ok"),
            "b": (b.get("final_verification") or {}).get("ok"),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", required=True)
    parser.add_argument("--b", required=True)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    a = json.loads(Path(args.a).read_text())
    b = json.loads(Path(args.b).read_text())
    a["__path__"], b["__path__"] = args.a, args.b
    report = compare(a, b, args.tol)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2))
        print(f"[ok] {out}")
    print(
        f"same_source={report['same_source']} success {report['a']['success']}->{report['b']['success']} "
        f"actions {report['actions']['a']}->{report['actions']['b']} "
        f"max_delta={report['actions']['max_abs_delta']:.6g} "
        f"first_divergence={report['actions']['first_divergence']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
