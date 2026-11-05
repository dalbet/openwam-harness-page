#!/usr/bin/env python
"""Where do two rounds with the same scene seed and the same policy seed diverge?

The seed matrix does not assume that a pinned sampling seed makes two runs
identical. It pins the *policy's* initial noise and reports what is left. This
script compares two episode records from the inside out, in the order a real
episode meets the world:

1. ``initial_state_fingerprint`` — movable objects + robot (``*_pos`` observables)
2. ``initial_fixture_fingerprint`` — static fixtures the observables do not
   publish (flat stove, cabinet, wine rack), which LIBERO's placement sampler
   draws without a seed
3. the first chunk's policy input: ``main``/``wrist``/``state`` hashes
4. the first chunk's sampling: ``noise_seed`` and ``noise_sha256``
5. the first chunk's action: ``action0`` / ``action0_sha256``
6. the first chunk index, then the first action index, at which the two
   action streams differ

The first stage that differs is the answer. Everything after it is a
consequence, and the record says so instead of attributing the whole spread to
"model randomness".

    python scripts/seed_divergence_report.py \
        --pair <episode_a.json> <episode_b.json> --label s3-ps101 \
        --out records/acceptance/seed_divergence_20261105.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))


def _load(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _first_chunk(record: dict[str, Any], step_index: int = 0) -> tuple[dict[str, Any], dict[str, Any]]:
    step = (record.get("steps") or [])[step_index]
    rows = ((step.get("pi05") or {}).get("generation") or {}).get("rows") or []
    return step, (rows[0] if rows else {})


def _chunks(record: dict[str, Any], step_index: int = 0) -> list[dict[str, Any]]:
    step = (record.get("steps") or [])[step_index]
    return ((step.get("pi05") or {}).get("generation") or {}).get("rows") or []


def _first_action_difference(a: list[dict[str, Any]], b: list[dict[str, Any]]) -> dict[str, Any]:
    """The first (chunk, action) pair whose commanded action differs."""
    for index in range(min(len(a), len(b))):
        left = a[index].get("action0")
        right = b[index].get("action0")
        if left is None or right is None:
            return {"chunk": index, "action": None, "reason": "action0 missing from a chunk row"}
        if [round(float(v), 6) for v in left] != [round(float(v), 6) for v in right]:
            return {
                "chunk": index,
                "action": 0,
                "at_sim_action": {"a": a[index].get("at_sim_action"), "b": b[index].get("at_sim_action")},
                "a_action": [round(float(v), 6) for v in left],
                "b_action": [round(float(v), 6) for v in right],
                "max_abs_diff": round(
                    max(abs(float(x) - float(y)) for x, y in zip(left, right)), 8
                ),
            }
    if len(a) != len(b):
        return {
            "chunk": min(len(a), len(b)),
            "action": None,
            "reason": f"the shorter stream ran out first ({len(a)} vs {len(b)} chunks)",
        }
    return {"chunk": None, "action": None, "reason": "the compared chunks are identical"}


def compare(path_a: str, path_b: str, label: str) -> dict[str, Any]:
    a, b = _load(path_a), _load(path_b)
    step_a, first_a = _first_chunk(a)
    step_b, first_b = _first_chunk(b)
    obs_a, obs_b = first_a.get("obs") or {}, first_b.get("obs") or {}
    sampling_a, sampling_b = a.get("sampling") or {}, b.get("sampling") or {}
    init_a = a.get("initial_init_state") or {}
    init_b = b.get("initial_init_state") or {}

    def init_stage(name: str, key: str, what: str) -> dict[str, Any]:
        return {
            "stage": f"init_{name}",
            "same": init_a.get(key) is not None and init_a.get(key) == init_b.get(key),
            "a": init_a.get(key),
            "b": init_b.get(key),
            "what": what,
            "source": "env.init_state (worker: " + (init_a.get("schema") or "absent") + ")",
        }

    stages = [
        # Physics first, in the order the reset writes it, then the pixels. An
        # images-only difference is a renderer question; anything above it is
        # a scene question.
        init_stage("fixtures", "fixtures_sha256", "sim.model.body_pos/body_quat for every body (placement sampler output)"),
        init_stage("movable", "movable_sha256", "movable objects: qpos-derived position + body xquat"),
        init_stage("joints", "joints_sha256", "every joint's qpos and qvel"),
        init_stage("robot", "robot_sha256", "eef pose, gripper opening, controller references, arm joints"),
        {
            "stage": "init_images",
            "same": bool(init_a.get("images")) and init_a.get("images") == init_b.get("images"),
            "a": init_a.get("images"),
            "b": init_b.get("images"),
            "what": "sha256 of both raw camera frames, read in the same worker call as the physics",
            "source": "env.init_state",
        },
        {
            "stage": "scene_init_state",
            "same": a.get("initial_state_fingerprint") == b.get("initial_state_fingerprint"),
            "a": a.get("initial_state_fingerprint"),
            "b": b.get("initial_state_fingerprint"),
            "what": "movable objects + robot from the *_pos observables",
        },
        {
            "stage": "static_fixtures",
            "same": (a.get("initial_fixture_fingerprint") or {}).get("sha256")
            == (b.get("initial_fixture_fingerprint") or {}).get("sha256"),
            "a": (a.get("initial_fixture_fingerprint") or {}).get("sha256"),
            "b": (b.get("initial_fixture_fingerprint") or {}).get("sha256"),
            "what": "flat stove, cabinet, wine rack: written by LIBERO's placement sampler, unseeded",
        },
        {
            "stage": "policy_input_main_image",
            "same": obs_a.get("main_sha256") == obs_b.get("main_sha256"),
            "a": (obs_a.get("main_sha256") or "")[:16],
            "b": (obs_b.get("main_sha256") or "")[:16],
            "what": "the agentview frame the policy actually receives (rotated + resized)",
        },
        {
            "stage": "policy_input_wrist_image",
            "same": obs_a.get("wrist_sha256") == obs_b.get("wrist_sha256"),
            "a": (obs_a.get("wrist_sha256") or "")[:16],
            "b": (obs_b.get("wrist_sha256") or "")[:16],
            "what": "the wrist frame the policy actually receives",
        },
        {
            "stage": "policy_input_state",
            "same": obs_a.get("state_sha256") == obs_b.get("state_sha256"),
            "a": (obs_a.get("state_sha256") or "")[:16],
            "b": (obs_b.get("state_sha256") or "")[:16],
            "what": "the 10-D EEF proprio vector",
        },
        {
            "stage": "sampling_noise",
            "same": first_a.get("noise_seed") == first_b.get("noise_seed")
            and first_a.get("noise_sha256") == first_b.get("noise_sha256"),
            "a": {"seed": first_a.get("noise_seed"), "sha": (first_a.get("noise_sha256") or "")[:16]},
            "b": {"seed": first_b.get("noise_seed"), "sha": (first_b.get("noise_sha256") or "")[:16]},
            "what": "the explicit initial diffusion noise sent with the request",
        },
        {
            "stage": "first_action",
            "same": first_a.get("action0_sha256") == first_b.get("action0_sha256"),
            "a": [round(float(v), 6) for v in (first_a.get("action0") or [])],
            "b": [round(float(v), 6) for v in (first_b.get("action0") or [])],
            "what": "the first commanded 7-D action of the step",
        },
    ]
    first_divergence = next((row["stage"] for row in stages if not row["same"]), None)
    action_difference = _first_action_difference(_chunks(a), _chunks(b))
    return {
        "label": label,
        "a": {"path": path_a, "episode_id": a.get("episode_id"), "round": a.get("round")},
        "b": {"path": path_b, "episode_id": b.get("episode_id"), "round": b.get("round")},
        "sampling": {
            "a": {"scene_seed": sampling_a.get("scene_seed"), "policy_seed": sampling_a.get("policy_seed")},
            "b": {"scene_seed": sampling_b.get("scene_seed"), "policy_seed": sampling_b.get("policy_seed")},
        },
        "outcome": {
            "a": {"user_goal_success": a.get("user_goal_success"), "failure_code": a.get("failure_code")},
            "b": {"user_goal_success": b.get("user_goal_success"), "failure_code": b.get("failure_code")},
        },
        "first_step_actions": {"a": step_a.get("sim_actions"), "b": step_b.get("sim_actions")},
        "stages": stages,
        "first_divergence": first_divergence,
        "first_action_difference": action_difference,
        "reading": (
            f"the two rounds are identical up to {first_divergence}, which is where the paths separate"
            if first_divergence
            else "every compared stage is identical; the difference, if any, is downstream of the first action"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", nargs=2, action="append", required=True, metavar=("A", "B"))
    parser.add_argument("--label", action="append", required=False, default=[])
    parser.add_argument("--out", default=str(PROJECT / "records/acceptance/seed_divergence_20261105.json"))
    args = parser.parse_args()

    pairs = []
    for index, (path_a, path_b) in enumerate(args.pair):
        label = args.label[index] if index < len(args.label) else f"pair_{index}"
        pairs.append(compare(path_a, path_b, label))

    payload = {
        "schema": "embodied_engine.seed_divergence.v1",
        "note": (
            "compares rounds from the inside out: scene init, static fixtures, the policy's own input "
            "hashes, the pinned noise, the first action, then the action stream. The first stage that "
            "differs is reported as first_divergence; later stages are consequences of it."
        ),
        "pairs": pairs,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    for pair in pairs:
        print(
            f"{pair['label']}: first_divergence={pair['first_divergence']} "
            f"first_action_diff_chunk={(pair['first_action_difference'] or {}).get('chunk')}",
            flush=True,
        )
    print(f"written: {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
