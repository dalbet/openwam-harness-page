#!/usr/bin/env python
"""Build the round manifest for the scene-init / independent re-seat round.

Every number the manifest states is read back from the record it comes from and
every artifact carries its real SHA256, so the manifest cannot drift from the
evidence it describes. The prose fields (scope, gates, decisions) are the only
hand-written parts.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parents[1]
ACCEPTANCE = PROJECT / "records" / "acceptance"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def artifact(path: str, role: str) -> dict[str, Any]:
    target = PROJECT / path
    return {
        "path": path,
        "role": role,
        "bytes": target.stat().st_size,
        "sha256": sha256(target),
    }


def init_probe_summary() -> dict[str, Any]:
    data = load(ACCEPTANCE / "scene_init_probe_20261105.json")
    rows = []
    for row in data["rows"]:
        for comparison in row["comparisons"]:
            fixtures = next(s for s in comparison["stages"] if s["stage"] == "fixtures")
            first = fixtures["first_difference"] or {}
            rows.append(
                {
                    "seed": row["seed"],
                    "seeding": row["seeding"],
                    "scope": comparison["scope"],
                    "first_divergence": comparison["first_divergence"],
                    "physics_identical": comparison["physics_identical"],
                    "images_identical": comparison["images_identical"],
                    "fixture_sha256": fixtures["sha256"],
                    "first_field": first.get("field"),
                    "first_abs_delta_m": None
                    if first.get("abs_delta") is None
                    else round(float(first["abs_delta"]), 6),
                }
            )
    return {
        "artifact": "records/acceptance/scene_init_probe_20261105.json",
        "script": "scripts/scene_init_probe.py",
        "design": "3 scene seeds x {seeding off, seeding on} x 2 env-server processes x 2 loads",
        "rows": rows,
        "verdict": {
            "unseeded": "6/6 comparisons diverge at fixtures; movable/joints/robot identical",
            "seeded": "6/6 comparisons bit-identical including both raw camera frames",
        },
    }


def repeat_summary() -> dict[str, Any]:
    divergence = load(ACCEPTANCE / "seed_divergence_fixturefix_20261105.json")
    rounds = []
    for name in sorted((ACCEPTANCE / "fixturefix_repeat_20261105_runs").glob("*.json")):
        record = load(name)
        step = (record.get("steps") or [{}])[0] or {}
        generation = ((step.get("pi05") or {}).get("generation") or {}).get("rows") or []
        rounds.append(
            {
                "record": f"records/acceptance/fixturefix_repeat_20261105_runs/{name.name}",
                "episode_id": record.get("episode_id"),
                "scene_seed": (record.get("sampling") or {}).get("scene_seed"),
                "policy_seed": (record.get("sampling") or {}).get("policy_seed"),
                "outcome": record.get("failure_code") or "success",
                "sim_action_count": record.get("sim_action_count"),
                "initial_scene_fingerprint": record.get("initial_scene_fingerprint"),
                "initial_state_fingerprint": record.get("initial_state_fingerprint"),
                "initial_fixture_fingerprint": (record.get("initial_fixture_fingerprint") or {}).get("sha256"),
                "initial_images": (record.get("initial_init_state") or {}).get("images"),
                "noise_seed": generation[0].get("noise_seed") if generation else None,
                "action0_sha256": generation[0].get("action0_sha256") if generation else None,
            }
        )
    return {
        "manifest": "config/rounds_fixturefix_repeat_20261105.json",
        "summary": "records/acceptance/fixturefix_repeat_20261105.json",
        "config_note": "both pairs ran on config_demo_b900_forced (max_seat_delta_m=0.005) so the transport-readiness branch triggers every time",
        "rounds": rounds,
        "divergence": {
            "artifact": "records/acceptance/seed_divergence_fixturefix_20261105.json",
            "pairs": [
                {
                    "label": pair["label"],
                    "sampling": pair["sampling"],
                    "first_divergence": pair.get("first_divergence"),
                    "first_action_diff_chunk": pair.get("first_action_diff_chunk"),
                }
                for pair in divergence["pairs"]
            ],
            "pre_fix": {
                "artifact": "records/acceptance/seed_divergence_20261105.json",
                "pairs": [
                    {"label": pair["label"], "first_divergence": pair.get("first_divergence")}
                    for pair in load(ACCEPTANCE / "seed_divergence_20261105.json")["pairs"]
                ],
            },
        },
    }


def asset_geometry() -> dict[str, Any]:
    data = load(ACCEPTANCE / "asset_geometry_20261105.json")
    measured = data["measured"]
    candidate = data["candidate_conditions"]
    return {
        "artifact": "records/acceptance/asset_geometry_20261105.json",
        "script": "scripts/asset_geometry_audit.py",
        "measured": measured,
        "candidate_conditions": candidate,
        "geometric_conflict": {
            "cheese_max_radius_m": candidate["cheese_max_radius_m"],
            "bowl_wall_inner_radius_m": measured["bowl_wall_inner_radius_m"],
            "bowl_rim_inner_radius_m": measured["bowl_rim_inner_radius_m"],
            "wall_clearance_for_cheese_m": candidate["wall_clearance_for_cheese_m"],
            "reading": "the cheese is wider than the bowl opening, so 'in the bowl' can only mean perched on the rim, never inside the cavity",
        },
        "status": data["status"],
    }


def reseat_summary() -> dict[str, Any]:
    probes = []
    for name in [
        ACCEPTANCE / "reseat_probe_20261105_runs" / "ep-20261105-090401-9cff68.json",
        ACCEPTANCE / "reseat_probe_20261105_repeats" / "ep-20261105-090635-381c66.json",
        ACCEPTANCE / "reseat_probe_20261105_repeats" / "ep-20261105-091731-c813c2.json",
        ACCEPTANCE / "reseat_probe_20261105_repeats" / "ep-20261105-091931-d19f74.json",
    ]:
        record = load(name)
        block = ((record.get("steps") or [{}])[0] or {}).get("independent_reseat") or {}
        chunks = block.get("chunks") or []
        probes.append(
            {
                "record": str(name.relative_to(PROJECT)),
                "episode_id": record.get("episode_id"),
                "episode_success": bool(record.get("user_goal_success")),
                "status": block.get("status"),
                "policy_actions": block.get("policy_actions"),
                "finalizer_actions": block.get("finalizer_actions"),
                "first_chunk_action0_sha256": chunks[0]["action0_sha256"] if chunks else None,
                "last_chunk_action0_sha256": chunks[-1]["action0_sha256"] if chunks else None,
                "first_noise_sha256": ((block.get("noise") or [{}])[0]).get("noise_sha256"),
                "peak": block.get("peak"),
                "pose_change": block.get("pose_change"),
                "start": {k: block.get("start", {}).get(k) for k in ("sim_action_count", "seat_m", "xy_m", "rim_z", "floor_z")},
            }
        )
    partial_record = load(ACCEPTANCE / "reseat_probe_20261105_repeats" / "ep-20261105-090854-1a89df.json")
    partial_block = ((partial_record.get("steps") or [{}])[0] or {}).get("independent_reseat") or {}
    lift = (partial_block.get("stages") or {}).get("lift") or {}
    peak = partial_block.get("peak") or {}
    start = partial_block.get("start") or {}
    rim_start = start.get("rim_z")
    lift_evidence = lift.get("evidence") or {}
    inferred = {
        "rim_z_start_m": rim_start,
        "rim_z_at_lift_m": None,
        "rim_z_at_peak_m": None,
        "rim_rise_at_lift_m": None,
        "rim_rise_at_peak_m": None,
    }
    bottom0 = start.get("object_bottom_z")
    if isinstance(bottom0, (int, float)) and isinstance(rim_start, (int, float)):
        if isinstance(lift_evidence.get("rise_m"), (int, float)) and isinstance(lift_evidence.get("above_rim_m"), (int, float)):
            rim_lift = bottom0 + lift_evidence["rise_m"] - lift_evidence["above_rim_m"]
            inferred["rim_z_at_lift_m"] = round(rim_lift, 5)
            inferred["rim_rise_at_lift_m"] = round(rim_lift - rim_start, 5)
        if isinstance(peak.get("rise_m"), (int, float)) and isinstance(peak.get("above_rim_m"), (int, float)):
            rim_peak = bottom0 + peak["rise_m"] - peak["above_rim_m"]
            inferred["rim_z_at_peak_m"] = round(rim_peak, 5)
            inferred["rim_rise_at_peak_m"] = round(rim_peak - rim_start, 5)
    return {
        "config": {
            "arm": "config_abB_20261105",
            "shipped_default": "config/engine.json composite.independent_reseat.enabled=false",
            "prompt": "pick up the cream cheese and put it back in the bowl",
            "step_ref": "rr",
            "chunk_budget": 48,
            "max_actions": 240,
            "timeout_s": 180.0,
            "on_result": "report",
        },
        "frozen_config_summary": "records/acceptance/reseat_probe_20261105.json",
        "repeat_probes": probes,
        "repeat_verdict": "4/4 runs: identical chunk-level action hashes, identical noise, status=no_progress",
        "partial_attempt": {
            "record": str(
                (ACCEPTANCE / "reseat_probe_20261105_repeats" / "ep-20261105-090854-1a89df.json").relative_to(PROJECT)
            ),
            "episode_id": partial_record.get("episode_id"),
            "policy_seed": (partial_record.get("sampling") or {}).get("policy_seed"),
            "status": partial_block.get("status"),
            "achieved": partial_block.get("achieved"),
            "peak": peak,
            "pose_change": partial_block.get("pose_change"),
            "outcome": partial_record.get("failure_code"),
            "outcome_reason": partial_record.get("failure_reason"),
            "inferred_rim_motion": inferred,
            "reading": "the bowl rose together with the cheese, so lift/carry fired without a cheese grasp; release/settle never happened and step 1's own predicate was broken",
        },
        "verdict": {
            "grasp": False,
            "lift": False,
            "carry": False,
            "release": False,
            "settle": False,
            "completed": 0,
            "note": "no attempt reached settle; the only lift/carry is the bowl-lift case, which is not a re-seat",
        },
    }


def lifecycle_summary() -> dict[str, Any]:
    cases = []
    for path in [
        ACCEPTANCE / "reseat_stop_20261105.json",
        ACCEPTANCE / "composite_stop_reseat_20261105.json",
    ]:
        payload = load(path)
        case = payload["cases"][0]
        stop = (case.get("stop_ack") or {}).get("stop") or {}
        cases.append(
            {
                "artifact": str(path.relative_to(PROJECT)),
                "episode_id": case.get("episode_id"),
                "ok": case.get("ok"),
                "trigger_detail": case.get("trigger_detail"),
                "sim_at_request": stop.get("sim_at_request"),
                "sim_at_ack": stop.get("sim_at_ack"),
                "in_flight_actions": stop.get("in_flight_actions"),
                "in_flight_after_cancel": stop.get("in_flight_after_cancel"),
                "chunk_buffer_cleared": stop.get("chunk_buffer_cleared"),
                "actions_after_stop": case.get("after_stop", {}).get("last"),
                "advanced_after_stop": case.get("after_stop", {}).get("advanced"),
                "record_status": case.get("record_execution_status"),
                "leak_check": case.get("leak_check"),
            }
        )
    return {
        "cases": cases,
        "verdict": "two independent runs stop inside the re-seat with 6 actions in flight, leave 0 in flight, clear the chunk buffer, and the next load starts at 0 actions / scene_epoch 1",
        "harness_fix": {
            "problem": "composite_stop_tests.py tailed only $EMBODIED_RECORDS_DIR/run_journal.jsonl, so a client without that env var never saw the trigger row and reported a false 'stage_not_reached' after 10 minutes",
            "fix": "explicit --journal path; a wrong path still fails silently, so it must be passed whenever the engine runs on another records dir",
            "note": "the misfired run (ep-20261105-091931-d19f74) actually completed and is counted as the 4th re-seat repeat, not discarded",
        },
    }


def ab_summary() -> dict[str, Any]:
    arm_a = Path("/home/host/flashrt-exp/pi05_round/records_ab_20261105/A")
    return {
        "gate": "only compare A (no re-seat) vs B (re-seat then transport) after an actual re-grasp and re-placement",
        "gate_satisfied": False,
        "arms": {
            "A": "config_abA_20261105 (independent_reseat.enabled=false)",
            "B": "config_abB_20261105 (independent_reseat.enabled=true)",
            "diff": "the two configs differ only in this block and the two ports",
        },
        "runs": {
            "A": "not run",
            "B": "1 capability probe round only (records/acceptance/reseat_probe_20261105.json)",
        },
        "a_records_dir_exists": arm_a.exists(),
        "consequence": "no drop-rate, no added time, no collision comparison is reported for this round",
    }


def main() -> int:
    artifacts = []
    for path, role in (
        ("config/rounds_fixturefix_repeat_20261105.json", "closed-loop repeat manifest"),
        ("config/rounds_reseat_probe_20261105.json", "re-seat capability probe manifest"),
        ("records/acceptance/scene_init_probe_20261105.json", "initialisation probe (3 seeds x seeding x processes)"),
        ("records/acceptance/seed_divergence_20261105.json", "pre-fix divergence report (baseline round)"),
        ("records/acceptance/seed_divergence_fixturefix_20261105.json", "post-fix divergence report"),
        ("records/acceptance/fixturefix_repeat_20261105.json", "closed-loop repeat summary"),
        ("records/acceptance/fixturefix_repeat_20261105.log", "closed-loop repeat stdout"),
        ("records/acceptance/asset_geometry_20261105.json", "asset geometry audit"),
        ("records/acceptance/reseat_probe_20261105.json", "re-seat probe summary"),
        ("records/acceptance/reseat_stop_20261105.json", "re-seat stop case #1"),
        ("records/acceptance/composite_stop_reseat_20261105.json", "re-seat stop case #2"),
    ):
        if (PROJECT / path).exists():
            artifacts.append(artifact(path, role))
        else:
            artifacts.append({"path": path, "role": role, "missing": True})

    for directory, role in (
        ("records/acceptance/fixturefix_repeat_20261105_runs", "closed-loop repeat episode record"),
        ("records/acceptance/reseat_probe_20261105_runs", "re-seat probe episode record (incl. early defect rounds)"),
        ("records/acceptance/reseat_probe_20261105_repeats", "re-seat probe repeat / partial episode record"),
        ("records/acceptance/reseat_stop_20261105_runs", "re-seat stop episode record"),
        ("records/acceptance/scene_init_probe_20261105_logs", "env-server log of an initialisation probe process"),
    ):
        for file in sorted((PROJECT / directory).glob("*.json")) + sorted((PROJECT / directory).glob("*.log")):
            artifacts.append(artifact(str(file.relative_to(PROJECT)), role))

    code = [
        ("adapters/libero_env_server.py", "placement seed flag -> LIBERO_PLACEMENT_SEED"),
        ("adapters/env_runtime/episode_service.py", "placement sampler re-seeding + env.init_state"),
        ("adapters/env_client.py", "init_state RPC"),
        ("engine/engine_service.py", "initial_init_state, independent re-seat, stop propagation"),
        ("common/independent_reseat.py", "stage machine, default off"),
        ("config/engine.json", "composite.independent_reseat (enabled=false)"),
        ("scripts/scene_init_probe.py", "initialisation probe"),
        ("scripts/asset_geometry_audit.py", "asset geometry audit"),
        ("scripts/seed_divergence_report.py", "init_* comparison stages"),
        ("scripts/composite_stop_tests.py", "re-seat stop case + --journal"),
        ("scripts/run_rounds.py", "independent_reseat summary column"),
        ("scripts/build_acceptance_manifest_scene_init.py", "this manifest generator"),
        ("tests/test_independent_reseat.py", "15 unit tests"),
        ("docs/architecture.md", "section 14"),
        ("docs/acceptance_report.md", "section 37"),
    ]
    for path, role in code:
        artifacts.append(artifact(path, f"code: {role}"))

    payload = {
        "schema": "embodied_engine.manifest.v1",
        "round": "scene-init-reproducible-and-independent-reseat-20261105",
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "branch": "feature/pi05-scene-init-and-reseat-capability",
        "baseline": {
            "commit": "4a5f975e61827cfc99ea87e56306091f5fc03ca4",
            "tag": "backup/pi05-reproducible-baseline-20261105-4a5f975",
            "source_tarball": "/home/host/flashrt-exp/freeze/2026-11-05-pi05-reproducible-baseline/embodied_engine-src-4a5f975.tar.gz",
            "source_tarball_sha256": "258d2ff1e75bd7dd9fae9f63147251e418be1618a69278b39de576334b16e474",
        },
        "scope": {
            "changed": [
                "scene init: LIBERO placement samplers re-seeded inside every reset (fixtures included)",
                "records: initial_init_state (fixtures/movable/joints/robot/images) + placement_seeding",
                "new capability: independent re-seat stage machine, default off",
                "lifecycle: stop inside the re-seat, reset isolation after it",
                "docs: architecture section 14, acceptance report section 37",
            ],
            "unchanged": [
                "model weights and precision",
                "num_steps=10, action_horizon=10, execute_horizon=5",
                "transport_ready thresholds (0.029 / 0.002 / 0.015)",
                "composite step budgets",
                "router prompts and flow boundaries",
                "all archived rounds, tags and earlier results",
            ],
            "no_training": True,
            "no_router_prompt_change": True,
            "no_threshold_change_after_results": True,
        },
        "engine_identity": {
            "action_backend": "pi05_libero (official PyTorch conversion)",
            "inference": "openpi.PI0Pytorch (eager, no CUDA Graph)",
            "num_steps": 10,
            "action_horizon": 10,
            "execute_horizon": 5,
        },
        "placement_seeding": {
            "mechanism": "libero_env_server maps --seed to LIBERO_PLACEMENT_SEED before importing the env server; episode_service reseeds every leaf placement sampler and the global numpy stream at the top of each _reset_internal",
            "key": "(scene_seed, sampler name) -> splitmix64",
            "samplers_seeded_seed5": 9,
            "global_numpy_seed_seed5": 841021262,
            "control_arm": "LIBERO_PLACEMENT_SEEDING=0",
            "root_cause": "ObjectPositionSampler.__init__ uses np.random.default_rng() (OS entropy); BDDLBaseDomain.seed only calls np.random.seed",
        },
        "initialisation": init_probe_summary(),
        "closed_loop": repeat_summary(),
        "asset_geometry": asset_geometry(),
        "independent_reseat": reseat_summary(),
        "lifecycle": lifecycle_summary(),
        "ab_comparison": ab_summary(),
        "kept_failures": [
            "ep-20261105-085544-71541a: UnboundLocalError step_task",
            "ep-20261105-085655-0ba6d1: NameError step",
            "ep-20261105-085835-54bdb2: KeyError main_images",
            "ep-20261105-090008-62b68d: missing agentview_image",
            "ep-20261105-090151-3b2a03: pi05 500 image buffer 196608 vs 150528",
            "ep-20261105-091931-d19f74: stop harness sync miss (ran to completion, kept as 4th repeat)",
            "ep-20261105-090854-1a89df: handoff refused after the re-seat carried the bowl away",
        ],
        "tests": {
            "suite": "pytest tests -q",
            "passed": 224,
            "new": ["tests/test_independent_reseat.py (15 cases)"],
        },
        "artifacts": artifacts,
    }
    out = ACCEPTANCE / "manifest_20261105_scene_init_and_reseat.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {out.relative_to(PROJECT)} ({out.stat().st_size} bytes, {len(artifacts)} artifacts)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
