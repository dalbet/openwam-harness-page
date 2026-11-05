#!/usr/bin/env python
"""Is the initial scene the same twice?

The closed-loop matrix left one question open: two runs with the same scene
seed and the same policy seed did not follow the same path, and the first
difference was in the pixel input, not in the diffusion noise. This probe
answers that question on its own, without a policy:

for each seed, launch a fresh env server (the same launcher the engine uses),
call ``reset`` twice, and compare the two initial states *in a fixed order*:

1. ``fixtures`` — ``sim.model.body_pos/body_quat`` for every body. This is what
   the placement sampler writes, and it is the field that was changing.
2. ``movable`` — the ``*_pos`` bodies (qpos-driven, restored from LIBERO's
   ``init_files`` by ``set_init_state``).
3. ``joints`` — every joint's qpos and qvel.
4. ``robot`` — eef pose, gripper opening, controller references.
5. ``images`` — sha256 of both raw camera frames.

The first stage that differs is the answer, and the record names the first
*field* inside it with both values, so "the images differ" can never be
reported without saying whether the physics differed first.

Two arms are supported: ``--seeding on`` (the launcher sets
``LIBERO_PLACEMENT_SEED`` so the samplers are rebuilt from a fixed stream) and
``--seeding off`` (LIBERO's shipped behaviour: ``np.random.default_rng()`` per
sampler, i.e. OS entropy). The off arm is the control that shows the cause.

    python scripts/scene_init_probe.py --seeds 0 1 2 --repeats 2 \
        --out records/acceptance/scene_init_probe_20261105.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from adapters.env_client import EnvClient  # noqa: E402
from adapters.pi05_contract import build_pi05_obs  # noqa: E402
from common import config, policy_seed  # noqa: E402

STAGE_ORDER = ["fixtures", "movable", "joints", "robot", "images"]


def _digest(payload: Any) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def _fixture_view(state: dict[str, Any]) -> dict[str, Any]:
    """Body model poses: what the placement sampler writes for fixtures."""
    bodies = state.get("bodies") or {}
    return {
        name: {"model_pos": row.get("model_pos"), "model_quat": row.get("model_quat")}
        for name, row in bodies.items()
        if isinstance(row, dict) and ("model_pos" in row)
    }


def _movable_view(state: dict[str, Any]) -> dict[str, Any]:
    movable = state.get("movable") or {}
    bodies = state.get("bodies") or {}
    out = {}
    for name, row in movable.items():
        body = bodies.get(name) or {}
        out[name] = {
            "pos": (row or {}).get("pos"),
            "xquat": body.get("xquat") if isinstance(body, dict) else None,
        }
    return out


def _joint_view(state: dict[str, Any]) -> dict[str, Any]:
    joints = state.get("joints") or {}
    return {
        name: {"qpos": row.get("qpos"), "qvel": row.get("qvel")}
        for name, row in joints.items()
        if isinstance(row, dict) and "qpos" in row
    }


def _robot_view(state: dict[str, Any]) -> dict[str, Any]:
    robot = state.get("robot") or {}
    return {
        "eef_pos": robot.get("eef_pos"),
        "eef_quat": robot.get("eef_quat"),
        "gripper_opening_m": robot.get("gripper_opening_m"),
        "controllers": robot.get("controllers"),
        "joints": robot.get("joints"),
    }


def _image_view(state: dict[str, Any]) -> dict[str, Any]:
    images = state.get("images") or {}
    return {
        name: {"sha256": row.get("sha256"), "shape": row.get("shape"), "sum": row.get("sum")}
        for name, row in images.items()
        if isinstance(row, dict)
    }


VIEWS = {
    "fixtures": _fixture_view,
    "movable": _movable_view,
    "joints": _joint_view,
    "robot": _robot_view,
    "images": _image_view,
}


def _first_field_difference(a: Any, b: Any, path: str = "") -> dict[str, Any] | None:
    """First leaf that differs, with both values, in a deterministic key order."""
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            if key not in a or key not in b:
                return {"field": f"{path}.{key}".lstrip("."), "a": a.get(key), "b": b.get(key), "kind": "missing_key"}
            found = _first_field_difference(a[key], b[key], f"{path}.{key}")
            if found is not None:
                return found
        return None
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return {"field": f"{path}.length", "a": len(a), "b": len(b), "kind": "length"}
        for index, (left, right) in enumerate(zip(a, b)):
            found = _first_field_difference(left, right, f"{path}[{index}]")
            if found is not None:
                return found
        return None
    if a != b:
        try:
            delta = abs(float(a) - float(b))
        except (TypeError, ValueError):
            delta = None
        return {"field": path, "a": a, "b": b, "kind": "value", "abs_delta": delta}
    return None


def compare_states(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    stages = []
    for stage in STAGE_ORDER:
        view_left = VIEWS[stage](left)
        view_right = VIEWS[stage](right)
        difference = _first_field_difference(view_left, view_right)
        stages.append(
            {
                "stage": stage,
                "same": difference is None,
                "sha256": [_digest(view_left), _digest(view_right)],
                "fields": len(view_left),
                "first_difference": difference,
            }
        )
    first = next((row["stage"] for row in stages if not row["same"]), None)
    return {
        "stages": stages,
        "first_divergence": first,
        "physics_identical": all(row["same"] for row in stages if row["stage"] != "images"),
        "images_identical": next(row["same"] for row in stages if row["stage"] == "images"),
    }


def _launch_env(*, port: int, task: int, seed: int, seeding: bool, log_path: Path) -> subprocess.Popen:
    env = dict(os.environ)
    engine_cfg = config.engine_config()
    env.update(
        {
            "HF_HUB_OFFLINE": "1",
            "MUJOCO_GL": "egl",
            "PYOPENGL_PLATFORM": "egl",
            "LIBERO_TYPE": "standard",
            "LIBERO_PLACEMENT_SEEDING": "1" if seeding else "0",
            "PYTHONPATH": f"{PROJECT / 'adapters' / 'env_runtime'}:{engine_cfg['env']['rpent_root']}",
        }
    )
    command = [
        str(engine_cfg["env"]["python"]),
        str(PROJECT / "adapters" / "libero_env_server.py"),
        "--transport",
        "http",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--suite",
        str(engine_cfg["env"]["suite"]),
        "--task",
        str(task),
        "--seed",
        str(seed),
        "--max-episode-steps",
        str(engine_cfg["env"]["max_episode_steps"]),
        "--cuda-device",
        str(engine_cfg["env"]["cuda_device"]),
    ]
    log = log_path.open("w", encoding="utf-8")
    return subprocess.Popen(
        command,
        cwd=str(PROJECT),
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
        close_fds=True,
    )


def _wait(port: int, proc: subprocess.Popen, log_path: Path, timeout_s: float = 240.0) -> EnvClient:
    client = EnvClient(f"http://127.0.0.1:{port}")
    started = time.time()
    last = ""
    while time.time() - started < timeout_s:
        if proc.poll() is not None:
            tail = ""
            try:
                tail = log_path.read_text(encoding="utf-8", errors="replace")[-2000:]
            except OSError:
                pass
            raise RuntimeError(f"env server exited with {proc.returncode}\n{tail}")
        try:
            client.identity()
            return client
        except Exception as exc:  # noqa: BLE001 - not up yet
            last = f"{type(exc).__name__}: {exc}"
            time.sleep(1.0)
    raise RuntimeError(f"env server on port {port} not ready: {last}")


def _capture(client: EnvClient) -> dict[str, Any]:
    """One reset plus both readings: the policy-facing frames and the full state."""
    client.reset()
    init = client.init_state()
    snap = client.state_snapshot()
    obs = client.raw_obs()
    policy = build_pi05_obs(obs)
    frames = {}
    for key in ("agentview_image", "robot0_eye_in_hand_image"):
        frame = (obs or {}).get(key)
        if frame is not None:
            import numpy as np

            array = np.ascontiguousarray(np.asarray(frame))
            frames[key] = {
                "shape": list(array.shape),
                "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
                "sum": int(array.sum()),
            }
    init["policy_input"] = {
        "main_sha256": policy_seed.digest(policy["main_images"]),
        "wrist_sha256": policy_seed.digest(policy["wrist_images"]),
        "state_sha256": policy_seed.digest(policy["states"]),
        "state_dim": int(len(policy["states"])),
    }
    init["raw_frames"] = frames
    init["snapshot_state_fingerprint"] = _state_fingerprint(snap)
    init["snapshot_fixture_fingerprint"] = _fixture_fingerprint(snap)
    return init


def _state_fingerprint(snapshot: dict[str, Any]) -> str:
    sim = snapshot.get("sim") or {}
    objects = {
        name: [round(float(v), 5) for v in (pose or {}).get("pos") or []]
        for name, pose in (sim.get("objects") or {}).items()
        if isinstance(pose, dict)
    }
    payload = {
        "objects": objects,
        "eef": [round(float(v), 5) for v in (sim.get("eef_pos") or [])],
        "joints": {str(k): round(float(v), 6) for k, v in (snapshot.get("joints") or {}).items() if isinstance(v, (int, float))},
        "gripper": round(float(sim.get("gripper_opening_m") or 0.0), 6),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def _fixture_fingerprint(snapshot: dict[str, Any]) -> str:
    geometry = (snapshot.get("named_poses") or {}).get("site_geometry") or {}
    fixtures = {
        str(name): {
            "xyz": [round(float(v), 6) for v in (item.get("xyz") or [])],
            "quat": [round(float(v), 6) for v in (item.get("quat_wxyz") or [])],
        }
        for name, item in geometry.items()
        if isinstance(item, dict)
    }
    payload = {"fixtures": fixtures, "bodies": snapshot.get("named_poses", {}).get("bodies") or {}}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def probe_process(*, task: int, seed: int, repeats: int, seeding: bool, port: int, out_dir: Path, label: str) -> dict[str, Any]:
    log_path = out_dir / f"env_task{task}_seed{seed}_seeding{int(seeding)}_{label}.log"
    out_dir.mkdir(parents=True, exist_ok=True)
    proc = _launch_env(port=port, task=task, seed=seed, seeding=seeding, log_path=log_path)
    try:
        client = _wait(port, proc, log_path)
        captures = [_capture(client) for _ in range(repeats)]
        return {
            "label": label,
            "port": port,
            "log": str(log_path),
            "captures": captures,
            "placement": captures[0].get("placement"),
            "placement_seeding": captures[0].get("placement_seeding"),
        }
    finally:
        try:
            proc.terminate()
            proc.wait(timeout=30)
        except Exception:  # noqa: BLE001
            try:
                proc.kill()
            except Exception:  # noqa: BLE001
                pass


def probe_seed(*, task: int, seed: int, repeats: int, processes: int, seeding: bool, port: int, out_dir: Path) -> dict[str, Any]:
    runs = []
    for index in range(processes):
        runs.append(
            probe_process(
                task=task,
                seed=seed,
                repeats=repeats,
                seeding=seeding,
                port=port + index,
                out_dir=out_dir,
                label=f"p{index + 1}",
            )
        )
    comparisons = []
    for index in range(1, repeats):
        comparisons.append(
            {
                "scope": "in_process",
                "pair": [f"p1:load_1", f"p1:load_{index + 1}"],
                **compare_states(runs[0]["captures"][0], runs[0]["captures"][index]),
            }
        )
    for index in range(1, processes):
        comparisons.append(
            {
                "scope": "cross_process",
                "pair": ["p1:load_1", f"p{index + 1}:load_1"],
                **compare_states(runs[0]["captures"][0], runs[index]["captures"][0]),
            }
        )
    for row in comparisons:
        left, right = row["pair"]
        left_cap = run_capture(runs, left)
        right_cap = run_capture(runs, right)
        row["policy_input_same"] = (
            left_cap["policy_input"]["main_sha256"] == right_cap["policy_input"]["main_sha256"]
            and left_cap["policy_input"]["wrist_sha256"] == right_cap["policy_input"]["wrist_sha256"]
        )
    loads = []
    for run in runs:
        for index, capture in enumerate(run["captures"]):
            loads.append(
                {
                    "label": f"{run['label']}:load_{index + 1}",
                    "process": run["label"],
                    "state_fingerprint": capture.get("snapshot_state_fingerprint"),
                    "fixture_fingerprint": capture.get("snapshot_fixture_fingerprint"),
                    "policy_input": capture.get("policy_input"),
                    "raw_frames": capture.get("raw_frames"),
                    "fixture_sha256": _digest(_fixture_view(capture)),
                    "movable_sha256": _digest(_movable_view(capture)),
                    "joints_sha256": _digest(_joint_view(capture)),
                    "robot_sha256": _digest(_robot_view(capture)),
                    "placement": capture.get("placement"),
                }
            )
    return {
        "task_id": task,
        "seed": seed,
        "seeding": seeding,
        "repeats": repeats,
        "processes": processes,
        "placement": runs[0].get("placement"),
        "placement_seeding": runs[0].get("placement_seeding"),
        "logs": [run["log"] for run in runs],
        "loads": loads,
        "comparisons": comparisons,
    }


def run_capture(runs: list[dict[str, Any]], label: str) -> dict[str, Any]:
    process_label, load_label = label.split(":")
    index = int(load_label.split("_")[-1]) - 1
    for run in runs:
        if run["label"] == process_label:
            return run["captures"][index]
    raise KeyError(label)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=int, default=7)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--processes", type=int, default=2, help="independent env server processes per seed")
    parser.add_argument("--arms", nargs="+", default=["on"], choices=["on", "off"])
    parser.add_argument("--port", type=int, default=14571)
    parser.add_argument("--out", default=str(PROJECT / "records/acceptance/scene_init_probe_20261105.json"))
    parser.add_argument("--out-dir", default="/tmp/scene_init_probe")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    rows = []
    port = args.port
    for arm in args.arms:
        for seed in args.seeds:
            row = probe_seed(
                task=args.task,
                seed=seed,
                repeats=args.repeats,
                processes=args.processes,
                seeding=arm == "on",
                port=port,
                out_dir=out_dir,
            )
            port += args.processes + 1
            rows.append(row)
            for comparison in row["comparisons"]:
                print(
                    f"[task {row['task_id']} seed {row['seed']} seeding {arm}] "
                    f"{comparison['scope']:13s} {comparison['pair'][0]} vs {comparison['pair'][1]}: "
                    f"first_divergence={comparison['first_divergence']} "
                    f"physics_identical={comparison['physics_identical']} "
                    f"images_identical={comparison['images_identical']}",
                    flush=True,
                )

    payload = {
        "schema": "embodied_engine.scene_init_probe.v1",
        "task_id": args.task,
        "seeds": args.seeds,
        "repeats": args.repeats,
        "arms": args.arms,
        "order": STAGE_ORDER,
        "note": (
            "each seed is probed with independent env server processes, each doing `repeats` "
            "resets: in_process compares two resets of one process, cross_process compares the "
            "first reset of two processes, so both the reset path and the process start are covered"
        ),
        "rows": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"written: {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
