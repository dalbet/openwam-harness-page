#!/usr/bin/env python
"""Does the policy see the same *pixels* on two resets with the same scene seed?

This is the question the 2026-11-05 spread (2/6, 5/6, 5/6, 4/6 over one frozen
manifest) actually turns on. The four batches agreed on
``initial_state_fingerprint`` -- object xyz, joint angles, gripper opening -- to
five/six decimals, and their first policy request still emitted four different
first actions (A 0.4584/0.4335, B 0.4559/0.4239, C 0.4016/0.4315,
D 0.4341/0.4436 on the first two dims). So the divergence is upstream of the
sampler.

The probe separates the two possible upstream causes:

``same_state_reread``
    Read the raw observation twice with nothing stepped. Isolates the renderer
    and the observation RPC: if this differs, nothing downstream can be
    reproducible.

``reset_to_reset``
    Two ``/api/load`` calls with the same task and scene seed, i.e. exactly what
    the round runner does before every round. Compares the *policy* input built
    by ``build_pi05_obs`` (rotate-180 + PIL bilinear 224) and the pose snapshot
    side by side, so "same pose, different pixels" is measured rather than
    argued.

Nothing is stepped and no policy request is sent, so this can run while the
policy server stays loaded.

    EMBODIED_CONFIG_DIR=... python scripts/policy_obs_determinism_probe.py \
        --engine http://127.0.0.1:14821 --task 7 --seed 3 --out records/acceptance/policy_obs_determinism.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from adapters.env_client import EnvClient  # noqa: E402
from adapters.pi05_contract import build_pi05_obs  # noqa: E402
from common import config  # noqa: E402

OUT = ROOT / "records" / "acceptance" / "policy_obs_determinism.json"
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _post(url: str, body: dict, timeout: float = 400.0) -> dict:
    data = json.dumps(body).encode()
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with OPENER.open(request, timeout=timeout) as response:
        return json.loads(response.read().decode())


def _sha(array) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(array)).tobytes()).hexdigest()


def _diff(a: np.ndarray, b: np.ndarray) -> dict:
    delta = np.abs(a.astype(np.int16) - b.astype(np.int16))
    mask = delta.max(axis=2) > 0
    strong = delta.max(axis=2) > 16
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return {
        "shape": list(a.shape),
        "max_abs_diff": int(delta.max()),
        "mean_abs_diff": round(float(delta.mean()), 4),
        "differing_pixels": int(mask.sum()),
        "differing_pixels_gt16": int(strong.sum()),
        "differing_fraction": round(float(mask.mean()), 4),
        "bbox": None if not len(rows) else [int(rows[0]), int(cols[0]), int(rows[-1]), int(cols[-1])],
    }


def _pose(snapshot: dict) -> dict:
    sim = snapshot.get("sim") or {}
    geometry = (snapshot.get("named_poses") or {}).get("site_geometry") or {}
    return {
        "scene_epoch": snapshot.get("scene_epoch"),
        "eef_pos": [round(float(v), 6) for v in (sim.get("eef_pos") or [])],
        "gripper_opening_m": sim.get("gripper_opening_m"),
        "joints": {str(k): round(float(v), 6) for k, v in (snapshot.get("joints") or {}).items()},
        "objects": {
            str(name): [round(float(v), 6) for v in (item.get("pos") or [])]
            for name, item in (sim.get("objects") or {}).items()
        },
        # Object *orientation* is not in ``sim.objects``; the site geometry is
        # the only place the snapshot exposes a quaternion, so it is what can
        # rule "an object rotated a little" in or out.
        "site_geometry": {
            str(name): {
                "xyz": [round(float(v), 6) for v in (item.get("xyz") or [])],
                "quat_wxyz": [round(float(v), 6) for v in (item.get("quat_wxyz") or [])],
            }
            for name, item in geometry.items()
        },
    }


def _geometry_diff(a: dict, b: dict) -> dict:
    """Largest orientation/position change between two snapshots, per site."""
    worst_angle = {"site": None, "deg": 0.0}
    worst_pos = {"site": None, "m": 0.0}
    for name, row in (a or {}).items():
        other = (b or {}).get(name)
        if not other:
            continue
        qa, qb = row.get("quat_wxyz") or [], other.get("quat_wxyz") or []
        if len(qa) == 4 and len(qb) == 4:
            dot = abs(sum(x * y for x, y in zip(qa, qb)))
            dot = min(1.0, max(-1.0, dot))
            import math

            deg = 2.0 * math.degrees(math.acos(dot))
            if deg > worst_angle["deg"]:
                worst_angle = {"site": name, "deg": round(deg, 6)}
        pa, pb = row.get("xyz") or [], other.get("xyz") or []
        if len(pa) == 3 and len(pb) == 3:
            dist = float(np.linalg.norm(np.asarray(pa) - np.asarray(pb)))
            if dist > worst_pos["m"]:
                worst_pos = {"site": name, "m": round(dist, 9)}
    return {"max_rotation_deg": worst_angle, "max_position_m": worst_pos}


def _obs_row(client: EnvClient) -> dict:
    raw = client.raw_obs()
    obs = build_pi05_obs(raw)
    return {
        "main": np.asarray(obs["main_images"]),
        "wrist": np.asarray(obs["wrist_images"]),
        "states": np.asarray(obs["states"], dtype=np.float32),
        "hashes": {
            "main_sha256": _sha(obs["main_images"]),
            "wrist_sha256": _sha(obs["wrist_images"]),
            "state_sha256": _sha(obs["states"]),
        },
        "raw_hashes": {
            "agentview_sha256": _sha(raw.get("agentview_image")),
            "eye_in_hand_sha256": _sha(raw.get("robot0_eye_in_hand_image")),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", default=None, help="defaults to services.engine in the active config")
    parser.add_argument("--env", default=None, help="defaults to services.env in the active config")
    parser.add_argument("--task", type=int, default=7)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--passes", type=int, default=3)
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()

    engine_url = args.engine or config.service_url("engine")
    env_url = args.env or config.service_url("env")
    client = EnvClient(env_url)

    passes = []
    for index in range(args.passes):
        loaded = _post(f"{engine_url}/api/load", {"task_id": int(args.task), "seed": int(args.seed)})
        if loaded.get("status") != "ok":
            raise SystemExit(f"load failed: {loaded.get('error') or loaded}")
        first = _obs_row(client)          # what the policy would receive
        reread = _obs_row(client)         # same state, second read: renderer + RPC
        pose = _pose(client.state_snapshot())
        passes.append({"index": index, "obs": first, "reread": reread, "pose": pose})
        print(
            f"pass {index}: scene_epoch={pose['scene_epoch']} "
            f"main={first['hashes']['main_sha256'][:12]} "
            f"reread_same={first['hashes']['main_sha256'] == reread['hashes']['main_sha256']}",
            flush=True,
        )

    def pairwise(rows: list[dict], key: str) -> list[dict]:
        out = []
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                a, b = rows[i][key], rows[j][key]
                out.append(
                    {
                        "a": i,
                        "b": j,
                        "identical": a["hashes"] == b["hashes"],
                        "main": _diff(a["main"], b["main"]),
                        "wrist": _diff(a["wrist"], b["wrist"]),
                        "states_identical": bool(np.array_equal(a["states"], b["states"])),
                        "state_max_abs_diff": float(np.abs(a["states"] - b["states"]).max()),
                    }
                )
        return out

    same_state = [
        {
            "pass": row["index"],
            "identical": row["obs"]["hashes"] == row["reread"]["hashes"],
            "raw_identical": row["obs"]["raw_hashes"] == row["reread"]["raw_hashes"],
            "main": _diff(row["obs"]["main"], row["reread"]["main"]),
        }
        for row in passes
    ]
    across = pairwise(passes, "obs")
    pose_rows = [row["pose"] for row in passes]
    pose_identical = all(
        json.dumps(pose_rows[0][key], sort_keys=True) == json.dumps(row[key], sort_keys=True)
        for row in pose_rows[1:]
        for key in ("eef_pos", "gripper_opening_m", "objects", "joints")
    )
    geometry = [
        {"a": i, "b": j, **_geometry_diff(pose_rows[i]["site_geometry"], pose_rows[j]["site_geometry"])}
        for i in range(len(pose_rows))
        for j in range(i + 1, len(pose_rows))
    ]
    obs_identical = all(item["identical"] for item in across)

    payload = {
        "schema": "embodied_engine.policy_obs_determinism.v1",
        "engine": engine_url,
        "env": env_url,
        "task_id": int(args.task),
        "scene_seed": int(args.seed),
        "passes": len(passes),
        "same_state_reread_identical": all(item["identical"] for item in same_state),
        "same_state_reread": same_state,
        "pose_identical_across_resets": pose_identical,
        "geometry_diff_across_resets": geometry,
        "policy_obs_identical_across_resets": obs_identical,
        "reset_to_reset": across,
        "scene_epochs": [row["scene_epoch"] for row in pose_rows],
        "hashes": [
            {"pass": row["index"], **{**row["obs"]["hashes"], **row["obs"]["raw_hashes"]}} for row in passes
        ],
        "verdict": (
            "policy input reproducible: the same scene seed yields the same pixels and state, so a differing "
            "action must come from the sampler"
            if obs_identical
            else "policy input NOT reproducible: the same scene seed yields different pixels at the same pose, "
            "so the sampler is not the first divergence"
        ),
        "caveat": "pixels are compared on the 224x224 policy input after rotate-180 and PIL bilinear resize; "
        "no claim is made about bit-identity across software or hardware versions",
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n")
    print(
        json.dumps(
            {
                "same_state_reread_identical": payload["same_state_reread_identical"],
                "pose_identical_across_resets": pose_identical,
                "policy_obs_identical_across_resets": obs_identical,
                "diff_fraction": [
                    {"pair": f"{r['a']}-{r['b']}", "main": r["main"]["differing_fraction"],
                     "wrist": r["wrist"]["differing_fraction"], "state_max": r["state_max_abs_diff"]}
                    for r in across
                ],
                "verdict": payload["verdict"],
            },
            ensure_ascii=False,
            indent=1,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
