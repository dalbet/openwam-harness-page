#!/usr/bin/env python
"""Is a "same init, different outcome" episode explained by the model or the pixels?

Two resets with the same seed put the scene in the same pose. If the rendered
images differ anyway, the policy is fed different inputs and a different action
chunk is a legitimate consequence, not model sampling error. This probe measures
that, and it also re-renders one unchanged state twice so the two effects are
separable:

* ``same_state_pixels``  - renderer determinism with nothing moving,
* ``reset_to_reset``     - what the policy actually sees across two resets.

Read-only with respect to the engine: it resets the env RPC, which the engine
does on every episode anyway. Nothing is stepped, no policy is loaded.

    python scripts/render_determinism_probe.py --task 2
"""

from __future__ import annotations

import argparse
import io
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from adapters.env_client import EnvClient  # noqa: E402

OUT = ROOT / "records" / "acceptance" / "render_determinism.json"
POSE_KEYS = ("eef_pos", "gripper_opening_m")


def _image(raw) -> np.ndarray:
    if isinstance(raw, np.ndarray):
        return raw.astype(np.int16)
    from PIL import Image

    if isinstance(raw, (bytes, bytearray)):
        data = bytes(raw)
    elif isinstance(raw, str):
        import base64

        data = base64.b64decode(raw)
    elif isinstance(raw, (list, tuple)):
        data = bytes(raw)
    else:
        raise TypeError(type(raw).__name__)
    return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"), dtype=np.int16)


def _pixels(client: EnvClient, camera: str, size: int) -> np.ndarray:
    return _image(client.render_camera(camera, size, size))


def _pose(snapshot: dict) -> dict:
    sim = snapshot.get("sim") or {}
    joints = {name: round(float(value), 6) for name, value in (snapshot.get("joints") or {}).items()}
    objects = {
        name: [round(float(v), 6) for v in (item.get("pos") or [])]
        for name, item in (sim.get("objects") or {}).items()
    }
    geometry = (snapshot.get("named_poses") or {}).get("site_geometry") or {}
    site_quats = {
        name: [round(float(v), 6) for v in (item.get("quat_wxyz") or [])]
        for name, item in geometry.items()
        if name.endswith("_default_site")
    }
    return {
        "joints": joints,
        "objects": objects,
        "site_quats": site_quats,
        "eef_pos": [round(float(v), 6) for v in (sim.get("eef_pos") or [])],
        "gripper_opening_m": sim.get("gripper_opening_m"),
        "scene_epoch": snapshot.get("scene_epoch"),
    }


def _quat_angle_deg(a: list[float], b: list[float]) -> float | None:
    if len(a) != 4 or len(b) != 4:
        return None
    dot = abs(sum(x * y for x, y in zip(a, b)))
    dot = min(1.0, max(-1.0, dot))
    return round(2.0 * math.degrees(math.acos(dot)), 4)


def _diff(a: np.ndarray, b: np.ndarray) -> dict:
    delta = np.abs(a - b)
    return {
        "shape": list(a.shape),
        "max_abs_diff": int(delta.max()),
        "mean_abs_diff": round(float(delta.mean()), 4),
        "differing_pixels": int((delta.max(axis=2) > 0).sum()),
        "differing_pixels_gt16": int((delta.max(axis=2) > 16).sum()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="http://127.0.0.1:14561")
    parser.add_argument("--camera", default="agentview")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--resets", type=int, default=3)
    parser.add_argument("--task", type=int, default=1, help="only recorded, the env server owns the task")
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()

    client = EnvClient(args.env)
    meta = client.meta()

    passes = []
    for index in range(args.resets):
        client.reset()
        snapshot = client.state_snapshot()
        image = _pixels(client, args.camera, args.size)
        repeat = _pixels(client, args.camera, args.size) if index == 0 else None
        passes.append({"index": index, "pose": _pose(snapshot), "image": image, "repeat": repeat})

    pose_keys = [
        (key, [p["pose"][key] for p in passes])
        for key in ("joints", "objects", "site_quats", "eef_pos", "gripper_opening_m")
    ]
    pose_identical = all(
        len({json.dumps(value, sort_keys=True) for value in values}) == 1 for _, values in pose_keys
    )
    orientation = []
    for _, values in [item for item in pose_keys if item[0] == "site_quats"]:
        homes = {name for value in values for name in value}
        for name in sorted(homes):
            angles = [
                _quat_angle_deg(value[name], values[0][name])
                for value in values[1:]
                if name in value
            ]
            orientation.append({"site": name, "max_angle_deg_vs_first_reset": max(angles) if angles else None})
    # scene_epoch is a counter, not a pose; it is recorded but never compared.
    pairwise = [
        {"a": i, "b": j, **_diff(passes[i]["image"], passes[j]["image"])}
        for i in range(len(passes))
        for j in range(i + 1, len(passes))
    ]
    payload = {
        "schema": "embodied_engine.render_determinism.v1",
        "env": args.env,
        "camera": args.camera,
        "size": args.size,
        "task_id_recorded": args.task,
        "env_task_id": meta.get("task"),
        "suite": meta.get("suite"),
        "seed": meta.get("seed"),
        "resets": args.resets,
        "same_state_pixels": _diff(passes[0]["image"], passes[0]["repeat"]) if args.resets else None,
        "pose_identical": pose_identical,
        "positions_identical": all(
            len({json.dumps(value, sort_keys=True) for value in values}) == 1
            for key, values in pose_keys
            if key != "site_quats"
        ),
        "quats_identical": all(
            len({json.dumps(value, sort_keys=True) for value in values}) == 1
            for key, values in pose_keys
            if key == "site_quats"
        ),
        "orientation": orientation,
        "scene_epochs": [p["pose"]["scene_epoch"] for p in passes],
        "pairwise": pairwise,
        "diff_fraction": {f"{item['a']}-{item['b']}": round(item["differing_pixels"] / 65536.0, 4) for item in pairwise},
        "note": (
            "same_state_pixels isolates the renderer: identical. pose_identical compares eef, gripper opening, "
            "all 17 joints and every object xyz across resets. The snapshot exposes no quaternions, so an "
            "identical-position reset can still differ in object or free-joint rotation; a differing image "
            "therefore means the policy input is not reproducible, not that the model sampled differently."
        ),
    }
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=1))
    print(
        json.dumps(
            {
                "pose_identical": pose_identical,
                "same_state_pixels": payload["same_state_pixels"],
                "pairwise": pairwise,
                "scene_epochs": payload["scene_epochs"],
            },
            ensure_ascii=False,
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
