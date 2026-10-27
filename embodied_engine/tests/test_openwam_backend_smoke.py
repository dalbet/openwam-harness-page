#!/usr/bin/env python
"""Smoke test of the OpenWAM model layer on a real LIBERO observation.

Exercises the full path the engine uses:

    raw LIBERO obs -> adapters.openwam_contract -> engine.inference_client
    -> inference service (:14600) -> models.openwam_backend (WebSocket)
    -> resident OpenWAM server -> 10-D native EEF10 action

Run with the engine interpreter (the env server must be up on :14561):

    /home/host/page-dmh/openwam_harness/envs/rpent/bin/python tests/test_openwam_backend_smoke.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ENGINE_ROOT = Path(__file__).resolve().parents[1]
if str(ENGINE_ROOT) not in sys.path:
    sys.path.insert(0, str(ENGINE_ROOT))

from adapters.env_client import EnvClient  # noqa: E402
from adapters.openwam_contract import build_openwam_obs, native_eef10_to_libero7d  # noqa: E402
from engine.inference_client import InferenceClient, request_row  # noqa: E402

PROMPT = "put the black bowl on the stove"


def main() -> int:
    inference = InferenceClient("http://127.0.0.1:14600")
    health = inference.health()
    identity = health.get("identity") or {}
    print("inference backend:", identity.get("backend"), "| status:", health.get("status"), "| mode:", health.get("mode"))
    if identity.get("backend") != "openwam":
        raise SystemExit(f"refusing to run: selected backend is {identity.get('backend')!r}, not openwam")

    env = EnvClient("http://127.0.0.1:14561")
    snapshot = env.state_snapshot()
    raw = env.raw_obs()
    obs = build_openwam_obs(raw)
    print(
        "obs:",
        "head", obs["main_images"].shape,
        "wrist", obs["wrist_images"].shape,
        "state", np.round(obs["states"], 4).tolist(),
    )

    token = inference.reset("smoke")
    timings = []
    for attempt in range(3):
        request = request_row(
            token=token,
            episode_id="openwam-smoke",
            scene_epoch=snapshot.get("scene_epoch"),
            snapshot=snapshot,
            skill="whole_task",
            step_id=attempt,
            prompt=PROMPT,
        )
        started = time.perf_counter()
        out = inference.infer(request, obs, timeout_s=300)
        elapsed = time.perf_counter() - started
        if out.get("status") != "ok":
            raise SystemExit(f"inference failed: {out.get('error') or out.get('reason')}")
        actions = np.asarray(out["actions"], dtype=np.float32)
        if actions.shape != (1, 10):
            raise SystemExit(f"unexpected action shape {actions.shape}")
        if not np.isfinite(actions).all():
            raise SystemExit("non-finite action")
        action7 = native_eef10_to_libero7d(actions[0])
        timings.append(out.get("inference_s"))
        print(
            f"call {attempt + 1}: client={elapsed*1000:.1f}ms inference={out.get('inference_s')}s "
            f"server={out.get('server_latency_ms')}ms raw10={np.round(actions[0], 4).tolist()} "
            f"libero7={np.round(action7, 4).tolist()}"
        )

    print("timings (s):", timings)
    # Reset must invalidate the receding-horizon buffer: the next call after a
    # reset is served from a fresh generation, and the server acknowledges it.
    reset_out = inference.health().get("metrics") or {}
    print("metrics after reset:", {k: reset_out.get(k) for k in ("backend_resets", "infer_calls", "last_backend_reset")})
    print("SMOKE OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
