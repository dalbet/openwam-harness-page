#!/usr/bin/env python
"""Is a policy request reproducible when the sampling seed is pinned?

The 2026-11-05 batches agreed on ``initial_state_fingerprint`` and still emitted
four different first actions, because the diffusion noise was drawn from the
server's ambient ``torch.Generator`` per request. ``policy_seed`` now travels
with the request: the engine derives one integer per logical request from
``(policy_seed, step_ref, sim_action_count)`` and the inference service passes
it to the Pi0.5 backend, which builds the noise itself instead of drawing it.

This probe measures that, offline (no simulator step, no round):

``same_seed_repeat``      same seed, same observation, buffer cleared between
                          requests -> the two action chunks must be identical.
``interleaved``           the same seed again, after a different-seed request
                          was served in between -> still identical, so another
                          request cannot shift this one's stream.
``buffer_clear``          the buffer clear is the reset itself; the rows record
                          the token so a stale chunk cannot be mistaken for a
                          new sample.
``different_seed``        a different seed -> different actions, i.e. the seed
                          is actually read (a seed that changes nothing is a
                          no-op, not a feature).
``different_step``        same seed, different ``sim_action_count`` -> different
                          actions, because the stream is keyed by the logical
                          step; it is never reset to one sequence per chunk.
``legacy_no_seed``        no seed at all -> the shipped behaviour, recorded so
                          the two protocols are not confused.

Nothing here claims bit-exactness across software or hardware versions; it
compares requests served by the same resident server in the same process.

    EMBODIED_CONFIG_DIR=... python scripts/policy_action_seed_probe.py \
        --out records/acceptance/policy_action_seed_probe.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from adapters.env_client import EnvClient  # noqa: E402
from adapters.pi05_contract import build_pi05_obs  # noqa: E402
from common import config  # noqa: E402
from common import policy_seed  # noqa: E402
from common.policy_seed import derive  # noqa: E402
from engine.inference_client import InferenceClient, model_view, request_row  # noqa: E402

OUT = ROOT / "records" / "acceptance" / "policy_action_seed_probe.json"


def _sha(array) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(array, dtype=np.float32)).tobytes()).hexdigest()


def _rows(action_a, action_b) -> dict:
    a = np.asarray(action_a, dtype=np.float32)
    b = np.asarray(action_b, dtype=np.float32)
    if a.shape != b.shape:
        return {"shape_a": list(a.shape), "shape_b": list(b.shape), "comparable": False}
    delta = np.abs(a - b)
    per_dim = [round(float(v), 6) for v in delta.reshape(-1, delta.shape[-1]).max(axis=0)]
    return {
        "comparable": True,
        "identical": bool(np.array_equal(a, b)),
        "max_abs_diff": round(float(delta.max()), 8),
        "mean_abs_diff": round(float(delta.mean()), 8),
        "differing_elements": int((delta > 0).sum()),
        "elements": int(delta.size),
        "per_dim_max_abs_diff": per_dim,
        "a_first_action": [round(float(v), 6) for v in a[0]],
        "b_first_action": [round(float(v), 6) for v in b[0]],
    }


def _call(
    client: InferenceClient,
    *,
    obs,
    prompt: str,
    seed: int | None,
    sim_action_count: int,
    label: str,
    token: str | None = None,
) -> dict:
    token = token or client.reset(f"probe:{label}")
    row = request_row(
        token=token,
        episode_id=f"probe:{label}",
        scene_epoch=0,
        snapshot={"observation_id": f"probe:{label}", "sim_action_count": sim_action_count},
        skill="whole_task",
        step_id=0,
        prompt=prompt,
        step_ref="probe",
        noise_seed=seed,
        policy_seed=seed,
    )
    started = time.perf_counter()
    out = client.infer(row, obs, timeout_s=180.0)
    elapsed = round(time.perf_counter() - started, 4)
    actions = out.get("actions")
    if out.get("status") != "ok" or actions is None:
        return {"label": label, "seed": seed, "status": out.get("status"), "error": out.get("error"), "s": elapsed}
    return {
        "label": label,
        "seed": seed,
        "sim_action_count": sim_action_count,
        "status": "ok",
        "token": token,
        "actions_sha256": _sha(actions),
        "actions_shape": list(np.asarray(actions).shape),
        "first_action": [round(float(v), 6) for v in np.asarray(actions)[0]],
        "noise_seed": out.get("noise_seed"),
        "noise_source": out.get("noise_source"),
        "noise_sha256": out.get("noise_sha256"),
        # The generation server's own echo of the first four entries of the
        # array it consumed; compared below against what we sent.
        "noise_echo": out.get("noise_echo"),
        "seed_applied": out.get("seed_applied"),
        "chunk": out.get("chunk"),
        "s": elapsed,
        "_actions": actions,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", default=None, help="defaults to services.engine in the active config")
    parser.add_argument("--env", default=None, help="defaults to services.env in the active config")
    parser.add_argument("--inference", default=None, help="defaults to services.inference in the active config")
    parser.add_argument("--task", type=int, default=7)
    parser.add_argument("--scene-seed", type=int, default=3)
    parser.add_argument("--policy-seed", type=int, default=101)
    parser.add_argument("--other-seed", type=int, default=202)
    parser.add_argument("--prompt", default="turn on the stove")
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()

    env_url = args.env or config.service_url("env")
    inference_url = args.inference or config.service_url("inference")
    client = InferenceClient(inference_url)
    health = client.health()
    print(f"inference health: {health}", flush=True)

    obs = model_view(build_pi05_obs(EnvClient(env_url).raw_obs()))
    prompt = args.prompt
    seed = int(args.policy_seed)
    other = int(args.other_seed)

    # The derived per-request seeds are what the engine would send. Keeping the
    # derivation here (rather than hard-coding 101 twice) means the probe fails
    # if the mixer stops being a pure function of its inputs.
    step_seed = derive(seed, step_ref="probe", sim_action_count=0)
    other_step_seed = derive(other, step_ref="probe", sim_action_count=0)
    next_step_seed = derive(seed, step_ref="probe", sim_action_count=5)
    print(
        f"policy_seed={seed} step_seed={step_seed} other={other_step_seed} next_step={next_step_seed}",
        flush=True,
    )

    first = _call(client, obs=obs, prompt=prompt, seed=step_seed, sim_action_count=0, label="a_same_seed")
    repeat = _call(client, obs=obs, prompt=prompt, seed=step_seed, sim_action_count=0, label="b_same_seed_repeat")
    different = _call(client, obs=obs, prompt=prompt, seed=other_step_seed, sim_action_count=0, label="c_other_seed")
    interleaved = _call(client, obs=obs, prompt=prompt, seed=step_seed, sim_action_count=0, label="d_interleaved")
    next_step = _call(client, obs=obs, prompt=prompt, seed=next_step_seed, sim_action_count=5, label="e_next_step")
    legacy_a = _call(client, obs=obs, prompt=prompt, seed=None, sim_action_count=0, label="f_legacy")
    legacy_b = _call(client, obs=obs, prompt=prompt, seed=None, sim_action_count=0, label="g_legacy_repeat")
    # A cleared episode's token must not be able to ask for actions in the new
    # one: the reset above rotated it, so the old string is refused.
    stale = _call(
        client,
        obs=obs,
        prompt=prompt,
        seed=step_seed,
        sim_action_count=0,
        label="h_stale_token",
        token=first.get("token"),
    )

    comparisons = {
        "same_seed_repeat": _rows(first.get("_actions"), repeat.get("_actions")),
        "interleaved": _rows(first.get("_actions"), interleaved.get("_actions")),
        "different_seed_contrast": _rows(first.get("_actions"), different.get("_actions")),
        "different_step_contrast": _rows(first.get("_actions"), next_step.get("_actions")),
        "legacy_repeat": _rows(legacy_a.get("_actions"), legacy_b.get("_actions")),
    }
    checks = {
        "same_seed_repeat_identical": bool(comparisons["same_seed_repeat"].get("identical")),
        "interleaved_identical": bool(comparisons["interleaved"].get("identical")),
        "different_seed_differs": not bool(comparisons["different_seed_contrast"].get("identical", True)),
        "different_step_differs": not bool(comparisons["different_step_contrast"].get("identical", True)),
        "noise_sha_matches_on_repeat": first.get("noise_sha256") == repeat.get("noise_sha256"),
        "seed_applied": all(row.get("seed_applied") for row in (first, repeat, interleaved)),
        "legacy_repeat_identical": bool(comparisons["legacy_repeat"].get("identical")),
        "stale_token_refused": stale.get("status") == "conflict",
        "noise_echo_matches_sent": bool(first.get("noise_echo"))
        and np.allclose(
            np.asarray(first["noise_echo"], dtype=np.float32),
            policy_seed.noise_for(step_seed, horizon=10, dim=32)[0][:4],
            atol=1e-6,
        ),
    }

    rows = [first, repeat, different, interleaved, next_step, legacy_a, legacy_b, stale]
    for row in rows:
        row.pop("_actions", None)
    payload = {
        "schema": "embodied_engine.policy_action_seed_probe.v1",
        "inference": inference_url,
        "env": env_url,
        "task_id": int(args.task),
        "scene_seed": int(args.scene_seed),
        "policy_seed": seed,
        "other_policy_seed": other,
        "prompt": prompt,
        "observation_sha256": _sha(obs["main_images"]),
        "noise_keys": {
            "step_seed": step_seed,
            "other_step_seed": other_step_seed,
            "next_step_seed": next_step_seed,
        },
        "checks": checks,
        "comparisons": comparisons,
        "rows": rows,
        "note": (
            "same resident inference service, same observation, no simulator step; "
            "'identical' is exact float equality of the returned action chunk"
        ),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")

    for name, ok in checks.items():
        print(f"{name}: {'OK' if ok else 'FAIL'}", flush=True)
    print(
        f"same_seed max|d|={comparisons['same_seed_repeat'].get('max_abs_diff')} "
        f"interleaved max|d|={comparisons['interleaved'].get('max_abs_diff')} "
        f"other_seed max|d|={comparisons['different_seed_contrast'].get('max_abs_diff')}",
        flush=True,
    )
    print(f"written: {out}", flush=True)
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
