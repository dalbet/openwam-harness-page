"""Per-request sampling seed for the action policy — the reproducible-baseline knob.

Why this exists
---------------
The 2026-11-05 round ran the *same* frozen manifest four times and got 2/6, 5/6,
5/6 and 4/6. The scene side was bit-identical (`initial_state_fingerprint`
matched across all four batches) but the first step alone took 81/86/93/94
actions, so the spread was in the action stream, not in the initial state. A
number like that is unusable as a baseline until the sample is pinned.

What is pinned, and what is not
-------------------------------
Only the *initial diffusion noise* of each request is pinned. Everything else
stays exactly as shipped: same weights, same precision, same CUDA path, same
``num_steps``, same ``execute_horizon``. The Euler ODE sampler is deterministic
given its noise (verified in ``tools/policy_seed_probe.py``: identical noise →
bit-wise identical chunks, different noise → different chunks), so pinning the
noise pins the sample.

Seed provenance is a *pure integer function* of stable coordinates:

    (policy_seed, step_ref, sim_action_count_at_observation)

* ``policy_seed`` — the experiment's number, never the scene/init seed.
* ``step_ref`` — the composite step id ("c1"), so step 1 never re-uses step 3's
  noise even when both are observed at overlapping action counts.
* ``sim_action_count`` — the observation's own position in the episode, so each
  chunk gets a *different* stream (the requirement "do not reset to the same
  sequence for every chunk") while a re-send of the same logical request (same
  observation, same counter) reproduces the *same* stream. A retry therefore
  cannot silently turn into a second sample.

No ``hash()`` (salted per process by PYTHONHASHSEED), no episode UUID, no wall
clock: two processes on two machines derive the same seed for the same
coordinates. The mixing is splitmix64, integer-only.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

MASK64 = (1 << 64) - 1
GOLDEN = 0x9E3779B97F4A7C15
SEED_MAX = 0x7FFFFFFF


def _mix64(value: int) -> int:
    x = (int(value) + GOLDEN) & MASK64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & MASK64
    return (x ^ (x >> 31)) & MASK64


def _mix_bytes(acc: int, blob: bytes) -> int:
    for byte in blob:
        acc = _mix64(acc ^ int(byte))
    return acc


def derive(
    policy_seed: int,
    *,
    step_ref: str | None,
    sim_action_count: int | None,
    salt: str | None = None,
) -> int:
    """Stable 31-bit seed for one logical request.

    ``salt`` is only for experiments that must draw a *second* stream from the
    same observation (e.g. the probe's "different seed" control). The execution
    path never passes it, so a retry of the same logical request is the same
    seed by construction.
    """
    acc = _mix64(int(policy_seed) & MASK64)
    acc = _mix_bytes(acc, str(step_ref or "").encode("utf-8"))
    acc = _mix64(acc ^ (int(sim_action_count if sim_action_count is not None else -1) + 1))
    if salt:
        acc = _mix_bytes(acc, str(salt).encode("utf-8"))
    return int(acc & SEED_MAX)


def noise_for(seed: int, *, horizon: int, dim: int) -> np.ndarray:
    """The exact array the policy server consumes, from ``seed`` alone.

    ``np.random.default_rng(int)`` seeds a PCG64 stream; ``standard_normal``
    (ziggurat) is documented to be reproducible across platforms and numpy
    releases, which is what makes the recorded ``noise_sha256`` comparable.
    """
    rng = np.random.default_rng(int(seed) & MASK64)
    return np.ascontiguousarray(rng.standard_normal((int(horizon), int(dim))), dtype=np.float32)


def digest(array: Any) -> str:
    """SHA256 of an array's canonical bytes — the value the record keeps."""
    arr = np.ascontiguousarray(np.asarray(array))
    return hashlib.sha256(arr.tobytes()).hexdigest()


def resolve(cfg: dict[str, Any] | None, cli_value: int | None = None) -> tuple[int | None, str]:
    """Which policy seed this process uses, and where it came from.

    Precedence: explicit request value > ``EMBODIED_POLICY_SEED`` >
    ``action_model.policy_seed`` in the engine config > none (legacy behaviour,
    the ambient per-(prompt, chunk-index) noise the shipped path always used).
    """
    if cli_value is not None:
        return int(cli_value), "request"
    import os

    env = os.environ.get("EMBODIED_POLICY_SEED")
    if env not in (None, ""):
        return int(env), "env:EMBODIED_POLICY_SEED"
    action = (cfg or {}).get("action_model") or {}
    value = action.get("policy_seed")
    if value is not None:
        return int(value), "config:action_model.policy_seed"
    return None, "unset"


__all__ = ["derive", "digest", "noise_for", "resolve", "SEED_MAX"]
