"""Whole-task runner for the Pi0.5 backend — a sibling of ``WholeTaskRunner``.

The OpenWAM runner and this one share the *engine* half of the rollout and
differ only in the two model-specific seams:

    observation  ->  build_pi05_obs   (8-D proprio, 224px, pi05 orientation)
    action       ->  pi05_action7     (already 7-D OSC; clip only)
    generation   ->  10-action chunk, execute 5, then re-observe

Everything else — scene-epoch guard, stop handling, action budget, native-goal
and engine-predicate verification, post-goal window, measured gripper release,
per-action timing ledger — is inherited unchanged from ``WholeTaskRunner`` so a
success here means the same thing it means there.  Subclassing rather than
copying is deliberate: a copy would let the two paths drift and would make the
S0/S1 comparison a comparison of two different completion definitions.

**Buffer semantics (identical for S1 and S2).** A chunk is generated from the
observation at its first action and the next ``execute_horizon`` actions are
executed open-loop.  When the queue empties, a *new* observation is taken and a
*new* chunk is generated.  No action is ever reused across a regeneration, and
no request is answered from a buffer that predates the newest observation —
that is the state dependence a Pi0.5 policy actually has.  Unlike the OpenWAM
candidate-C path there is no server-side "skip the images on a cached step":
here a cached step never contacts the server at all, on either arm.
"""

from __future__ import annotations

import os
import time
from collections import deque
from typing import Any

import numpy as np

from adapters.pi05_contract import (
    LIBERO_ACTION7_DIM,
    PI05_EXECUTE_HORIZON,
    PI05_GENERATE_HORIZON,
    build_pi05_obs,
    pi05_action7,
)
from common import policy_seed
from engine.inference_client import request_row
from engine.openwam_rollout import WholeTaskRunner

# Opt-in A/B switch (default off = the shipped behaviour). ``env.step`` already
# returns the post-step cameras; the baseline then asks the env server for the
# same pixels a second time via ``raw_obs``. Reusing the step observation
# removes one RPC + one base64 round trip per generated chunk and changes
# nothing else (same pixels, same contract, same snapshot). Off by default so
# the delivered path stays the one that was accepted.
PI05_REUSE_STEP_OBS = os.environ.get("PI05_REUSE_STEP_OBS", "0") == "1"


class Pi05WholeTaskRunner(WholeTaskRunner):
    """One bounded whole-task Pi0.5 rollout inside an already-reset scene."""

    def __init__(self, *args, execute_horizon: int = PI05_EXECUTE_HORIZON,
                 generate_horizon: int = PI05_GENERATE_HORIZON,
                 policy_seed: int | None = None, **kwargs):
        super().__init__(*args, policy_seed=policy_seed, **kwargs)
        self.execute_horizon = max(1, min(int(execute_horizon), int(generate_horizon)))
        self.generate_horizon = int(generate_horizon)
        # ``None`` = the shipped path (noise keyed to the backend's chunk
        # counter). An int pins the sample: the noise of every request is a pure
        # function of (policy_seed, step_ref, sim_action_count), so two runs of
        # the same plan draw the same streams and a retry reproduces its sample.
        self.seed_rows: list[dict[str, Any]] = []
        self._queue: deque[np.ndarray] = deque()
        self._chunk_id = 0
        self._chunk_row: dict[str, Any] = {}
        self.generation_count = 0
        self.generated_actions = 0
        self.consumed_actions = 0
        self.discarded_actions = 0
        self.generation_rows: list[dict[str, Any]] = []
        self._obs_source = "fresh"
        self.contract_checked = False

    # ---- model-specific seams -------------------------------------------

    def _observe_model_input(self) -> dict[str, Any]:
        if self._queue:
            # No new observation is needed to dequeue an action that was
            # already generated from this chunk's observation.
            self._obs_source = "chunk_buffer"
            return {}
        started = time.perf_counter()
        raw = None
        reused = False
        if PI05_REUSE_STEP_OBS and isinstance(self.obs, dict):
            # The observation ``env.step`` just returned already carries both
            # cameras for this exact state, so asking the env server for the
            # same pixels again would be a second transfer of identical data.
            if "agentview_image" in self.obs and "robot0_eye_in_hand_image" in self.obs:
                raw = self.obs
                reused = True
        if raw is None:
            raw = self.env.raw_obs()
        obs = build_pi05_obs(raw)
        self.observe_s += time.perf_counter() - started
        self._obs_source = "step_obs" if reused else "fresh"
        self._obs_reused = bool(getattr(self, "_obs_reused", False) or reused)
        if not self.contract_checked:
            shapes = {
                "main_images": list(np.asarray(obs["main_images"]).shape),
                "wrist_images": list(np.asarray(obs["wrist_images"]).shape),
                "states": list(np.asarray(obs["states"]).shape),
            }
            if shapes["states"] != [8] or shapes["main_images"][:2] != [224, 224]:
                raise RuntimeError(f"pi05 observation contract violated: {shapes}")
            self.contract_checked = True
            self.journal("pi05_contract", {**shapes, **obs.get("contract", {})})
        return obs

    def _infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        self._check_stop()
        if self._queue:
            action = self._queue.popleft()
            self.consumed_actions += 1
            return {
                "actions": np.asarray([action], dtype=np.float32),
                "inference_s": 0.0,
                "inference_total_s": 0.0,
                "queued_s": 0.0,
                "encode_s": 0.0,
                "send_s": 0.0,
                "recv_s": 0.0,
                "server_latency_ms": None,
                "server_step": None,
                "chunk": {**self._chunk_row, "cache_hit": True, "chunk_generation": False,
                          "buffer_index": self.execute_horizon - len(self._queue) - 1,
                          "consumed_actions": self.consumed_actions},
                "action_raw": [float(v) for v in action],
                "obs_source": "chunk_buffer",
            }
        if not obs:
            raise RuntimeError("pi05 runner asked for an action with no observation and an empty queue")
        sim_actions = self.sim_actions
        noise_seed = None
        if self.policy_seed is not None:
            noise_seed = policy_seed.derive(
                self.policy_seed, step_ref=self.step_ref, sim_action_count=sim_actions
            )
        request = request_row(
            token=self.token,
            episode_id=self.episode_id,
            scene_epoch=self.snapshot.get("scene_epoch"),
            snapshot=self.snapshot,
            skill="whole_task",
            step_id=sim_actions,
            prompt=self.prompt,
            step_ref=self.step_ref,
            noise_seed=noise_seed,
            policy_seed=self.policy_seed,
        )
        # What the policy is about to see, before any model time is spent. Two
        # runs can agree on the scene and still feed different pixels; hashing
        # here is what tells those two cases apart in the record instead of
        # blaming the sampler (cost: ~0.4 ms per chunk, no extra RPC).
        obs_hashes = {
            "main_sha256": policy_seed.digest(obs.get("main_images")),
            "wrist_sha256": policy_seed.digest(obs.get("wrist_images")),
            "state_sha256": policy_seed.digest(obs.get("states")),
            "state0": [round(float(v), 6) for v in np.asarray(obs.get("states")).reshape(-1)[:3]],
        }
        out = self.inference.infer(request, obs, timeout_s=self.timeout_s)
        if out.get("status") != "ok":
            raise RuntimeError(
                f"Pi0.5 inference failed ({out.get('status')}): {out.get('error') or out.get('reason')}"
            )
        actions = np.asarray(out["actions"], dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] < LIBERO_ACTION7_DIM:
            raise RuntimeError(
                f"Pi0.5 returned actions with shape {actions.shape}, expected [n,>=7] LIBERO OSC"
            )
        window = actions[: self.execute_horizon]
        self._queue.extend(np.asarray(row, dtype=np.float32) for row in window[1:])
        self._chunk_id += 1
        self.generation_count += 1
        self.generated_actions += int(actions.shape[0])
        self.consumed_actions += 1
        self._chunk_row = dict(out.get("chunk") or {})
        chunk_id = out.get("chunk_id") or getattr(self.inference, "_generation_index", None)
        self._chunk_row.setdefault("chunk_id", chunk_id)
        self.generation_rows.append(
            {
                "generation": self.generation_count,
                "at_sim_action": self.sim_actions,
                "client_inference_s": out.get("inference_s"),
                "engine_total_s": out.get("total_s"),
                "server_latency_ms": out.get("server_latency_ms"),
                "queued_s": out.get("queued_s"),
                "encode_s": out.get("encode_s"),
                "send_s": out.get("send_s"),
                "generated_actions": int(actions.shape[0]),
                "executed_actions": int(window.shape[0]),
                "mode": out.get("mode"),
                "precision": out.get("precision"),
                "noise_seed": out.get("noise_seed"),
                "noise_source": out.get("noise_source"),
                "noise_sha256": out.get("noise_sha256"),
                "noise_echo": out.get("noise_echo"),
                # Whether the inference service reports the seed as applied by
                # the backend. ``None`` on the legacy path (no seed was sent);
                # ``False`` would mean the backend ignored it, which the record
                # must not hide.
                "seed_applied": out.get("seed_applied"),
                "seed_unsupported": out.get("seed_unsupported"),
                "obs": obs_hashes,
                "action0": [round(float(v), 6) for v in window[0][:7]],
                "action0_sha256": policy_seed.digest(window[0][:7]),
            }
        )
        self.inference_s += float(out.get("inference_s") or 0.0)
        return {
            "actions": window,
            "inference_s": out.get("inference_s"),
            "inference_total_s": out.get("total_s"),
            "queued_s": out.get("queued_s"),
            "encode_s": out.get("encode_s"),
            "send_s": out.get("send_s"),
            "recv_s": out.get("recv_s"),
            "server_latency_ms": out.get("server_latency_ms"),
            "server_step": out.get("server_step"),
            "chunk": {**self._chunk_row, "cache_hit": False, "chunk_generation": True,
                      "chunk_id": self._chunk_id,
                      "execute_horizon": self.execute_horizon,
                      "generated_actions": int(actions.shape[0]),
                      "buffer_index": 0,
                      "consumed_actions": self.consumed_actions},
            "action_raw": [float(v) for v in window[0]],
            "obs_source": "fresh",
        }

    def _to_action7(self, raw_action) -> np.ndarray:
        return pi05_action7(raw_action)

    def _raw_gripper(self, raw_action) -> tuple[str, float | None]:
        """Pi0.5's native action is already the 7-D OSC command (grip in slot 6)."""
        value = np.asarray(raw_action, dtype=np.float64).reshape(-1)
        return "grip_cmd_raw", (float(value[6]) if value.shape[0] > 6 else None)

    # ---- record ---------------------------------------------------------

    def run(self) -> dict[str, Any]:
        result = super().run()
        self.discarded_actions = len(self._queue)
        gens = self.generation_rows
        lat = [float(r["client_inference_s"]) for r in gens if isinstance(r.get("client_inference_s"), (int, float))]
        srv = [float(r["server_latency_ms"]) for r in gens if isinstance(r.get("server_latency_ms"), (int, float))]
        result["pi05"] = {
            "contract": {
                "proprio": "libero8d",
                "images": ["agentview", "eye_in_hand"],
                "image_size": [224, 224],
                "rotate_180": True,
                "action_space": "libero7d OSC (clip only)",
                "generate_horizon": self.generate_horizon,
                "execute_horizon": self.execute_horizon,
                "denoise_steps": None,
                "obs_source_policy": (
                    "step_obs" if getattr(self, "_obs_reused", False) else "raw_obs"
                ),
                "step_obs_reuse_enabled": bool(PI05_REUSE_STEP_OBS),
            },
            "buffer": {
                "generations": self.generation_count,
                "generated_actions_total": self.generated_actions,
                "executed_actions_total": self.consumed_actions,
                "discarded_at_end": self.discarded_actions,
                "policy": "re-observe before every generation; no action crosses a generation",
            },
            "generation": {
                "n": len(gens),
                "client_inference_s": self._summary(lat),
                "server_latency_ms": self._summary(srv),
                "rows": gens,
            },
            "sampling": {
                "policy_seed": self.policy_seed,
                "seed_active": self.policy_seed is not None,
                "seed_key": "policy_seed + step_ref + sim_action_count"
                if self.policy_seed is not None
                else "none: backend per-(prompt, chunk-index) legacy stream",
                "noise_injected": "explicit array in every request (server echoes noise_used)",
                "accelerators_changed": False,
                "obs_hashes": [
                    {"at_sim_action": row.get("at_sim_action"), **(row.get("obs") or {})} for row in gens
                ],
            },
        }
        result["release"]["source"] = "measured_gripper_separation"
        result["release"]["note"] = (
            "release measured from finger separation on the environment side; "
            "the Pi0.5 backend has no executor receipt"
        )
        result["action_contract"] = "pi05_libero7d"
        return result


__all__ = ["Pi05WholeTaskRunner"]
