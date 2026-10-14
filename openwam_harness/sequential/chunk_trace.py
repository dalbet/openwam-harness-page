"""Record each Pi0.5 chunk without changing the checkpoint, normalization, or sampling."""

from __future__ import annotations

import numpy as np


def attach_chunk_trace(primitives) -> None:
    if getattr(primitives, "_trace_attached", False):
        return
    primitives._trace_attached = True
    primitives._chunk_trace = []
    primitives._env_action_count = 0
    env = primitives.env
    original = env.chunk_step

    def chunk_step(actions, *args, **kwargs):
        before = int(primitives._env_action_count)
        try:
            result = original(actions, *args, **kwargs)
        except Exception as exc:
            if "cancel" in str(exc).lower():
                primitives._chunk_trace.append({"cancelled": True, "error": str(exc), "env_actions_before": before})
            raise
        array = np.asarray(actions)
        if array.ndim <= 1:
            shape = [1, int(array.shape[-1])] if array.size else [0]
            sent = 1 if array.size else 0
        else:
            shape = [int(value) for value in array.shape]
            sent = int(shape[0])
        terminated = result[2] if isinstance(result, tuple) and len(result) > 2 else None
        truncated = result[3] if isinstance(result, tuple) and len(result) > 3 else None
        term_flags = _flags(terminated)
        applied = len(term_flags) if term_flags is not None else sent
        primitives._env_action_count = before + applied
        primitives._chunk_trace.append({
            "vla_shape": shape,
            "actions_sent": sent,
            "env_actions_before": before,
            "env_actions_after": primitives._env_action_count,
            "terminated": term_flags,
            "truncated": _flags(truncated),
            "cancelled": False,
            "observation_id": f"env-actions-{before}:{primitives._env_action_count}",
        })
        return result

    env.chunk_step = chunk_step


def take_chunk_trace(primitives) -> list[dict]:
    rows = list(getattr(primitives, "_chunk_trace", []))
    primitives._chunk_trace = []
    return rows


def _flags(value):
    if value is None:
        return None
    try:
        flat = np.asarray(value).reshape(-1)
    except Exception:
        return [bool(value)]
    return [bool(item) for item in flat.tolist()]
