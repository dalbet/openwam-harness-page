"""Numpy-in-JSON codec for the engine's HTTP boundaries.

Same tagged shape as RPent's HTTP RPC (``__ndarray__`` base64), so an array
decoded here is byte-identical to the one the env server produced.
"""

from __future__ import annotations

import base64
from typing import Any

import numpy as np


def encode(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return {
            "__ndarray__": base64.b64encode(np.ascontiguousarray(value).tobytes()).decode("ascii"),
            "dtype": str(value.dtype),
            "shape": list(value.shape),
        }
    if isinstance(value, np.generic):
        return {"__npscalar__": value.item(), "dtype": str(value.dtype)}
    if isinstance(value, dict):
        return {str(key): encode(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode(item) for item in value]
    return value


def decode(value: Any) -> Any:
    if isinstance(value, dict):
        if "__ndarray__" in value and set(value) <= {"__ndarray__", "dtype", "shape"}:
            raw = base64.b64decode(value["__ndarray__"])
            return np.frombuffer(raw, dtype=value["dtype"]).reshape(value.get("shape", (-1,))).copy()
        if "__npscalar__" in value and set(value) <= {"__npscalar__", "dtype"}:
            return np.dtype(value["dtype"]).type(value["__npscalar__"])
        return {key: decode(item) for key, item in value.items()}
    if isinstance(value, list):
        return [decode(item) for item in value]
    return value
