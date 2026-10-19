"""One sim-assisted point for a failed text segment.

The pixel is the visible surface nearest the bound instance. A low text score
is not lowered, and another object's surface is not reused.
"""

from __future__ import annotations

import math
from typing import Any

MAX_SURFACE_M = 0.03


def pixel_for_instance(
    world: Any,
    target: list[float],
    others: list[list[float]] | None = None,
    *,
    max_distance_m: float = MAX_SURFACE_M,
) -> dict[str, Any] | None:
    """Return row/col when the visible surface is this instance."""
    height = len(world)
    if height == 0:
        return None
    width = len(world[0])
    best = None
    best_d = None
    for row in range(height):
        line = world[row]
        for col in range(width):
            point = line[col]
            if point is None or len(point) < 3:
                continue
            if not all(math.isfinite(float(point[i])) for i in range(3)):
                continue
            distance = _dist(point, target)
            if best_d is None or distance < best_d:
                best = (row, col, [float(point[i]) for i in range(3)])
                best_d = distance
    if best is None or best_d is None or best_d > max_distance_m:
        return None
    row, col, surface = best
    for other in others or []:
        if _dist(surface, other) + 0.01 < best_d:
            return None
    return {
        "ok": True,
        "row": int(row),
        "col": int(col),
        "distance_m": round(best_d, 4),
        "image_hw": [height, width],
        "evidence_source": "sim_state",
        "vision": False,
    }


def _dist(left, right) -> float:
    return math.dist([float(left[i]) for i in range(3)], [float(right[i]) for i in range(3)])
