"""Explainable "is anything happening" metric for one rollout.

The only progress signal the stack used to have was an action budget: a policy
that never touches the object still burned all 600 actions, which is how one
episode cost 519.9 s. This module turns the recorded observations into windows
that can be judged without another model call:

* ``object_motion_m`` - how far the task object moved inside the window,
* ``eef_motion_m`` - how far the hand moved inside the same window.

A window where the hand keeps moving but the object does not is the signature
of a miss: approach, close, retreat, repeat. A window where neither moves is
the signature of a stuck arm. Both are reported with the action range so the
claim is checkable against the trajectory.

Deliberately *not* an abort rule: the first half of a drawer-then-place episode
opens the drawer while the bowl stays still, so a blind "no object motion"
abort would cut successful episodes. :func:`stall_windows` only reports; the
caller decides, and the decision has to be justified against these numbers.
"""

from __future__ import annotations

import math
from typing import Any

DEFAULT_WINDOW_ACTIONS = 120
DEFAULT_OBJECT_MOTION_M = 0.005
DEFAULT_EEF_MOTION_M = 0.02


def _pos(row: dict[str, Any], key: str) -> list[float] | None:
    value = (row or {}).get(key)
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None
    try:
        return [float(value[0]), float(value[1]), float(value[2])]
    except (TypeError, ValueError):
        return None


def _dist(a: list[float] | None, b: list[float] | None) -> float | None:
    if a is None or b is None:
        return None
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


def object_total_motion_m(observations: list[dict[str, Any]]) -> float | None:
    """Largest displacement of the object from its first recorded position."""
    first = _pos(observations[0], "object_pose") if observations else None
    if first is None:
        return None
    best = 0.0
    for row in observations:
        distance = _dist(first, _pos(row, "object_pose"))
        if distance is not None:
            best = max(best, distance)
    return round(best, 5)


def first_object_motion_index(observations: list[dict[str, Any]], threshold_m: float) -> int | None:
    """Index of the first observation where the object has moved off its start."""
    if not observations:
        return None
    first = _pos(observations[0], "object_pose")
    for index, row in enumerate(observations):
        distance = _dist(first, _pos(row, "object_pose"))
        if distance is not None and distance >= threshold_m:
            return index
    return None


def _opening_range(observations: list[dict[str, Any]], start: int, end: int) -> tuple[float | None, float | None]:
    values = [
        float(row["gripper_opening_m"])
        for row in observations[start : end + 1]
        if isinstance(row.get("gripper_opening_m"), (int, float))
    ]
    if not values:
        return None, None
    return round(min(values), 5), round(max(values), 5)


def stall_windows(
    observations: list[dict[str, Any]],
    *,
    window: int = DEFAULT_WINDOW_ACTIONS,
    object_motion_m: float = DEFAULT_OBJECT_MOTION_M,
    eef_motion_m: float = DEFAULT_EEF_MOTION_M,
) -> list[dict[str, Any]]:
    """Windows where the hand moved but the object did not.

    Windows are non-overlapping and anchored at the first observation, so the
    same trajectory always yields the same list. ``touched`` separates the two
    meanings of such a window: before the object has ever moved it is an
    approach or a drawer-first phase, after it has moved it is a miss.
    """
    if window <= 0 or len(observations) <= window:
        return []
    start_motion = first_object_motion_index(observations, object_motion_m)
    stalls: list[dict[str, Any]] = []
    start = 0
    while start + window < len(observations):
        head = _pos(observations[start], "object_pose")
        tail = _pos(observations[start + window], "object_pose")
        head_eef = _pos(observations[start], "eef_pos")
        tail_eef = _pos(observations[start + window], "eef_pos")
        obj = _dist(head, tail)
        eef = _dist(head_eef, tail_eef)
        if obj is not None and eef is not None and obj < object_motion_m and eef >= eef_motion_m:
            opening_min, opening_max = _opening_range(observations, start, start + window)
            stalls.append(
                {
                    "n_start": start,
                    "n_end": start + window,
                    "object_motion_m": round(obj, 5),
                    "eef_motion_m": round(eef, 5),
                    "touched": bool(start_motion is not None and start_motion <= start),
                    "gripper_open_min_m": opening_min,
                    "gripper_open_max_m": opening_max,
                }
            )
        start += window
    return stalls


def summarize(
    observations: list[dict[str, Any]],
    *,
    window: int = DEFAULT_WINDOW_ACTIONS,
    object_motion_m: float = DEFAULT_OBJECT_MOTION_M,
) -> dict[str, Any]:
    if not observations:
        # No observations means no evidence; callers must render this as unknown.
        return {
            "schema": "embodied_engine.progress.v1",
            "observations": 0,
            "window_actions": window,
            "object_motion_threshold_m": object_motion_m,
            "object_total_motion_m": None,
            "first_object_motion_n": None,
            "never_touched": None,
            "stall_windows": [],
            "stall_actions": 0,
            "stall_actions_after_contact": 0,
            "note": "no observations",
        }
    stalls = stall_windows(observations, window=window, object_motion_m=object_motion_m)
    total = object_total_motion_m(observations)
    first_motion = first_object_motion_index(observations, object_motion_m)
    return {
        "schema": "embodied_engine.progress.v1",
        "observations": len(observations),
        "window_actions": window,
        "object_motion_threshold_m": object_motion_m,
        "object_total_motion_m": total,
        "first_object_motion_n": first_motion,
        "never_touched": first_motion is None,
        "stall_windows": stalls,
        "stall_actions": sum(item["n_end"] - item["n_start"] for item in stalls),
        "stall_actions_after_contact": sum(
            item["n_end"] - item["n_start"] for item in stalls if item["touched"]
        ),
        "note": "reported, not enforced: the drawer-first half of a composite episode moves the hand without the object by design",
    }


__all__ = [
    "DEFAULT_EEF_MOTION_M",
    "DEFAULT_OBJECT_MOTION_M",
    "DEFAULT_WINDOW_ACTIONS",
    "first_object_motion_index",
    "object_total_motion_m",
    "stall_windows",
    "summarize",
]
