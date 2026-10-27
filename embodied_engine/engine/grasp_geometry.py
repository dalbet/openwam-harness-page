"""Measured gripper/object geometry from a LIBERO state snapshot.

Finger separation alone is one fact. Whether the object is between the pads is
a second, independent fact, and whether the object follows the arm is a third.
No single one of them proves "the object is still held" or "the object was
released"; the release contract in :mod:`engine.finalize` combines them and
reports every raw number so the claim stays auditable.

The snapshot already carries the MuJoCo body positions
(``named_poses.bodies``), so nothing here steps the simulator.
"""

from __future__ import annotations

import math
from typing import Any

PAD_BODIES = ("gripper0_right_leftfinger", "gripper0_right_rightfinger")
GRIP_SITE = "gripper0_right_grip_site"


def _xyz(value: Any) -> list[float] | None:
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        try:
            return [float(value[0]), float(value[1]), float(value[2])]
        except (TypeError, ValueError):
            return None
    return None


def _sub(a: list[float], b: list[float]) -> list[float]:
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]


def _norm(a: list[float]) -> float:
    return math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])


def _dot(a: list[float], b: list[float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def object_pos(snapshot: dict[str, Any], object_name: str | None) -> list[float] | None:
    if not object_name:
        return None
    row = ((snapshot.get("sim") or {}).get("objects") or {}).get(object_name)
    if not isinstance(row, dict):
        return None
    return _xyz(row.get("pos"))


def eef_pos(snapshot: dict[str, Any]) -> list[float] | None:
    return _xyz((snapshot.get("sim") or {}).get("eef_pos"))


def finger_frame(snapshot: dict[str, Any], object_name: str | None = None) -> dict[str, Any]:
    """Pad positions plus the object's position in the pad frame.

    ``gap_m`` is the perpendicular distance from the object centre to the line
    through the two finger bodies; ``along_m`` is the projection along that
    line. A held object sits close to the pad line with ``|along_m|`` inside
    the finger span. A released object sits below/away from it.
    """
    bodies = ((snapshot.get("named_poses") or {}).get("bodies") or {})
    left = _xyz(bodies.get(PAD_BODIES[0]))
    right = _xyz(bodies.get(PAD_BODIES[1]))
    grip = _xyz((((snapshot.get("named_poses") or {}).get("sites")) or {}).get(GRIP_SITE))
    out: dict[str, Any] = {
        "left_pad": left,
        "right_pad": right,
        "grip_site": grip,
        "separation_m": None,
        "object_gap_m": None,
        "object_along_m": None,
        "pad_mid": None,
    }
    if left is not None and right is not None:
        axis = _sub(right, left)
        separation = _norm(axis)
        out["separation_m"] = round(separation, 5)
        out["pad_mid"] = [(left[i] + right[i]) / 2.0 for i in range(3)]
        pos = object_pos(snapshot, object_name)
        if pos is not None and separation > 1e-6:
            unit = [value / separation for value in axis]
            delta = _sub(pos, out["pad_mid"])
            along = _dot(delta, unit)
            perp = _norm([delta[i] - along * unit[i] for i in range(3)])
            out["object_gap_m"] = round(perp, 5)
            out["object_along_m"] = round(along, 5)
    return out


def eef_object_m(snapshot: dict[str, Any], object_name: str | None) -> float | None:
    pos = object_pos(snapshot, object_name)
    eef = eef_pos(snapshot)
    if pos is None or eef is None:
        return None
    return round(_norm(_sub(pos, eef)), 5)


def opening_m(snapshot: dict[str, Any]) -> float | None:
    value = (snapshot.get("sim") or {}).get("gripper_opening_m")
    return float(value) if isinstance(value, (int, float)) else None


def relation_facts(
    snapshot: dict[str, Any], step: dict[str, Any], task: dict[str, Any]
) -> dict[str, Any]:
    """Relation/height state for a pick/place-style step, without touching sim."""
    from engine import completion

    return completion.goal_state(snapshot, step, task)


__all__ = [
    "PAD_BODIES",
    "GRIP_SITE",
    "eef_object_m",
    "eef_pos",
    "finger_frame",
    "object_pos",
    "opening_m",
    "relation_facts",
]
