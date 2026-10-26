"""Quaternion helper used by the LIBERO snapshot."""

from __future__ import annotations

import math


def mat_to_quat_wxyz(mat: list[float]) -> list[float]:
    m = [float(v) for v in mat[:9]]
    trace = m[0] + m[4] + m[8]
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2
        return [0.25 * s, (m[7] - m[5]) / s, (m[2] - m[6]) / s, (m[3] - m[1]) / s]
    if m[0] > m[4] and m[0] > m[8]:
        s = math.sqrt(1.0 + m[0] - m[4] - m[8]) * 2
        return [(m[7] - m[5]) / s, 0.25 * s, (m[1] + m[3]) / s, (m[2] + m[6]) / s]
    if m[4] > m[8]:
        s = math.sqrt(1.0 + m[4] - m[0] - m[8]) * 2
        return [(m[2] - m[6]) / s, (m[1] + m[3]) / s, 0.25 * s, (m[5] + m[7]) / s]
    s = math.sqrt(1.0 + m[8] - m[0] - m[4]) * 2
    return [(m[3] - m[1]) / s, (m[2] + m[6]) / s, (m[5] + m[7]) / s, 0.25 * s]
