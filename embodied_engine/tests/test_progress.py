"""The stall metric must inform, not decide.

The regression it protects is concrete: a rule "no object motion in 120 actions
=> abort" would have killed task 3 rounds 7 and 8, which reached native goal at
observation 122 and 121. These tests pin that the metric reports the window and
leaves the verdict to the caller.
"""

from __future__ import annotations

from typing import Callable

from engine import progress


def _obs(n: int, object_at: Callable[[int], tuple], eef_at: Callable[[int], tuple], opening: float = 0.08) -> list[dict]:
    return [
        {
            "object_pose": list(object_at(i)),
            "eef_pos": list(eef_at(i)),
            "gripper_opening_m": opening,
        }
        for i in range(n)
    ]


def test_never_touched_object_is_reported_without_stall_verdict():
    rows = _obs(600, lambda i: (0.0, 0.0, 0.0), lambda i: (0.001 * i, 0.0, 0.0))
    summary = progress.summarize(rows)
    assert summary["never_touched"] is True
    assert summary["first_object_motion_n"] is None
    assert summary["object_total_motion_m"] == 0.0
    assert len(summary["stall_windows"]) == 4
    assert all(window["touched"] is False for window in summary["stall_windows"])
    assert summary["stall_actions_after_contact"] == 0


def test_window_after_contact_is_marked_touched():
    rows = _obs(300, lambda i: (0.0, 0.0, 0.0) if i < 5 else (0.1, 0.0, 0.0), lambda i: (0.001 * i, 0.0, 0.0))
    summary = progress.summarize(rows, window=20)
    assert summary["first_object_motion_n"] == 5
    assert summary["stall_windows"], "hand moving, object still, after contact"
    assert summary["stall_windows"][0]["touched"] is True
    assert summary["stall_actions_after_contact"] > 0


def test_successful_composite_would_be_aborted_by_a_blind_rule():
    """task 3 rounds 7/8: object starts moving at n=122, after one quiet window."""
    rows = _obs(184, lambda i: (0.0, 0.0, 0.0) if i < 122 else (0.2, 0.0, 0.0), lambda i: (0.001 * i, 0.0, 0.0))
    summary = progress.summarize(rows)
    assert summary["stall_windows"][0]["n_end"] == 120
    assert summary["stall_windows"][0]["touched"] is False
    assert summary["first_object_motion_n"] == 122
    assert summary["never_touched"] is False, "a blind abort at 120 would lose this episode"


def test_object_motion_above_threshold_clears_window():
    rows = _obs(240, lambda i: (0.001 * i, 0.0, 0.0), lambda i: (0.001 * i, 0.0, 0.0))
    assert progress.summarize(rows)["stall_windows"] == []


def test_short_episode_yields_no_windows():
    rows = _obs(120, lambda i: (0.0, 0.0, 0.0), lambda i: (0.0, 0.0, 0.0))
    assert progress.stall_windows(rows, window=120) == []


def test_missing_observations_are_unknown_not_success():
    summary = progress.summarize([])
    assert summary["observations"] == 0
    assert summary["never_touched"] is None
    assert summary["object_total_motion_m"] is None


def test_thresholds_are_the_frozen_ones():
    assert progress.DEFAULT_WINDOW_ACTIONS == 120
    assert progress.DEFAULT_OBJECT_MOTION_M == 0.005
    assert progress.DEFAULT_EEF_MOTION_M == 0.02
