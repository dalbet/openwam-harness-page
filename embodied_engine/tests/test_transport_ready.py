"""Transport readiness: a second, independent question about a placed object.

The step goal says "the cheese is in the bowl". This check says whether the
cheese is seated deep enough for the *next* step to carry the bowl. These tests
pin the rule and, just as importantly, pin the cases where the answer must be
"not measured" instead of a guess.
"""

from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from common import transport_ready as tr  # noqa: E402

BOWL_Z = 0.9289
FLOOR_Z = BOWL_Z - 0.012
RIM_Z = BOWL_Z + 0.04
CHEESE_BOTTOM_MARGIN = 0.040608
BOWL_XY = (-0.2, 0.2)
STEP = {
    "step_id": "c1",
    "step_index": 1,
    "step_task_key": "put_cream_cheese_in_bowl",
    "completion": {
        "kind": "relation",
        "object_n": "cream_cheese_1",
        "target_n": "akita_black_bowl_1",
        "relation": "in",
    },
}
CATALOG = {
    "put_cream_cheese_in_bowl": {
        "objects": ["cream_cheese_1"],
        "targets": ["akita_black_bowl_1"],
    },
    "put_bowl_on_stove": {
        "objects": ["akita_black_bowl_1"],
        "targets": ["main_table_stove_front_region"],
    },
    "turn_on_stove": {"objects": ["flat_stove_1"], "targets": ["flat_stove_1_button"]},
}


def sample(*, xy: float, seat: float, action: int = 1, offset_xy: tuple[float, float] | None = None):
    """One post-release sample row shaped exactly like ``finalize._sample_rows``."""
    bottom = FLOOR_Z + seat
    dx, dy = offset_xy or (xy, 0.0)
    relation = {
        "ok": True,
        "predicate": "bowl_interior",
        "xy_m": round(xy, 5),
        "xy_limit_m": 0.025,
        "object_bottom_z": round(bottom, 5),
        "floor_z": round(FLOOR_Z, 5),
        "rim_z": round(RIM_Z, 5),
        "target_pose": [BOWL_XY[0], BOWL_XY[1], BOWL_Z],
        "evidence_source": "asset_geometry",
    }
    return {
        "observation_id": f"obs-{action}",
        "sim_action_count": action,
        "native_goal": False,
        "relation_ok": True,
        "relation": {
            "ok": True,
            "predicate": "relation+release_height",
            "relation": relation,
            "height": {
                "ready": True,
                "kind": "interior_floor",
                "object_bottom_z": round(bottom, 5),
                "surface_z": round(FLOOR_Z, 5),
                "delta_m": round(seat, 5),
            },
        },
        "object_pos": [BOWL_XY[0] + dx, BOWL_XY[1] + dy, bottom + CHEESE_BOTTOM_MARGIN],
    }


def three(**kwargs):
    return [sample(action=n + 1, **kwargs) for n in range(3)]


class OptionsTest(unittest.TestCase):
    def test_defaults_are_frozen_values(self):
        opts = tr.options(None)
        self.assertEqual(opts["max_seat_delta_m"], 0.029)
        self.assertEqual(opts["min_seat_delta_m"], -0.002)
        self.assertEqual(opts["min_xy_margin_m"], 0.002)
        self.assertEqual(opts["reseat"]["max_calls"], 1)
        self.assertEqual(opts["reseat"]["on_failure"], "stop")

    def test_config_overrides_only_declared_keys(self):
        cfg = {"composite": {"transport_ready": {"max_seat_delta_m": 0.031, "reseat": {"max_steps": 100}}}}
        opts = tr.options(cfg)
        self.assertEqual(opts["max_seat_delta_m"], 0.031)
        self.assertEqual(opts["reseat"]["max_steps"], 100)
        self.assertEqual(opts["reseat"]["max_calls"], 1)

    def test_disabled_check_is_not_a_verdict(self):
        cfg = {"composite": {"transport_ready": {"enabled": False}}}
        block = tr.evaluate(step=STEP, relation=None, samples=three(xy=0.01, seat=0.012), cfg=cfg)
        self.assertFalse(block["checked"])
        self.assertIsNone(block["ok"])
        self.assertEqual(block["reason_code"], "disabled")


class AppliesTest(unittest.TestCase):
    def test_applies_only_when_a_later_step_carries_the_target(self):
        steps = [STEP, {"step_task_key": "put_bowl_on_stove"}]
        self.assertTrue(tr.applies(steps, 0, CATALOG))

    def test_last_step_never_applies(self):
        steps = [STEP, {"step_task_key": "put_bowl_on_stove"}]
        self.assertFalse(tr.applies(steps, 1, CATALOG))

    def test_no_later_step_carries_the_container(self):
        steps = [STEP, {"step_task_key": "turn_on_stove"}]
        self.assertFalse(tr.applies(steps, 0, CATALOG))

    def test_non_relation_step_never_applies(self):
        steps = [dict(STEP, completion={"kind": "native_goal"}), {"step_task_key": "put_bowl_on_stove"}]
        self.assertFalse(tr.applies(steps, 0, CATALOG))


class EvaluateTest(unittest.TestCase):
    def test_nominal_seat_is_ready(self):
        block = tr.evaluate(step=STEP, relation=None, samples=three(xy=0.010, seat=0.012), cfg=None)
        self.assertTrue(block["checked"])
        self.assertTrue(block["ok"])
        self.assertEqual(block["risk"], "nominal")
        self.assertEqual(block["features"]["seat_delta_m"], 0.012)

    def test_archived_pass_at_the_top_of_the_nominal_cluster_is_ready(self):
        # accept_2step_zh_i2 measured 0.0225 m seat and was carried fine.
        block = tr.evaluate(step=STEP, relation=None, samples=three(xy=0.020, seat=0.0225), cfg=None)
        self.assertTrue(block["ok"], block["reasons"])

    def test_perched_seat_is_a_warning_not_a_rejection(self):
        # 0.0357 m is the value of both one failing and one passing archive
        # round, and 11 of 12 archived perched rounds transported fine, so the
        # rule records the warning instead of rejecting (or re-seating) them.
        block = tr.evaluate(step=STEP, relation=None, samples=three(xy=0.019, seat=0.0357), cfg=None)
        self.assertTrue(block["ok"])
        self.assertEqual(block["warning"], "perched_above_floor")
        self.assertEqual(block["risk"], "perched")
        self.assertIsNone(block["reason_code"])
        self.assertEqual(block["features"]["rim_clearance_m"], round(RIM_Z - (FLOOR_Z + 0.0357), 5))

    def test_below_the_interior_floor_means_it_fell_beside_the_bowl(self):
        block = tr.evaluate(step=STEP, relation=None, samples=three(xy=0.049, seat=-0.004), cfg=None)
        self.assertFalse(block["ok"])
        self.assertEqual(block["risk"], "fell")
        self.assertEqual(block["reason_code"], "fell_beside_bowl")

    def test_edge_needs_a_real_margin(self):
        block = tr.evaluate(step=STEP, relation=None, samples=three(xy=0.024, seat=0.012), cfg=None)
        self.assertFalse(block["ok"])
        self.assertEqual(block["reason_code"], "edge_of_bowl")
        self.assertAlmostEqual(block["features"]["xy_margin_m"], 0.001, places=5)

    def test_relative_drift_is_measured_between_samples(self):
        rows = three(xy=0.010, seat=0.012)
        rows[2] = sample(xy=0.010, seat=0.012, action=3, offset_xy=(0.030, 0.0))
        block = tr.evaluate(step=STEP, relation=None, samples=rows, cfg=None)
        self.assertEqual(block["reason_code"], "still_moving")
        self.assertAlmostEqual(block["features"]["relative_drift"]["max_distance_m"], 0.02, places=5)

    def test_a_single_read_measures_geometry_but_not_drift(self):
        # One row is enough for seat depth and edge margin; it is not enough
        # for drift, which must be reported as unmeasured rather than as zero.
        block = tr.evaluate(step=STEP, relation=None, samples=[sample(xy=0.010, seat=0.012)], cfg=None)
        self.assertTrue(block["measurable"])
        self.assertTrue(block["ok"])
        self.assertFalse(block["drift_measured"])
        self.assertEqual(block["features"]["relative_drift"]["samples"], 1)
        self.assertIsNone(block["features"]["relative_drift"]["max_distance_m"])

    def test_no_geometry_is_not_a_verdict(self):
        block = tr.evaluate(step=STEP, relation=None, samples=[], evidence={}, cfg=None)
        self.assertFalse(block["measurable"])
        self.assertIsNone(block["ok"])
        self.assertEqual(block["reason_code"], "not_applicable")

    def test_broken_relation_is_reported_before_geometry(self):
        rows = three(xy=0.010, seat=0.012)
        rows[-1]["relation"]["relation"]["ok"] = False
        rows[-1]["relation_ok"] = False
        block = tr.evaluate(step=STEP, relation=None, samples=rows, cfg=None)
        self.assertFalse(block["ok"])
        self.assertIn("relation_not_met", block["reasons"])

    def test_a_non_interior_target_is_not_applicable(self):
        rows = three(xy=0.010, seat=0.012)
        for row in rows:
            row["relation"]["relation"] = {"ok": True, "predicate": "plate_support"}
        block = tr.evaluate(step=STEP, relation=None, samples=rows, cfg=None)
        self.assertFalse(block["checked"])
        self.assertEqual(block["reason_code"], "not_applicable")
        self.assertIsNone(block["ok"])

    def test_falls_back_to_the_single_step_relation_when_samples_are_missing(self):
        relation = {
            "ok": True,
            "predicate": "relation+release_height",
            "relation": {
                "ok": True,
                "predicate": "bowl_interior",
                "xy_m": 0.02,
                "xy_limit_m": 0.025,
                "object_bottom_z": FLOOR_Z + 0.012,
                "floor_z": FLOOR_Z,
                "rim_z": RIM_Z,
                "target_pose": [BOWL_XY[0], BOWL_XY[1], BOWL_Z],
            },
            "height": {"ready": True},
        }
        block = tr.evaluate(step=STEP, relation=relation, samples=[], cfg=None)
        self.assertTrue(block["checked"])
        self.assertEqual(block["features"]["seat_delta_m"], 0.012)
        self.assertTrue(block["ok"])
        self.assertTrue(block["single_read"])
        self.assertEqual(block["evidence"]["source"], "evidence.relation")
        self.assertFalse(block["drift_measured"])

    def test_summary_is_a_short_row_that_keeps_the_reason(self):
        block = tr.evaluate(step=STEP, relation=None, samples=three(xy=0.0245, seat=0.012), cfg=None)
        row = tr.summarise(block)
        self.assertFalse(row["ok"])
        self.assertEqual(row["reason_code"], "edge_of_bowl")
        self.assertEqual(row["seat_delta_m"], 0.012)

    def test_summary_keeps_the_perched_warning_without_failing(self):
        block = tr.evaluate(step=STEP, relation=None, samples=three(xy=0.019, seat=0.0357), cfg=None)
        row = tr.summarise(block)
        self.assertTrue(row["ok"])
        self.assertEqual(row["warning"], "perched_above_floor")
        self.assertIsNone(row["reason_code"])

class EvidenceFallbackTest(unittest.TestCase):
    """The two sample shapes in the archive must give the same verdict."""

    def _evidence(self, seats, xys):
        inner = {
            "ok": True,
            "predicate": "bowl_interior",
            "xy_m": round(xys[-1], 5),
            "xy_limit_m": 0.025,
            "object_bottom_z": round(FLOOR_Z + seats[-1], 5),
            "floor_z": round(FLOOR_Z, 5),
            "rim_z": round(RIM_Z, 5),
            "target_pose": [BOWL_XY[0], BOWL_XY[1], BOWL_Z],
            "evidence_source": "asset_geometry",
        }
        return {
            "ok": True,
            "method": "engine_predicate:relation+release_height",
            "predicate": "relation+release_height",
            "samples": [True] * len(seats),
            "relation": inner,
            "height": {"ready": True, "kind": "interior_floor"},
            "relation_samples": [
                {
                    "ok": True,
                    "predicate": "bowl_interior",
                    "xy_m": round(xy, 5),
                    "object_bottom_z": round(FLOOR_Z + seat, 5),
                    "target_pose": [BOWL_XY[0], BOWL_XY[1], BOWL_Z],
                }
                for seat, xy in zip(seats, xys)
            ],
        }

    def test_relation_samples_are_enough_when_placement_samples_are_absent(self):
        evidence = self._evidence([0.012] * 3, [0.010] * 3)
        block = tr.evaluate(step=STEP, relation=evidence, samples=None, evidence=evidence, cfg=None)
        self.assertTrue(block["checked"])
        self.assertTrue(block["measurable"])
        self.assertTrue(block["ok"])
        self.assertEqual(block["evidence"]["source"], "evidence.relation_samples")
        self.assertEqual(block["features"]["seat_delta_m"], 0.012)
        self.assertEqual(block["features"]["relative_drift"]["mode"], "xy_and_seat")

    def test_geometry_defaults_are_merged_into_each_sample(self):
        evidence = self._evidence([0.0357] * 3, [0.019] * 3)
        block = tr.evaluate(step=STEP, relation=evidence, samples=None, evidence=evidence, cfg=None)
        self.assertEqual(block["features"]["rim_clearance_m"], round(RIM_Z - (FLOOR_Z + 0.0357), 5))
        self.assertEqual(block["warning"], "perched_above_floor")
        self.assertTrue(block["ok"])

    def test_a_falling_cheese_in_relation_samples_is_a_hard_failure(self):
        evidence = self._evidence([0.012, 0.011, -0.004], [0.010, 0.030, 0.049])
        block = tr.evaluate(step=STEP, relation=evidence, samples=None, evidence=evidence, cfg=None)
        self.assertFalse(block["ok"])
        self.assertEqual(block["risk"], "fell")
        self.assertEqual(block["reason_code"], "fell_beside_bowl")

    def test_placement_samples_win_over_the_step_evidence(self):
        evidence = self._evidence([0.0357] * 3, [0.019] * 3)
        block = tr.evaluate(
            step=STEP, relation=evidence, samples=three(xy=0.010, seat=0.012), evidence=evidence, cfg=None
        )
        self.assertEqual(block["evidence"]["source"], "placement.samples")
        self.assertTrue(block["ok"])
        self.assertIsNone(block["warning"])

    def test_evaluate_never_mutates_the_step(self):
        before = copy.deepcopy(STEP)
        tr.evaluate(step=STEP, relation=None, samples=three(xy=0.019, seat=0.0357), cfg=None)
        self.assertEqual(before, STEP)


class ChangeTests(unittest.TestCase):
    """The re-seat's before/after geometry, so a no-op is never read as a repair."""

    def test_a_still_object_reports_no_change(self):
        row = tr.change({"seat_delta_m": 0.0128, "xy_m": 0.0061}, {"seat_delta_m": 0.0128, "xy_m": 0.0061})
        self.assertFalse(row["changed"])
        self.assertEqual(row["seat_delta_change_m"], 0.0)
        self.assertIn("did not change", row["note"])

    def test_a_seated_object_reports_change(self):
        row = tr.change({"seat_delta_m": 0.0357, "xy_m": 0.019}, {"seat_delta_m": 0.011, "xy_m": 0.008})
        self.assertTrue(row["changed"])
        self.assertEqual(row["seat_delta_change_m"], -0.0247)
        self.assertEqual(row["note"], "the re-seat moved the object")

    def test_missing_geometry_is_not_a_claim_either_way(self):
        row = tr.change({"seat_delta_m": None}, {"seat_delta_m": 0.01})
        self.assertFalse(row["changed"])
        self.assertIsNone(row["seat_delta_change_m"])

    def test_change_is_not_returned_as_a_motion_measurement(self):
        # 1 mm of settling is inside the tolerance and must not count as a repair.
        row = tr.change({"seat_delta_m": 0.0128, "xy_m": 0.0061}, {"seat_delta_m": 0.0138, "xy_m": 0.0061})
        self.assertFalse(row["changed"])


if __name__ == "__main__":
    unittest.main()
