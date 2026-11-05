"""Independent re-seat: stage rules, frozen windows, and refused claims.

The point of these tests is the same as the transport-readiness ones: the
numbers must be pinned so a later run cannot quietly move them, and the cases
that must *not* count as a re-seat have to fail loudly (a finalizer release, a
hover with no transport, a release beside the bowl, an object that never left
the rim).
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from common import independent_reseat as ir  # noqa: E402

BOWL_XY = (-0.089, 0.004)
FINGER_ON_CHEESE = {
    "body1": "robot0_rightfinger",
    "body2": "cream_cheese_1_main",
    "geom1": "robot0_rightfinger_collision",
    "geom2": "cream_cheese_1_g0",
    "dist": 0.0,
    "robot1": True,
    "robot2": False,
    "gripper1": True,
    "gripper2": False,
}
FLOOR_Z = 0.8864
RIM_Z = 0.9384
START_BOTTOM = 0.8991
CHEESE_OFFSET = 0.040608
BOWL_START = [BOWL_XY[0], BOWL_XY[1], FLOOR_Z - 0.012]


def reading(
    *,
    dx: float = 0.0,
    dy: float = 0.0,
    bottom: float = START_BOTTOM,
    opening: float = 0.078,
    eef_object: float | None = 0.2,
    contacts: list | None = None,
    action: int = 0,
    relation_ok: bool = True,
    container: list | None = None,
) -> dict:
    pos = [BOWL_XY[0] + dx, BOWL_XY[1] + dy, bottom + CHEESE_OFFSET]
    return ir.measure(
        {
            "sim_action_count": action,
            "observation_id": f"obs-{action}",
            "object_pos": pos,
            "target_pose": [BOWL_XY[0], BOWL_XY[1], FLOOR_Z],
            "container_pos": list(BOWL_START if container is None else container),
            "eef_pos": [pos[0], pos[1], pos[2] + 0.1],
            "opening_m": opening,
            "eef_object_m": eef_object,
            "gripper_contacts": contacts,
            "relation_ok": relation_ok,
            "xy_m": (dx * dx + dy * dy) ** 0.5,
            "object_bottom_z": bottom,
            "floor_z": FLOOR_Z,
            "rim_z": RIM_Z,
        }
    )


class OptionsTest(unittest.TestCase):
    def test_off_by_default(self) -> None:
        opts = ir.options(None)
        self.assertFalse(opts["enabled"])
        self.assertEqual(opts["on_result"], "report")

    def test_config_override_merges_nested(self) -> None:
        opts = ir.options(
            {
                "composite": {
                    "independent_reseat": {
                        "enabled": True,
                        "settle": {"seat_max_m": 0.031},
                    }
                }
            }
        )
        self.assertTrue(opts["enabled"])
        self.assertEqual(opts["settle"]["seat_max_m"], 0.031)
        # untouched siblings keep their frozen defaults
        self.assertEqual(opts["settle"]["seat_min_m"], 0.006)
        self.assertEqual(opts["settle"]["max_drift_m"], 0.002)

    def test_candidate_step(self) -> None:
        self.assertTrue(
            ir.candidate_step(
                {
                    "skill": "pick_place",
                    "completion": {"kind": "relation", "object_n": "cream_cheese_1", "target_n": "akita_black_bowl_1"},
                }
            )
        )
        self.assertFalse(ir.candidate_step({"skill": "turn_on", "completion": {"kind": "joint"}}))
        self.assertFalse(ir.candidate_step({"skill": "pick_place", "completion": {"kind": "relation"}}))


class TriggerTest(unittest.TestCase):
    def test_in_bowl_state_triggers(self) -> None:
        block = ir.trigger(reading())
        self.assertTrue(block["ok"])
        self.assertTrue(block["bottom_above_floor"])
        self.assertTrue(block["xy_within_opening"])

    def test_beside_the_bowl_does_not_trigger(self) -> None:
        row = reading(dx=0.06)
        row["relation_ok"] = False
        self.assertFalse(ir.trigger(row)["ok"])

    def test_below_floor_does_not_trigger(self) -> None:
        row = reading(bottom=FLOOR_Z - 0.01)
        self.assertFalse(ir.trigger(row)["ok"])


class StageMachineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.opts = ir.options({"composite": {"independent_reseat": {"enabled": True}}})
        self.start = reading()
        self.tracker = ir.Tracker(start=self.start, opts=self.opts)

    def _grasp(self, action: int) -> dict:
        return self.tracker.observe(
            reading(
                opening=0.02,
                eef_object=0.01,
                contacts=[FINGER_ON_CHEESE],
                action=action,
            )
        )

    def test_required_rise_is_rim_plus_clearance(self) -> None:
        expected = round((RIM_Z - START_BOTTOM) + 0.02, 5)
        self.assertAlmostEqual(self.tracker._required_rise(), expected, places=5)
        self.assertGreater(expected, 0.05)

    def test_full_sequence_completes(self) -> None:
        self._grasp(5)
        self.tracker.observe(
            reading(bottom=RIM_Z + 0.03, opening=0.02, eef_object=0.01, contacts=[FINGER_ON_CHEESE], action=10)
        )
        self.tracker.observe(
            reading(dx=0.02, bottom=RIM_Z + 0.03, opening=0.02, eef_object=0.02, contacts=[FINGER_ON_CHEESE], action=15)
        )
        # released over the bowl, above the rim, fingers free
        self.tracker.observe(reading(dx=0.004, bottom=RIM_Z + 0.01, opening=0.078, eef_object=0.1, contacts=[], action=20))
        self.tracker.observe(reading(dx=0.004, bottom=START_BOTTOM + 0.002, opening=0.078, eef_object=0.1, contacts=[], action=25))
        self.tracker.observe(reading(dx=0.004, bottom=START_BOTTOM + 0.002, opening=0.078, eef_object=0.1, contacts=[], action=30))
        self.tracker.observe(reading(dx=0.004, bottom=START_BOTTOM + 0.002, opening=0.078, eef_object=0.1, contacts=[], action=35))
        result = self.tracker.result()
        self.assertEqual(result["status"], "completed")
        self.assertTrue(result["completed"])
        self.assertEqual(result["achieved"], list(ir.STAGE_ORDER))
        self.assertEqual(result["missing"], [])

    def test_finalizer_only_release_is_not_a_reseat(self) -> None:
        # execution layer opens the fingers without the policy ever grasping
        self.tracker.observe(reading(opening=0.078, action=5))
        self.tracker.observe(reading(opening=0.078, action=10))
        result = self.tracker.result()
        self.assertEqual(result["status"], "no_progress")
        self.assertFalse(result["completed"])
        self.assertEqual(result["achieved"], [])

    def test_hover_without_lift_is_no_progress(self) -> None:
        self._grasp(5)
        self.tracker.observe(reading(bottom=START_BOTTOM + 0.005, opening=0.02, eef_object=0.01, contacts=[FINGER_ON_CHEESE], action=10))
        result = self.tracker.result()
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["achieved"], ["grasp"])
        self.assertNotIn("lift", result["achieved"])

    def test_lift_without_transport_does_not_pass_carry(self) -> None:
        self._grasp(5)
        self.tracker.observe(reading(bottom=RIM_Z + 0.03, opening=0.02, eef_object=0.01, contacts=[FINGER_ON_CHEESE], action=10))
        result = self.tracker.result()
        self.assertIn("lift", result["achieved"])
        self.assertNotIn("carry", result["achieved"])
        self.assertNotEqual(result["status"], "completed")

    def test_release_beside_the_bowl_does_not_count(self) -> None:
        self._grasp(5)
        self.tracker.observe(reading(bottom=RIM_Z + 0.03, opening=0.02, eef_object=0.01, contacts=[FINGER_ON_CHEESE], action=10))
        self.tracker.observe(reading(dx=0.06, bottom=RIM_Z + 0.03, opening=0.02, eef_object=0.02, contacts=[FINGER_ON_CHEESE], action=15))
        self.tracker.observe(reading(dx=0.06, bottom=RIM_Z, opening=0.078, eef_object=0.1, contacts=[], action=20))
        result = self.tracker.result()
        self.assertIn("carry", result["achieved"])
        self.assertNotIn("release", result["achieved"])

    def test_settle_outside_seat_window_does_not_count(self) -> None:
        self._grasp(5)
        self.tracker.observe(reading(bottom=RIM_Z + 0.03, opening=0.02, eef_object=0.01, contacts=[FINGER_ON_CHEESE], action=10))
        self.tracker.observe(reading(dx=0.02, bottom=RIM_Z + 0.03, opening=0.02, eef_object=0.02, contacts=[FINGER_ON_CHEESE], action=15))
        self.tracker.observe(reading(dx=0.004, bottom=RIM_Z + 0.01, opening=0.078, eef_object=0.1, contacts=[], action=20))
        # lands perched on the rim: seat = rim - floor = 0.052 m, above the window
        for action in (25, 30, 35):
            self.tracker.observe(reading(dx=0.004, bottom=RIM_Z, opening=0.078, eef_object=0.1, contacts=[], action=action))
        result = self.tracker.result()
        self.assertIn("release", result["achieved"])
        self.assertNotIn("settle", result["achieved"])
        self.assertEqual(result["status"], "partial")

    def test_unstable_settle_does_not_count(self) -> None:
        self._grasp(5)
        self.tracker.observe(reading(bottom=RIM_Z + 0.03, opening=0.02, eef_object=0.01, contacts=[FINGER_ON_CHEESE], action=10))
        self.tracker.observe(reading(dx=0.02, bottom=RIM_Z + 0.03, opening=0.02, eef_object=0.02, contacts=[FINGER_ON_CHEESE], action=15))
        self.tracker.observe(reading(dx=0.004, bottom=RIM_Z + 0.01, opening=0.078, eef_object=0.1, contacts=[], action=20))
        self.tracker.observe(reading(dx=0.004, bottom=START_BOTTOM + 0.002, opening=0.078, eef_object=0.1, contacts=[], action=25))
        self.tracker.observe(reading(dx=0.004, bottom=START_BOTTOM + 0.002, opening=0.078, eef_object=0.1, contacts=[], action=30))
        # still sliding: 6 mm between the last two samples
        self.tracker.observe(reading(dx=0.011, bottom=START_BOTTOM + 0.002, opening=0.078, eef_object=0.1, contacts=[], action=35))
        self.assertNotIn("settle", self.tracker.result()["achieved"])

    def test_summary_is_small(self) -> None:
        self._grasp(5)
        block = self.tracker.result()
        block["policy_actions"] = 6
        block["finalizer_actions"] = 0
        block["trigger"] = ir.trigger(self.start)
        row = ir.summarise(block)
        self.assertEqual(row["status"], "partial")
        self.assertEqual(row["policy_actions"], 6)
        self.assertEqual(row["finalizer_actions"], 0)
        self.assertTrue(row["trigger_ok"])

    # -- the four false-evidence cases this round fixed -------------------

    def test_bowl_lifted_with_cheese_riding_it_is_not_a_lift(self) -> None:
        """The archived false positive: fingers on the bowl, cheese came along.

        The cheese bottom rises 6.9 cm in the world, so the old absolute-rise
        rule recorded ``lift``. Measured against the rim it rose 0 mm: it never
        left the bowl.
        """
        self.tracker.observe(
            reading(
                bottom=START_BOTTOM + 0.069,
                opening=0.02,
                eef_object=0.01,
                contacts=[FINGER_ON_CHEESE],
                action=5,
                container=[BOWL_XY[0], BOWL_XY[1], BOWL_START[2] + 0.069],
            )
        )
        result = self.tracker.result()
        self.assertNotIn("lift", result["achieved"])
        self.assertGreater(result["peak"]["rise_m"], 0.05)
        reasons = [row["reason"] for row in result["refusals"]]
        self.assertIn("container_lifted_with_object", reasons)

    def test_gripper_on_the_bowl_is_not_a_grasp_of_the_cheese(self) -> None:
        on_bowl = {
            "body1": "robot0_rightfinger",
            "body2": "akita_black_bowl_1_main",
            "dist": 0.0,
            "robot1": True,
            "gripper1": True,
        }
        self.tracker.observe(
            reading(opening=0.02, eef_object=0.01, contacts=[on_bowl, FINGER_ON_CHEESE], action=5)
        )
        result = self.tracker.result()
        self.assertNotIn("grasp", result["achieved"])
        self.assertIn("gripper_on_container", [row["reason"] for row in result["refusals"]])

    def test_unavailable_contact_probe_is_refused_not_scored(self) -> None:
        row = ir.measure(
            {
                "sim_action_count": 5,
                "object_pos": [BOWL_XY[0], BOWL_XY[1], START_BOTTOM + CHEESE_OFFSET],
                "container_pos": list(BOWL_START),
                "opening_m": 0.02,
                "eef_object_m": 0.01,
                "contacts": None,
                "object_bottom_z": START_BOTTOM,
                "floor_z": FLOOR_Z,
                "rim_z": RIM_Z,
                "xy_m": 0.0,
            }
        )
        self.assertIsNone(row["gripper_object_contact"])
        self.assertFalse(row["contacts_available"])
        self.tracker.observe(row)
        result = self.tracker.result()
        self.assertNotIn("grasp", result["achieved"])
        self.assertEqual(result["status"], "no_progress")
        self.assertIn("contact_evidence_unavailable", [r["reason"] for r in result["refusals"]])
        self.assertFalse(result["evidence"]["contacts_available"])

    def test_empty_contact_list_is_a_measured_absence(self) -> None:
        row = reading(contacts=[], opening=0.02, eef_object=0.01)
        self.assertFalse(row["gripper_object_contact"])
        self.assertTrue(row["contacts_available"])

    def test_container_guard_is_limited_to_fixture_stages(self) -> None:
        """``carry`` may brush the bowl; ``grasp``/``lift``/``settle`` may not."""
        self.assertEqual(
            sorted(self.opts["container_guard"]["stages"]), ["grasp", "lift", "settle"]
        )
        self._grasp(5)
        self.tracker.observe(
            reading(bottom=RIM_Z + 0.03, opening=0.02, eef_object=0.01, contacts=[FINGER_ON_CHEESE], action=10)
        )
        # the arm brushes the bowl 2 cm while carrying: carry is still reached
        self.tracker.observe(
            reading(
                dx=0.02,
                bottom=RIM_Z + 0.03,
                opening=0.02,
                eef_object=0.02,
                contacts=[FINGER_ON_CHEESE],
                action=15,
                container=[BOWL_XY[0] + 0.02, BOWL_XY[1], BOWL_START[2]],
            )
        )
        result = self.tracker.result()
        self.assertIn("lift", result["achieved"])
        self.assertIn("carry", result["achieved"])
        self.assertGreater(result["container"]["move_m"], 0.01)


if __name__ == "__main__":
    unittest.main()
