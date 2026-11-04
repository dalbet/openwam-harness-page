"""The bounded final-goal repair: frozen budget and strict decision parsing.

No simulator and no model: these pin the two places a repair could silently
become something else -- an unbounded retry loop, or free-form control output
from the language model.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from agent.agent_service import _parse_repair  # noqa: E402
from common import composite_plan  # noqa: E402


class RepairOptionsTest(unittest.TestCase):
    def test_disabled_and_defaults(self):
        opts = composite_plan.repair_options({"composite": {}})
        self.assertFalse(opts["enabled"])
        self.assertEqual(opts["max_calls"], 1)
        self.assertEqual(opts["max_steps"], 240)
        self.assertTrue(opts["require_final_step_released"])

    def test_frozen_budget_is_read_from_config(self):
        opts = composite_plan.repair_options(
            {"composite": {"repair": {"enabled": True, "max_calls": 5, "max_steps": 90, "timeout_s": 12}}}
        )
        self.assertTrue(opts["enabled"])
        # At most one repair call per episode, whatever the file says.
        self.assertEqual(opts["max_calls"], 1)
        self.assertEqual(opts["max_steps"], 90)
        self.assertEqual(opts["timeout_s"], 12.0)


class RepairDecisionParseTest(unittest.TestCase):
    KEYS = {"open_middle_drawer", "turn_on_stove"}

    def test_repair_with_candidate_key(self):
        decision, key, errors = _parse_repair(
            '{"decision":"repair","key":"open_middle_drawer","reason":"the drawer closed"}', self.KEYS
        )
        self.assertEqual(decision, "repair")
        self.assertEqual(key, "open_middle_drawer")
        self.assertEqual(errors, [])

    def test_abort_has_no_key(self):
        decision, key, _errors = _parse_repair('{"decision":"abort","key":null,"reason":"none"}', self.KEYS)
        self.assertEqual(decision, "abort")
        self.assertIsNone(key)

    def test_invented_key_is_rejected(self):
        decision, key, errors = _parse_repair('{"decision":"repair","key":"wave_magic"}', self.KEYS)
        self.assertIsNone(decision)
        self.assertIsNone(key)
        self.assertEqual(errors[0]["code"], "not_a_candidate")

    def test_free_form_output_is_rejected(self):
        decision, _key, errors = _parse_repair("I would open the drawer and then place the bowl.", self.KEYS)
        self.assertIsNone(decision)
        self.assertEqual(errors[0]["code"], "not_json")


if __name__ == "__main__":
    unittest.main()
