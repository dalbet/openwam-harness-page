"""CPU checks for three-state verification, recovery budget, and stale versions."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from executor import Session, run_plan
from planner import load_human_plan
from recovery import RecoveryBudget, propose, signature
from store import LocalizationCache, candidate_card
from test_contract import PLAN, Scripted, _run
from verify import native_success, verify_step


class HeldButUnstable(Scripted):
    """First pick is empty. A later pick looks lifted, but the object keeps moving."""

    def __init__(self):
        super().__init__(empty=True)
        self.picks = 0

    def execute(self, action, args):
        if action == "pi0_pick":
            self.picks += 1
            if self.picks >= 2:
                self.empty = False
                self.skip_samples = True
        return super().execute(action, args)

    def observe(self):
        obs = super().observe()
        if self.picks >= 2:
            obs["sim"]["objects"]["alphabet_soup_1"]["pos"] = [0.05 * self.observations, 0.0, 0.62]
            obs["sim"]["eef_pos"] = [0.0, 0.0, 0.62]
            obs["sim"]["gripper_opening_m"] = 0.02
        return obs

ROOT = Path(__file__).resolve().parent


def _pick_plan():
    plan = json.loads(json.dumps(load_human_plan(PLAN, "libero_10_t0")))
    plan["steps"] = [plan["steps"][4]]
    plan["steps"][0]["depends_on"] = []
    plan["steps"][0]["preconditions"] = []
    return plan


class StageCTest(unittest.TestCase):
    def test_empty_gripper_fails_and_held_grasp_passes(self):
        session = Session("libero_10_t0")
        session.begin_episode()
        empty = Scripted(empty=True)
        result = run_plan(_pick_plan(), empty, session, lambda _subtask: "success", task=_pick_plan()["goal"])
        self.assertEqual(result["status"], "RECOVERY_EXHAUSTED")
        self.assertLessEqual(empty.calls.count("set_gripper"), 2)
        self.assertNotIn("reset", empty.calls)
        held = Scripted()
        session = Session("libero_10_t0")
        session.begin_episode()
        result = run_plan(_pick_plan(), held, session, lambda _subtask: "success", task=_pick_plan()["goal"])
        self.assertEqual(result["steps"][0]["verification"]["status"], "success")
        self.assertEqual(result["steps"][0]["verification"]["reason_code"], "GRASP_HELD")
        self.assertNotIn("set_gripper", held.calls)

    def test_missing_evidence_is_unknown_and_not_rewritten(self):
        session = Session("libero_10_t0")
        session.begin_episode()
        blind = Scripted(blind=True)
        result = run_plan(_pick_plan(), blind, session, lambda _subtask: "success", task=_pick_plan()["goal"])
        verification = result["steps"][0]["verification"]
        self.assertEqual(verification["status"], "unknown")
        self.assertIn("manual_confirmation", verification)
        self.assertNotEqual(verification["status"], "success")
        self.assertNotIn("set_gripper", blind.calls)

    def test_unknown_after_recovery_does_not_continue(self):
        plan = json.loads(json.dumps(load_human_plan(PLAN, "libero_10_t0")))
        plan["steps"] = plan["steps"][4:6]
        plan["steps"][0]["depends_on"] = []
        plan["steps"][0]["preconditions"] = []
        plan["steps"][1]["depends_on"] = []
        plan["steps"][1]["preconditions"] = []
        plan["steps"][1]["args"] = {"xyz": [0.0, 0.1, 0.7], "gripper": 1.0, "object_ref": "alphabet_soup"}
        asked = []
        session = Session("libero_10_t0")
        session.begin_episode()
        result = run_plan(
            plan,
            HeldButUnstable(),
            session,
            lambda subtask: asked.append(subtask) or "unknown",
            task=plan["goal"],
        )
        self.assertEqual(asked, ["soup_pick"])
        self.assertEqual(result["status"], "no_progress")
        self.assertEqual(result["steps"][0]["verification"]["reason_code"], "manual_unknown")
        self.assertIn("lifted samples", result["reason"])
        self.assertEqual([step["step_id"] for step in result["steps"]], ["pick_alphabet_soup"])

    def test_recovery_budget_and_duplicate_signature(self):
        budget = RecoveryBudget(limit=2)
        step = {"step_id": "pick", "object_ref": "alphabet_soup", "args": {"prompt": "pick up the alphabet soup", "object_ref": "alphabet_soup"}}
        first = propose("EMPTY_GRASP", step, budget=budget, holding=False, released=True, stalled=True, env_step_changed=False, attempt_index=0)
        self.assertEqual(first["kind"], "actions")
        self.assertTrue(budget.consume(step["step_id"]))
        for action in first["actions"]:
            if action["skill"] in {"pi0_pick", "move_to", "set_gripper"}:
                budget.note(step["step_id"], signature(action["skill"], action["args"]))
        self.assertFalse(budget.allow(step["step_id"], signature("set_gripper", {"gripper": -1.0, "steps": 8, "object_ref": "alphabet_soup"})))
        self.assertTrue(budget.consume(step["step_id"]))
        self.assertEqual(budget.remaining(step["step_id"]), 0)
        exhausted = propose("EMPTY_GRASP", step, budget=budget, holding=False, released=True, stalled=True, env_step_changed=False, attempt_index=1)
        self.assertEqual(exhausted["kind"], "exhausted")

    def test_service_error_does_not_repeat_a_committed_action(self):
        budget = RecoveryBudget()
        step = {"step_id": "move", "skill": "move_to", "object_ref": "alphabet_soup", "args": {"xyz": [0.0, 0.1, 0.6], "gripper": 1.0, "object_ref": "alphabet_soup"}}
        blocked = propose("SERVICE_ERROR", step, budget=budget, holding=True, released=False, stalled=False, env_step_changed=True, attempt_index=0)
        self.assertEqual(blocked["kind"], "service_no_retry")
        self.assertEqual(blocked["actions"], [])

    def test_normal_progress_does_not_replan_and_old_version_is_discarded(self):
        calls = []
        plan = load_human_plan(PLAN, "libero_10_t0")
        session = Session("libero_10_t0")
        session.begin_episode()

        def confirm(subtask):
            if subtask == "soup_place":
                session.plan_version += 1
                return "success"
            return "unknown"

        result = run_plan(
            plan,
            Scripted(),
            session,
            confirm,
            task=plan["goal"],
            replan_fn=lambda payload: calls.append(payload) or None,
        )
        self.assertEqual(calls, [])
        self.assertEqual(result["status"], "discarded")

    def test_native_success_ignores_truncation(self):
        self.assertTrue(native_success(True, False))
        self.assertFalse(native_success(False, True))
        self.assertFalse(native_success(True, True))

    def test_cache_and_card_do_not_keep_coordinates_as_a_procedure(self):
        cache = LocalizationCache()
        cache.put("basket", scene_epoch=1, camera="agentview", observation_id="obs", xyz=[0.1, 0.2, 0.4], env_step=1)
        cache.put("alphabet_soup", scene_epoch=1, camera="agentview", observation_id="obs", xyz=[0.0, 0.0, 0.5], env_step=2)
        cache.invalidate("pi0_pick", object_ref="alphabet_soup", scene_epoch=1)
        self.assertIsNone(cache.get("alphabet_soup", 1))
        self.assertIsNotNone(cache.get("basket", 1, "agentview"))
        card = candidate_card(load_human_plan(PLAN, "libero_10_t0"), code_version="test")
        self.assertEqual(card["status"], "candidate")
        self.assertNotIn("xyz", json.dumps(card["objects"]))
        self.assertEqual(card["goal_relation"], "in")

    def test_stale_observation_rejected(self):
        plan = load_human_plan(PLAN, "libero_10_t0")
        step = plan["steps"][4]
        verdict = verify_step(
            step,
            type("O", (), {"executed": True, "grasp_signal": True, "xyz": None, "error": None})(),
            [{"observation_id": "old", "scene_epoch": 1, "env_step": 0, "cameras": {"agentview": {}, "wrist": {}}, "sim": {}}],
            scene_epoch=2,
            min_step=1,
            objects=plan["objects"],
            support_z={},
            goal_relation="in",
        )
        self.assertEqual(verdict["status"], "failure")
        self.assertEqual(verdict["reason_code"], "STALE_OBSERVATION")

    def test_gripper_outside_band_names_the_condition(self):
        plan = load_human_plan(PLAN, "libero_10_t0")
        step = plan["steps"][4]
        observations = [
            _obs("a", 5, 0.0034, 1.0),
            _obs("b", 10, 0.0034, 2.0),
        ]
        verdict = verify_step(
            step,
            type("O", (), {"executed": True, "grasp_signal": True, "xyz": None, "error": None})(),
            observations,
            scene_epoch=1,
            min_step=None,
            objects=plan["objects"],
            support_z={"alphabet_soup_1": 0.45},
            goal_relation="in",
        )
        self.assertEqual(verdict["status"], "success")
        self.assertEqual(verdict["reason_code"], "GRASP_HELD")
        self.assertEqual(verdict["evidence_source"], "sim_state")
        self.assertIn("not a contact gap", verdict["metrics"]["gripper_band_note"])
        self.assertIsNone(verdict["metrics"]["contact"])
        self.assertIn("not contact distance", verdict["metrics"]["origin_note"])

    def test_same_snapshot_is_not_a_second_sample(self):
        from executor import _observations_from_samples

        rows = _observations_from_samples(
            [
                {"observation_id": "a", "env_action_count": 5, "timestamp": 1.0, "sim": {"objects": {}}},
                {"observation_id": "b", "env_action_count": 5, "timestamp": 1.1, "sim": {"objects": {}}},
            ],
            1,
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["observation_id"], "a")

    def test_frame_bus_drops_a_stale_snapshot(self):
        import time
        from frame_bus import FrameBus
        import numpy as np

        sent = []

        def sender(agent, wrist, meta):
            time.sleep(0.08)
            sent.append(meta["observation_id"])

        bus = FrameBus()
        bus.set_sender(sender)
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        for index in range(4):
            bus.publish(image, image, {"observation_id": f"f{index}"})
        time.sleep(0.6)
        bus.close()
        self.assertGreater(bus.dropped, 0)
        self.assertLess(bus.sent, bus.published)
        self.assertGreaterEqual(bus.sent, 1)

    def test_transport_tracks_relative_pose_not_origin_distance(self):
        step = {
            "completion_check": "object_transported",
            "object_ref": "bowl",
            "target_ref": "plate",
            "args": {"motion": "travel"},
        }
        objects = {
            "bowl": {"sim_name": "akita_black_bowl_1"},
            "plate": {"sim_name": "plate_1", "role": "container"},
        }
        before = _scene("a", [0.0, 0.0, 1.0], [0.0, 0.04, 1.06], 1)
        after = _scene("b", [0.05, 0.16, 1.0], [0.05, 0.2, 1.06], 2)
        outcome = type("O", (), {"executed": True, "grasp_signal": None, "xyz": None, "error": None})()
        held = verify_step(
            step, outcome, [after], scene_epoch=1, min_step=None, objects=objects,
            support_z={}, goal_relation="on", before=before,
        )
        self.assertEqual(held["reason_code"], "TRANSPORTED")
        self.assertGreater(held["metrics"]["origin_to_eef_m"], 0.03)
        slipped = _scene("c", [0.05, 0.16, 0.9], [0.05, 0.2, 1.06], 3)
        lost = verify_step(
            step, outcome, [slipped], scene_epoch=1, min_step=None, objects=objects,
            support_z={}, goal_relation="on", before=before,
        )
        self.assertEqual(lost["reason_code"], "GRASP_LOST")

    def test_carry_offset_moves_the_object_onto_the_target(self):
        from schema import resolve_args

        resolved = resolve_args(
            {
                "xyz_from": "segment:plate",
                "z_offset": 0.12,
                "gripper": 1,
                "carry_offset": "bowl",
                "motion": "vertical",
                "object_ref": "bowl",
            },
            {"segment:plate": [0.05, 0.2, 0.9], "carry:bowl": [0.0, -0.04, -0.06]},
        )
        self.assertEqual(resolved["xyz"], [0.05, 0.24, 1.08])
        self.assertNotIn("carry_offset", resolved)

    def test_native_release_completes_without_closing_again(self):
        class NativeOnRelease(Scripted):
            def execute(self, action, args):
                self.gripper_commands.append(args.get("gripper") if action == "set_gripper" else None)
                outcome = super().execute(action, args)
                if action == "set_gripper" and float(args.get("gripper") or 0) < 0:
                    self.native = True
                return outcome

            def __init__(self):
                super().__init__()
                self.gripper_commands = []

        backend = NativeOnRelease()
        result = _run(load_human_plan(PLAN, "libero_10_t0"), backend, lambda _subtask: "unknown")
        self.assertEqual(result["status"], "completed")
        self.assertTrue(result["native_success"])
        self.assertEqual(result["reason"], "NATIVE_SUCCESS")
        opened = [value for value in backend.gripper_commands if value is not None]
        self.assertEqual(opened, [-1.0])

    def test_recovery_uses_the_object_phrase(self):
        from recovery import _phrase

        self.assertEqual(
            _phrase({"grounding_query": "black bowl", "args": {"xyz_from": "segment:plate"}}),
            "black bowl",
        )

    def test_display_flip_is_vertical_once_and_does_not_upscale(self):
        import numpy as np
        from display_view import encode_display, prepare_camera

        image = np.zeros((256, 256, 3), dtype=np.uint8)
        image[0, :, :] = 255
        view = prepare_camera(image, source="raw_obs.agentview_image")
        self.assertEqual(view["flip"], "vertical")
        self.assertEqual(int(view["array"][-1, 0, 0]), 255)
        self.assertEqual(int(view["array"][0, 0, 0]), 0)
        jpg, meta = encode_display(view)
        self.assertGreater(len(jpg), 100)
        self.assertEqual(meta["out_h"], 256)
        self.assertEqual(meta["jpeg_quality"], 85)
        self.assertEqual(meta["max_edge"], 512)


def _scene(name, obj, eef, version):
    return {
        "observation_id": name,
        "scene_epoch": 1,
        "env_step": version,
        "timestamp": version,
        "cameras": {"agentview": {"id": name}, "wrist": {"id": name}},
        "sim": {
            "objects": {
                "akita_black_bowl_1": {"pos": obj, "identified": True},
                "plate_1": {"pos": [0.05, 0.2, 0.9], "identified": True},
            },
            "eef_pos": eef,
            "gripper_opening_m": 0.004,
            "contact": None,
            "evidence_source": "sim_state",
        },
    }


def _obs(name, version, opening, stamp):
    return {
        "observation_id": name,
        "scene_epoch": 1,
        "env_step": version,
        "timestamp": stamp,
        "cameras": {"agentview": {"id": name}, "wrist": {"id": name}},
        "sim": {
            "objects": {"alphabet_soup_1": {"pos": [0.0, 0.0, 0.62], "identified": True}},
            "eef_pos": [0.0, 0.0, 0.62],
            "gripper_opening_m": opening,
            "contact": None,
            "evidence_source": "sim_state",
        },
    }


if __name__ == "__main__":
    unittest.main()
