"""The env-service side of ``robot_state_restore``.

These tests use a fake MuJoCo model that deliberately puts the robot joints
*not* at the front of ``qpos``: a hardcoded slice would pass a test where the
arm sits at index 0, and would corrupt the drawer/object joints in the real
scene. Everything here is checked against the address tables instead.

The collision pre-check is exercised in its two decision branches (refused and
unavailable): the real ``mj_forward`` probe needs a live model and is covered
by the on-robot probe, not by this file.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from adapters.env_runtime import episode_service as es  # noqa: E402

ARM = [f"robot0_joint{i}" for i in range(1, 8)]
FINGERS = ["gripper0_right_finger_joint1", "gripper0_right_finger_joint2"]
ROBOT_JOINTS = ARM + FINGERS


class FakeModel:
    """njnt/nq/nv/nu plus the address tables the service actually reads."""

    # (name, mujoco joint type, qpos width, dof width, body)
    JOINTS = (
        ("wooden_cabinet_1_middle_level", 3, 1, 1, "wooden_cabinet_1_middle_level"),  # drawer
        *[(name, 3, 1, 1, "robot0_base") for name in ARM],
        *[(name, 2, 1, 1, "gripper0_right") for name in FINGERS],
        ("akita_black_bowl_1_joint0", 0, 7, 6, "akita_black_bowl_1"),  # free object
        ("wooden_cabinet_1_top_level", 3, 1, 1, "wooden_cabinet_1_top_level"),
    )
    GEOMS = (
        ("robot0_link7_collision", "robot0_link7"),
        ("gripper0_right_finger1_collision", "gripper0_right_finger1"),
        ("gripper0_right_finger2_collision", "gripper0_right_finger2"),
        ("cabinet_handle_collision", "wooden_cabinet_1_middle_level"),
        ("bowl_collision", "akita_black_bowl_1"),
    )

    def __init__(self):
        self.body_names = [
            "world",
            "robot0_base",
            "robot0_link7",
            "gripper0_right",
            "gripper0_right_finger1",
            "gripper0_right_finger2",
            "wooden_cabinet_1_middle_level",
            "wooden_cabinet_1_top_level",
            "akita_black_bowl_1",
        ]
        self.geom_names = [row[0] for row in self.GEOMS]
        self.geom_bodyid = [self.body_index(row[1]) for row in self.GEOMS]
        self.joint_names = [row[0] for row in self.JOINTS]
        self.njnt = len(self.JOINTS)
        self.jnt_type = [row[1] for row in self.JOINTS]
        self.jnt_bodyid = [self.body_index(row[4]) for row in self.JOINTS]
        qpos = 0
        dof = 0
        self.jnt_qposadr = []
        self.jnt_dofadr = []
        for _name, _type, qwidth, dwidth, _body in self.JOINTS:
            self.jnt_qposadr.append(qpos)
            self.jnt_dofadr.append(dof)
            qpos += qwidth
            dof += dwidth
        self.nq = qpos
        self.nv = dof
        self.nu = len(ARM) + len(FINGERS)
        # Every actuator drives exactly one robot joint, in model order.
        self.actuator_trnid = [[index, 0] for index in self.robot_joint_ids()]
        # Present but unusable on purpose: the raw-mujoco probe must report the
        # capability gap rather than silently skip the check.
        self._model = None

    def robot_joint_ids(self) -> list[int]:
        return [index for index, name in enumerate(self.joint_names) if name in ROBOT_JOINTS]

    def body_index(self, name: str) -> int:
        return self.body_names.index(name)

    def joint_name2id(self, name: str) -> int:
        return self.joint_names.index(name)

    def body_id2name(self, index: int) -> str | None:
        return self.body_names[index]

    def geom_id2name(self, index: int) -> str | None:
        return self.geom_names[index]


class FakeContact:
    def __init__(self, geom1: int, geom2: int, dist: float):
        self.geom1 = geom1
        self.geom2 = geom2
        self.dist = dist


class FakeData:
    def __init__(self, model: FakeModel):
        self.qpos = [0.0] * model.nq
        self.qvel = [0.0] * model.nv
        self.ctrl = [0.0] * model.nu
        self.time = 3.5
        self.ncon = 0
        self.contact: list[FakeContact] = []
        self.body_xpos = [[0.0, 0.0, 0.0] for _ in model.body_names]
        self.body_xquat = [[0.0, 0.0, 0.0, 1.0] for _ in model.body_names]
        self._data = None
        self.forward_calls = 0

    def forward(self) -> None:
        self.forward_calls += 1


class FakeSim:
    def __init__(self, model: FakeModel, data: FakeData):
        self.model = model
        self.data = data

    def forward(self) -> None:
        self.data.forward()


class FakePart:
    def __init__(self):
        self.goal_pos = [0.0, 0.0, 0.0]
        self.goal_ori = [0.0] * 9
        self.calls: list[str] = []

    def update(self, force: bool = False):
        self.calls.append(f"update:{bool(force)}")

    def reset_goal(self):
        self.calls.append("reset_goal")


class FakeController:
    def __init__(self):
        self.parts = {"arm": FakePart(), "gripper": FakePart()}
        self.part_controllers = self.parts
        self.calls: list[str] = []

    def update_state(self):
        self.calls.append("update_state")

    def reset(self):
        self.calls.append("reset")


class FakeRobotModel:
    naming_prefix = "robot0"


class FakeRobot:
    def __init__(self):
        self.robot_model = FakeRobotModel()
        self.robot_joints = list(ARM)
        self.gripper_joints = {"right": list(FINGERS)}
        self.gripper = {"right": type("G", (), {"naming_prefix": "gripper0_right"})()}
        self.composite_controller = FakeController()
        self.controller = None


class FakeEnv:
    def __init__(self):
        self.model = FakeModel()
        self.data = FakeData(self.model)
        self.sim = FakeSim(self.model, self.data)
        self.robots = [FakeRobot()]
        self.observation = {
            "robot0_eef_pos": [0.0, -0.30, 1.10],
            "robot0_eef_quat": [0.0, 0.0, 0.0, 1.0],
            "robot0_gripper_qpos": [-0.02, 0.02],
        }
        self.post_process_calls = 0
        self.observable_calls = 0

    def _get_observations(self, force_update: bool = False):
        return dict(self.observation)

    def _post_process(self):
        self.post_process_calls += 1

    def _update_observables(self, force: bool = False):
        self.observable_calls += 1

    # ---- helpers for the tests ----
    def qpos_of(self, name: str) -> float:
        index = self.model.joint_name2id(name)
        return self.data.qpos[self.model.jnt_qposadr[index]]

    def set_qpos(self, name: str, value: float) -> None:
        index = self.model.joint_name2id(name)
        self.data.qpos[self.model.jnt_qposadr[index]] = value


class RobotIndexTest(unittest.TestCase):
    def setUp(self):
        self.env = FakeEnv()

    def test_robot_joints_come_from_address_tables_not_slices(self):
        indexes = es._robot_joint_indexes(self.env, self.env.model)
        self.assertTrue(indexes["ok"], indexes)
        self.assertEqual(indexes["source"], "robot_joints+gripper_joints")
        self.assertEqual(sorted(indexes["joint_names"]), sorted(ROBOT_JOINTS))
        self.assertEqual(len(indexes["qpos_indexes"]), 9)
        expected = []
        for name in ROBOT_JOINTS:
            joint = self.env.model.joint_name2id(name)
            expected.append(self.env.model.jnt_qposadr[joint])
        self.assertEqual(indexes["qpos_indexes"], expected)
        # The fixture and the free object keep their own addresses.
        self.assertNotIn(0, indexes["qpos_indexes"])
        self.assertNotIn(self.env.model.jnt_qposadr[self.env.model.joint_name2id(
            "akita_black_bowl_1_joint0")], indexes["qpos_indexes"])

    def test_gripper_joints_are_inside_the_restored_set(self):
        indexes = es._robot_joint_indexes(self.env, self.env.model)
        names = set(indexes["joint_names"])
        for name in FINGERS:
            self.assertIn(name, names)
        for actuator in indexes["actuator_ids"]:
            self.assertIn(
                int(self.env.model.actuator_trnid[actuator][0]),
                set(self.env.model.robot_joint_ids()),
            )
        self.assertEqual(len(indexes["actuator_ids"]), len(ROBOT_JOINTS))

    def test_a_free_joint_declared_as_robot_is_refused(self):
        self.env.robots[0].robot_joints = list(ARM) + ["akita_black_bowl_1_joint0"]
        indexes = es._robot_joint_indexes(self.env, self.env.model)
        self.assertFalse(indexes["ok"])
        self.assertEqual(indexes["reason_code"], "robot_free_joint")

    def test_missing_robot_joint_is_refused_not_silently_dropped(self):
        self.env.robots[0].robot_joints = list(ARM) + ["robot0_joint_does_not_exist"]
        indexes = es._robot_joint_indexes(self.env, self.env.model)
        self.assertFalse(indexes["ok"])
        self.assertEqual(indexes["reason_code"], "robot_joint_missing_from_model")


class NonRobotStateTest(unittest.TestCase):
    def test_nonrobot_state_covers_objects_and_fixtures_but_not_the_arm(self):
        env = FakeEnv()
        env.set_qpos("wooden_cabinet_1_middle_level", -0.21)
        env.set_qpos("akita_black_bowl_1_joint0", 0.4)
        indexes = es._robot_joint_indexes(env, env.model)
        state = es._nonrobot_state(
            env.model, env.data, set(indexes["qpos_indexes"])
        )
        self.assertEqual(state["joints"]["wooden_cabinet_1_middle_level"], [-0.21])
        self.assertIn("akita_black_bowl_1_joint0", state["joints"])
        self.assertEqual(len(state["joints"]["akita_black_bowl_1_joint0"]), 7)
        for name in ROBOT_JOINTS:
            self.assertNotIn(name, state["joints"])
        self.assertIn("akita_black_bowl_1_joint0", state["free_bodies"])

    def test_digest_changes_when_a_scene_joint_moves(self):
        env = FakeEnv()
        indexes = es._robot_joint_indexes(env, env.model)
        robot = set(indexes["qpos_indexes"])
        before = es._nonrobot_state(env.model, env.data, robot)["digest"]
        again = es._nonrobot_state(env.model, env.data, robot)["digest"]
        self.assertEqual(before, again)
        env.set_qpos("wooden_cabinet_1_middle_level", -0.3)
        self.assertNotEqual(before, es._nonrobot_state(env.model, env.data, robot)["digest"])


class RestoreTest(unittest.TestCase):
    def setUp(self):
        self.env = FakeEnv()
        self.env.set_qpos("wooden_cabinet_1_middle_level", -0.21)
        self.env.set_qpos("akita_black_bowl_1_joint0", 0.4)
        for offset, name in enumerate(ARM):
            self.env.set_qpos(name, 0.1 * (offset + 1))
        self.env.set_qpos(FINGERS[0], -0.02)
        self.env.set_qpos(FINGERS[1], 0.02)
        reply = es._robot_capture(
            self.env, {"capture_id": "rs1", "reason": "episode_init", "sim_action_count": 7}
        )
        self.assertTrue(reply["ok"], reply)
        self.capture_id = "rs1"

    def _request(self, **overrides):
        body = {
            "capture_id": self.capture_id,
            "penetration_tol_m": 0.002,
            "reject_gripper_contact": True,
            "require_collision_check": True,
        }
        body.update(overrides)
        return body

    def _move_arm_away(self):
        for offset, name in enumerate(ARM):
            self.env.set_qpos(name, 1.0 + 0.1 * offset)
        self.env.data.qvel[0] = 3.0

    def test_restore_writes_only_robot_joints_and_zeroes_their_velocity(self):
        object_before = list(self.env.data.qpos[
            self.env.model.jnt_qposadr[self.env.model.joint_name2id("akita_black_bowl_1_joint0")]:
            self.env.model.jnt_qposadr[self.env.model.joint_name2id("akita_black_bowl_1_joint0")] + 7
        ])
        drawer_before = self.env.qpos_of("wooden_cabinet_1_middle_level")
        top_before = self.env.qpos_of("wooden_cabinet_1_top_level")
        self._move_arm_away()
        time_before = self.env.data.time
        reply = es._robot_restore(self.env, self._request())
        # The raw-mujoco probe is unavailable in this fake, so a build that
        # requires the check must refuse instead of claiming safety.
        self.assertFalse(reply["ok"])
        self.assertEqual(reply["reason_code"], "raw_mujoco_unavailable")
        self.assertEqual(self.env.qpos_of(ARM[0]), 1.0)
        self.assertEqual(self.env.data.time, time_before)

        # With the check not required, the write itself must still be exact.
        reply = es._robot_restore(self.env, self._request(require_collision_check=False))
        self.assertTrue(reply["ok"], reply)
        verification = reply["block"]["verification"]
        self.assertLessEqual(verification["robot_qpos_error"], 1e-9)
        self.assertLessEqual(verification["robot_qvel_abs_max"], 1e-9)
        self.assertTrue(verification["non_robot_preserved"])
        self.assertTrue(verification["scene_time_unchanged"])
        for offset, name in enumerate(ARM):
            self.assertAlmostEqual(self.env.qpos_of(name), 0.1 * (offset + 1), places=9)
        self.assertEqual(self.env.qpos_of(FINGERS[0]), -0.02)
        self.assertEqual(self.env.qpos_of(FINGERS[1]), 0.02)
        # Nothing else moved.
        self.assertEqual(self.env.qpos_of("wooden_cabinet_1_middle_level"), drawer_before)
        self.assertEqual(self.env.qpos_of("wooden_cabinet_1_top_level"), top_before)
        object_after = self.env.data.qpos[
            self.env.model.jnt_qposadr[self.env.model.joint_name2id("akita_black_bowl_1_joint0")]:
            self.env.model.jnt_qposadr[self.env.model.joint_name2id("akita_black_bowl_1_joint0")] + 7
        ]
        self.assertEqual(object_before, list(object_after))
        self.assertEqual(self.env.data.time, time_before)

    def test_restore_syncs_controllers_and_refreshes_the_observation(self):
        self._move_arm_away()
        reply = es._robot_restore(self.env, self._request(require_collision_check=False))
        self.assertTrue(reply["ok"], reply)
        robot = self.env.robots[0]
        self.assertEqual(robot.composite_controller.calls, ["update_state", "reset"])
        for part in robot.composite_controller.parts.values():
            self.assertEqual(part.calls, ["update:True", "reset_goal"])
        self.assertGreaterEqual(self.env.post_process_calls, 1)
        self.assertGreaterEqual(self.env.observable_calls, 1)
        self.assertGreaterEqual(self.env.data.forward_calls, 1)

    def test_gripper_in_contact_refuses_before_any_write(self):
        self._move_arm_away()
        self.env.data.ncon = 1
        self.env.data.contact = [FakeContact(1, 3, 0.0)]  # finger vs handle geom
        reply = es._robot_restore(self.env, self._request(require_collision_check=False))
        self.assertFalse(reply["ok"])
        self.assertEqual(reply["reason_code"], "gripper_contact")
        self.assertEqual(self.env.qpos_of(ARM[0]), 1.0)
        self.assertEqual(self.env.data.qvel[0], 3.0)

    def test_candidate_penetration_refuses_and_leaves_the_live_state_alone(self):
        self._move_arm_away()
        original = es._candidate_collisions
        es._candidate_collisions = lambda env, capture, tol: {
            "checked": True,
            "isolated_copy": True,
            "penetration_max_m": 0.05,
            "tol_m": tol,
            "refused": True,
            "robot_vs_nonrobot": [],
            "contact_count": 1,
        }
        try:
            reply = es._robot_restore(self.env, self._request())
        finally:
            es._candidate_collisions = original
        self.assertFalse(reply["ok"])
        self.assertEqual(reply["reason_code"], "candidate_penetration")
        self.assertEqual(self.env.qpos_of(ARM[0]), 1.0)
        self.assertEqual(self.env.data.forward_calls, 0)

    def test_unchecked_candidate_is_a_recorded_gap_not_a_claim(self):
        self._move_arm_away()
        original = es._candidate_collisions
        es._candidate_collisions = lambda env, capture, tol: {
            "checked": False,
            "reason_code": "candidate_forward_failed",
            "reason": "mj_forward refused",
        }
        try:
            reply = es._robot_restore(self.env, self._request())
        finally:
            es._candidate_collisions = original
        self.assertFalse(reply["ok"])
        self.assertEqual(reply["reason_code"], "candidate_forward_failed")
        self.assertIn("collision safe", reply["reason"])
        self.assertIs(reply["block"]["candidate"]["checked"], False)

    def test_unknown_capture_is_refused(self):
        reply = es._robot_restore(self.env, self._request(capture_id="nope"))
        self.assertFalse(reply["ok"])
        self.assertEqual(reply["reason_code"], "capture_unknown")

    def test_write_verification_failure_is_reported(self):
        # A sim that does not keep the written value models a masked write.
        self._move_arm_away()
        data = self.env.data
        original_qpos = data.qpos

        class Reverting(list):
            def __setitem__(self, index, value):
                return super().__setitem__(index, original_qpos[index])

        data.qpos = Reverting(original_qpos)
        reply = es._robot_restore(self.env, self._request(require_collision_check=False))
        self.assertFalse(reply["ok"])
        self.assertEqual(reply["reason_code"], "write_verification_failed")


if __name__ == "__main__":
    unittest.main()
