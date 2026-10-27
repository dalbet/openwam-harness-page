#!/usr/bin/env python
"""Parity check: the engine's LIBERO adapter vs the upstream OpenWAM client.

Run with the OpenWAM inference environment (it has numpy/Pillow and the same
python version as the engine):

    .venv-openwam/bin/python tests/test_openwam_contract.py

The engine must speak exactly the contract the checkpoint was trained on. This
test imports the upstream modules from the vendored repo and compares both
observation construction (rotation + LANCZOS slot resize + EEF10 proprio) and
action conversion (native EEF10 -> LIBERO 7-D OSC) on random inputs. A drift on
either side fails here instead of silently degrading task success.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ENGINE_ROOT = Path(__file__).resolve().parents[1]
OPENWAM_ROOT = Path("/home/host/page-dmh/openwam_harness/OpenWAM")
sys.path.insert(0, str(ENGINE_ROOT))
sys.path.insert(0, str(OPENWAM_ROOT))

from adapters import openwam_contract as engine  # noqa: E402
from benchmarks.libero import openwam2libero_interface as upstream_libero  # noqa: E402
from benchmarks.utils import action_conversion as upstream_conv  # noqa: E402
from benchmarks.utils import client as upstream_client  # noqa: E402

RNG = np.random.default_rng(7)


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"  ok  {message}")


def test_quat_rot6d_roundtrip() -> None:
    print("[1] quaternion <-> rot6d <-> axis-angle")
    for _ in range(200):
        quat = RNG.normal(size=4)
        quat /= np.linalg.norm(quat)
        mine = engine.quat_xyzw_to_rot6d(quat[None])[0]
        theirs = upstream_conv.quat_xyzw_to_rot6d(quat[None])[0]
        check(np.allclose(mine, theirs, atol=1e-5), "rot6d matches upstream") if _ == 0 else None
        assert np.allclose(mine, theirs, atol=1e-5), "rot6d drift"
        back = engine.rot6d_to_quat_xyzw(mine)
        assert np.allclose(mine, engine.quat_xyzw_to_rot6d(back[None])[0], atol=2e-5), "roundtrip drift"
        aa_mine = engine.rot6d_to_axis_angle(mine)
        aa_theirs = upstream_conv.rot6d_to_axis_angle(mine)
        assert np.allclose(aa_mine, aa_theirs, atol=1e-5), f"axis-angle drift {aa_mine} vs {aa_theirs}"
    print("  200 random rotations: rot6d / quaternion / axis-angle agree")


def test_eef10_proprio() -> None:
    print("[2] raw LIBERO proprio -> EEF10")
    for _ in range(50):
        pos = RNG.normal(scale=0.3, size=3)
        quat = RNG.normal(size=4)
        quat /= np.linalg.norm(quat)
        width = RNG.uniform(0.0, 0.08)
        qpos = np.array([width / 2.0, -width / 2.0])
        mine = engine.eef10_from_raw(pos, quat, qpos)
        theirs = upstream_conv.libero_obs_to_eef10(pos, quat, qpos)
        check(mine.shape == (10,), "EEF10 is 10-D") if _ == 0 else None
        assert np.allclose(mine, theirs, atol=1e-5), f"proprio drift {mine} vs {theirs}"
    print("  50 random obs: EEF10 proprio agrees")


def test_action_conversion() -> None:
    print("[3] native EEF10 action -> LIBERO 7-D OSC")
    for _ in range(200):
        action = np.concatenate(
            [
                RNG.uniform(-1, 1, 3),
                engine.quat_xyzw_to_rot6d((lambda q: q / np.linalg.norm(q))(RNG.normal(size=4))[None])[0],
                [RNG.uniform(-1, 1)],
            ]
        ).astype(np.float32)
        mine = engine.native_eef10_to_libero7d(action)
        theirs = upstream_libero.native_eef10_to_libero7d(action)
        check(mine.shape == (7,), "converted action is 7-D") if _ == 0 else None
        assert np.allclose(mine, theirs, atol=1e-5), f"action drift {mine} vs {theirs}"
        assert np.all(np.abs(mine) <= 1.0), "action is inside the LIBERO command box"
    print("  200 random actions: 7-D OSC command agrees")


def test_image_contract() -> None:
    print("[4] observation images: 180 deg rotation + LANCZOS slot resize")
    head = RNG.integers(0, 256, size=(256, 256, 3), dtype=np.uint8)
    wrist = RNG.integers(0, 256, size=(256, 256, 3), dtype=np.uint8)
    raw = {
        "agentview_image": head,
        "robot0_eye_in_hand_image": wrist,
        "robot0_eef_pos": np.array([0.1, 0.2, 0.3]),
        "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
        "robot0_gripper_qpos": np.array([0.02, -0.02]),
    }
    obs = engine.build_openwam_obs(raw)
    expected_head = upstream_client.resize_for_lshape_slot(head[::-1, ::-1], "head_camera")
    expected_wrist = upstream_client.resize_for_lshape_slot(wrist[::-1, ::-1], "left_wrist_camera")
    check(obs["main_images"].shape == (256, 320, 3), "head slot is 320x256 (WxH)")
    check(obs["wrist_images"].shape == (128, 160, 3), "wrist slot is 160x128 (WxH)")
    assert np.array_equal(obs["main_images"], np.asarray(expected_head)), "head image drift"
    assert np.array_equal(obs["wrist_images"], np.asarray(expected_wrist)), "wrist image drift"
    check(obs["states"].shape == (10,), "state is 10-D EEF10")
    print("  canvas slots and pixels agree with the upstream client")


def main() -> int:
    print("engine adapter:", ENGINE_ROOT / "adapters" / "openwam_contract.py")
    print("upstream      :", OPENWAM_ROOT / "benchmarks")
    test_quat_rot6d_roundtrip()
    test_eef10_proprio()
    test_action_conversion()
    test_image_contract()
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
