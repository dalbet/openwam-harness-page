"""Official LIBERO observation/action contract for OpenWAM checkpoints.

OpenWAM-Alpha-Sim-LIBERO was trained from ``configs/dataloader/libero.yaml``
(``action_mode: eef``, ``unify_action: true`` with ``0-9`` → 80-D, 384x320
multi-view canvas, ``num_frames: 33`` → 32-step action horizon). Its deploy
client therefore speaks a fixed contract:

    observation → head 256x256 agentview rotated 180°, wrist 256x256 rotated
                  180° (down-sampled to 320x256 / 160x128 LANCZOS),
                  proprio = EEF10 = [eef_pos(3), rot6d(quat)(6), grip(1)]
    action      → raw native EEF10 = [d_xyz(3), rot6d(6), open_scale_grip(1)]
                  which LIBERO's 7-D OSC expects as
                  [d_xyz(3), d_axis_angle(3), +1=close grip]

The conversion functions below are ports of the upstream client
(``OpenWAM/benchmarks/utils/action_conversion.py`` and
``OpenWAM/benchmarks/libero/openwam2libero_interface.py``); the port is
verified numerically against those modules by
``tests/test_openwam_contract.py`` so a drift in either side is detectable.
Nothing here is Pi0.5-specific: Pi0.5 keeps its own 8-D proprio and 5-action
chunk path.
"""

from __future__ import annotations

from typing import Any

import numpy as np

LIBERO_EEF10_DIM = 10
LIBERO_ACTION7_DIM = 7
# Panda finger separation at fully open (m), same constant as upstream.
LIBERO_GRIPPER_WIDTH_OPEN = 0.08
# Upstream L-shape slots: head label is (W, H).
HEAD_SLOT = (320, 256)
WRIST_SLOT = (160, 128)


def quat_xyzw_to_rot6d(quat: np.ndarray) -> np.ndarray:
    """xyzw quaternion -> first two rotation-matrix columns (6-D)."""
    q = np.asarray(quat, dtype=np.float64)
    if q.shape[-1] != 4:
        raise ValueError(f"quat must end with dimension 4, got shape {q.shape}")
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    q = q / np.maximum(norm, 1e-8)
    x, y, z, w = np.moveaxis(q, -1, 0)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    col1 = np.stack([1.0 - 2.0 * (yy + zz), 2.0 * (xy + wz), 2.0 * (xz - wy)], axis=-1)
    col2 = np.stack([2.0 * (xy - wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz + wx)], axis=-1)
    return np.concatenate([col1, col2], axis=-1).astype(np.float32)


def rot6d_to_quat_xyzw(r6d: np.ndarray) -> np.ndarray:
    """6-D rotation -> xyzw quaternion (Gram-Schmidt + Shepperd)."""
    a1, a2 = np.asarray(r6d, dtype=np.float64)[:3], np.asarray(r6d, dtype=np.float64)[3:6]
    b1 = a1 / max(float(np.linalg.norm(a1)), 1e-8)
    b2 = a2 - float(np.dot(b1, a2)) * b1
    b2 = b2 / max(float(np.linalg.norm(b2)), 1e-8)
    b3 = np.cross(b1, b2)
    matrix = np.stack([b1, b2, b3], axis=1)
    t = matrix[0, 0] + matrix[1, 1] + matrix[2, 2]
    if t > 0:
        s = 0.5 / np.sqrt(t + 1.0)
        w = 0.25 / s
        x = (matrix[2, 1] - matrix[1, 2]) * s
        y = (matrix[0, 2] - matrix[2, 0]) * s
        z = (matrix[1, 0] - matrix[0, 1]) * s
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        s = 2.0 * np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])
        w = (matrix[2, 1] - matrix[1, 2]) / s
        x = 0.25 * s
        y = (matrix[0, 1] + matrix[1, 0]) / s
        z = (matrix[0, 2] + matrix[2, 0]) / s
    elif matrix[1, 1] > matrix[2, 2]:
        s = 2.0 * np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])
        w = (matrix[0, 2] - matrix[2, 0]) / s
        x = (matrix[0, 1] + matrix[1, 0]) / s
        y = 0.25 * s
        z = (matrix[1, 2] + matrix[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])
        w = (matrix[1, 0] - matrix[0, 1]) / s
        x = (matrix[0, 2] + matrix[2, 0]) / s
        y = (matrix[1, 2] + matrix[2, 1]) / s
        z = 0.25 * s
    return np.array([x, y, z, w], dtype=np.float32)


def quat_xyzw_to_axis_angle(quat: np.ndarray) -> np.ndarray:
    """xyzw quaternion -> axis-angle rotation vector (norm <= pi)."""
    q = np.asarray(quat, dtype=np.float64)
    q = q / max(float(np.linalg.norm(q)), 1e-8)
    if q[3] < 0:
        q = -q
    w = float(np.clip(q[3], -1.0, 1.0))
    den = float(np.sqrt(max(0.0, 1.0 - w * w)))
    if den < 1e-8:
        return np.zeros(3, dtype=np.float32)
    angle = 2.0 * float(np.arccos(w))
    return (q[:3] / den * angle).astype(np.float32)


def rot6d_to_axis_angle(r6d: np.ndarray) -> np.ndarray:
    return quat_xyzw_to_axis_angle(rot6d_to_quat_xyzw(np.asarray(r6d, dtype=np.float64).reshape(-1)))


def libero_gripper_qpos_to_cmd(width: float) -> float:
    """Achieved finger separation (m) -> trained open-scale (+1 = open)."""
    return float(np.clip(2.0 * float(width) / LIBERO_GRIPPER_WIDTH_OPEN - 1.0, -1.0, 1.0))


def libero_open_scale_to_gripper_cmd(value: float) -> float:
    """Trained open-scale (+1 = open) -> LIBERO command (+1 = close)."""
    return float(np.clip(-float(value), -1.0, 1.0))


def eef10_from_raw(eef_pos, eef_quat_xyzw, gripper_qpos) -> np.ndarray:
    """Live LIBERO raw obs -> the exact 10-D proprio the checkpoint trained on."""
    pos = np.asarray(eef_pos, np.float32).reshape(-1)
    quat = np.asarray(eef_quat_xyzw, np.float32).reshape(-1)
    qpos = np.asarray(gripper_qpos, np.float32).reshape(-1)
    if pos.shape[0] != 3 or quat.shape[0] != 4 or qpos.shape[0] != 2:
        raise ValueError(
            f"LIBERO proprio dims must be pos3/quatxyzw4/gripper2, got {pos.shape[0]}/{quat.shape[0]}/{qpos.shape[0]}"
        )
    width = float(qpos[0]) - float(qpos[1])
    grip = np.array([libero_gripper_qpos_to_cmd(width)], np.float32)
    return np.concatenate([pos, quat_xyzw_to_rot6d(quat[None])[0], grip]).astype(np.float32)


def native_eef10_to_libero7d(action, *, clip: bool = True) -> np.ndarray:
    """Raw native EEF10 command -> LIBERO 7-D OSC command (no pose composition)."""
    value = np.asarray(action, dtype=np.float32).reshape(-1)
    if value.shape != (LIBERO_EEF10_DIM,):
        raise ValueError(f"expected a native LIBERO EEF10 action, got {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError("native LIBERO EEF10 action contains NaN or infinity")
    position = value[:3].copy()
    rotation = rot6d_to_axis_angle(value[3:9]).astype(np.float32)
    gripper = np.array([libero_open_scale_to_gripper_cmd(float(value[9]))], dtype=np.float32)
    if clip:
        position = np.clip(position, -1.0, 1.0)
        rotation = np.clip(rotation, -1.0, 1.0)
        gripper = np.clip(gripper, -1.0, 1.0)
    return np.concatenate([position, rotation, gripper]).astype(np.float32)


def rotate_180(image) -> np.ndarray:
    """LIBERO renders upside-down relative to the training reader."""
    value = np.asarray(image)
    if value.ndim != 3 or value.shape[-1] != 3:
        raise ValueError(f"camera image must be HxWx3 RGB, got {value.shape}")
    return np.ascontiguousarray(value[::-1, ::-1])


def resize_slot(image, slot: str) -> np.ndarray:
    """LANCZOS resize to the upstream L-shape slot size (head/wrist)."""
    from PIL import Image

    sizes = {"head_camera": HEAD_SLOT, "left_wrist_camera": WRIST_SLOT, "right_wrist_camera": WRIST_SLOT}
    if slot not in sizes:
        raise ValueError(f"unknown L-shape slot {slot!r}; expected one of {tuple(sizes)}")
    value = np.asarray(image)
    if value.ndim != 3 or value.shape[-1] != 3:
        raise ValueError(f"{slot} image must be HxWx3 RGB, got {value.shape}")
    pil = Image.fromarray(np.ascontiguousarray(value.astype(np.uint8, copy=False)))
    target = sizes[slot]
    if pil.size != target:
        pil = pil.resize(target, Image.Resampling.LANCZOS)
    return np.asarray(pil, dtype=np.uint8)


def build_openwam_obs(raw_obs: dict[str, Any], *, rotate: bool = True) -> dict[str, Any]:
    """Turn one RPent/LIBERO raw obs into the OpenWAM observation contract.

    Returns the same field names the engine already uses for Pi0.5
    (``main_images`` / ``wrist_images`` / ``states``) so the inference service
    stays backend-agnostic; only the *contents* follow the OpenWAM contract:
    slot-sized rotated images and a 10-D EEF10 proprio.
    """
    for key in ("agentview_image", "robot0_eye_in_hand_image", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"):
        if key not in raw_obs:
            raise KeyError(f"raw LIBERO obs is missing {key}")
    head = np.asarray(raw_obs["agentview_image"], dtype=np.uint8)
    wrist = np.asarray(raw_obs["robot0_eye_in_hand_image"], dtype=np.uint8)
    if rotate:
        head = rotate_180(head)
        wrist = rotate_180(wrist)
    head = resize_slot(head, "head_camera")
    wrist = resize_slot(wrist, "left_wrist_camera")
    state = eef10_from_raw(raw_obs["robot0_eef_pos"], raw_obs["robot0_eef_quat"], raw_obs["robot0_gripper_qpos"])
    return {
        "main_images": head,
        "wrist_images": wrist,
        "states": state,
        "raw_proprio": {
            "eef_pos": [float(v) for v in np.asarray(raw_obs["robot0_eef_pos"]).reshape(-1)[:3]],
            "gripper_qpos": [float(v) for v in np.asarray(raw_obs["robot0_gripper_qpos"]).reshape(-1)[:2]],
        },
    }


__all__ = [
    "HEAD_SLOT",
    "LIBERO_ACTION7_DIM",
    "LIBERO_EEF10_DIM",
    "LIBERO_GRIPPER_WIDTH_OPEN",
    "WRIST_SLOT",
    "build_openwam_obs",
    "eef10_from_raw",
    "libero_gripper_qpos_to_cmd",
    "libero_open_scale_to_gripper_cmd",
    "native_eef10_to_libero7d",
    "quat_xyzw_to_axis_angle",
    "quat_xyzw_to_rot6d",
    "resize_slot",
    "rot6d_to_axis_angle",
    "rot6d_to_quat_xyzw",
    "rotate_180",
]
