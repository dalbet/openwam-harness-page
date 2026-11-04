"""Pi0.5 observation/action contract — deliberately separate from OpenWAM's.

Nothing here is shared with :mod:`adapters.openwam_contract`.  The two models
were trained on different LIBERO conventions and the point of keeping them
apart is that neither can silently inherit the other's:

    OpenWAM   proprio = EEF10  [eef_pos(3), rot6d(6), grip(1)]
              images  = 256->320x256 / 160x128 LANCZOS L-shape slots, rotated 180
              action  = native EEF10, converted to the 7-D OSC command

    Pi0.5     proprio = LIBERO 8-D [eef_pos(3), axis_angle(3), gripper_qpos(2)]
              images  = agentview + eye-in-hand, rotated 180, resized 224x224
              action  = already the 7-D OSC command; only clipped, never rotated
                        or re-scaled

The 8-D proprio is the same vector RLinf's LIBERO wrapper publishes, which is
what the production Pi0.5 server already consumes, so this path reproduces the
existing deployment rather than inventing a new one.

**Why the resize happens here and not in the backends.**  The two backends
disagree about input size: openpi resizes internally only when the image is not
already 224x224, and FlashRT copies straight into a fixed 224x224 buffer (it
would raise, not resize).  Doing one shared 256->224 bilinear resize before
either backend means both see byte-identical pixels *and* openpi's own resize
becomes a no-op, so the comparison is not contaminated by two different
resamplers.  ``resize_pi05`` is the only place that decides this.
"""

from __future__ import annotations

import numpy as np

from adapters.openwam_contract import quat_xyzw_to_axis_angle

PI05_IMAGE_SIZE = 224
PI05_PROPRIO_DIM = 8
LIBERO_ACTION7_DIM = 7
# pi05_libero: action_horizon = 10 (what the model was trained to emit).
PI05_GENERATE_HORIZON = 10
# The project's Pi0.5 deployment executes 5 of the 10 before replanning; S1 and
# S2 use the same number so the comparison is about the backend, not the window.
PI05_EXECUTE_HORIZON = 5


def rotate_180(image: np.ndarray) -> np.ndarray:
    """LIBERO renders upside-down relative to the Pi0.5 training reader."""
    value = np.asarray(image)
    if value.ndim != 3 or value.shape[-1] != 3:
        raise ValueError(f"camera image must be HxWx3 RGB, got {value.shape}")
    return np.ascontiguousarray(value[::-1, ::-1])


def resize_pi05(image: np.ndarray, size: int = PI05_IMAGE_SIZE) -> np.ndarray:
    """Single shared 256->224 bilinear resize (no-op when already 224)."""
    value = np.asarray(image, dtype=np.uint8)
    if value.shape[0] == size and value.shape[1] == size:
        return np.ascontiguousarray(value)
    from PIL import Image

    pil = Image.fromarray(np.ascontiguousarray(value))
    pil = pil.resize((size, size), Image.Resampling.BILINEAR)
    return np.asarray(pil, dtype=np.uint8)


def libero_proprio(raw_obs: dict) -> np.ndarray:
    """8-D LIBERO proprio: eef_pos(3) + axis_angle(3) + gripper_qpos(2)."""
    pos = np.asarray(raw_obs["robot0_eef_pos"], dtype=np.float64).reshape(-1)[:3]
    quat = np.asarray(raw_obs["robot0_eef_quat"], dtype=np.float64).reshape(-1)[:4]
    grip = np.asarray(raw_obs["robot0_gripper_qpos"], dtype=np.float64).reshape(-1)[:2]
    axis = np.asarray(quat_xyzw_to_axis_angle(quat), dtype=np.float64).reshape(-1)[:3]
    state = np.concatenate([pos, axis, grip]).astype(np.float32)
    if state.shape != (PI05_PROPRIO_DIM,):
        raise ValueError(f"pi05 proprio must be 8-D, got {state.shape}")
    return state


def build_pi05_obs(raw_obs: dict, *, rotate: bool = True, size: int = PI05_IMAGE_SIZE) -> dict:
    """One LIBERO raw observation -> the Pi0.5 model input contract."""
    for key in ("agentview_image", "robot0_eye_in_hand_image",
                "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"):
        if key not in raw_obs:
            raise KeyError(f"raw LIBERO obs is missing {key}")
    head = np.asarray(raw_obs["agentview_image"], dtype=np.uint8)
    wrist = np.asarray(raw_obs["robot0_eye_in_hand_image"], dtype=np.uint8)
    if rotate:
        head = rotate_180(head)
        wrist = rotate_180(wrist)
    return {
        "main_images": resize_pi05(head, size),
        "wrist_images": resize_pi05(wrist, size),
        "states": libero_proprio(raw_obs),
        "contract": {"images": 2, "size": [size, size], "rotate_180": bool(rotate),
                     "proprio": "libero8d", "resize": "PIL_BILINEAR_shared"},
    }


def pi05_action7(raw_action) -> np.ndarray:
    """Pi0.5 already emits the 7-D OSC command; only clip to LIBERO's box.

    LIBERO's OSC_POSE controller expects [dx, dy, dz, dax, day, daz, grip] with
    +1 = close, which is exactly the action space pi05_libero was trained on.
    No rotation/6-D/re-scale conversion is applied — that would be the OpenWAM
    contract and would silently change the policy's command.
    """
    value = np.asarray(raw_action, dtype=np.float64).reshape(-1)
    if value.shape[0] < LIBERO_ACTION7_DIM:
        raise ValueError(f"pi05 action must have >= 7 dims, got {value.shape}")
    return np.clip(value[:LIBERO_ACTION7_DIM], -1.0, 1.0)


__all__ = [
    "LIBERO_ACTION7_DIM",
    "PI05_EXECUTE_HORIZON",
    "PI05_GENERATE_HORIZON",
    "PI05_IMAGE_SIZE",
    "PI05_PROPRIO_DIM",
    "build_pi05_obs",
    "libero_proprio",
    "pi05_action7",
    "resize_pi05",
    "rotate_180",
]
