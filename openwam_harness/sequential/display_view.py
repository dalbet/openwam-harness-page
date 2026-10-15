"""One display transform for both cameras.

The page shows the environment camera buffer with a single vertical flip.
That is the same orientation as ``agentview.png`` / ``wrist.png``.
It is not the Pi0.5 input: ``get_libero_image`` rotates the same buffer 180 degrees.
"""

from __future__ import annotations

import io
from typing import Any

import numpy as np

DISPLAY_MAX_EDGE = 512
DISPLAY_JPEG_QUALITY = 85
DISPLAY_JPEG_SUBSAMPLING = 0
AGENT_KEY = "agentview_image"
WRIST_KEY = "robot0_eye_in_hand_image"


def prepare_camera(image, *, source: str) -> dict[str, Any]:
    """Copy one raw camera and flip it vertically once."""
    array = np.asarray(image)
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    if array.ndim == 3 and array.shape[0] in {1, 3} and array.shape[-1] not in {1, 3}:
        array = np.moveaxis(array, 0, -1)
    if array.shape[-1] == 4:
        array = array[..., :3]
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    flipped = np.ascontiguousarray(array[::-1])
    return {
        "array": flipped,
        "source": source,
        "height": int(flipped.shape[0]),
        "width": int(flipped.shape[1]),
        "channels": "RGB",
        "flip": "vertical",
        "transforms": 1,
    }


def prepare_display_pair(raw: dict) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read the two raw cameras. VLA ``main_images`` is not a display source."""
    if AGENT_KEY not in raw or WRIST_KEY not in raw:
        raise KeyError("raw_obs is missing agentview_image or robot0_eye_in_hand_image")
    return (
        prepare_camera(raw[AGENT_KEY], source=f"raw_obs.{AGENT_KEY}"),
        prepare_camera(raw[WRIST_KEY], source=f"raw_obs.{WRIST_KEY}"),
    )


def encode_display(view: dict[str, Any]) -> tuple[bytes, dict[str, Any]]:
    """JPEG the already-flipped view. Shrink only when an edge exceeds 512."""
    from PIL import Image

    array = view["array"]
    picture = Image.fromarray(array, mode="RGB")
    picture.thumbnail((DISPLAY_MAX_EDGE, DISPLAY_MAX_EDGE), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    picture.save(
        buf,
        format="JPEG",
        quality=DISPLAY_JPEG_QUALITY,
        subsampling=DISPLAY_JPEG_SUBSAMPLING,
    )
    meta = {
        "source": view["source"],
        "src_h": view["height"],
        "src_w": view["width"],
        "out_h": int(picture.size[1]),
        "out_w": int(picture.size[0]),
        "channels": view["channels"],
        "flip": view["flip"],
        "jpeg_quality": DISPLAY_JPEG_QUALITY,
        "jpeg_subsampling": DISPLAY_JPEG_SUBSAMPLING,
        "max_edge": DISPLAY_MAX_EDGE,
    }
    return buf.getvalue(), meta
