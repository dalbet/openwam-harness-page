"""Latest-frame queue for display snapshots.

The environment thread only copies a snapshot. Encoding and websocket sends
run here, and a newer snapshot replaces one that has not been sent.
"""

from __future__ import annotations

import io
import threading
import time
from typing import Any, Callable


class FrameBus:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest: dict[str, Any] | None = None
        self._event = threading.Event()
        self._stop = False
        self._sender: Callable[[bytes, bytes, dict], None] | None = None
        self.published = 0
        self.sent = 0
        self.dropped = 0
        self.skipped = 0
        self.gaps: list[float] = []
        self._last_sent = None
        self._thread = threading.Thread(target=self._loop, name="display-frames", daemon=True)
        self._thread.start()

    def set_sender(self, sender: Callable[[bytes, bytes, dict], None] | None) -> None:
        self._sender = sender

    def publish(self, agent, wrist, meta: dict) -> None:
        """Queue one display snapshot. The caller must already own the arrays."""
        with self._lock:
            if self._latest is not None:
                self.dropped += 1
            self._latest = {"agent": agent, "wrist": wrist, "meta": dict(meta)}
            self.published += 1
        self._event.set()

    def stats(self) -> dict[str, Any]:
        with self._lock:
            gaps = list(self.gaps)
        return {
            "published": self.published,
            "sent": self.sent,
            "dropped": self.dropped,
            "skipped": self.skipped,
            "max_gap_s": None if not gaps else round(max(gaps), 4),
            "gaps_s": [round(item, 4) for item in gaps[-12:]],
        }

    def close(self) -> None:
        self._stop = True
        self._event.set()

    def _loop(self) -> None:
        while not self._stop:
            self._event.wait(0.5)
            self._event.clear()
            while not self._stop:
                with self._lock:
                    snap = self._latest
                    self._latest = None
                if snap is None:
                    break
                try:
                    agent = _jpeg(snap["agent"])
                    wrist = _jpeg(snap["wrist"])
                except Exception:
                    continue
                now = time.time()
                if self._last_sent is not None:
                    self.gaps.append(now - self._last_sent)
                self._last_sent = now
                self.sent += 1
                sender = self._sender
                if sender is not None:
                    try:
                        sender(agent, wrist, snap["meta"])
                    except Exception:
                        continue


def _jpeg(image) -> bytes:
    import numpy as np
    from PIL import Image

    array = np.asarray(image)
    if array.ndim == 3 and array.shape[0] in {1, 3} and array.shape[-1] not in {1, 3}:
        array = np.moveaxis(array, 0, -1)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    picture = Image.fromarray(array).convert("RGB")
    picture.thumbnail((512, 512))
    buf = io.BytesIO()
    picture.save(buf, format="JPEG", quality=70)
    return buf.getvalue()
