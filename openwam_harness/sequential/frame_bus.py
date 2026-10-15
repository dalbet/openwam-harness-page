"""Latest-frame queue for display snapshots.

The environment thread only copies a snapshot. Encoding and websocket sends
run here, and a newer snapshot replaces one that has not been sent.
"""

from __future__ import annotations

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
                    started = time.perf_counter()
                    agent, agent_meta = _encode(snap["agent"])
                    wrist, wrist_meta = _encode(snap["wrist"])
                    encode_s = time.perf_counter() - started
                except Exception:
                    continue
                snap["meta"]["encode_ms"] = round(encode_s * 1000, 2)
                snap["meta"]["agent"] = agent_meta
                snap["meta"]["wrist"] = wrist_meta
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


def _encode(view) -> tuple[bytes, dict]:
    from display_view import encode_display

    if isinstance(view, dict) and "array" in view:
        return encode_display(view)
    from display_view import prepare_camera

    return encode_display(prepare_camera(view, source="array"))
