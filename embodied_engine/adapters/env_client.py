"""Client for the LIBERO env RPC server.

The wire format is RPent's HTTP JSON RPC (numpy arrays tagged as
``__ndarray__``). Only the methods the engine uses are wrapped here, so the
engine code does not depend on RPent's toolkit layer.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

RPENT_ROOT = Path(
    "/home/host/page-dmh/openwam_harness/RPent"
)
if str(RPENT_ROOT) not in sys.path:
    sys.path.insert(0, str(RPENT_ROOT))

from rpent.utils.rpc.http_rpc import HttpRpcClient  # noqa: E402


class EnvClient:
    """Single-env LIBERO RPC client with the engine's action contract."""

    def __init__(self, base_url: str):
        self._rpc = HttpRpcClient(base_url)
        self._base_url = base_url

    def identity(self) -> dict[str, Any]:
        return self._rpc.call("env.identity", timeout_s=15)

    def meta(self) -> dict[str, Any]:
        return self._rpc.call("env.get_env_meta", timeout_s=15)

    def reset(self) -> dict[str, Any]:
        obs, _info = self._rpc.call("env.reset", timeout_s=120)
        return obs

    def step(self, action) -> tuple[dict, float, bool, bool, dict]:
        obs, reward, term, trunc, info = self._rpc.call(
            "env.step", (np.asarray(action, dtype=np.float64),), timeout_s=60
        )
        return obs, float(reward), bool(term), bool(trunc), info

    def state_snapshot(self) -> dict[str, Any]:
        return self._rpc.call("env.state_snapshot", timeout_s=60)

    def raw_obs(self) -> dict[str, Any]:
        return self._rpc.call("env.raw_obs", timeout_s=60)

    def robot_state_capture(self, reason: str = "episode_init") -> dict[str, Any]:
        """Freeze the robot's own state inside the env service.

        Returns the stamped evidence row: the robot joint spans, its
        controllers, the untouched non-robot digest, and the capture id a
        later restore must name. Nothing is written.
        """
        return self._rpc.call("env.robot_state_capture", (str(reason),), timeout_s=60)

    def robot_state_restore(self, request: dict[str, Any]) -> dict[str, Any]:
        """Restore one captured robot state; the scene is not touched.

        ``sim_action_count`` does not move: this is a state write, not an
        action. Settle actions, if any, are ordinary ``step`` calls made by
        the caller and are counted separately.
        """
        return self._rpc.call("env.robot_state_restore", (dict(request),), timeout_s=120)

    def robot_state_inspect(self) -> dict[str, Any]:
        """Read the robot state plus its contact list; nothing is written."""
        return self._rpc.call("env.robot_state_inspect", timeout_s=60)

    def render_camera(
        self, camera_name: str, height: int, width: int, depth: bool = False
    ):
        return self._rpc.call(
            "env.render_camera",
            (camera_name,),
            {"height": height, "width": width, "depth": depth},
            timeout_s=120,
        )

    def close(self) -> None:
        try:
            self._rpc.call("env.close", timeout_s=30)
        except Exception:
            pass

    @staticmethod
    def observation_id(obs: dict | None) -> str | None:
        if not isinstance(obs, dict):
            return None
        value = obs.get("observation_id") or obs.get("seq_id")
        return value if isinstance(value, str) else None
