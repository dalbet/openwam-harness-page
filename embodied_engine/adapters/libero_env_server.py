#!/usr/bin/env python
"""LIBERO env server for the embodied engine.

Thin launcher around RPent's ``robots.libero.env_server``. The runtime patch
in ``adapters/env_runtime`` is installed before the facade class is built, so
the server exposes ``env.state_snapshot`` / ``env.identity`` and publishes a
native LIBERO success as an event instead of latching termination in free
mode. The patch is a copy of the sequential system's episode_service with an
isolated mode file; see ``docs/architecture.md`` for provenance.

Launch:
    envs/rpent/bin/python adapters/libero_env_server.py \
        --suite libero_goal --task 1 --seed 0 --port 14561 --cuda-device 3
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNTIME = HERE / "env_runtime"
sys.path.insert(0, str(RUNTIME))

# These must be set before anything imports mujoco; the RPent server repeats
# the defaults but its module-level assert rejects an early mujoco import.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("ROBOT_PLATFORM", "LIBERO")
os.environ.setdefault("LIBERO_TYPE", "standard")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

# Worker processes spawned by RLinf inherit PYTHONPATH; putting the runtime
# directory there makes them load sitecustomize and attach the MuJoCo named
# pose snapshot. The parent process needs the same path for the facade wrap.
existing = os.environ.get("PYTHONPATH", "")
parts = [p for p in existing.split(os.pathsep) if p]
if str(RUNTIME) not in parts:
    os.environ["PYTHONPATH"] = os.pathsep.join([str(RUNTIME), *parts])

import episode_service  # noqa: E402

episode_service.install()

from robots.libero import env_server  # noqa: E402


if __name__ == "__main__":
    env_server.main()
