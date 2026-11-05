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


def _flag_value(flag: str) -> str | None:
    argv = sys.argv[1:]
    for index, item in enumerate(argv):
        if item == flag and index + 1 < len(argv):
            return argv[index + 1]
        if item.startswith(f"{flag}="):
            return item.split("=", 1)[1]
    return None


# Tell the worker processes which stream the placement samplers must draw
# from, *before* any env is built. LIBERO's ``ObjectPositionSampler`` defaults
# ``rng`` to ``np.random.default_rng()`` (OS entropy), so ``np.random.seed()``
# in ``BDDLBaseDomain.seed`` never reached it; the fixtures were re-drawn on
# every reset. The seed travels as an env var because the workers are spawned
# by robosuite's vector env and inherit this process's environment.
# ``LIBERO_PLACEMENT_SEEDING=0`` reproduces the old unseeded behaviour for a
# control arm; it is recorded in ``env.identity`` either way.
_placement_seed = _flag_value("--seed")
if os.environ.get("LIBERO_PLACEMENT_SEEDING", "1") != "0" and _placement_seed is not None:
    os.environ["LIBERO_PLACEMENT_SEED"] = str(_placement_seed)
else:
    os.environ.pop("LIBERO_PLACEMENT_SEED", None)

from robots.libero import env_server  # noqa: E402


if __name__ == "__main__":
    env_server.main()
