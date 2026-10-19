"""Free-mode native success is an event on the environment service.

Benchmark mode still publishes the gym terminated flag and the client keeps
its latch. Free mode records the native goal and does not turn that event
into an episode end. Truncation, a real service error, and an explicit stop
still end the episode. This module does not clear a client latch and does
not reset the scene.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
MODE_PATH = ROOT / "records" / "episode_mode.json"


def publish_mode(mode: str) -> None:
    if mode not in {"benchmark", "free"}:
        raise ValueError(mode)
    MODE_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODE_PATH.write_text(json.dumps({"mode": mode}, ensure_ascii=False), encoding="utf-8")


def read_mode() -> str:
    try:
        payload = json.loads(MODE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "benchmark"
    mode = payload.get("mode") if isinstance(payload, dict) else None
    return mode if mode in {"benchmark", "free"} else "benchmark"


def classify_step(terminated: bool, truncated: bool, mode: str) -> dict[str, Any]:
    """Translate one gym step into the flag the RPC client is allowed to latch."""
    if mode not in {"benchmark", "free"}:
        mode = "benchmark"
    if truncated:
        return {
            "terminated": False,
            "truncated": True,
            "native_goal": bool(terminated),
            "continuable": False,
            "kind": "time_limit",
        }
    if terminated and mode == "free":
        return {
            "terminated": False,
            "truncated": False,
            "native_goal": True,
            "continuable": True,
            "kind": "native_goal",
        }
    if terminated:
        return {
            "terminated": True,
            "truncated": False,
            "native_goal": True,
            "continuable": False,
            "kind": "native_goal",
        }
    return {
        "terminated": False,
        "truncated": False,
        "native_goal": False,
        "continuable": True,
        "kind": "running",
    }


def install() -> None:
    """Wrap the LIBERO facade and the worker env class as they are defined.

    The env server runs as ``__main__``, so an import hook never sees it.
    ``__build_class__`` runs in that process and in the spawned env worker.
    """
    if os.environ.get("SEQUENTIAL_EPISODE_SERVICE") == "0":
        return
    import builtins

    if getattr(builtins, "_sequential_episode_hook", False):
        return
    real_import = builtins.__import__
    real_build = builtins.__build_class__

    def hooked(name, globals=None, locals=None, fromlist=(), level=0):
        module = real_import(name, globals, locals, fromlist, level)
        module_name = str(getattr(module, "__name__", ""))
        if module_name.endswith("env_server") or module_name == "__main__":
            facade = module.__dict__.get("LiberoEnvFacade")
            if isinstance(facade, type):
                _wrap(facade)
        if module_name.endswith("envs.env_wrapper"):
            _attach_named_poses(module)
        return module

    def build(func, name, *bases, **kwargs):
        cls = real_build(func, name, *bases, **kwargs)
        if name == "LiberoEnvFacade":
            _wrap(cls)
        if name == "ControlEnv":
            _attach_named_poses_class(cls)
        return cls

    builtins.__import__ = hooked
    builtins.__build_class__ = build
    builtins._sequential_episode_hook = True


def _wrap(cls) -> None:
    if getattr(cls, "_sequential_episode_wrapped", False):
        return
    original_step = cls.step
    original_chunk = cls.chunk_step
    original_reset = cls.reset
    original_register = cls._register_rpc

    def step(self, action):
        obs, reward, term, trunc, info = original_step(self, action)
        return _publish(self, obs, reward, term, trunc, info)

    def chunk_step(self, actions, *args, **kwargs):
        obs, reward, term, trunc, info = original_chunk(self, actions, *args, **kwargs)
        return _publish(self, obs, reward, term, trunc, info)

    def reset(self, *args, **kwargs):
        result = original_reset(self, *args, **kwargs)
        self._sequential_native_goal = False
        self._sequential_truncated = False
        return result

    def native_goal_status(self):
        return {
            "native_goal": bool(getattr(self, "_sequential_native_goal", False)),
            "truncated": bool(getattr(self, "_sequential_truncated", False)),
            "mode": read_mode(),
        }

    def _register_rpc(self):
        original_register(self)
        self._rpc["env.native_goal"] = self.native_goal_status
        self._rpc["env.named_poses"] = self.named_poses
        self._readonly_methods.add("env.native_goal")
        self._readonly_methods.add("env.named_poses")

    def named_poses(self):
        return _read_worker_poses(self)

    step._sequential_episode_wrapped = True
    cls.step = step
    cls.named_poses = named_poses
    cls.chunk_step = chunk_step
    cls.reset = reset
    cls.native_goal_status = native_goal_status
    cls._register_rpc = _register_rpc
    cls._sequential_episode_wrapped = True


def _attach_named_poses(module) -> None:
    """Add an exact body/site snapshot on the class the env worker constructs."""
    cls = module.__dict__.get("ControlEnv") or module.__dict__.get("OffScreenRenderEnv")
    if isinstance(cls, type):
        _attach_named_poses_class(cls)


def _attach_named_poses_class(cls) -> None:
    if not isinstance(cls, type) or getattr(cls, "_sequential_named_poses", False):
        return

    def sequential_named_poses(self):
        try:
            sim = self.sim
            model = sim.model
            data = sim.data
            bodies = _indexed_xyz(model, data, "nbody", "body_xpos", "body_names", getattr(model, "body_id2name", None))
            sites = _indexed_xyz(model, data, "nsite", "site_xpos", "site_names", getattr(model, "site_id2name", None))
            geometry = _site_geometry(model, data)
            return {"bodies": bodies, "sites": sites, "site_geometry": geometry, "evidence_source": "mujoco_site"}
        except Exception as exc:
            return {"bodies": {}, "sites": {}, "error": f"{type(exc).__name__}: {exc}"}

    cls.sequential_named_poses = property(sequential_named_poses)
    cls._sequential_named_poses = True


def _indexed_xyz(model, data, count_name, xpos_name, names_attr, id2name) -> dict[str, list[float]]:
    count = int(getattr(model, count_name))
    raw_names = getattr(model, names_attr, None)
    names = [str(item) for item in list(raw_names)[:count]] if raw_names is not None else []
    if len(names) < count and callable(id2name):
        names = []
        for index in range(count):
            try:
                names.append(str(id2name(index) or ""))
            except Exception:
                names.append("")
    xpos = getattr(data, xpos_name)
    found = {}
    for index, name in enumerate(names):
        if not name:
            continue
        try:
            row = xpos[index]
            found[name] = [float(row[0]), float(row[1]), float(row[2])]
        except Exception:
            continue
    return found


def _site_geometry(model, data) -> dict[str, dict]:
    """World pose of each site. quat is MuJoCo wxyz. size is the site half-extent."""
    from regions import mat_to_quat_wxyz

    count = int(getattr(model, "nsite"))
    names = _site_names(model, count)
    xpos = getattr(data, "site_xpos")
    xmat = getattr(data, "site_xmat")
    size = getattr(model, "site_size", None)
    found = {}
    for index, name in enumerate(names):
        if not name:
            continue
        try:
            mat = [float(v) for v in list(xmat[index])]
            row = xpos[index]
            half = [float(v) for v in list(size[index][:3])] if size is not None else None
            found[name] = {
                "xyz": [float(row[0]), float(row[1]), float(row[2])],
                "quat_wxyz": [round(v, 6) for v in mat_to_quat_wxyz(mat)],
                "mat": [round(v, 6) for v in mat],
                "size": None if half is None else [round(v, 6) for v in half],
            }
        except Exception:
            continue
    return found


def _site_names(model, count: int) -> list[str]:
    raw_names = getattr(model, "site_names", None)
    names = [str(item) for item in list(raw_names)[:count]] if raw_names is not None else []
    id2name = getattr(model, "site_id2name", None)
    if len(names) < count and callable(id2name):
        names = []
        for index in range(count):
            try:
                names.append(str(id2name(index) or ""))
            except Exception:
                names.append("")
    return names


def _read_worker_poses(facade) -> dict[str, Any]:
    """Ask the existing worker getattr command for the site snapshot. No env.step."""
    try:
        inner = getattr(facade, "_env", None)
        vector = getattr(inner, "env", None)
        workers = getattr(vector, "workers", None) or []
        if not workers:
            return {"bodies": {}, "sites": {}, "error": "named pose worker missing"}
        worker = workers[0]
        worker.parent_remote.send(["getattr", "sequential_named_poses"])
        payload = worker.parent_remote.recv()
    except Exception as exc:
        return {"bodies": {}, "sites": {}, "error": f"{type(exc).__name__}: {exc}"}
    if not isinstance(payload, dict):
        return {"bodies": {}, "sites": {}, "error": f"named pose payload is {type(payload).__name__}"}
    payload.setdefault("bodies", {})
    payload.setdefault("sites", {})
    return payload


def _publish(self, obs, reward, term, trunc, info):
    decision = classify_step(_any_flag(term), _any_flag(trunc), read_mode())
    if decision["native_goal"]:
        self._sequential_native_goal = True
    if decision["truncated"]:
        self._sequential_truncated = True
    native = bool(getattr(self, "_sequential_native_goal", False))
    if isinstance(info, dict):
        info_out = dict(info)
        info_out["native_goal"] = native
        info_out["episode_continuable"] = decision["continuable"]
        info_out["episode_kind"] = decision["kind"]
    elif isinstance(info, list):
        info_out = [dict(item) if isinstance(item, dict) else item for item in info]
        if info_out and isinstance(info_out[-1], dict):
            info_out[-1]["native_goal"] = native
            info_out[-1]["episode_continuable"] = decision["continuable"]
            info_out[-1]["episode_kind"] = decision["kind"]
    else:
        info_out = {"info": info, "native_goal": native, "episode_continuable": decision["continuable"], "episode_kind": decision["kind"]}
    if decision["kind"] == "running" or (decision["terminated"] == _any_flag(term) and decision["truncated"] == _any_flag(trunc)):
        return obs, reward, term, trunc, info_out
    return obs, reward, _replace_flag(term, decision["terminated"]), _replace_flag(trunc, decision["truncated"]), info_out


def _any_flag(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    if isinstance(value, (list, tuple)):
        return any(_any_flag(item) for item in value)
    any_fn = getattr(value, "any", None)
    if callable(any_fn):
        return bool(any_fn())
    return bool(value)


def _replace_flag(value, new: bool):
    zeros = getattr(value, "shape", None)
    if zeros is not None:
        import numpy as np

        return np.zeros_like(value, dtype=bool) if not new else np.ones_like(value, dtype=bool)
    if isinstance(value, list):
        return [new for _item in value]
    return new
