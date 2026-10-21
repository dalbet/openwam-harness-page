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


def publish_mode(mode: str, task_scope: str = "official_single") -> None:
    if mode not in {"benchmark", "free"}:
        raise ValueError(mode)
    if task_scope not in {"official_single", "combo"}:
        raise ValueError(task_scope)
    MODE_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODE_PATH.write_text(
        json.dumps({"mode": mode, "task_scope": task_scope}, ensure_ascii=False),
        encoding="utf-8",
    )


def _mode_payload() -> dict[str, Any]:
    try:
        payload = json.loads(MODE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def read_mode() -> str:
    mode = _mode_payload().get("mode")
    return mode if mode in {"benchmark", "free"} else "benchmark"


def read_task_scope() -> str:
    scope = _mode_payload().get("task_scope")
    return scope if scope in {"official_single", "combo"} else "official_single"


def classify_step(terminated: bool, truncated: bool, mode: str, task_scope: str = "official_single") -> dict[str, Any]:
    """Translate one gym step into the flag the RPC client is allowed to latch."""
    if mode not in {"benchmark", "free"}:
        mode = "benchmark"
    if task_scope not in {"official_single", "combo"}:
        task_scope = "official_single"
    if truncated:
        return {
            "terminated": False,
            "truncated": True,
            "native_goal": bool(terminated),
            "continuable": False,
            "clear_terminated": False,
            "kind": "time_limit",
        }
    if terminated and mode == "free" and task_scope == "combo":
        return {
            "terminated": False,
            "truncated": False,
            "native_goal": True,
            "continuable": True,
            "clear_terminated": True,
            "kind": "native_goal",
        }
    if terminated:
        return {
            "terminated": True,
            "truncated": False,
            "native_goal": True,
            "continuable": False,
            "clear_terminated": False,
            "kind": "native_goal",
        }
    return {
        "terminated": False,
        "truncated": False,
        "native_goal": False,
        "continuable": True,
        "clear_terminated": False,
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
        _refuse_if_terminal(self, "env.step")
        obs, reward, term, trunc, info = original_step(self, action)
        published = _publish(self, obs, reward, term, trunc, info)
        return _stamp_action(self, published, 1)

    def chunk_step(self, actions, *args, **kwargs):
        _refuse_if_terminal(self, "env.chunk_step")
        obs, reward, term, trunc, info = original_chunk(self, actions, *args, **kwargs)
        published = _publish(self, obs, reward, term, trunc, info)
        count = len(actions) if isinstance(actions, (list, tuple)) else 1
        return _stamp_action(self, published, count)

    def reset(self, *args, **kwargs):
        result = original_reset(self, *args, **kwargs)
        _ensure_identity(self)
        self._sequential_native_goal = False
        self._sequential_truncated = False
        self._sequential_refusing_step = False
        self._scene_generation = int(getattr(self, "_scene_generation", 0)) + 1
        self._sim_action_count = 0
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
        self._rpc["env.state_snapshot"] = self.state_snapshot
        self._rpc["env.identity"] = self.identity
        self._readonly_methods.add("env.native_goal")
        self._readonly_methods.add("env.named_poses")
        self._readonly_methods.add("env.state_snapshot")
        self._readonly_methods.add("env.identity")

    def named_poses(self):
        return _read_worker_poses(self)

    def identity(self):
        _ensure_identity(self)
        return {
            "service_instance_id": self._service_instance_id,
            "scene_generation": self._scene_generation,
            "sim_action_count": self._sim_action_count,
            "task_scope": read_task_scope(),
            "mode": read_mode(),
            "code": "episode-service-state-1",
        }

    def state_snapshot(self):
        """Read current sim state on the worker. Does not call env.step or move the gripper."""
        _ensure_identity(self)
        self._query_seq = int(getattr(self, "_query_seq", 0)) + 1
        poses = _read_worker_poses(self)
        from state_contract import stamp_snapshot

        sim = poses.get("sim") if isinstance(poses, dict) and isinstance(poses.get("sim"), dict) else {"evidence_source": "readonly_rpc", "objects": {}}
        named = {
            key: poses.get(key)
            for key in ("bodies", "sites", "site_geometry", "evidence_source", "error")
            if isinstance(poses, dict) and key in poses
        }
        body = {
            "named_poses": named,
            "sim": sim,
            "joints": poses.get("joints") if isinstance(poses, dict) else {},
        }
        return stamp_snapshot(
            body,
            service_instance_id=self._service_instance_id,
            scene_epoch=int(getattr(self, "_sequential_scene_epoch", self._scene_generation)),
            scene_generation=self._scene_generation,
            sim_action_count=self._sim_action_count,
            query_seq=self._query_seq,
            terminated=bool(getattr(self, "_sequential_native_goal", False) or getattr(self, "_sequential_refusing_step", False)),
            truncated=bool(getattr(self, "_sequential_truncated", False)),
            source="readonly_rpc",
        )

    step._sequential_episode_wrapped = True
    cls.step = step
    cls.named_poses = named_poses
    cls.chunk_step = chunk_step
    cls.reset = reset
    cls.native_goal_status = native_goal_status
    cls.state_snapshot = state_snapshot
    cls.identity = identity
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
            reading = _worker_reading(self, bodies)
            return {
                "bodies": bodies,
                "sites": sites,
                "site_geometry": geometry,
                "sim": reading["sim"],
                "joints": reading["joints"],
                "evidence_source": "mujoco_site",
            }
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


def _worker_reading(env, bodies: dict) -> dict[str, Any]:
    """One cached observation plus the current MuJoCo arrays. Does not call env.step."""
    raw = {}
    try:
        reader = getattr(env, "_get_observations", None)
        if reader is None and getattr(env, "env", None) is not None:
            reader = getattr(env.env, "_get_observations", None)
        if callable(reader):
            raw = reader(force_update=False) or {}
    except TypeError:
        try:
            raw = reader() or {}
        except Exception:
            raw = {}
    except Exception:
        raw = {}
    objects = {}
    for key, value in raw.items() if isinstance(raw, dict) else []:
        if not isinstance(key, str) or not key.endswith("_pos"):
            continue
        if "robot0" in key or "to_robot" in key:
            continue
        try:
            pos = [float(item) for item in list(value)[:3]]
        except (TypeError, ValueError):
            continue
        if len(pos) == 3:
            objects[key[:-4]] = {"pos": pos, "identified": True, "evidence_source": "readonly_rpc"}
    if not objects:
        for name, pos in bodies.items():
            if name.startswith("robot") or "world" in name:
                continue
            objects[name] = {"pos": pos, "identified": True, "evidence_source": "mujoco_body"}
    eef = raw.get("robot0_eef_pos") if isinstance(raw, dict) else None
    try:
        eef_pos = None if eef is None else [float(item) for item in list(eef)[:3]]
    except (TypeError, ValueError):
        eef_pos = None
    grip = raw.get("robot0_gripper_qpos") if isinstance(raw, dict) else None
    opening = None
    try:
        if grip is not None:
            opening = abs(float(grip[0])) + abs(float(grip[1]))
    except (TypeError, ValueError, IndexError):
        opening = None
    joints = {}
    try:
        model = env.sim.model
        data = env.sim.data
        count = int(getattr(model, "njnt"))
        names = []
        raw_names = getattr(model, "joint_names", None)
        if raw_names is not None:
            names = [str(item) for item in list(raw_names)[:count]]
        qpos = data.qpos
        # A free joint owns 7 qpos slots, so a joint index is not a qpos
        # address. Prefer the model's own address table.
        addresses = getattr(model, "jnt_qposadr", None)
        if addresses is not None:
            try:
                addresses = [int(item) for item in list(addresses)[:count]]
            except Exception:
                addresses = None
        getter = getattr(model, "get_joint_qpos_addr", None)
        for index, name in enumerate(names):
            if not name:
                continue
            address = None
            if addresses is not None and index < len(addresses):
                address = addresses[index]
            elif callable(getter):
                try:
                    address = int(getter(name))
                except Exception:
                    address = None
            if address is None:
                address = index
            joints[name] = float(qpos[address])
    except Exception:
        joints = {}
    return {
        "sim": {
            "objects": objects,
            "eef_pos": eef_pos,
            "gripper_opening_m": opening,
            "evidence_source": "readonly_rpc",
        },
        "joints": joints,
    }


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


def _ensure_identity(facade) -> None:
    import uuid

    if not getattr(facade, "_service_instance_id", None):
        facade._service_instance_id = uuid.uuid4().hex
        facade._sim_action_count = 0
        facade._query_seq = 0
        facade._scene_generation = 0
        facade._sequential_refusing_step = False


def _refuse_if_terminal(facade, name: str) -> None:
    _ensure_identity(facade)
    if getattr(facade, "_sequential_refusing_step", False):
        raise RuntimeError(f"{name} refused after term/trunc")


def _stamp_action(facade, published, applied: int):
    obs, reward, term, trunc, info = published
    _ensure_identity(facade)
    facade._sim_action_count = int(facade._sim_action_count) + max(0, int(applied))
    if _any_flag(term) or _any_flag(trunc):
        facade._sequential_refusing_step = True
    identity = {
        "service_instance_id": facade._service_instance_id,
        "scene_generation": facade._scene_generation,
        "sim_action_count": facade._sim_action_count,
        "observation_id_meaning": "readonly_or_action_query",
    }
    if isinstance(info, dict):
        info_out = dict(info)
        info_out.update(identity)
    elif isinstance(info, list):
        info_out = [dict(item) if isinstance(item, dict) else item for item in info]
        if info_out and isinstance(info_out[-1], dict):
            info_out[-1].update(identity)
    else:
        info_out = {"info": info, **identity}
    return obs, reward, term, trunc, info_out


def _publish(self, obs, reward, term, trunc, info):
    decision = classify_step(_any_flag(term), _any_flag(trunc), read_mode(), task_scope=read_task_scope())
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
