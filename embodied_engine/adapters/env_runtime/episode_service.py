"""Free-mode native success is an event on the environment service.

Benchmark mode still publishes the gym terminated flag and the client keeps
its latch. Free mode records the native goal and does not turn that event
into an episode end; the executor may finish the current transfer only.
Truncation, a real service error, and an explicit stop still end the episode.
This module does not clear a client latch and does not reset the scene.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
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
    if terminated and mode == "free":
        # The executor limits what runs after this event. The environment
        # itself can still step, so the RPC client must not latch terminated.
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
        self._rpc["env.robot_state_capture"] = self.robot_state_capture
        self._rpc["env.robot_state_restore"] = self.robot_state_restore
        self._rpc["env.robot_state_inspect"] = self.robot_state_inspect
        self._readonly_methods.add("env.native_goal")
        self._readonly_methods.add("env.named_poses")
        self._readonly_methods.add("env.state_snapshot")
        self._readonly_methods.add("env.identity")
        self._readonly_methods.add("env.robot_state_capture")
        self._readonly_methods.add("env.robot_state_inspect")

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
            "code": "episode-service-state-2",
            "capabilities": [
                "env.state_snapshot",
                "env.robot_state_capture",
                "env.robot_state_restore",
            ],
            "robot_state_restore": {
                "reads": "worker rpc (getattr attribute carrying a base64 request)",
                "writes": "robot joints only, inside the env worker",
                "capture_scope": "worker process memory",
            },
        }

    def robot_state_capture(self, reason: str = "episode_init"):
        """Freeze this episode's robot state inside the worker process.

        The engine only sees the evidence dict; the arrays a later restore
        writes stay in the process that read them. A respawned worker reports
        ``capture_unknown`` instead of restoring a state it never read.
        """
        _ensure_identity(self)
        self._robot_capture_seq = int(getattr(self, "_robot_capture_seq", 0)) + 1
        capture_id = f"{self._service_instance_id}:rs{self._robot_capture_seq}"
        self._query_seq = int(getattr(self, "_query_seq", 0)) + 1
        reply = _worker_robot_rpc(
            self,
            {
                "op": "capture",
                "capture_id": capture_id,
                "reason": str(reason)[:200],
                "sim_action_count": int(self._sim_action_count),
            },
        )
        captures = getattr(self, "_robot_captures", None)
        if captures is None:
            captures = {}
            self._robot_captures = captures
        if reply.get("ok"):
            captures[capture_id] = reply.get("capture") or {}
        from state_contract import stamp_snapshot

        row = stamp_snapshot(
            {
                "robot": reply,
                "capture_id": capture_id,
                "reason": str(reason)[:200],
            },
            service_instance_id=self._service_instance_id,
            scene_epoch=int(getattr(self, "_sequential_scene_epoch", self._scene_generation)),
            scene_generation=self._scene_generation,
            sim_action_count=self._sim_action_count,
            query_seq=self._query_seq,
            terminated=False,
            truncated=False,
            source="robot_state_capture",
        )
        row["ok"] = bool(reply.get("ok"))
        row["reason_code"] = reply.get("reason_code")
        row["reason"] = reply.get("reason")
        return row

    def robot_state_restore(self, request: dict | None = None):
        """Restore one captured robot state over RPC; never touches the scene.

        No action is applied here, so ``sim_action_count`` does not move: the
        write is a state change, not a step. Callers that need a fresh image
        must take counted settle actions afterwards.
        """
        _refuse_if_terminal(self, "env.robot_state_restore")
        _ensure_identity(self)
        body = dict(request or {})
        capture_id = str(body.get("capture_id") or "")
        if capture_id not in (getattr(self, "_robot_captures", None) or {}):
            return {
                "ok": False,
                "reason_code": "capture_unknown",
                "reason": "this env service never captured that robot state (or it was respawned)",
                "capture_id": capture_id,
                "service_instance_id": self._service_instance_id,
            }
        self._query_seq = int(getattr(self, "_query_seq", 0)) + 1
        reply = _worker_robot_rpc(self, {**body, "op": "restore", "capture_id": capture_id})
        from state_contract import stamp_snapshot

        row = stamp_snapshot(
            {"robot": reply, "capture_id": capture_id, "request": _safe_request(body)},
            service_instance_id=self._service_instance_id,
            scene_epoch=int(getattr(self, "_sequential_scene_epoch", self._scene_generation)),
            scene_generation=self._scene_generation,
            sim_action_count=self._sim_action_count,
            query_seq=self._query_seq,
            terminated=bool(getattr(self, "_sequential_refusing_step", False)),
            truncated=bool(getattr(self, "_sequential_truncated", False)),
            source="robot_state_restore",
        )
        row["ok"] = bool(reply.get("ok"))
        row["reason_code"] = reply.get("reason_code")
        row["reason"] = reply.get("reason")
        return row

    def robot_state_inspect(self):
        """Read the robot state and its contact list; nothing is written.

        The handoff uses this as the barrier's contact probe: whether the
        fingers are still on a handle is measured here, never inferred from
        the step's task type.
        """
        _refuse_if_terminal(self, "env.robot_state_inspect")
        _ensure_identity(self)
        self._query_seq = int(getattr(self, "_query_seq", 0)) + 1
        reply = _worker_robot_rpc(self, {"op": "inspect"})
        from state_contract import stamp_snapshot

        row = stamp_snapshot(
            {"robot": reply, "source": "robot_state_inspect"},
            service_instance_id=self._service_instance_id,
            scene_epoch=int(getattr(self, "_sequential_scene_epoch", self._scene_generation)),
            scene_generation=self._scene_generation,
            sim_action_count=self._sim_action_count,
            query_seq=self._query_seq,
            terminated=bool(getattr(self, "_sequential_refusing_step", False)),
            truncated=bool(getattr(self, "_sequential_truncated", False)),
            source="robot_state_inspect",
        )
        row["ok"] = bool(reply.get("ok"))
        row["reason_code"] = reply.get("reason_code")
        row["reason"] = reply.get("reason")
        return row

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
        native_goal = bool(getattr(self, "_sequential_native_goal", False))
        truncated = bool(getattr(self, "_sequential_truncated", False))
        refusing = bool(getattr(self, "_sequential_refusing_step", False))
        body = {
            "named_poses": named,
            "sim": sim,
            "joints": poses.get("joints") if isinstance(poses, dict) else {},
            "native_goal": native_goal,
            "continuable": not truncated and not refusing,
        }
        return stamp_snapshot(
            body,
            service_instance_id=self._service_instance_id,
            scene_epoch=int(getattr(self, "_sequential_scene_epoch", self._scene_generation)),
            scene_generation=self._scene_generation,
            sim_action_count=self._sim_action_count,
            query_seq=self._query_seq,
            terminated=bool(refusing and not truncated),
            truncated=truncated,
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
    cls.robot_state_capture = robot_state_capture
    cls.robot_state_restore = robot_state_restore
    cls.robot_state_inspect = robot_state_inspect
    cls._register_rpc = _register_rpc
    cls._sequential_episode_wrapped = True


def _attach_named_poses(module) -> None:
    """Add an exact body/site snapshot on the class the env worker constructs."""
    cls = module.__dict__.get("ControlEnv") or module.__dict__.get("OffScreenRenderEnv")
    if isinstance(cls, type):
        _attach_named_poses_class(cls)


def _attach_named_poses_class(cls) -> None:
    if not isinstance(cls, type):
        return
    _attach_robot_state_class(cls)
    if getattr(cls, "_sequential_named_poses", False):
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
    _attach_robot_state_class(cls)


def _attach_robot_state_class(cls) -> None:
    """Attach the robot-only capture/restore entry point to the worker env.

    The rlinf worker loop forwards exactly two commands we can use here:
    ``["getattr", name]`` and ``["setattr", {key, value}]``. ``setattr``
    targets ``env.unwrapped``, which a LIBERO ``ControlEnv`` does not have, so
    the request travels inside the attribute *name*: the facade sends
    ``getattr "<prefix><base64 json>"`` and ``__getattr__`` decodes it. One
    call, no shared files, no arguments the loop cannot carry.
    """
    if not isinstance(cls, type) or getattr(cls, "_sequential_robot_state", False):
        return

    def sequential_getattr(self, name):
        if not isinstance(name, str) or not name.startswith(ROBOT_RPC_PREFIX):
            raise AttributeError(name)
        memo = self.__dict__.setdefault("_sequential_robot_memo", {})
        cached = memo.get(name)
        if cached is not None:
            # ``hasattr`` in the worker loop calls this once and throws the
            # value away; the following ``getattr`` is the one that is sent
            # back. Replay the stored answer instead of applying the write
            # twice.
            memo.pop(name, None)
            return cached
        try:
            value = _robot_rpc(self, name[len(ROBOT_RPC_PREFIX):])
        except Exception as exc:  # noqa: BLE001 - the worker loop would die
            value = {
                "ok": False,
                "reason_code": "worker_exception",
                "reason": f"{type(exc).__name__}: {exc}",
            }
        memo[name] = value
        return value

    cls.__getattr__ = sequential_getattr
    cls._sequential_robot_state = True


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
    quat = raw.get("robot0_eef_quat") if isinstance(raw, dict) else None
    eef_quat = None
    try:
        if quat is not None:
            values = [float(item) for item in list(quat)[:4]]
            norm = sum(value * value for value in values) ** 0.5
            if len(values) == 4 and norm > 1e-9:
                eef_quat = [value / norm for value in values]
    except (TypeError, ValueError, IndexError):
        eef_quat = None
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
            "eef_quat": eef_quat,
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


# ---------------------------------------------------------------------------
# Robot-only state capture / restore (worker side)
# ---------------------------------------------------------------------------
# The engine must never mutate MuJoCo across processes, so both halves run
# here on the worker that owns ``sim``:
#
#   capture  reads the robot joint spans the *model* declares (never a sliced
#            array) and keeps the exact qpos in worker memory under an id;
#   restore  looks that id up, checks the current gripper contacts, checks the
#            candidate configuration on an isolated ``mujoco.MjData`` copy of
#            the same model, then writes only the robot qpos/qvel, syncs the
#            robot's own controllers, forwards, and re-reads the observation.
#
# Non-robot qpos (objects, drawer, stove button) is read before and after and
# must be bit-identical: a restore that leaks into the scene is a failure, not
# a detail.

ROBOT_RPC_PREFIX = "sequential_robot_rpc:"
ROBOT_RESTORE_CODE = "episode-service-robot-restore-1"
JOINT_QPOS_WIDTH = {0: 7, 1: 4, 2: 1, 3: 1}  # free, ball, slide, hinge
JOINT_DOF_WIDTH = {0: 6, 1: 3, 2: 1, 3: 1}


def _safe_request(body: dict[str, Any]) -> dict[str, Any]:
    keep = (
        "capture_id",
        "reason",
        "mode",
        "penetration_tol_m",
        "reject_gripper_contact",
        "require_collision_check",
        "settle_actions",
    )
    return {key: body.get(key) for key in keep if key in body}


def _worker_robot_rpc(facade, request: dict[str, Any]) -> dict[str, Any]:
    """One attribute-name RPC to the worker that owns ``sim``. No env.step."""
    try:
        encoded = (
            base64.urlsafe_b64encode(json.dumps(request, default=str).encode("utf-8"))
            .decode("ascii")
            .rstrip("=")
        )
        inner = getattr(facade, "_env", None)
        vector = getattr(inner, "env", None)
        workers = getattr(vector, "workers", None) or []
        if not workers:
            return {"ok": False, "reason_code": "worker_missing", "reason": "env worker is not running"}
        worker = workers[0]
        worker.parent_remote.send(["getattr", f"{ROBOT_RPC_PREFIX}{encoded}"])
        payload = worker.parent_remote.recv()
    except Exception as exc:  # noqa: BLE001 - report, never crash the facade
        return {"ok": False, "reason_code": "worker_rpc_failed", "reason": f"{type(exc).__name__}: {exc}"}
    if isinstance(payload, dict):
        return payload
    return {"ok": False, "reason_code": "worker_payload", "reason": f"payload is {type(payload).__name__}"}


def _robot_rpc(env, encoded: str) -> dict[str, Any]:
    pad = "=" * (-len(encoded) % 4)
    request = json.loads(base64.urlsafe_b64decode(encoded + pad).decode("utf-8"))
    op = str(request.get("op") or "")
    if op == "capture":
        return _robot_capture(env, request)
    if op == "restore":
        return _robot_restore(env, request)
    if op == "inspect":
        return {"ok": True, "state": _robot_state(env)}
    return {"ok": False, "reason_code": "unknown_op", "reason": f"op={op}"}


def _model_joint_names(model) -> list[str]:
    count = int(getattr(model, "njnt", 0))
    raw = getattr(model, "joint_names", None)
    names = [str(item) for item in list(raw)[:count]] if raw is not None else []
    if len(names) < count:
        getter = getattr(model, "joint_id2name", None)
        names = []
        for index in range(count):
            try:
                names.append(str(getter(index) or ""))
            except Exception:
                names.append("")
    return names


def _model_geom_name(model, geom_id: int) -> str:
    try:
        getter = getattr(model, "geom_id2name", None)
        if callable(getter):
            return str(getter(int(geom_id)) or f"geom{geom_id}")
        names = getattr(model, "geom_names", None)
        if names is not None and int(geom_id) < len(names):
            return str(list(names)[int(geom_id)] or f"geom{geom_id}")
    except Exception:
        pass
    return f"geom{geom_id}"


def _model_body_name(model, body_id: int) -> str:
    try:
        getter = getattr(model, "body_id2name", None)
        if callable(getter):
            return str(getter(int(body_id)) or f"body{body_id}")
        names = getattr(model, "body_names", None)
        if names is not None and int(body_id) < len(names):
            return str(list(names)[int(body_id)] or f"body{body_id}")
    except Exception:
        pass
    return f"body{body_id}"


def _robot_prefixes(env) -> dict[str, list[str]]:
    """Arm and gripper naming prefixes, read from the robots themselves."""
    arms: list[str] = []
    grippers: list[str] = []
    for robot in list(getattr(env, "robots", None) or []):
        prefix = getattr(getattr(robot, "robot_model", None), "naming_prefix", None)
        if prefix and str(prefix) not in arms:
            arms.append(str(prefix))
        gripper = getattr(robot, "gripper", None)
        models = list(gripper.values()) if isinstance(gripper, dict) else [gripper]
        for holder in models:
            prefix = getattr(holder, "naming_prefix", None)
            if prefix and str(prefix) not in grippers:
                grippers.append(str(prefix))
    return {"arms": arms, "grippers": grippers}


def _robot_joint_names(env, model) -> tuple[list[str], str]:
    """Robot joint names, preferring the robot's own joint lists.

    robosuite keeps the authoritative names in ``robot.robot_joints`` and
    ``robot.gripper_joints``; prefix matching is only the fallback, so a
    renamed gripper cannot silently fall out of the restored set.
    """
    names: list[str] = []
    for robot in list(getattr(env, "robots", None) or []):
        for name in list(getattr(robot, "robot_joints", None) or []):
            if str(name) not in names:
                names.append(str(name))
        gripper_joints = getattr(robot, "gripper_joints", None)
        for joints in dict(gripper_joints or {}).values():
            for name in list(joints or []):
                if str(name) not in names:
                    names.append(str(name))
    if names:
        return names, "robot_joints+gripper_joints"
    prefixes = _robot_prefixes(env)
    all_prefixes = prefixes["arms"] + prefixes["grippers"]
    return [name for name in _model_joint_names(model) if name and any(name.startswith(p) for p in all_prefixes)], "naming_prefix"


def _robot_joint_indexes(env, model) -> dict[str, Any]:
    """Robot qpos/qvel addresses, derived from the model address tables."""
    prefixes = _robot_prefixes(env)
    if not prefixes["arms"]:
        return {
            "ok": False,
            "reason_code": "robot_prefix_unresolved",
            "reason": "no robot naming prefix was readable from the env",
        }
    model_names = _model_joint_names(model)
    wanted, source = _robot_joint_names(env, model)
    wanted_set = set(wanted)
    missing = sorted(wanted_set - set(model_names))
    joints: list[dict[str, Any]] = []
    qpos_indexes: list[int] = []
    qvel_indexes: list[int] = []
    for joint_id, name in enumerate(model_names):
        if name not in wanted_set:
            continue
        joint_type = int(getattr(model, "jnt_type")[joint_id])
        if joint_type == 0:
            return {
                "ok": False,
                "reason_code": "robot_free_joint",
                "reason": f"joint {name} is a free joint; a robot joint is never restored this way",
            }
        qpos_adr = int(getattr(model, "jnt_qposadr")[joint_id])
        qvel_adr = int(getattr(model, "jnt_dofadr")[joint_id])
        qpos_width = JOINT_QPOS_WIDTH.get(joint_type, 1)
        dof_width = JOINT_DOF_WIDTH.get(joint_type, 1)
        joints.append(
            {
                "name": name,
                "joint_id": joint_id,
                "type": joint_type,
                "qpos_adr": qpos_adr,
                "qpos_width": qpos_width,
                "qvel_adr": qvel_adr,
                "dof_width": dof_width,
            }
        )
        qpos_indexes.extend(range(qpos_adr, qpos_adr + qpos_width))
        qvel_indexes.extend(range(qvel_adr, qvel_adr + dof_width))
    if not joints:
        return {
            "ok": False,
            "reason_code": "robot_joints_unresolved",
            "reason": f"none of {sorted(wanted_set)[:8]} is in the model joint table",
        }
    if missing:
        return {
            "ok": False,
            "reason_code": "robot_joint_missing_from_model",
            "reason": f"the robot declares joints the model does not carry: {missing}",
        }
    robot_joint_ids = {row["joint_id"] for row in joints}
    actuator_ids: list[int] = []
    for actuator in range(int(getattr(model, "nu", 0))):
        try:
            transmission = getattr(model, "actuator_trnid")[actuator]
            joint_id = int(list(transmission)[0])
        except Exception:
            continue
        if joint_id in robot_joint_ids:
            actuator_ids.append(actuator)
    return {
        "ok": True,
        "source": source,
        "prefixes": prefixes["arms"] + prefixes["grippers"],
        "arms": prefixes["arms"],
        "grippers": prefixes["grippers"],
        "joints": joints,
        "joint_names": [row["name"] for row in joints],
        "robot_joint_ids": sorted(robot_joint_ids),
        "qpos_indexes": qpos_indexes,
        "qvel_indexes": qvel_indexes,
        "actuator_ids": actuator_ids,
    }


def _joint_qpos_map(model) -> list[dict[str, Any]]:
    names = _model_joint_names(model)
    rows = []
    for joint_id, name in enumerate(names):
        if not name:
            continue
        joint_type = int(getattr(model, "jnt_type")[joint_id])
        rows.append(
            {
                "name": name,
                "type": joint_type,
                "adr": int(getattr(model, "jnt_qposadr")[joint_id]),
                "width": JOINT_QPOS_WIDTH.get(joint_type, 1),
                "dof_adr": int(getattr(model, "jnt_dofadr")[joint_id]),
                "dof_width": JOINT_DOF_WIDTH.get(joint_type, 1),
                "body": _model_body_name(model, int(getattr(model, "jnt_bodyid")[joint_id])),
            }
        )
    return rows


def _nonrobot_state(model, data, robot_qpos: set[int]) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    free_bodies: dict[str, Any] = {}
    # Velocities are recorded but deliberately kept *out* of the digest: the
    # restore check compares the scene byte-for-byte, and folding a velocity
    # into the digest would make that comparison depend on what the sim
    # happened to be doing at the moment of the read.
    speeds: dict[str, Any] = {}
    digest = hashlib.sha256()
    for row in _joint_qpos_map(model):
        if row["adr"] in robot_qpos:
            continue
        values = [round(float(data.qpos[row["adr"] + k]), 9) for k in range(row["width"])]
        rows[row["name"]] = values
        speeds[row["name"]] = [
            round(float(data.qvel[row["dof_adr"] + k]), 9) for k in range(row["dof_width"])
        ]
        digest.update(f"{row['name']}:{values}".encode("utf-8"))
        if row["type"] == 0:
            try:
                body_id = int(getattr(model, "jnt_bodyid")[model.joint_name2id(row["name"])])
                pos = [round(float(v), 6) for v in list(data.body_xpos[body_id])[:3]]
                quat = [round(float(v), 6) for v in list(data.body_xquat[body_id])[:4]]
                free_bodies[row["name"]] = {"pos": pos, "quat": quat}
                digest.update(f"{row['name']}@{pos}{quat}".encode("utf-8"))
            except Exception:
                pass
    return {"joints": rows, "qvel": speeds, "free_bodies": free_bodies, "digest": digest.hexdigest()[:32]}


def _contact_rows(model, data, prefixes: dict[str, list[str]], limit: int = 64) -> list[dict[str, Any]]:
    def robot_geom(geom_id: int) -> bool:
        try:
            body = _model_body_name(model, int(getattr(model, "geom_bodyid")[geom_id]))
        except Exception:
            return False
        return any(body.startswith(prefix) for prefix in prefixes["arms"] + prefixes["grippers"])

    def gripper_geom(geom_id: int) -> bool:
        try:
            body = _model_body_name(model, int(getattr(model, "geom_bodyid")[geom_id]))
        except Exception:
            return False
        return any(body.startswith(prefix) for prefix in prefixes["grippers"])

    rows: list[dict[str, Any]] = []
    count = int(getattr(data, "ncon", 0))
    for index in range(min(count, limit)):
        contact = data.contact[index]
        geom1, geom2 = int(contact.geom1), int(contact.geom2)
        rows.append(
            {
                "geom1": _model_geom_name(model, geom1),
                "geom2": _model_geom_name(model, geom2),
                "body1": _model_body_name(model, int(getattr(model, "geom_bodyid")[geom1])),
                "body2": _model_body_name(model, int(getattr(model, "geom_bodyid")[geom2])),
                "dist": round(float(contact.dist), 5),
                "robot1": robot_geom(geom1),
                "robot2": robot_geom(geom2),
                "gripper1": gripper_geom(geom1),
                "gripper2": gripper_geom(geom2),
            }
        )
    return rows


def _contact_summary(rows: list[dict[str, Any]], total: int) -> dict[str, Any]:
    robot_pairs = [row for row in rows if bool(row["robot1"]) != bool(row["robot2"])]
    gripper_pairs = [
        row
        for row in robot_pairs
        if (row["gripper1"] and not row["robot2"]) or (row["gripper2"] and not row["robot1"])
    ]
    penetration = max([-float(row["dist"]) for row in robot_pairs if float(row["dist"]) < 0] or [0.0])
    return {
        "count": int(total),
        "listed": len(rows),
        "robot_vs_nonrobot": robot_pairs[:16],
        "gripper_vs_nonrobot": gripper_pairs[:8],
        "penetration_max_m": round(float(penetration), 5),
    }


def _floats(value, limit: int = 16) -> list[float] | None:
    if value is None:
        return None
    try:
        return [round(float(v), 6) for v in list(value)[:limit]]
    except (TypeError, ValueError):
        return None


def _controller_state(env) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for robot in list(getattr(env, "robots", None) or []):
        controller = getattr(robot, "composite_controller", None)
        row: dict[str, Any] = {
            "robot": getattr(getattr(robot, "robot_model", None), "naming_prefix", None),
            "composite": controller is not None,
            "parts": {},
        }
        parts = getattr(controller, "part_controllers", None) if controller is not None else None
        for part_name, part in dict(parts or {}).items():
            row["parts"][str(part_name)] = {
                "type": type(part).__name__,
                "goal_pos": _floats(getattr(part, "goal_pos", None), 3),
                "goal_ori": _floats(getattr(part, "goal_ori", None), 9),
                "ori_ref": _floats(getattr(part, "ori_ref", None), 9),
                "relative_ori": _floats(getattr(part, "relative_ori", None), 3),
                "goal_update_mode": getattr(part, "_goal_update_mode", None),
                "interpolators": [
                    name
                    for name in ("interpolator_pos", "interpolator_ori")
                    if getattr(part, name, None) is not None
                ],
            }
        legacy = getattr(robot, "controller", None)
        if legacy is not None:
            row["legacy_controller"] = {
                "type": type(legacy).__name__,
                "goal_pos": _floats(getattr(legacy, "goal_pos", None), 3),
                "goal_ori": _floats(getattr(legacy, "goal_ori", None), 9),
            }
        rows.append(row)
    return rows


def _cached_obs(env) -> dict[str, Any]:
    try:
        reader = getattr(env, "_get_observations", None)
        if not callable(reader) and getattr(env, "env", None) is not None:
            reader = getattr(env.env, "_get_observations", None)
        if not callable(reader):
            return {}
        try:
            return reader(force_update=False) or {}
        except TypeError:
            return reader() or {}
    except Exception:
        return {}


def _obs_pose(obs: dict[str, Any]) -> dict[str, Any]:
    eef = obs.get("robot0_eef_pos") if isinstance(obs, dict) else None
    quat = obs.get("robot0_eef_quat") if isinstance(obs, dict) else None
    grip = obs.get("robot0_gripper_qpos") if isinstance(obs, dict) else None
    opening = None
    try:
        if grip is not None:
            opening = round(abs(float(grip[0])) + abs(float(grip[1])), 6)
    except (TypeError, ValueError, IndexError):
        opening = None
    return {
        "eef_pos": _floats(eef, 3),
        "eef_quat": _floats(quat, 4),
        "gripper_opening_m": opening,
    }


def _robot_state(env) -> dict[str, Any]:
    """Read-only: robot spans, controller goals, non-robot qpos, contacts."""
    sim = getattr(env, "sim", None)
    if sim is None:
        return {"ok": False, "reason_code": "sim_unavailable", "reason": "env has no sim"}
    model, data = sim.model, sim.data
    indexes = _robot_joint_indexes(env, model)
    if not indexes.get("ok"):
        return indexes
    obs = _cached_obs(env)
    pose = _obs_pose(obs)
    prefixes = _robot_prefixes(env)
    rows = _contact_rows(model, data, prefixes)
    return {
        "ok": True,
        "code": ROBOT_RESTORE_CODE,
        "robot": {
            "prefixes": indexes["prefixes"],
            "joints": indexes["joint_names"],
            "details": indexes["joints"],
            "qpos_indexes": indexes["qpos_indexes"],
            "qvel_indexes": indexes["qvel_indexes"],
            "qpos": [round(float(data.qpos[adr]), 9) for adr in indexes["qpos_indexes"]],
            "qvel": [round(float(data.qvel[adr]), 9) for adr in indexes["qvel_indexes"]],
            "actuator_ids": indexes["actuator_ids"],
            "ctrl": [round(float(data.ctrl[a]), 6) for a in indexes["actuator_ids"]],
            **pose,
        },
        "controller": _controller_state(env),
        "non_robot": _nonrobot_state(model, data, set(indexes["qpos_indexes"])),
        "contacts": _contact_summary(rows, int(getattr(data, "ncon", 0))),
        "sim": {
            "time": round(float(data.time), 6),
            "nq": int(getattr(model, "nq", 0)),
            "nv": int(getattr(model, "nv", 0)),
            "nu": int(getattr(model, "nu", 0)),
            "na": int(getattr(model, "na", 0)),
            "qpos_modified": False,
        },
    }


def _digest(state: dict[str, Any]) -> dict[str, Any]:
    """Small comparable slice of one robot state read."""
    robot = state.get("robot") or {}
    return {
        "eef_pos": robot.get("eef_pos"),
        "eef_quat": robot.get("eef_quat"),
        "gripper_opening_m": robot.get("gripper_opening_m"),
        "qpos": robot.get("qpos"),
        "qvel": robot.get("qvel"),
        "ctrl": robot.get("ctrl"),
        "non_robot_digest": (state.get("non_robot") or {}).get("digest"),
        "non_robot_joints": (state.get("non_robot") or {}).get("joints"),
        "contacts": state.get("contacts"),
        "controller": state.get("controller"),
        "sim_time": (state.get("sim") or {}).get("time"),
    }


def _robot_capture(env, request: dict[str, Any]) -> dict[str, Any]:
    capture_id = str(request.get("capture_id") or "")
    if not capture_id:
        return {"ok": False, "reason_code": "capture_id_missing", "reason": "capture needs an id"}
    state = _robot_state(env)
    if not state.get("ok"):
        return {**state, "capture_id": capture_id}
    capture = {
        "capture_id": capture_id,
        "qpos_indexes": state["robot"]["qpos_indexes"],
        "qvel_indexes": state["robot"]["qvel_indexes"],
        "qpos": state["robot"]["qpos"],
        "joints": state["robot"]["joints"],
        "eef_pos": state["robot"]["eef_pos"],
        "eef_quat": state["robot"]["eef_quat"],
        "gripper_opening_m": state["robot"]["gripper_opening_m"],
        "non_robot_digest": state["non_robot"]["digest"],
        "sim_time": state["sim"]["time"],
        "sim_action_count": request.get("sim_action_count"),
        "reason": request.get("reason"),
    }
    env.__dict__.setdefault("_sequential_robot_captures", {})[capture_id] = capture
    return {
        "ok": True,
        "capture_id": capture_id,
        "capture": capture,
        "state": state,
        "code": ROBOT_RESTORE_CODE,
    }


def _candidate_collisions(env, capture: dict[str, Any], tolerance_m: float) -> dict[str, Any]:
    """Check the candidate robot configuration on a *separate* MjData copy.

    The main sim data is never written while checking: the candidate qpos goes
    into a fresh ``mujoco.MjData`` built from the same model, ``mj_forward``
    computes contact distances there, and the answer is a penetration depth.
    """
    sim = getattr(env, "sim", None)
    model = getattr(sim, "model", None)
    data = getattr(sim, "data", None)
    raw_model = getattr(model, "_model", None)
    raw_data = getattr(data, "_data", None)
    if raw_model is None or raw_data is None:
        return {
            "checked": False,
            "reason_code": "raw_mujoco_unavailable",
            "reason": "robosuite MjModel/MjData did not expose the raw mujoco objects",
        }
    try:
        import mujoco

        candidate = mujoco.MjData(raw_model)
        candidate.qpos[:] = raw_data.qpos
        candidate.qvel[:] = 0.0
        for adr, value in zip(capture["qpos_indexes"], capture["qpos"]):
            candidate.qpos[int(adr)] = float(value)
        mujoco.mj_forward(raw_model, candidate)
        rows = _contact_rows(model, candidate, _robot_prefixes(env))
        summary = _contact_summary(rows, int(getattr(candidate, "ncon", 0)))
    except Exception as exc:  # noqa: BLE001 - report the capability gap
        return {
            "checked": False,
            "reason_code": "candidate_forward_failed",
            "reason": f"{type(exc).__name__}: {exc}",
        }
    penetration = float(summary["penetration_max_m"])
    return {
        "checked": True,
        "isolated_copy": True,
        "method": (
            "fresh mujoco.MjData on the same MjModel: current qpos plus the captured robot qpos, "
            "qvel zeroed, mj_forward, contact distances read there"
        ),
        "penetration_max_m": penetration,
        "tol_m": float(tolerance_m),
        "refused": bool(penetration > float(tolerance_m)),
        "robot_vs_nonrobot": summary["robot_vs_nonrobot"],
        "contact_count": summary["count"],
    }


def _sync_controllers(env) -> dict[str, Any]:
    """Point every robot controller at the pose that was just written.

    ``reset_goal`` reads the controller's ``ref_pos``/``ref_ori_mat`` buffers,
    and those are only refreshed by ``update()``. Calling ``reset`` alone would
    re-hold the *previous* pose, so the reference buffers are forced first.
    """
    rows: list[dict[str, Any]] = []
    for robot in list(getattr(env, "robots", None) or []):
        row: dict[str, Any] = {"robot": getattr(getattr(robot, "robot_model", None), "naming_prefix", None)}
        controller = getattr(robot, "composite_controller", None)
        if controller is not None:
            try:
                update = getattr(controller, "update_state", None)
                if callable(update):
                    update()
                row["update_state"] = True
            except Exception as exc:  # noqa: BLE001 - report, keep going
                row["update_state_error"] = f"{type(exc).__name__}: {exc}"
            parts = dict(getattr(controller, "part_controllers", None) or {})
            part_rows: dict[str, Any] = {}
            for part_name, part in parts.items():
                entry: dict[str, Any] = {}
                try:
                    update = getattr(part, "update", None)
                    if callable(update):
                        update(force=True)
                    entry["update"] = True
                except Exception as exc:  # noqa: BLE001
                    entry["update_error"] = f"{type(exc).__name__}: {exc}"
                try:
                    reset_goal = getattr(part, "reset_goal", None)
                    if callable(reset_goal):
                        reset_goal()
                    entry["reset_goal"] = True
                except Exception as exc:  # noqa: BLE001
                    entry["reset_goal_error"] = f"{type(exc).__name__}: {exc}"
                part_rows[str(part_name)] = entry
            row["parts"] = part_rows
            try:
                reset = getattr(controller, "reset", None)
                if callable(reset):
                    reset()
                row["composite_reset"] = True
            except Exception as exc:  # noqa: BLE001
                row["composite_reset_error"] = f"{type(exc).__name__}: {exc}"
        legacy = getattr(robot, "controller", None)
        if legacy is not None:
            try:
                update = getattr(legacy, "update", None)
                if callable(update):
                    update(force=True)
                reset_goal = getattr(legacy, "reset_goal", None)
                if callable(reset_goal):
                    reset_goal()
                row["legacy_reset_goal"] = True
            except Exception as exc:  # noqa: BLE001
                row["legacy_reset_error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
    return {"robots": rows, "after": _controller_state(env)}


def _refresh_observation(env) -> dict[str, Any]:
    """Make the cached observation describe the restored pose, not the old one."""
    out: dict[str, Any] = {"post_process": False, "observables": False, "observations": False}
    try:
        post = getattr(env, "_post_process", None)
        if callable(post):
            post()
        out["post_process"] = True
    except Exception as exc:  # noqa: BLE001
        out["post_process_error"] = f"{type(exc).__name__}: {exc}"
    try:
        update = getattr(env, "_update_observables", None)
        if callable(update):
            update(force=True)
        out["observables"] = True
    except Exception as exc:  # noqa: BLE001
        out["observables_error"] = f"{type(exc).__name__}: {exc}"
    obs = _cached_obs(env)
    out["observations"] = bool(obs)
    out.update(_obs_pose(obs))
    return out


def _robot_restore(env, request: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    capture_id = str(request.get("capture_id") or "")
    capture = (env.__dict__.get("_sequential_robot_captures") or {}).get(capture_id)
    if not capture:
        return {
            "ok": False,
            "reason_code": "capture_unknown",
            "reason": "this worker has no such capture (it may have been respawned)",
            "capture_id": capture_id,
        }
    tolerance = float(request.get("penetration_tol_m") or 0.002)
    reject_contact = bool(request.get("reject_gripper_contact", True))
    require_check = bool(request.get("require_collision_check", True))

    sim = getattr(env, "sim", None)
    if sim is None:
        return {"ok": False, "reason_code": "sim_unavailable", "reason": "env has no sim"}
    model, data = sim.model, sim.data
    before = _robot_state(env)
    if not before.get("ok"):
        return before
    block: dict[str, Any] = {
        "code": ROBOT_RESTORE_CODE,
        "capture_id": capture_id,
        "policy": {
            "penetration_tol_m": tolerance,
            "reject_gripper_contact": reject_contact,
            "require_collision_check": require_check,
        },
        "before": _digest(before),
    }

    # 1. An empty gripper only: any finger contact with a non-robot geom is a
    #    held object or a handle still between the fingers.
    grip = (before.get("contacts") or {}).get("gripper_vs_nonrobot") or []
    block["gripper_contacts"] = grip
    if reject_contact and grip:
        return {
            "ok": False,
            "reason_code": "gripper_contact",
            "reason": "the fingers are in contact with a non-robot geom; refusing to restore the arm",
            "block": block,
            "elapsed_s": round(time.perf_counter() - started, 4),
        }

    # 2. Candidate configuration, checked away from the live sim.
    candidate = _candidate_collisions(env, capture, tolerance)
    block["candidate"] = candidate
    if not candidate.get("checked"):
        if require_check:
            return {
                "ok": False,
                "reason_code": candidate.get("reason_code") or "collision_check_unavailable",
                "reason": (
                    "the candidate configuration could not be collision-checked, so this build does not "
                    "claim the restore is collision safe: " + str(candidate.get("reason"))
                ),
                "block": block,
                "elapsed_s": round(time.perf_counter() - started, 4),
            }
    elif candidate.get("refused"):
        return {
            "ok": False,
            "reason_code": "candidate_penetration",
            "reason": (
                f"the captured arm pose would overlap a scene geom by "
                f"{candidate['penetration_max_m'] * 1000:.1f} mm"
            ),
            "block": block,
            "elapsed_s": round(time.perf_counter() - started, 4),
        }

    # 3. Write the robot joints only, then zero their velocities.
    for adr, value in zip(capture["qpos_indexes"], capture["qpos"]):
        data.qpos[int(adr)] = float(value)
    for adr in capture["qvel_indexes"]:
        data.qvel[int(adr)] = 0.0
    sim.forward()
    block["write"] = {"qpos_indexes": len(capture["qpos_indexes"]), "qvel_zeroed": len(capture["qvel_indexes"])}

    # 4. Controller goals/integral state follow the written pose.
    block["controller_sync"] = _sync_controllers(env)
    sim.forward()

    # 5. Verify the write and that nothing else moved.
    after = _robot_state(env)
    write_error = max(
        [abs(float(a) - float(b)) for a, b in zip(after["robot"]["qpos"], capture["qpos"])] or [0.0]
    )
    velocity_error = max([abs(float(v)) for v in after["robot"]["qvel"]] or [0.0])
    non_robot_same = after["non_robot"]["digest"] == before["non_robot"]["digest"]
    block["after_write"] = _digest(after)
    block["verification"] = {
        "robot_qpos_error": round(float(write_error), 9),
        "robot_qvel_abs_max": round(float(velocity_error), 9),
        "non_robot_preserved": bool(non_robot_same),
        "non_robot_digest_before": before["non_robot"]["digest"],
        "non_robot_digest_after": after["non_robot"]["digest"],
        "scene_time_unchanged": (before["sim"]["time"] == after["sim"]["time"]),
    }

    # 6. Fresh observation for the restored pose (no action applied).
    block["observation"] = _refresh_observation(env)
    ok = bool(write_error <= 1e-6 and velocity_error <= 1e-9 and non_robot_same)
    return {
        "ok": ok,
        "reason_code": "" if ok else "write_verification_failed",
        "reason": "" if ok else "the written robot state or the untouched scene state did not verify",
        "block": block,
        "capture": {
            "capture_id": capture_id,
            "qpos": capture["qpos"],
            "joints": capture["joints"],
            "eef_pos": capture["eef_pos"],
            "eef_quat": capture["eef_quat"],
            "gripper_opening_m": capture["gripper_opening_m"],
            "sim_time": capture["sim_time"],
        },
        "elapsed_s": round(time.perf_counter() - started, 4),
    }
