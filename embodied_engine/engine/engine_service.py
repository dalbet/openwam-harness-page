#!/usr/bin/env python
"""Inference-engine / rollout service for the three-layer LIBERO system.

Binds to 127.0.0.1:14800. It owns the LIBERO env process on port 14561,
talks to the Agent service (14700), the inference service (14600), and
records every episode. The Agent layer is the only source of plans; skill
execution, release/retreat, and completion evidence are deterministic.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
import uuid
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from aiohttp import web

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from adapters.env_client import EnvClient  # noqa: E402
from common import config, records  # noqa: E402
from common import plan_contract
from engine import completion  # noqa: E402
from engine.inference_client import InferenceClient  # noqa: E402
from engine.openwam_rollout import WholeTaskRunner  # noqa: E402
from engine.skills import SkillFailure, SkillRunner, Stopped  # noqa: E402


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


def _scene_fingerprint(snapshot: dict[str, Any]) -> str:
    """Stable hash of the initial object poses and EEF pose.

    A/B runs on the same seed must start from the same scene; the episode
    record keeps this so the claim is checkable instead of assumed.
    """
    sim = snapshot.get("sim") or {}
    objects = {
        name: [round(float(value), 5) for value in (pose or {}).get("pos") or []]
        for name, pose in (sim.get("objects") or {}).items()
        if isinstance(pose, dict)
    }
    payload = {
        "objects": objects,
        "eef": [round(float(value), 5) for value in (sim.get("eef_pos") or [])],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def _post_json(url: str, body: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        try:
            return json.loads(detail)
        except json.JSONDecodeError:
            return {"status": "error", "error": f"HTTP {exc.code}: {detail[:1500]}"}


def _get_json(url: str, timeout_s: float = 10.0) -> dict[str, Any]:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(url, timeout=timeout_s) as response:
        return json.loads(response.read().decode("utf-8"))


def _recovery_failure(failure: dict[str, Any]) -> dict[str, Any]:
    """Recovery gets one scene snapshot; strip duplicate snapshots from failure
    evidence and prior attempts so the Qwen prompt stays bounded."""
    evidence = failure.get("evidence")
    if isinstance(evidence, dict):
        evidence = {key: value for key, value in evidence.items() if key != "snapshot"}
    return {
        "reason_code": failure.get("reason_code"),
        "message": failure.get("message"),
        "evidence": evidence,
    }


def _port_listener(port: int) -> int | None:
    try:
        out = subprocess.run(
            ["ss", "-ltnp", f"sport = :{port}"],
            capture_output=True, text=True, timeout=5, check=False
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    marker = "pid="
    for line in out.splitlines():
        index = line.find(marker)
        if index < 0:
            continue
        tail = line[index + len(marker):]
        digits = ""
        for char in tail:
            if char.isdigit():
                digits += char
            else:
                break
        if digits:
            return int(digits)
    return None


def _cmdline(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\x00", b" ").decode("utf-8", "replace")
    except OSError:
        return ""


class EnvManager:
    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        self.host = cfg["services"]["env"]["host"]
        self.port = int(cfg["services"]["env"]["port"])
        self.client: EnvClient | None = None
        self.task_id: int | None = None
        self.seed = int(cfg["env"]["seed"])
        self.identity: dict[str, Any] = {}
        self.pid: int | None = None
        self.lock = threading.RLock()

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def _is_ours(self, pid: int) -> bool:
        line = _cmdline(pid)
        return "adapters/libero_env_server.py" in line

    def stop(self) -> None:
        with self.lock:
            pid = _port_listener(self.port)
            if pid is not None:
                if not self._is_ours(pid):
                    raise RuntimeError(f"env port {self.port} is held by foreign pid {pid}: {_cmdline(pid)[:300]}")
                try:
                    group = os.getpgid(pid)
                    if group == os.getpgid(0):
                        os.kill(pid, signal.SIGTERM)
                    else:
                        os.killpg(group, signal.SIGTERM)
                except (ProcessLookupError, PermissionError):
                    pass
                for _ in range(80):
                    if _port_listener(self.port) != pid:
                        break
                    time.sleep(0.1)
                left = _port_listener(self.port)
                if left == pid:
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
            if self.client is not None:
                try:
                    self.client.close()
                except Exception:
                    pass
            self.client = None
            self.task_id = None
            self.pid = None
            try:
                (config.RECORDS_DIR / "env_server.pid").unlink()
            except OSError:
                pass

    def ensure(self, task_id: int, seed: int | None = None) -> EnvClient:
        wanted = int(self.cfg["env"]["seed"] if seed is None else seed)
        with self.lock:
            if self.client is not None and self.task_id == task_id and self.seed == wanted:
                try:
                    ident = self.client.identity()
                    if ident.get("service_instance_id"):
                        self.identity = ident
                        return self.client
                except Exception:
                    pass
            self.stop()
            self._start(task_id, wanted)
            assert self.client is not None
            return self.client

    def _start(self, task_id: int, seed: int) -> None:
        cfg = self.cfg["env"]
        listener = _port_listener(self.port)
        if listener is not None:
            if not self._is_ours(listener):
                raise RuntimeError(f"env port {self.port} is held by foreign pid {listener}: {_cmdline(listener)[:300]}")
            self.stop()
        config.ensure_dirs()
        mode_path = config.RECORDS_DIR / "episode_mode.json"
        mode_path.write_text(json.dumps({"mode": "free", "task_scope": "official_single"}), encoding="utf-8")
        log_path = config.LOGS_DIR / f"env_libero_goal_task{task_id}.log"
        log = log_path.open("w", encoding="utf-8")
        env = dict(os.environ)
        env.update(
            {
                "HF_HUB_OFFLINE": "1",
                "MUJOCO_GL": "egl",
                "PYOPENGL_PLATFORM": "egl",
                "LIBERO_TYPE": "standard",
                "PYTHONPATH": f"{PROJECT / 'adapters' / 'env_runtime'}:{cfg['rpent_root']}",
            }
        )
        command = [
            cfg["python"],
            str(PROJECT / "adapters" / "libero_env_server.py"),
            "--transport", "http",
            "--host", self.host,
            "--port", str(self.port),
            "--suite", str(cfg["suite"]),
            "--task", str(task_id),
            "--seed", str(seed),
            "--max-episode-steps", str(cfg["max_episode_steps"]),
            "--cuda-device", str(cfg["cuda_device"]),
        ]
        proc = subprocess.Popen(
            command, cwd=str(PROJECT), env=env, stdout=log, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, start_new_session=True, close_fds=True
        )
        self.pid = proc.pid
        (config.RECORDS_DIR / "env_server.pid").write_text(str(proc.pid), encoding="utf-8")
        started = time.time()
        client = EnvClient(self.base_url)
        last_error = ""
        while time.time() - started < 240:
            if proc.poll() is not None:
                raise RuntimeError(f"env server exited with {proc.returncode}; see {log_path}")
            try:
                ident = client.identity()
                meta = client.meta()
                if meta.get("task") != task_id or meta.get("suite") != cfg["suite"] or meta.get("seed") != seed:
                    raise RuntimeError(
                        f"env meta mismatch: {meta} expected task={task_id} suite={cfg['suite']} seed={seed}"
                    )
                self.client = client
                self.task_id = task_id
                self.seed = seed
                self.identity = {**ident, "meta": meta, "pid": proc.pid, "load_s": round(time.time() - started, 1)}
                return
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                time.sleep(0.5)
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        raise RuntimeError(f"env server did not become ready in 240s: {last_error}; see {log_path}")

    def view(self) -> dict[str, Any]:
        return {
            "suite": self.cfg["env"]["suite"],
            "seed": self.seed,
            "configured_seed": self.cfg["env"]["seed"],
            "task_id": self.task_id,
            "port": self.port,
            "identity": self.identity,
            "ready": self.client is not None,
        }


class Engine:
    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        self.env = EnvManager(cfg)
        self.inference = InferenceClient(
            f"http://{cfg['services']['inference']['host']}:{cfg['services']['inference']['port']}"
        )
        self.agent_url = f"http://{cfg['services']['agent']['host']}:{cfg['services']['agent']['port']}"
        self.lock = threading.RLock()
        self.run_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.rollout_thread: threading.Thread | None = None
        self.phase = "starting"
        self.detail = "engine started; no scene loaded"
        self.task_id: int | None = None
        self.task: dict[str, Any] | None = None
        self.seed = int(cfg["env"]["seed"])
        self.instruction = ""
        self.episode_id: str | None = None
        self.run_id: int | None = None
        self.scene_epoch: int | None = None
        self.sim_action_count = 0
        self.failure_reason = ""
        self.failure_code = ""
        self.last_result: dict[str, Any] = {}
        self.last_stop: dict[str, Any] = {}
        self.plan: dict[str, Any] | None = None
        # Bumped on every lifecycle transition (load/reset/stop/new rollout). A
        # planner response that was in flight across one of those transitions
        # carries an older generation and must not be written into the new
        # episode's state.
        self.planner_generation = 0
        self.steps_done: list[dict[str, Any]] = []
        self.latest_frames: dict[str, Any] = {}
        self.frame_seq = 0
        self.health_cache: dict[str, Any] = {"at": 0.0, "value": {}}
        self.metrics = {
            "episodes_started": 0,
            "episodes_succeeded": 0,
            "episodes_failed": 0,
            "episodes_stopped": 0,
            "planner_calls": 0,
            "recovery_calls": 0,
            "last_planner_s": None,
            "last_episode_s": None,
        }

    # ---- identity / state ----

    def _health(self) -> dict[str, Any]:
        now = time.time()
        with self.lock:
            if now - self.health_cache["at"] < 5.0 and self.health_cache["value"]:
                return self.health_cache["value"]
        agent = {}
        try:
            agent = _get_json(f"{self.agent_url}/health", timeout_s=5)
        except Exception as exc:
            agent = {"status": "unreachable", "error": f"{type(exc).__name__}: {exc}"}
        inference = self.inference.health()
        action_cfg = self.cfg["action_model"]
        infer_ident = inference.get("identity") or {}
        openwam_row = infer_ident.get("openwam") or {}
        if infer_ident:
            action_identity = {
                "backend": infer_ident.get("backend") or action_cfg["backend"],
                "name": infer_ident.get("name") or action_cfg["name"],
                "checkpoint": infer_ident.get("checkpoint") or action_cfg["checkpoint"],
                "revision": infer_ident.get("revision") or action_cfg["revision"],
                "action_chunk": infer_ident.get("action_horizon") or action_cfg["action_chunk"],
                "action_dim": infer_ident.get("action_dim") or action_cfg["action_dim"],
                "action_space": infer_ident.get("action_space"),
                "execute_horizon": infer_ident.get("execute_horizon"),
                "denoise_steps": infer_ident.get("denoise_steps"),
                "inference_mode": infer_ident.get("inference_mode") or inference.get("mode"),
                "control_frequency_hz": action_cfg["control_frequency_hz"],
                "counterpart": infer_ident.get("counterpart"),
                "server": infer_ident.get("server"),
                "openwam": {**action_cfg.get("openwam", {}), **openwam_row},
            }
        else:
            action_identity = {
                "backend": action_cfg["backend"],
                "name": action_cfg["name"],
                "checkpoint": action_cfg["checkpoint"],
                "revision": action_cfg["revision"],
                "action_chunk": action_cfg["action_chunk"],
                "action_dim": action_cfg["action_dim"],
                "control_frequency_hz": action_cfg["control_frequency_hz"],
                "openwam": {
                    **action_cfg.get("openwam", {}),
                    "integration": "not_on_execution_path",
                    "reason": f"inference service unreachable: {inference.get('error') or inference.get('status')}",
                },
            }
        identity = {
            "agent": {
                "service": agent.get("service", "agent"),
                "status": agent.get("status", "unreachable"),
                "mode": agent.get("mode", config.agent_mode()),
                "modes": list(config.AGENT_MODES),
                "model": agent.get("model", self.cfg["agent"]["llm_model"]),
                "llm_model": agent.get("llm_model", self.cfg["agent"]["llm_model"]),
                "d1": agent.get("d1"),
                "planner_vision": agent.get("planner_vision", False),
                "llm": agent.get("llm"),
                "jev": agent.get("jev", {"enabled": False}),
            },
            "action_model": action_identity,
            "inference": {
                "service": inference.get("service", "inference"),
                "status": inference.get("status", "unreachable"),
                "mode": inference.get("mode", self.cfg["inference"]["mode"]),
                "backend": inference.get("backend"),
                "identity": inference.get("identity"),
                "metrics": inference.get("metrics"),
            },
            "env": self.env.view(),
            "engine": {"phase": self.phase, "detail": self.detail},
        }
        with self.lock:
            self.health_cache = {"at": now, "value": identity}
        return identity

    def status(self) -> dict[str, Any]:
        health = self._health()
        with self.lock:
            return {
                "service": "engine",
                "status": "ok",
                "phase": self.phase,
                "detail": self.detail,
                "task_id": self.task_id,
                "seed": self.seed,
                "instruction": self.instruction,
                "episode_id": self.episode_id,
                "run_id": self.run_id,
                "planner_generation": self.planner_generation,
                "scene_epoch": self.scene_epoch,
                "sim_action_count": self.sim_action_count,
                "frame_seq": self.frame_seq,
                "failure_reason": self.failure_reason,
                "failure_code": self.failure_code,
                "last_result": self.last_result,
                "last_stop": self.last_stop,
                "plan": self.plan,
                "steps": self.steps_done,
                "metrics": self.metrics,
                "identity": health,
            }

    def _set_phase(self, phase: str, detail: str, **extra: Any) -> None:
        with self.lock:
            self.phase = phase
            self.detail = detail
            for key, value in extra.items():
                setattr(self, key, value)
        records.append_journal(
            {"t": time.time(), "stage": phase, "episode_id": self.episode_id, "detail": detail}
        )

    def _set_frames(self, obs: dict[str, Any], snapshot: dict[str, Any]) -> None:
        if not isinstance(obs, dict) or obs.get("main_images") is None:
            return
        with self.lock:
            self.latest_frames = {
                "agent": np.asarray(obs["main_images"], dtype=np.uint8),
                "wrist": np.asarray(obs.get("wrist_images"), dtype=np.uint8) if obs.get("wrist_images") is not None else None,
                "sim_action_count": snapshot.get("sim_action_count"),
                "observation_id": snapshot.get("observation_id"),
                "scene_epoch": snapshot.get("scene_epoch"),
                "episode_id": self.episode_id,
                "frame_seq": self.frame_seq + 1,
            }
            self.frame_seq += 1
            self.sim_action_count = int(snapshot.get("sim_action_count") or 0)

    def frame(self, view: str) -> bytes:
        with self.lock:
            row = dict(self.latest_frames)
        key = "agent" if view in {"agent", "agentview", "main"} else "wrist"
        arr = row.get(key)
        if arr is None:
            raise KeyError(view)
        image = Image.fromarray(np.asarray(arr, dtype=np.uint8))
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=90, subsampling=2)
        return buf.getvalue()

    # ---- scene / lifecycle ----

    def load(self, task_id: int, seed: int | None = None) -> dict[str, Any]:
        task = config.task_for(task_id)
        wanted_seed = int(self.cfg["env"]["seed"] if seed is None else seed)
        with self.run_lock:
            if self.phase in {"planning", "running", "recovering", "stopping"}:
                raise RuntimeError("cannot load while an episode is active")
            self._set_phase(
                "loading",
                f"starting LIBERO {self.cfg['env']['suite']} task {task_id} seed {wanted_seed}",
            )
            try:
                env = self.env.ensure(task_id, wanted_seed)
                obs = env.reset()
                snap = env.state_snapshot()
                self.inference.reset(f"load_task_{task_id}_seed{wanted_seed}")
                with self.lock:
                    self.task_id = task_id
                    self.task = task
                    self.seed = wanted_seed
                    self.instruction = str(task["instruction"])
                    self.episode_id = None
                    self.run_id = None
                    self.planner_generation += 1
                    self.scene_epoch = snap.get("scene_epoch")
                    self.failure_reason = ""
                    self.failure_code = ""
                    self.plan = None
                    self.steps_done = []
                    self.last_result = {}
                    self.sim_action_count = int(snap.get("sim_action_count") or 0)
                self._set_frames(obs, snap)
                self._set_phase("ready", f"loaded {task['bddl']} ({task['kind']})")
                return {
                    "status": "ok",
                    "task_id": task_id,
                    "seed": wanted_seed,
                    "instruction": self.instruction,
                    "env_id": f"libero_goal/{task['bddl']}",
                    "identity": self._health(),
                    "scene_epoch": self.scene_epoch,
                }
            except Exception as exc:
                self._set_phase("failed", f"load failed: {type(exc).__name__}: {exc}", failure_reason=str(exc), failure_code="load_failed")
                raise

    def reset(self) -> dict[str, Any]:
        self.stop(reason="reset")
        thread = self.rollout_thread
        if thread is not None and thread.is_alive():
            # A reset while the old rollout still owns the env would race the
            # new scene and could let stale verification land in it.
            raise RuntimeError("rollout thread is still active after stop; reset refused")
        if self.task_id is None:
            return {"status": "ok", "phase": "ready", "detail": "no scene loaded"}
        with self.run_lock:
            env = self.env.ensure(self.task_id, self.seed)
            obs = env.reset()
            snap = env.state_snapshot()
            self.inference.reset("reset")
            with self.lock:
                self.episode_id = None
                self.run_id = None
                self.planner_generation += 1
                self.scene_epoch = snap.get("scene_epoch")
                self.failure_reason = ""
                self.failure_code = ""
                self.plan = None
                self.steps_done = []
                self.last_result = {}
                self.sim_action_count = int(snap.get("sim_action_count") or 0)
            self._set_frames(obs, snap)
            self._set_phase("ready", f"reset task {self.task_id}")
            return {"status": "ok", "phase": "ready", "scene_epoch": self.scene_epoch}

    def stop(self, reason: str = "user_stop") -> dict[str, Any]:
        with self.lock:
            active = self.rollout_thread is not None and self.rollout_thread.is_alive()
            sim_at_request = self.sim_action_count
            if active:
                # Any planner call still in flight belongs to a dead rollout.
                self.planner_generation += 1
                self.stop_event.set()
                self._set_phase("stopping", f"stop requested: {reason}")
        # Cancelling the inference request also clears the model-side
        # receding-horizon buffer, so no generated action can be replayed into
        # the next episode.
        self.inference.cancel(reason)
        thread = self.rollout_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=float(self.cfg["timing"]["stop_join_s"]))
        alive = bool(thread and thread.is_alive())
        sim_at_ack = sim_at_request
        if not alive:
            try:
                sim_at_ack = int(self.env.state_snapshot().get("sim_action_count") or sim_at_request)
            except Exception:
                sim_at_ack = sim_at_request
        budget = {
            "reason": reason,
            "sim_at_request": int(sim_at_request),
            "sim_at_ack": int(sim_at_ack),
            # An env.step already handed to the simulator when the flag rose
            # cannot be recalled; the contract is "no new action is issued",
            # so this is 0 or 1 and is reported instead of hidden.
            "in_flight_actions": max(0, int(sim_at_ack) - int(sim_at_request)),
            "thread_alive": alive,
        }
        with self.lock:
            self.last_stop = budget
        return {"status": "ok", "active": alive, "phase": self.phase, "stop": budget}

    IDLE_PHASES = {"starting", "ready", "succeeded", "failed", "stopped"}

    def set_agent_mode(self, mode: str) -> dict[str, Any]:
        """Switch the Agent model only while no rollout owns the episode.

        The mode is persisted in config/engine.json, which the Agent service
        re-reads per request. A switch during planning/running/recovering or
        while the rollout thread is alive is refused, so a plan is never
        produced by one model and recorded as another.
        """
        if mode not in config.AGENT_MODES:
            return {"status": "bad_request", "error": f"unknown agent mode {mode!r}", "mode": config.agent_mode()}
        with self.lock:
            phase = self.phase
            active = self.rollout_thread is not None and self.rollout_thread.is_alive()
            if active or phase not in self.IDLE_PHASES:
                return {
                    "status": "busy",
                    "error": f"agent mode can only change while idle; phase={phase}",
                    "phase": phase,
                    "mode": config.agent_mode(),
                }
        if mode != "qwen":
            d1_cfg = self.cfg["agent"]["d1"]
            try:
                health = _get_json(f"{d1_cfg['url']}/health", timeout_s=3)
            except Exception as exc:
                health = {"status": "unreachable", "error": f"{type(exc).__name__}: {exc}"}
            if health.get("status") != "ok":
                return {
                    "status": "unavailable",
                    "error": "d1 service is not ready; start it before selecting a d1 mode",
                    "d1": health,
                    "mode": config.agent_mode(),
                }
        config.set_agent_mode(mode)
        with self.lock:
            self.health_cache = {"at": 0.0, "value": {}}
        return {"status": "ok", "mode": mode, "phase": phase}


    def start_run(
        self, task_id: int, instruction: str | None, seed: int | None = None
    ) -> dict[str, Any]:
        task = config.task_for(task_id)
        with self.run_lock:
            if self.rollout_thread is not None and self.rollout_thread.is_alive():
                raise RuntimeError("an episode is already running")
            seed_changed = seed is not None and int(seed) != self.seed
            needs_load = seed_changed or self.task_id != task_id or self.env.client is None
        if needs_load:
            # ``load`` takes the same lock; call it after releasing it here.
            self.load(task_id, int(seed) if seed is not None else self.seed)
        with self.lock:
            self.stop_event = threading.Event()
            self.episode_id = f"ep-{time.strftime('%Y%m%d-%H%M%S', time.localtime())}-{uuid.uuid4().hex[:6]}"
            self.task = task
            self.task_id = task_id
            self.instruction = (instruction or task["instruction"]).strip() or task["instruction"]
            self.failure_reason = ""
            self.failure_code = ""
            self.plan = None
            self.steps_done = []
            self.last_result = {}
            self.metrics["episodes_started"] += 1
            self.run_id = self._next_run_id(task_id)
            self.planner_generation += 1
            planner_generation = self.planner_generation
            episode_id = self.episode_id
            run_id = self.run_id
        self._set_phase("running", "preparing episode")
        thread = threading.Thread(
            target=self._run_episode,
            args=(task, self.instruction, episode_id, run_id, self.stop_event, planner_generation),
            name=f"episode-{episode_id}",
            daemon=True,
        )
        self.rollout_thread = thread
        thread.start()
        return {"status": "started", "episode_id": episode_id, "run_id": run_id, "task_id": task_id}

    def _prepare_env(self, task_id: int):
        """Ensure a live env server and reset it, respawning once on a crash.

        A robosuite reset can segfault after many resets in one process; the
        old code then died before entering its try block, leaving the episode
        wedged in "preparing". Respawning bounds that to one reload.
        """
        try:
            env = self.env.ensure(task_id, self.seed)
            obs = env.reset()
            return env, obs, env.state_snapshot()
        except Exception as first:  # noqa: BLE001 - retry once on a fresh process
            self.metrics["env_respawn"] = int(self.metrics.get("env_respawn", 0)) + 1
            records.append_journal(
                {"t": time.time(), "stage": "env_respawn", "task_id": task_id,
                 "reason": f"{type(first).__name__}: {first}"}
            )
            self.env.stop()
            self.env._start(task_id, self.seed)
            env = self.env.ensure(task_id, self.seed)
            obs = env.reset()
            return env, obs, env.state_snapshot()

    def _next_run_id(self, task_id: int) -> int:
        count = 0
        for path in config.RUNS_DIR.glob("*.json"):
            try:
                row = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if row.get("task_id") == task_id:
                count += 1
        return count + 1

    # ---- rollout ----

    def _action_backend(self) -> str:
        """The backend that is actually serving, not the one in the stale config.

        The inference service is restarted by ``scripts/set_backend.sh`` while
        the engine keeps running, so ``self.cfg`` can name a backend that is no
        longer serving. The inference service is authoritative; the config is
        only a fallback when it is unreachable.
        """
        try:
            row = self.inference.health()
            name = row.get("backend")
            if name:
                return str(name).lower()
        except Exception:
            pass
        return str(self.cfg["action_model"].get("backend") or "pi05").lower()

    def _run_whole_task(
        self,
        *,
        env: EnvClient,
        obs: dict[str, Any],
        snapshot: dict[str, Any],
        step: dict[str, Any],
        task: dict[str, Any],
        token: str,
        episode_id: str,
        stop_event: threading.Event,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        """Run one whole-task policy rollout; returns (record, obs, snapshot)."""
        openwam = self.cfg["action_model"]["openwam"]
        runner = WholeTaskRunner(
            env=env,
            inference=self.inference,
            token=token,
            episode_id=episode_id,
            task=task,
            step=step,
            obs=obs,
            snapshot=snapshot,
            baseline=snapshot,
            stop_event=stop_event,
            journal=lambda stage, row: records.append_journal(
                {"t": time.time(), "stage": stage, "episode_id": episode_id, "row": row}
            ),
            max_steps=int(openwam.get("max_steps") or 600),
            timeout_s=float(openwam.get("timeout_s") or 300),
            gpu_index=openwam.get("gpu_index"),
            frame_sink=self._set_frames,
        )
        result = runner.run()
        return result, runner.obs, runner.snapshot

    def _run_episode(
        self,
        task: dict[str, Any],
        instruction: str,
        episode_id: str,
        run_id: int,
        stop_event: threading.Event,
        planner_generation: int,
    ) -> None:
        started = time.time()
        started_iso = _now_iso()
        recovery_log: list[dict[str, Any]] = []
        try:
            env, obs, snapshot = self._prepare_env(int(task["task_id"]))
        except Exception as exc:  # noqa: BLE001 - must always reach a terminal state
            message = f"env preparation failed: {type(exc).__name__}: {exc}"
            self._set_phase("failed", message, failure_code="env_reset_failed", failure_reason=message)
            with self.lock:
                self.last_result = {
                    "episode_id": episode_id,
                    "run_id": run_id,
                    "task_id": task["task_id"],
                    "execution_status": "failed",
                    "success": False,
                    "native_task_success": False,
                    "failure_code": "env_reset_failed",
                    "reason": message,
                    "record": None,
                    "duration_s": round(time.time() - started, 3),
                }
            return
        baseline = snapshot
        initial_scene = _scene_fingerprint(snapshot)
        token = self.inference.reset(f"episode:{episode_id}")
        with self.lock:
            self.scene_epoch = snapshot.get("scene_epoch")
            self.sim_action_count = int(snapshot.get("sim_action_count") or 0)
        self._set_frames(obs, snapshot)
        timing = {
            "planner_s": 0.0,
            "execution_s": 0.0,
            "release_s": 0.0,
            "verification_s": 0.0,
            "inference_s_sum": 0.0,
            "env_step_s_sum": 0.0,
            "snapshot_s_sum": 0.0,
            "observe_s_sum": 0.0,
            "convert_s_sum": 0.0,
            "frame_s_sum": 0.0,
            "plan_to_first_action_s": 0.0,
            "chunk_gap_s_sum": 0.0,
            "chunk_decision_wait_s_sum": 0.0,
            "prefetch_inference_overlap_s_est": 0.0,
            "recovery_s": 0.0,
        }
        step_results: list[dict[str, Any]] = []
        step_attempts: dict[tuple[Any, ...], int] = {}
        home_eef = (baseline.get("sim") or {}).get("eef_pos")
        plan = None
        attempts: list[dict[str, Any]] = []
        failure: dict[str, Any] | None = None
        execution_status = "failed"
        native_ever = False
        plan_completed_at: float | None = None
        openwam_stats: dict[str, Any] | None = None
        try:
            self._set_phase("planning", f"{config.agent_mode()} structured plan")
            plan_started = time.perf_counter()
            plan_response = _post_json(
                f"{self.agent_url}/plan",
                {
                    "instruction": instruction,
                    "task_id": task["task_id"],
                    "snapshot": snapshot,
                    "attempts": [],
                    "backend": self._action_backend(),
                },
                timeout_s=180,
            )
            timing["planner_s"] += time.perf_counter() - plan_started
            self.metrics["planner_calls"] += 1
            self.metrics["last_planner_s"] = round(timing["planner_s"], 3)
            with self.lock:
                stale = planner_generation != self.planner_generation
                if stale:
                    # stop/reset/load happened while the Agent was planning, so
                    # this answer belongs to an episode that no longer exists.
                    self.metrics["stale_plans_discarded"] = int(self.metrics.get("stale_plans_discarded", 0)) + 1
            if stale:
                records.append_journal(
                    {
                        "t": time.time(),
                        "stage": "planner_result_discarded",
                        "episode_id": episode_id,
                        "planner_generation": planner_generation,
                        "current_generation": self.planner_generation,
                        "planner_s": round(timing["planner_s"], 3),
                        "agent_mode": plan_response.get("agent_mode"),
                    }
                )
                raise Stopped("planner result from an expired generation")
            with self.lock:
                self.plan = {
                    "status": plan_response.get("status"),
                    "steps": plan_response.get("plan", {}).get("steps") if isinstance(plan_response.get("plan"), dict) else None,
                    "errors": plan_response.get("errors"),
                    "attempts": plan_response.get("attempts"),
                    "total_s": plan_response.get("total_s"),
                    "source": plan_response.get("plan_source")
                    or ((plan_response.get("plan") or {}).get("plan_source") if isinstance(plan_response.get("plan"), dict) else None)
                    or "qwen3.5-9b",
                    "agent_mode": plan_response.get("agent_mode"),
                    "decision": plan_response.get("decision"),
                    "fallback": plan_response.get("fallback"),
                    "planner_vision": False,
                }
            if plan_response.get("status") != "ok":
                raise SkillFailure(
                    "plan_invalid",
                    "Agent plan was not valid",
                    {"response": plan_response},
                )
            plan = plan_response["plan"]
            plan_completed_at = time.perf_counter()
            steps = list(plan["steps"])
            # A plan is validated against the catalog the planner was given;
            # this is the second gate, because a wrong executor would run the
            # right skill against the wrong action contract (8-D proprio into a
            # 10-D EEF10 backend) and fail halfway through an episode.
            backend = self._action_backend()
            planned = [step.get("skill") for step in steps]
            if backend == "openwam" and planned != ["whole_task"]:
                raise SkillFailure(
                    "plan_backend_mismatch",
                    "openwam backend requires exactly one whole_task step",
                    {"backend": backend, "planned_skills": planned},
                )
            if backend != "openwam" and "whole_task" in planned:
                raise SkillFailure(
                    "plan_backend_mismatch",
                    f"{backend} backend cannot execute whole_task",
                    {"backend": backend, "planned_skills": planned},
                )
            # The user-visible goal is the final step of the initial plan.
            # A replan may reorder or add steps, but dropping this step would
            # silently solve a different task.
            goal_step = dict(steps[-1]) if steps else None
            index = 0
            retries = 0
            replans = 0
            max_recoveries = int(self.cfg["execution"]["max_recoveries"])
            while index < len(steps):
                if stop_event.is_set():
                    raise Stopped("stop requested")
                step = steps[index]
                pre_state = (
                    completion.goal_state(snapshot, step, task, snapshot)
                    if step.get("skill") == "open_drawer"
                    else None
                )
                if pre_state and pre_state.get("ok"):
                    # A replan can re-issue a drawer step that is already
                    # satisfied; re-running it only burns actions and moves
                    # the arm away from the handle.
                    step_results.append(
                        {
                            "step": step,
                            "ok": True,
                            "skipped": "already_open",
                            "state": pre_state,
                            "chunks_used": 0,
                            "sim_actions": 0,
                        }
                    )
                    with self.lock:
                        self.steps_done = [s for s in step_results]
                    index += 1
                    continue
                self._set_phase(
                    "running",
                    f"step {index + 1}/{len(steps)} {step['skill']} {step.get('object_n') or ''} -> {step.get('target_n') or ''}".strip(),
                )
                skill_started = time.perf_counter()
                attempt_key = (step.get("skill"), step.get("object_n"), step.get("target_n"), step.get("relation"))
                attempt_no = step_attempts.get(attempt_key, 0) + 1
                step_attempts[attempt_key] = attempt_no
                runner_step = dict(step)
                if attempt_no > 1:
                    # Retries must not start from the pose where the previous
                    # attempt got stuck; the runner brings the arm back to the
                    # episode's initial EEF pose first (bounded, see skills.py).
                    runner_step["attempt"] = attempt_no
                if step.get("skill") == "whole_task":
                    record, obs, snapshot = self._run_whole_task(
                        env=env,
                        obs=obs,
                        snapshot=snapshot,
                        step=runner_step,
                        task=task,
                        token=token,
                        episode_id=episode_id,
                        stop_event=stop_event,
                    )
                    timing["execution_s"] += time.perf_counter() - skill_started
                    timing["inference_s_sum"] += record["timing"]["inference_s_sum"]
                    timing["env_step_s_sum"] += record["timing"]["env_step_s_sum"]
                    timing["snapshot_s_sum"] += record["timing"]["snapshot_s_sum"]
                    timing["observe_s_sum"] += record["timing"].get("observe_s_sum", 0.0)
                    timing["convert_s_sum"] += record["timing"].get("convert_s_sum", 0.0)
                    timing["frame_s_sum"] += record["timing"].get("frame_s_sum", 0.0)
                    if openwam_stats is None:
                        wall = record.get("wall") or {}
                        first_action_s = wall.get("first_action_s")
                        if plan_completed_at is not None and isinstance(first_action_s, (int, float)):
                            timing["plan_to_first_action_s"] = (
                                skill_started - plan_completed_at + float(first_action_s)
                            )
                        openwam_stats = {
                            "wall": wall,
                            "chunk_stats": record.get("chunk_stats"),
                            "prompt": record.get("prompt"),
                            "sim_actions": record.get("sim_actions"),
                            "chunks_used": record.get("chunks_used"),
                        }
                    record["step"] = step
                    record["attempt"] = attempt_no
                    step_results.append(record)
                    with self.lock:
                        self.steps_done = [s for s in step_results]
                    native_ever = native_ever or bool(record.get("ok"))
                    self._set_frames(obs, snapshot)
                    if not record.get("ok"):
                        evidence = record.get("verification") or {}
                        failure = {
                            "reason_code": evidence.get("reason_code") or "native_goal_not_reached",
                            "message": f"whole_task finished {record.get('sim_actions')} actions without the native goal",
                            "evidence": evidence,
                            "step": step,
                            "snapshot": snapshot,
                        }
                        # The policy is whole-task: a sub-skill replan is not a
                        # legal recovery. Only an explicit retry/abort from the
                        # Agent is honoured, and the failure stays recorded.
                        self._set_phase("recovering", "whole_task native goal not reached")
                        recover_started = time.perf_counter()
                        recovery = _post_json(
                            f"{self.agent_url}/recover",
                            {
                                "instruction": instruction,
                                "task_id": task["task_id"],
                                "snapshot": snapshot,
                                "failed_step": step,
                                "failure": _recovery_failure(failure),
                                "completed_steps": [],
                                "attempts": [
                                    {
                                        "step": attempt.get("step"),
                                        "failure": _recovery_failure(attempt.get("failure") or {}),
                                        "chunks_used": attempt.get("chunks_used"),
                                    }
                                    for attempt in attempts
                                ],
                                "backend": self._action_backend(),
                            },
                            timeout_s=180,
                        )
                        timing["recovery_s"] += time.perf_counter() - recover_started
                        self.metrics["recovery_calls"] += 1
                        decision = recovery.get("decision")
                        recovery_log.append(
                            {
                                "step": step,
                                "failure_code": failure["reason_code"],
                                "decision": decision,
                                "policy": recovery.get("policy") or "qwen",
                                "reason": recovery.get("reason"),
                                "agent_s": recovery.get("total_s"),
                            }
                        )
                        attempts.append({"step": step, "failure": failure})
                        if recovery.get("status") != "ok":
                            raise SkillFailure("recovery_invalid", "Agent recovery decision was invalid", {"recovery": recovery})
                        if decision == "retry" and retries < max_recoveries:
                            retries += 1
                            step_results.pop()
                            continue
                        raise SkillFailure(
                            failure["reason_code"],
                            failure["message"],
                            {"verification": evidence, "recovery": recovery},
                        )
                    index += 1
                    retries = 0
                    continue
                runner = SkillRunner(
                    env=env,
                    inference=self.inference,
                    token=token,
                    episode_id=episode_id,
                    task=task,
                    step=runner_step,
                    obs=obs,
                    snapshot=snapshot,
                    baseline=snapshot,
                    stop_event=stop_event,
                    journal=lambda stage, row: records.append_journal(
                        {"t": time.time(), "stage": stage, "episode_id": episode_id, "row": row}
                    ),
                    max_chunk_age_actions=int(self.cfg["inference"]["max_chunk_age_actions"]),
                    allow_prefetch=bool(self.cfg["inference"]["mode"] == "async_chunk_prefetch"),
                    chunk_budget=self._budget(step["skill"]),
                    chunk_budget_hard=self._budget_hard(step["skill"]),
                    frame_sink=self._set_frames,
                    home_eef=home_eef,
                )
                try:
                    result = runner.run()
                except Stopped:
                    raise
                except SkillFailure as exc:
                    failure = {
                        "reason_code": exc.reason_code,
                        "message": exc.message,
                        "evidence": exc.evidence,
                        "step": step,
                        "snapshot": runner.snapshot,
                        "chunks_used": runner.chunk_index,
                        "grasp": {"held": runner.held, "held_streak": runner.held_streak},
                    }
                    record = {
                        "step": step,
                        "ok": False,
                        "failure": failure,
                        "chunks_used": runner.chunk_index,
                        "attempt": attempt_no,
                        "rehome": runner.rehome,
                        "sim_actions": runner.sim_actions,
                        "timing": {
                            "skill_s": round(time.perf_counter() - skill_started, 3),
                            "inference_s_sum": round(runner.inference_s, 3),
                            "env_step_s_sum": round(runner.env_step_s, 3),
                            "snapshot_s_sum": round(runner.snapshot_s, 3),
                            "chunk_gap_s_sum": round(runner.chunk_gap_s, 3),
                            "chunk_decision_wait_s_sum": round(runner.decision_wait_s, 3),
                            "prefetch_inference_overlap_s_est": round(runner.prefetch_saved_s, 3),
                            "release_s": round(runner.last_release_s, 3),
                            "verification_s": round(runner.last_verify_s, 3),
                        },
                        "prefetch": {
                            "used": runner.prefetch_used,
                            "discarded": runner.prefetch_discarded,
                            "reinfer": runner.prefetch_reinfer,
                        },
                        "chunks": runner.chunks,
                        "observations": runner.observations,
                        "action_trace": runner.action_trace,
                    }
                    step_results.append(record)
                    native_ever = native_ever or bool(runner.native_events)
                    timing["execution_s"] += time.perf_counter() - skill_started
                    self._accumulate_timing(timing, runner)
                    obs = runner.obs
                    snapshot = runner.snapshot
                    self._set_frames(obs, snapshot)
                    with self.lock:
                        self.steps_done = [s for s in step_results]
                    if retries + replans >= max_recoveries:
                        raise
                    self._set_phase("recovering", f"{step['skill']} failed: {exc.reason_code}")
                    recover_started = time.perf_counter()
                    attempts.append(
                        {
                            "step": step,
                            "failure": failure,
                            "chunks_used": runner.chunk_index,
                        }
                    )
                    recovery = _post_json(
                        f"{self.agent_url}/recover",
                        {
                            "instruction": instruction,
                            "task_id": task["task_id"],
                            "snapshot": snapshot,
                            "failed_step": step,
                            "failure": _recovery_failure(failure),
                            "completed_steps": [row.get("step") for row in step_results],
                            "backend": self._action_backend(),
                            "attempts": [
                                {
                                    "step": attempt.get("step"),
                                    "failure": _recovery_failure(attempt.get("failure") or {}),
                                    "chunks_used": attempt.get("chunks_used"),
                                }
                                for attempt in attempts
                            ],
                        },
                        timeout_s=180,
                    )
                    timing["recovery_s"] += time.perf_counter() - recover_started
                    self.metrics["recovery_calls"] += 1
                    recovery_log.append(
                        {
                            "step": step,
                            "failure_code": exc.reason_code,
                            "chunks_used": runner.chunk_index,
                            "decision": recovery.get("decision"),
                            "policy": recovery.get("policy") or "qwen",
                            "reason": recovery.get("reason"),
                            "agent_s": recovery.get("total_s"),
                        }
                    )
                    records.append_journal(
                        {"t": time.time(), "stage": "recovery", "episode_id": episode_id, "decision": recovery.get("decision"), "reason": recovery.get("reason")}
                    )
                    if recovery.get("status") != "ok":
                        raise SkillFailure("recovery_invalid", "Agent recovery decision was invalid", {"recovery": recovery})
                    decision = recovery.get("decision")
                    if decision == "retry":
                        retries += 1
                        step_results.pop()
                        self._set_phase("running", f"retry {step['skill']} ({retries}/{max_recoveries})")
                        continue
                    if decision == "replan":
                        new_steps = recovery.get("steps") or []
                        if not new_steps:
                            raise SkillFailure("recovery_empty_replan", "replan returned no validated steps", {"recovery": recovery})
                        if not plan_contract.preserves_goal(new_steps, goal_step):
                            records.append_journal(
                                {"t": time.time(), "stage": "replan_rejected", "episode_id": episode_id,
                                 "reason": "does not preserve the initial goal step", "steps": new_steps}
                            )
                            retries += 1
                            step_results.pop()
                            continue
                        replans += 1
                        step_results.pop()
                        steps = steps[:index] + list(new_steps)
                        self._set_phase("running", f"replanned {len(new_steps)} step(s)")
                        continue
                    raise SkillFailure("recovery_abort", str(recovery.get("reason") or "Agent aborted"), {"recovery": recovery})
                else:
                    timing["execution_s"] += time.perf_counter() - skill_started
                    self._accumulate_timing(timing, runner)
                    obs = runner.obs
                    snapshot = runner.snapshot
                    result["step"] = step
                    step_results.append(result)
                    self._set_frames(obs, snapshot)
                    with self.lock:
                        self.steps_done = [s for s in step_results]
                    native_ever = native_ever or bool(result.get("native_events"))
                    if not result.get("ok"):
                        failure = {
                            "reason_code": (result.get("verification") or {}).get("reason_code") or "verification_failed",
                            "message": f"{step['skill']} finished but completion evidence failed",
                            "evidence": result.get("verification"),
                            "step": step,
                            "snapshot": snapshot,
                        }
                        if retries + replans >= max_recoveries:
                            raise SkillFailure(failure["reason_code"], failure["message"], failure["evidence"])
                        self._set_phase("recovering", f"{step['skill']} verification failed")
                        recover_started = time.perf_counter()
                        recovery = _post_json(
                            f"{self.agent_url}/recover",
                            {
                                "instruction": instruction,
                                "task_id": task["task_id"],
                                "snapshot": snapshot,
                                "failed_step": step,
                                "failure": _recovery_failure(failure),
                                "completed_steps": [row.get("step") for row in step_results],
                                "backend": self._action_backend(),
                                "attempts": [
                                    {
                                        "step": attempt.get("step"),
                                        "failure": _recovery_failure(attempt.get("failure") or {}),
                                        "chunks_used": attempt.get("chunks_used"),
                                    }
                                    for attempt in attempts
                                ],
                            },
                            timeout_s=180,
                        )
                        timing["recovery_s"] += time.perf_counter() - recover_started
                        self.metrics["recovery_calls"] += 1
                        recovery_log.append(
                            {
                                "step": step,
                                "failure_code": failure["reason_code"],
                                "chunks_used": result.get("chunks_used"),
                                "decision": recovery.get("decision"),
                                "policy": recovery.get("policy") or "qwen",
                                "reason": recovery.get("reason"),
                                "agent_s": recovery.get("total_s"),
                            }
                        )
                        attempts.append({"step": step, "failure": failure})
                        if recovery.get("status") != "ok":
                            raise SkillFailure("recovery_invalid", "Agent recovery decision was invalid", {"recovery": recovery})
                        if recovery.get("decision") == "retry":
                            retries += 1
                            step_results.pop()
                            continue
                        if recovery.get("decision") == "replan" and recovery.get("steps"):
                            new_steps = list(recovery["steps"])
                            if not plan_contract.preserves_goal(new_steps, goal_step):
                                records.append_journal(
                                    {"t": time.time(), "stage": "replan_rejected", "episode_id": episode_id,
                                     "reason": "does not preserve the initial goal step", "steps": new_steps}
                                )
                                retries += 1
                                step_results.pop()
                                continue
                            replans += 1
                            step_results.pop()
                            steps = steps[:index] + new_steps
                            continue
                        raise SkillFailure("recovery_abort", str(recovery.get("reason") or "Agent aborted"), {"recovery": recovery})
                    index += 1
                    retries = 0
            # fresh cross-skill verification: every step must still hold.
            # Placing into an open drawer pushes its front panel and can close
            # it by 1-3 cm, so an earlier open_drawer step may regress. When
            # that is the only failure, re-open the drawer once (bounded) and
            # verify again instead of failing an otherwise complete task.
            verify_started = time.perf_counter()
            verify_repairs = 0
            while True:
                self._set_phase("verifying", "final task verification")
                samples = []
                for _ in range(3):
                    obs, _r, _term, _trunc, _info = env.step(np.zeros(7, dtype=np.float64) + np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float64))
                    snapshot = env.state_snapshot()
                    samples.append(snapshot)
                self._set_frames(obs, snapshot)
                final_steps = []
                all_ok = True
                failed = []
                for position, result in enumerate(step_results):
                    step = result["step"]
                    release = result.get("release")
                    if step.get("skill") == "whole_task":
                        # Whole-task policies own the whole trajectory: the
                        # only user-goal check is the native LIBERO goal, and
                        # it must hold on every fresh sample. Release is
                        # measured and reported, never required or invented.
                        flags = [bool(s.get("native_goal")) for s in samples]
                        ok = all(flags)
                        openings = [
                            (s.get("sim") or {}).get("gripper_opening_m") for s in samples
                        ]
                        measured = [float(v) for v in openings if isinstance(v, (int, float))]
                        check = {
                            "ok": ok,
                            "method": "native_libero_goal",
                            "samples": flags,
                            "reason_code": None if ok else "native_goal_not_reached",
                            "gripper_opening_m": [round(v, 4) for v in measured],
                            "release_observed": bool(
                                result.get("release", {}).get("events")
                            ) or any(v > 0.05 for v in measured),
                            "release_source": "measured_gripper_separation",
                            "note": (
                                "official LIBERO goal atoms; the policy's own gripper "
                                "commands are measured, not replaced by executor receipts"
                            ),
                        }
                    elif step.get("skill") == "pick_place":
                        check = completion.verification(samples, step, task, baseline, release=release)
                    else:
                        check = completion.verification(samples, step, task, baseline)
                    ok = bool(check.get("ok"))
                    final_steps.append({"step": step, "ok": ok, "verification": check})
                    all_ok = all_ok and ok
                    if not ok:
                        failed.append((position, step))
                native_ever = native_ever or any(bool(s.get("native_goal")) for s in samples)
                if all_ok:
                    break
                repairable = [(position, step) for position, step in failed if step.get("skill") == "open_drawer"]
                if verify_repairs >= 1 or not repairable or len(repairable) != len(failed):
                    raise SkillFailure(
                        "final_verification_failed",
                        "final cross-skill verification failed",
                        {"steps": final_steps, "snapshot": snapshot},
                    )
                verify_repairs += 1
                position, step = repairable[0]
                self._set_phase("repairing", f"re-open drawer {step.get('target_n')}")
                repair_started = time.perf_counter()
                repair_runner = SkillRunner(
                    env=env,
                    inference=self.inference,
                    token=token,
                    episode_id=episode_id,
                    task=task,
                    step=step,
                    obs=obs,
                    snapshot=snapshot,
                    baseline=snapshot,
                    stop_event=stop_event,
                    journal=lambda stage, row: records.append_journal(
                        {"t": time.time(), "stage": stage, "episode_id": episode_id, "row": row}
                    ),
                    max_chunk_age_actions=int(self.cfg["inference"]["max_chunk_age_actions"]),
                    allow_prefetch=bool(self.cfg["inference"]["mode"] == "async_chunk_prefetch"),
                    chunk_budget=self._budget(step["skill"]),
                    chunk_budget_hard=self._budget_hard(step["skill"]),
                    frame_sink=self._set_frames,
                )
                repair = repair_runner.run()
                repair["step"] = step
                timing["verification_s"] += time.perf_counter() - repair_started
                step_results[position] = repair
                obs = repair_runner.obs
                snapshot = repair_runner.snapshot
                with self.lock:
                    self.steps_done = [s for s in step_results]
            timing["verification_s"] += time.perf_counter() - verify_started
            execution_status = "completed"
            final_verification = {"ok": True, "samples": len(samples), "steps": final_steps}
        except Stopped:
            execution_status = "stopped"
            failure = {"reason_code": "user_stop", "message": "episode stopped by user", "evidence": {}}
            final_verification = {}
        except SkillFailure as exc:
            execution_status = "failed"
            failure = {
                **(failure or {}),
                "reason_code": exc.reason_code,
                "message": exc.message,
                "evidence": exc.evidence,
            }
            final_verification = {}
        except Exception as exc:
            execution_status = "failed"
            failure = {
                "reason_code": "engine_error",
                "message": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc()[-3000:],
            }
            final_verification = {}
        ended = time.time()
        if execution_status == "completed":
            failure = None
        final_snapshot = snapshot if isinstance(snapshot, dict) else {}
        native_final = bool(final_snapshot.get("native_goal"))
        record = {
            "schema": "embodied_engine.run.v1",
            "task_id": task["task_id"],
            "task_bddl": task["bddl"],
            "instruction": instruction,
            "episode_id": episode_id,
            "run_id": run_id,
            "scene_epoch": final_snapshot.get("scene_epoch"),
            "initial_scene_fingerprint": initial_scene,
            "started_at": started_iso,
            "ended_at": _now_iso(),
            "duration_s": round(ended - started, 3),
            "phase": execution_status,
            "execution_status": execution_status,
            "native_task_success": bool(native_ever or native_final),
            "native_goal_final": native_final,
            "user_goal_success": execution_status == "completed",
            "user_goal_status": "met" if execution_status == "completed" else "not_met",
            "failure_code": (failure or {}).get("reason_code"),
            "failure_reason": (failure or {}).get("message"),
            "failure_evidence": (failure or {}).get("evidence"),
            "plan": {
                "interpretation": (plan or {}).get("interpretation"),
                "success_condition": (plan or {}).get("success_condition"),
                "steps": (plan or {}).get("steps"),
                "source": (self.plan or {}).get("source") or "qwen3.5-9b",
                "agent_mode": (self.plan or {}).get("agent_mode") or config.agent_mode(),
                "planner_vision": False,
                "latency_s": round(timing["planner_s"], 3),
                "agent_response": self.plan,
            },
            "steps": step_results,
            "openwam": openwam_stats,
            "final_verification": final_verification,
            "timing": {
                **{key: round(value, 3) for key, value in timing.items()},
                "total_s_engine": round(ended - started, 3),
                "critical_path_note": "planner + skill wall time + final verification; inference and action execution overlap",
            },
            "sim_action_count": int(final_snapshot.get("sim_action_count") or 0),
            "episode_sim_actions": max(
                0,
                int(final_snapshot.get("sim_action_count") or 0)
                - int((baseline or {}).get("sim_action_count") or 0),
            ),
            "recovery_count": len([a for a in attempts]),
            "recovery_log": recovery_log,
            "human_interventions": 0,
            "inference_mode": (self._health().get("inference") or {}).get("mode") or self.cfg["inference"]["mode"],
            "action_backend": self._action_backend(),
            "model": self._health().get("action_model"),
            "agent_model": self._health().get("agent", {}).get("model"),
            "env": self.env.view(),
            "page": {"ui": "page_ui (restored old Page, backend bridge)", "human_confirmation": False},
        }
        path = records.write_run(record)
        records.append_journal(
            {
                "t": ended,
                "stage": "episode_end",
                "episode_id": episode_id,
                "task_id": task["task_id"],
                "phase": execution_status,
                "native": record["native_task_success"],
                "user_goal": record["user_goal_success"],
                "duration_s": record["duration_s"],
                "record": str(path),
            }
        )
        with self.lock:
            self.metrics["episodes_succeeded" if execution_status == "completed" else "episodes_stopped" if execution_status == "stopped" else "episodes_failed"] += 1
            self.metrics["last_episode_s"] = record["duration_s"]
            self.last_result = {
                "episode_id": episode_id,
                "task_id": task["task_id"],
                "success": execution_status == "completed",
                "execution_status": execution_status,
                "native_task_success": record["native_task_success"],
                "reason": (failure or {}).get("message") or "all skills and final verification passed",
                "failure_code": (failure or {}).get("reason_code"),
                "duration_s": record["duration_s"],
                "sim_action_count": record["sim_action_count"],
                "initial_scene_fingerprint": record.get("initial_scene_fingerprint"),
                "timing": record["timing"],
                "record": str(path),
            }
            if execution_status == "completed":
                self._set_phase("succeeded", f"task {task['task_id']} complete: {record['duration_s']}s, {record['sim_action_count']} sim actions")
            elif execution_status == "stopped":
                self._set_phase("stopped", "episode stopped")
            else:
                self._set_phase(
                    "failed",
                    f"task {task['task_id']} failed: {(failure or {}).get('reason_code')}: {(failure or {}).get('message')}",
                    failure_reason=str((failure or {}).get("reason") or (failure or {}).get("message") or ""),
                    failure_code=str((failure or {}).get("reason_code") or ""),
                )
        with self.lock:
            self.failure_reason = str((failure or {}).get("message") or "") if execution_status == "failed" else ""
            self.failure_code = str((failure or {}).get("reason_code") or "") if execution_status == "failed" else ""

    def _accumulate_timing(self, timing: dict[str, float], runner: SkillRunner) -> None:
        timing["inference_s_sum"] += runner.inference_s
        timing["env_step_s_sum"] += runner.env_step_s
        timing["snapshot_s_sum"] += runner.snapshot_s
        timing["chunk_gap_s_sum"] += runner.chunk_gap_s
        timing["chunk_decision_wait_s_sum"] += runner.decision_wait_s
        timing["prefetch_inference_overlap_s_est"] += runner.prefetch_saved_s
        timing["release_s"] += runner.last_release_s
        timing["verification_s"] += runner.last_verify_s

    def _budget_hard(self, skill: str) -> int:
        ex = self.cfg["execution"]
        if skill == "open_drawer":
            return int(ex.get("open_drawer_hard_chunks", self._budget(skill)))
        if skill == "pick_place":
            return int(ex.get("pick_hard_chunks", self._budget(skill)))
        return self._budget(skill)

    def _budget(self, skill: str) -> int:
        ex = self.cfg["execution"]
        return int(
            ex.get(
                {
                    "open_drawer": "open_drawer_max_chunks",
                    "push_object": "push_max_chunks",
                    "turn_on": "turn_on_max_chunks",
                }.get(skill, "pick_max_chunks")
            )
        )


ENGINE: Engine | None = None


def _engine() -> Engine:
    assert ENGINE is not None
    return ENGINE


async def health(request: web.Request) -> web.Response:
    return web.json_response({"service": "engine", "status": "ok", "identity": _engine()._health(), "phase": _engine().phase})


async def status(request: web.Request) -> web.Response:
    return web.json_response(_engine().status())


async def tasks(request: web.Request) -> web.Response:
    rows = []
    for row in config.tasks_by_id().values():
        rows.append(
            {
                "id": f"task_{row['task_id']}",
                "task_id": row["task_id"],
                "label": f"{row['task_id']}: {row['instruction']}",
                "instruction": row["instruction"],
                "kind": row["kind"],
                "bddl": row["bddl"],
            }
        )
    return web.json_response({"status": "ok", "scenes": rows})


async def load(request: web.Request) -> web.Response:
    body = await _json_body(request)
    try:
        seed = body.get("seed")
        out = await asyncio.to_thread(
            _engine().load, int(body.get("task_id")), None if seed is None else int(seed)
        )
        return web.json_response(out)
    except Exception as exc:
        return web.json_response({"status": "error", "error": f"{type(exc).__name__}: {exc}"}, status=503)


async def run(request: web.Request) -> web.Response:
    body = await _json_body(request)
    try:
        seed = body.get("seed")
        out = _engine().start_run(
            int(body.get("task_id")),
            body.get("instruction"),
            None if seed is None else int(seed),
        )
        return web.json_response(out)
    except KeyError as exc:
        return web.json_response({"status": "bad_request", "error": str(exc)}, status=400)
    except Exception as exc:
        return web.json_response({"status": "error", "error": f"{type(exc).__name__}: {exc}"}, status=409)


async def stop(request: web.Request) -> web.Response:
    body = await _json_body(request)
    out = await asyncio.to_thread(_engine().stop, str(body.get("reason") or "user_stop"))
    return web.json_response(out)


async def reset(request: web.Request) -> web.Response:
    try:
        out = await asyncio.to_thread(_engine().reset)
        return web.json_response(out)
    except Exception as exc:
        return web.json_response({"status": "error", "error": f"{type(exc).__name__}: {exc}"}, status=503)


async def frame(request: web.Request) -> web.Response:
    view = str(request.query.get("view") or "agentview")
    try:
        data = await asyncio.to_thread(_engine().frame, view)
        return web.Response(body=data, content_type="image/jpeg")
    except KeyError:
        return web.json_response({"status": "no_frame", "view": view}, status=404)


async def metrics(request: web.Request) -> web.Response:
    engine = _engine()
    return web.json_response(
        {
            "status": "ok",
            "phase": engine.phase,
            "metrics": engine.metrics,
            "inference": engine.inference.metrics(),
            "env": engine.env.view(),
        }
    )


async def agent_mode(request: web.Request) -> web.Response:
    if request.method == "GET":
        return web.json_response(
            {"status": "ok", "mode": config.agent_mode(), "modes": list(config.AGENT_MODES)}
        )
    body = await _json_body(request)
    try:
        out = await asyncio.to_thread(_engine().set_agent_mode, str(body.get("mode") or ""))
    except Exception as exc:
        return web.json_response({"status": "error", "error": f"{type(exc).__name__}: {exc}"}, status=503)
    status = {"ok": 200, "bad_request": 400, "busy": 409, "unavailable": 503}.get(out.get("status"), 400)
    return web.json_response(out, status=status)


async def _json_body(request: web.Request) -> dict[str, Any]:
    try:
        body = await request.json()
        return body if isinstance(body, dict) else {}
    except Exception:
        return {}


def make_app() -> web.Application:
    app = web.Application(client_max_size=16 * 1024 * 1024)
    app.router.add_get("/health", health)
    app.router.add_get("/metrics", metrics)
    app.router.add_get("/api/status", status)
    app.router.add_get("/api/tasks", tasks)
    app.router.add_get("/api/agent_mode", agent_mode)
    app.router.add_post("/api/agent_mode", agent_mode)
    app.router.add_post("/api/load", load)
    app.router.add_post("/api/run", run)
    app.router.add_post("/api/stop", stop)
    app.router.add_post("/api/reset", reset)
    app.router.add_get("/api/frame", frame)
    return app


def main() -> None:
    global ENGINE
    cfg = config.engine_config()
    config.ensure_dirs()
    ENGINE = Engine(cfg)
    host = cfg["services"]["engine"]["host"]
    port = int(cfg["services"]["engine"]["port"])
    print(f"engine listening on {host}:{port}", flush=True)
    web.run_app(make_app(), host=host, port=port, print=None)


if __name__ == "__main__":
    main()
