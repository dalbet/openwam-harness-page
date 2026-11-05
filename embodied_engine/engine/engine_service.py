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
from common import config, records, task_catalog  # noqa: E402
from common import geometry  # noqa: E402
from common import plan_contract
from common import policy_seed as policy_seed_module  # noqa: E402
from common import transport_ready  # noqa: E402
from engine import completion  # noqa: E402
from engine import composite  # noqa: E402
from engine import finalize  # noqa: E402
from engine import grasp_geometry  # noqa: E402
from engine.inference_client import InferenceClient  # noqa: E402
from engine.openwam_rollout import WholeTaskRunner  # noqa: E402
from engine.pi05_rollout import Pi05WholeTaskRunner  # noqa: E402
from engine.skills import SkillFailure, SkillRunner, Stopped  # noqa: E402


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


def _release_evidence(
    task: dict[str, Any],
    step_results: list[dict[str, Any]],
    snapshot: dict[str, Any],
    placement: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Report gripper release as measured evidence, never as inferred success.

    Release is only expected for pick/place tasks (including the composite
    open-top-drawer-and-place-bowl case). Opening a drawer or turning on a stove
    has no release phase, so those tasks are explicitly marked not applicable
    rather than failing a release check.

    ``placement`` is the newer contract (``evidence_source``,
    ``release_verified``, ``placement_verified``, ``retreat_status``); this
    legacy block mirrors it so old readers and old records stay comparable.
    """
    kind = str(task.get("kind") or "")
    applicable = kind in {"pick_place", "composite"}
    rows = []
    observed = False
    for index, row in enumerate(step_results):
        release = row.get("release") or {}
        events = release.get("events") or []
        openings = release.get("openings") or []
        if events:
            observed = True
        rows.append(
            {
                "step_index": index,
                "skill": (row.get("step") or {}).get("skill"),
                "events": events,
                "event_count": len(events),
                "openings_m": openings,
                "source": release.get("source"),
            }
        )
    placement = placement or {}
    sim = snapshot.get("sim") or {}
    final_opening = sim.get("gripper_opening_m")
    if placement:
        observed = bool(placement.get("release_executed"))
    elif isinstance(final_opening, (int, float)) and float(final_opening) > 0.05:
        observed = True
    if not applicable:
        state = "not_applicable"
    elif placement:
        state = "verified" if placement.get("placement_verified") else "unknown"
    else:
        state = "observed" if observed else "unknown"
    return {
        "applicable": applicable,
        "state": state,
        "observed": bool(observed),
        "final_gripper_opening_m": final_opening,
        "threshold_m": 0.05,
        "steps": rows,
        "placement": {
            key: placement.get(key)
            for key in (
                "release_executed",
                "release_verified",
                "placement_verified",
                "retreat_status",
                "evidence_source",
                "policy_owned",
                "reason_code",
            )
        }
        if placement
        else None,
        "note": (
            "finger separation and the object's position relative to the fingers, measured. "
            "Native goal success alone does not prove the object was released; a closed "
            "gripper alone does not prove it is still held."
        ),
    }


def _state_fingerprint(snapshot: dict[str, Any]) -> str:
    """Hash of the initial state the env *publishes* to the policy.

    ``_scene_fingerprint`` covers object positions and the EEF position, which
    is what "same scene" used to mean. The policy also consumes the wrist image
    and an EEF rotation, both of which follow the arm joint angles - two runs
    can agree on the object positions and still see different pixels. This hash
    adds the joint configuration so a "same initial state" claim can be checked
    against the state the model actually received.

    Scope limit, measured 2026-11-05: this only sees objects the env exposes as
    ``*_pos`` observables. Static props placed into ``model.body_pos`` (the flat
    stove, cabinet, wine rack) are absent, so two episodes can share this hash
    and still see a stove 12 mm away. ``_fixture_fingerprint`` covers that gap.
    """
    sim = snapshot.get("sim") or {}
    objects = {
        name: [round(float(value), 5) for value in (pose or {}).get("pos") or []]
        for name, pose in (sim.get("objects") or {}).items()
        if isinstance(pose, dict)
    }
    joints = {
        str(name): round(float(value), 6)
        for name, value in (snapshot.get("joints") or {}).items()
        if isinstance(value, (int, float))
    }
    payload = {
        "objects": objects,
        "eef": [round(float(value), 5) for value in (sim.get("eef_pos") or [])],
        "joints": joints,
        "gripper": round(float(sim.get("gripper_opening_m") or 0.0), 6),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def _fixture_fingerprint(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Fingerprint of the *static* props the observation hash cannot see.

    ``_state_fingerprint`` only covers what the env publishes as ``*_pos``
    observables, which is the movable task objects plus the robot. The flat
    stove, the cabinet and the wine rack are fixtures: their pose is written to
    ``model.body_pos`` by LIBERO's placement sampler and never appears in that
    observable set. Measured on 2026-11-05 (``policy_obs_determinism_probe``),
    two resets with the same scene seed put the four movable objects at *bit
    identical* positions while the flat stove moved 12.4 mm, the cabinet 9.0 mm
    and the wine rack 6.1 mm, and the policy input differed in 16-25% of its
    pixels. This hash makes that visible in every record without changing the
    meaning of the existing fingerprint.
    """
    geometry = (snapshot.get("named_poses") or {}).get("site_geometry") or {}
    fixtures = {
        str(name): {
            "xyz": [round(float(value), 6) for value in (item.get("xyz") or [])],
            "quat": [round(float(value), 6) for value in (item.get("quat_wxyz") or [])],
        }
        for name, item in geometry.items()
        if isinstance(item, dict)
    }
    payload = {"fixtures": fixtures, "bodies": snapshot.get("named_poses", {}).get("bodies") or {}}
    return {
        "sha256": hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16],
        "sites": len(fixtures),
        "scope": "named_poses.site_geometry + bodies; excludes movable *_pos observables (see state fingerprint)",
    }


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


def _watched_joints(
    steps: list[dict[str, Any]],
    index: int,
    snapshot: dict[str, Any],
    task: dict[str, Any],
) -> list[str]:
    """Earlier steps' goal joints, so step N can be traced against them.

    Read-only: a later step's action stream is unchanged. This is what lets a
    report say *when* a previously achieved drawer goal started to regress.
    """
    names: list[str] = []
    if index <= 0:
        return names
    joints = snapshot.get("joints") or {}
    for prior in steps[:index]:
        goal = composite.step_goal(prior) or {}
        if goal.get("skill") != "open_drawer":
            continue
        target = goal.get("target_n")
        try:
            level = completion.drawer_level(target) or task.get("drawer_level")
            name, _qpos = completion.drawer_joint(
                snapshot, str(level), str(target) if target else None
            )
            if not name and level:
                # The owner prefix of an earlier step's target can differ from
                # the owner the scene actually reports; the level suffix is
                # still enough to find the joint the earlier goal was about.
                for candidate in joints:
                    if str(candidate).endswith(f"_{level}_level"):
                        name = str(candidate)
                        break
        except Exception:  # noqa: BLE001 - diagnostics only
            name = None
        if name and str(name) not in names:
            names.append(str(name))
    return names


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


def _step_digest(record: dict[str, Any]) -> dict[str, Any]:
    """Compact per-attempt summary that outlives the attempt it describes.

    A retried attempt is dropped from ``steps`` (the record keeps only the
    attempt that decided the episode), which used to make the earlier attempts
    unauditable: no action count, no object pose, no gripper state. The digest
    keeps exactly the fields needed to answer "what did that attempt actually
    do" without copying hundreds of observation rows.
    """
    observations = record.get("observations") or []

    def _pose(row: dict[str, Any], key: str) -> list[float] | None:
        value = (row or {}).get(key)
        if not isinstance(value, (list, tuple)) or len(value) < 3:
            return None
        try:
            return [round(float(v), 5) for v in value[:3]]
        except (TypeError, ValueError):
            return None

    opens = [
        float(row["gripper_opening_m"])
        for row in observations
        if isinstance(row, dict) and isinstance(row.get("gripper_opening_m"), (int, float))
    ]
    native_hits = [
        index for index, row in enumerate(observations)
        if isinstance(row, dict) and row.get("native_goal")
    ]
    first = observations[0] if observations else {}
    last = observations[-1] if observations else {}
    failure = record.get("failure") or {}
    grasp = failure.get("grasp") if isinstance(failure.get("grasp"), dict) else {}
    return {
        "attempt": record.get("attempt"),
        "skill": (record.get("step") or {}).get("skill"),
        "ok": record.get("ok"),
        "sim_actions": record.get("sim_actions"),
        "chunks_used": record.get("chunks_used"),
        "failure_code": failure.get("reason_code"),
        "failure_message": failure.get("message"),
        "held_at_failure": grasp.get("held"),
        "native_goal_hits": len(native_hits),
        "native_goal_first_index": native_hits[0] if native_hits else None,
        "gripper_open_first_m": opens[0] if opens else None,
        "gripper_open_last_m": opens[-1] if opens else None,
        "gripper_open_max_m": max(opens) if opens else None,
        "gripper_open_min_m": min(opens) if opens else None,
        "object_first": _pose(first, "object_pose"),
        "object_last": _pose(last, "object_pose"),
        "eef_first": _pose(first, "eef_pos"),
        "eef_last": _pose(last, "eef_pos"),
        "sim_action_count_first": (first or {}).get("sim_action_count"),
        "sim_action_count_last": (last or {}).get("sim_action_count"),
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


def _quat_of(obs: dict[str, Any] | None) -> list[float] | None:
    """x,y,z,w end-effector orientation from a raw observation, normalised."""
    value = (obs or {}).get("robot0_eef_quat")
    if value is None:
        return None
    try:
        quat = [float(v) for v in np.asarray(value).reshape(-1)[:4]]
    except (TypeError, ValueError):
        return None
    norm = float(np.linalg.norm(quat))
    if norm < 1e-9:
        return None
    return [component / norm for component in quat]


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
        # Sampling seed of the *episode* (not the scene). ``None`` = legacy.
        try:
            self.policy_seed, self.policy_seed_source = policy_seed_module.resolve(cfg)
        except Exception:  # noqa: BLE001 - a bad config value must not stop the service
            self.policy_seed, self.policy_seed_source = None, "unresolved"
        self.instruction = ""
        self.episode_id: str | None = None
        self.run_id: int | None = None
        self.scene_epoch: int | None = None
        self.sim_action_count = 0
        self.failure_reason = ""
        self.failure_code = ""
        self.last_result: dict[str, Any] = {}
        self.last_stop: dict[str, Any] = {}
        self.episode_started_ts: float | None = None
        self.plan: dict[str, Any] | None = None
        # Bumped on every lifecycle transition (load/reset/stop/new rollout). A
        # planner response that was in flight across one of those transitions
        # carries an older generation and must not be written into the new
        # episode's state.
        self.planner_generation = 0
        self.steps_done: list[dict[str, Any]] = []
        self.latest_frames: dict[str, Any] = {}
        self.frame_seq = 0
        # Encoded JPEGs keyed by (frame_seq, view): a repeated request for a
        # frame already seen is served from memory instead of being re-encoded.
        self.frame_jpeg: dict[str, tuple] = {}
        # Why frames did (not) land — the page bug "one frame per whole task"
        # is a frame-source question, so the counters are exposed in /status.
        self.frame_stats: dict[str, Any] = {
            "written": 0, "skipped": 0, "cleared": 0,
            "last_skip": "", "last_src": "", "last_clear": "",
        }
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
                # Which upstream generation server the router is actually
                # using, and which arm it was started as. The page compares
                # this against the arm the user selected, so a stale selection
                # can never be shown as the running backend.
                "server_url": infer_ident.get("server_url"),
                "serving_arm": infer_ident.get("arm"),
                "precision": (
                    infer_ident.get("precision_requested")
                    or ((infer_ident.get("server") or {}).get("detail") or {}).get("precision")
                ),
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
                "mode": agent.get("mode", "semantic_router"),
                "model": agent.get("model", self.cfg["agent"]["llm_model"]),
                "llm_model": agent.get("llm_model", self.cfg["agent"]["llm_model"]),
                "planner_vision": agent.get("planner_vision", False),
                "llm": agent.get("llm"),
                "routing": agent.get("routing"),
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
            "engine": {
                "phase": self.phase,
                "detail": self.detail,
                "sampling": {
                    "policy_seed": self.active_policy_seed(),
                    "policy_seed_source": self.policy_seed_source,
                    "scene_seed": int(self.seed),
                    "protocol": "policy_seed_v1"
                    if self.active_policy_seed() is not None
                    else "legacy_prompt_index",
                },
                "handoff": {
                    "mode": composite.handoff_mode(self.cfg),
                    "default_mode": composite.DEFAULT_HANDOFF["mode"],
                    "modes": list(composite.HANDOFF_MODES),
                    "options": composite.handoff_options(self.cfg),
                },
            },
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
                "frame_ts": (self.latest_frames or {}).get("frame_ts"),
                "frame_age_s": (
                    None
                    if (self.latest_frames or {}).get("frame_ts") is None
                    else round(time.time() - float(self.latest_frames["frame_ts"]), 3)
                ),
                "frame_src": (self.latest_frames or {}).get("frame_src"),
                "frame_episode_id": (self.latest_frames or {}).get("episode_id"),
                "frame_scene_epoch": (self.latest_frames or {}).get("scene_epoch"),
                "frame_stats": dict(self.frame_stats),
                "episode_started_ts": self.episode_started_ts,
                "episode_elapsed_s": (
                    None if not self.episode_started_ts
                    else round(time.time() - float(self.episode_started_ts), 2)
                ),
                "failure_reason": self.failure_reason,
                "failure_code": self.failure_code,
                "last_result": self.last_result,
                "last_stop": self.last_stop,
                "plan": self.plan,
                "steps": self.steps_done,
                "metrics": self.metrics,
                "identity": health,
            }

    def set_handoff_mode(self, mode: str, options: dict[str, Any] | None = None) -> dict[str, Any]:
        """Switch the step-handoff mode for *subsequent* episodes.

        The mode is runtime state of the engine, not of the episode: an episode
        snapshots it at init (and captures the robot state there when the mode
        needs it), so changing it mid-episode would make the record and the
        behaviour disagree. Refuse that instead.
        """
        if mode not in composite.HANDOFF_MODES:
            raise ValueError(
                f"unknown handoff mode {mode!r}; expected one of {list(composite.HANDOFF_MODES)}"
            )
        with self.lock:
            if self.rollout_thread is not None and self.rollout_thread.is_alive():
                raise RuntimeError("cannot change the handoff mode while an episode is running")
            block = dict(self.cfg.get("composite") or {})
            handoff_block = dict(block.get("handoff") or {})
            handoff_block["mode"] = mode
            for key, value in (options or {}).items():
                if key == "mode":
                    continue
                if key in composite.DEFAULT_HANDOFF:
                    handoff_block[key] = value
            block["handoff"] = handoff_block
            self.cfg["composite"] = block
            self.health_cache = {"at": 0.0, "value": {}}
            effective = composite.handoff_options(self.cfg)
        records.append_journal(
            {
                "t": time.time(),
                "stage": "handoff_mode_set",
                "episode_id": self.episode_id,
                "mode": mode,
                "options": effective,
            }
        )
        return {"status": "ok", "mode": mode, "options": effective}

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
        """Publish the frame the control loop already holds.

        Called from the rollout loop with the observation ``env.step`` just
        returned (``main_images`` / ``wrist_images``) and from the read-only
        probe paths with a raw ``env.raw_obs`` (``agentview_image`` /
        ``robot0_eye_in_hand_image``). Both are accepted: the page needs a
        frame per control step, and neither path steps the simulator or
        renders again. No frame is ever fetched inside this method.
        """
        if not isinstance(obs, dict):
            with self.lock:
                self.frame_stats["skipped"] += 1
                self.frame_stats["last_skip"] = "obs_not_dict"
            return
        agent = obs.get("main_images")
        wrist = obs.get("wrist_images")
        source = "model_obs"
        if agent is None:
            agent = obs.get("agentview_image")
            wrist = obs.get("robot0_eye_in_hand_image")
            source = "raw_obs"
        if agent is None:
            with self.lock:
                self.frame_stats["skipped"] += 1
                self.frame_stats["last_skip"] = "no_image_key"
            return
        with self.lock:
            self.latest_frames = {
                "agent": np.asarray(agent, dtype=np.uint8),
                "wrist": np.asarray(wrist, dtype=np.uint8) if wrist is not None else None,
                "sim_action_count": snapshot.get("sim_action_count"),
                "observation_id": snapshot.get("observation_id"),
                "scene_epoch": snapshot.get("scene_epoch"),
                "episode_id": self.episode_id,
                "frame_seq": self.frame_seq + 1,
                "frame_ts": time.time(),
                "frame_src": source,
            }
            self.frame_seq += 1
            self.frame_stats["written"] += 1
            self.frame_stats["last_src"] = source
            self.sim_action_count = int(snapshot.get("sim_action_count") or 0)

    def _clear_frames(self, reason: str) -> None:
        """Drop frames from the previous episode so the page cannot show them."""
        with self.lock:
            self.latest_frames = {}
            self.frame_jpeg.clear()
            self.frame_stats["cleared"] += 1
            self.frame_stats["last_clear"] = reason

    def frame(self, view: str) -> bytes:
        key = "agent" if view in {"agent", "agentview", "main"} else "wrist"
        with self.lock:
            row = dict(self.latest_frames)
            seq = row.get("frame_seq")
            cached = self.frame_jpeg.get(key)
            if cached is not None and cached[0] == seq:
                return cached[1]
        arr = row.get(key)
        if arr is None:
            raise KeyError(view)
        # Publish cost is measured here, not in the control loop: this method
        # runs on the HTTP thread that serves the page, so its time is part of
        # the episode's wall clock but not of any action's timing row.
        encode_started = time.perf_counter()
        image = Image.fromarray(np.asarray(arr, dtype=np.uint8))
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=90, subsampling=2)
        data = buf.getvalue()
        encode_s = time.perf_counter() - encode_started
        with self.lock:
            self.frame_stats["encode_n"] = int(self.frame_stats.get("encode_n") or 0) + 1
            self.frame_stats["encode_s"] = round(
                float(self.frame_stats.get("encode_s") or 0.0) + encode_s, 4
            )
            self.frame_stats["encode_last_s"] = round(encode_s, 5)
            # Keyed by frame_seq: a repeated request for the same frame is
            # served from here, so a slow client never makes the engine
            # re-encode (and never touches the control loop).
            if self.latest_frames.get("frame_seq") == seq:
                self.frame_jpeg[key] = (seq, data)
        return data

    # ---- scene / lifecycle ----

    def load(self, task_id: int, seed: int | None = None) -> dict[str, Any]:
        task = config.task_for(task_id)
        wanted_seed = int(self.cfg["env"]["seed"] if seed is None else seed)
        with self.run_lock:
            if self.phase in {"planning", "running", "recovering", "stopping"}:
                raise RuntimeError("cannot load while an episode is active")
            # A scene change must not leave the previous episode's last frame
            # on the page while the new environment is still starting.
            self._clear_frames("load")
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
            self._clear_frames("reset")
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
        cancel = self.inference.cancel(reason)
        thread = self.rollout_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=float(self.cfg["timing"]["stop_join_s"]))
        alive = bool(thread and thread.is_alive())
        sim_at_ack = sim_at_request
        if not alive:
            sim_at_ack = self._settled_sim_count(sim_at_request)
        try:
            inference_metrics = self.inference.metrics()
        except Exception:  # noqa: BLE001 - the stop budget must still be reported
            inference_metrics = {}
        budget = {
            "reason": reason,
            "sim_at_request": int(sim_at_request),
            "sim_at_ack": int(sim_at_ack),
            # An env.step already handed to the simulator when the flag rose
            # cannot be recalled; the contract is "no new action is issued",
            # so this is 0 or 1 and is reported instead of hidden.
            "in_flight_actions": max(0, int(sim_at_ack) - int(sim_at_request)),
            "thread_alive": alive,
            # A stale count is not enough on its own: the action request has to
            # have finished too, and the cancel barrier has to have completed.
            "settle_agreements": 3,
            "cancel_barrier_s": (cancel or {}).get("barrier_s"),
            "in_flight_after_cancel": (cancel or {}).get("in_flight_after_cancel"),
            "inference_idle": int(inference_metrics.get("in_flight") or 0) == 0,
            "chunk_buffer_cleared": bool((cancel or {}).get("status") == "ok"),
        }
        with self.lock:
            self.sim_action_count = int(sim_at_ack)
            self.last_stop = budget
        return {"status": "ok", "active": alive, "phase": self.phase, "stop": budget}

    def _settled_sim_count(self, fallback: int, *, agreements: int = 3, idle_s: float = 0.15) -> int:
        """Simulator action count after any in-flight step has finished.

        The rollout thread may have handed one ``env.step`` to the env server
        when the stop flag rose; joining the thread returns as soon as the
        caller sees the flag, which can be *while* that step is still running
        in the env process. A single snapshot taken at that moment under-reports
        the count by one. Two equal reads are not enough either (a slow step can
        still land between them), so this requires ``agreements`` consecutive
        identical reads spaced by ``idle_s`` and keeps the maximum seen.
        """
        best = int(fallback)
        client = self.env.client
        if client is None:
            return best
        previous = None
        stable = 0
        deadline = time.time() + 3.0
        while time.time() < deadline:
            try:
                current = int(client.state_snapshot().get("sim_action_count") or fallback)
            except Exception:
                break
            best = max(best, current)
            if previous is not None and current == previous:
                stable += 1
            else:
                stable = 0
            previous = current
            if stable >= max(1, agreements - 1):
                break
            time.sleep(idle_s)
        return best

    def start_run(
        self,
        task_id: int,
        instruction: str | None,
        seed: int | None = None,
        plan_keys: list[str] | None = None,
        policy_seed: int | None = None,
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
        self._clear_frames("run_start")
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
            self.episode_started_ts = time.time()
            self.run_id = self._next_run_id(task_id)
            self.planner_generation += 1
            planner_generation = self.planner_generation
            episode_id = self.episode_id
            run_id = self.run_id
            # Sampling seed for this episode, kept separate from ``self.seed``
            # (the scene/init seed) on purpose: the two are different numbers
            # and are reported separately.
            self.policy_seed, self.policy_seed_source = policy_seed_module.resolve(
                self.cfg, policy_seed
            )
        self._set_phase("running", "preparing episode")
        thread = threading.Thread(
            target=self._run_episode,
            args=(
                task,
                self.instruction,
                episode_id,
                run_id,
                self.stop_event,
                planner_generation,
                list(plan_keys or []) or None,
            ),
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

    def active_policy_seed(self) -> int | None:
        """The sampling seed every request of this episode is derived from.

        ``None`` means the shipped legacy path: no seed is sent, and each chunk
        keeps the backend's per-(prompt, chunk-index) noise. Reported, never
        guessed — the record carries the value and where it came from.
        """
        return None if self.policy_seed is None else int(self.policy_seed)

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

    def _run_whole_task(        self,
        *,
        env: EnvClient,
        obs: dict[str, Any],
        snapshot: dict[str, Any],
        step: dict[str, Any],
        task: dict[str, Any],
        token: str,
        episode_id: str,
        stop_event: threading.Event,
        goal: dict[str, Any] | None = None,
        max_steps: int | None = None,
        watch_joints: list[str] | None = None,
        step_task: dict[str, Any] | None = None,
        watch_goals: list[dict[str, Any]] | None = None,
        carry_objects: list[str] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        """Run one whole-task policy rollout; returns (record, obs, snapshot).

        ``goal`` is the engine predicate of a composite's early step. When it
        is set the rollout ends on that predicate and the loaded scene's
        native flag can never terminate the step.
        """
        openwam = self.cfg["action_model"]["openwam"]
        composite_opts = composite.options(self.cfg)
        backend = self._action_backend()
        # The rollout loop, verification, release and record shape are shared;
        # only the observation/action/generation seams differ. Choosing the
        # class here keeps the two models' contracts from bleeding into each
        # other (an 8-D proprio in OpenWAM or EEF10 in pi05 both fail loudly).
        runner_cls = Pi05WholeTaskRunner if backend == "pi05" else WholeTaskRunner
        pi05_cfg = dict(self.cfg["action_model"].get("pi05") or {})
        extra = (
            {"execute_horizon": int(pi05_cfg.get("execute_horizon") or 5),
             "generate_horizon": int(pi05_cfg.get("generate_horizon") or 10)}
            if runner_cls is Pi05WholeTaskRunner
            else {}
        )
        runner = runner_cls(
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
            policy_seed=self.active_policy_seed(),
            journal=lambda stage, row: records.append_journal(
                {"t": time.time(), "stage": stage, "episode_id": episode_id, "row": row}
            ),
            max_steps=int(max_steps or (pi05_cfg if runner_cls is Pi05WholeTaskRunner else openwam).get("max_steps") or 600),
            timeout_s=float((pi05_cfg if runner_cls is Pi05WholeTaskRunner else openwam).get("timeout_s") or 300),
            gpu_index=openwam.get("gpu_index"),
            frame_sink=self._set_frames,
            post_goal=self._post_goal_cfg(),
            goal=goal,
            goal_confirm_samples=int(composite_opts["goal_confirm_samples"]),
            step_ref=str(step.get("step_id") or "step"),
            watch_joints=watch_joints,
            watch_goals=watch_goals,
            carry_objects=carry_objects,
            **extra,
        )
        result = runner.run()
        if goal is not None and step_task is not None and composite.release_required(step_task):
            # An early composite step that places an object still owes a real
            # release: its own predicate says "on the target", not "let go".
            # The policy ended on that predicate, so the execution layer
            # finishes the release/retreat/fresh-sample sequence here, before
            # the handoff is allowed to move the arm.
            placement, placement_obs, placement_snapshot = self._finish_predicate_placement(
                env=env,
                step=step,
                step_task=step_task,
                record=result,
                snapshot=runner.snapshot,
                episode_id=episode_id,
                stop_event=stop_event,
            )
            result["placement"] = placement
            if placement.get("reason_code") == "user_stop":
                raise Stopped("stop requested during step placement finalizer")
            return result, placement_obs, placement_snapshot
        return result, runner.obs, runner.snapshot

    def _post_goal_cfg(self) -> dict[str, Any]:
        """The post-goal window/finalizer block of the *active* backend.

        Pi0.5 and OpenWAM have different action spaces, chunk lengths, and
        measured release behaviour, so one backend's window must not silently
        become the other's.
        """
        action = self.cfg["action_model"]
        backend = self._action_backend()
        openwam = dict(action.get("openwam") or {})
        if backend == "pi05":
            pi05_block = dict(action.get("pi05") or {})
            declared = pi05_block.get("post_goal")
            return dict(declared or openwam.get("post_goal") or {})
        return dict(openwam.get("post_goal") or {})

    def _finish_predicate_placement(
        self,
        *,
        env: EnvClient,
        step: dict[str, Any],
        step_task: dict[str, Any],
        record: dict[str, Any],
        snapshot: dict[str, Any],
        episode_id: str,
        stop_event: threading.Event,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        """Execution-layer release for a composite step judged by a predicate.

        Same bounded sequence as the anchor's finalizer (release, retreat,
        stability samples) but the precondition is this step's own engine
        predicate and the relation binding comes from the step's catalog row.
        Every action is tagged ``action_source="execution_finalizer"``.
        """
        relation_step = self._placement_step(step_task)
        post_cfg = self._post_goal_cfg()
        if relation_step is None:
            return (
                {
                    "schema": "embodied_engine.composite.step.v1",
                    "required": True,
                    "attempted": False,
                    "release_executed": False,
                    "release_verified": False,
                    "placement_verified": False,
                    "retreat_status": "skipped",
                    "execution_status": "failed",
                    "evidence_source": "none",
                    "policy_owned": False,
                    "action_source": "none",
                    "reason_code": "no_relation_binding",
                    "note": "the step's catalog row has no object/target pair to verify",
                },
                record.get("obs") or {},
                snapshot,
            )
        finalizer = finalize.PlacementFinalizer(
            env=env,
            task=step_task,
            relation_step=relation_step,
            execution_cfg=self.cfg["execution"],
            post_goal_cfg=post_cfg,
            stop_event=stop_event,
            episode_id=episode_id,
            journal=lambda stage, row: records.append_journal(
                {"t": time.time(), "stage": stage, "episode_id": episode_id, "row": row}
            ),
            frame_sink=self._set_frames,
            reset_policy=lambda reason: self.inference.reset(reason),
            candidate_check=lambda snap: composite.state_check(snap, step, step_task),
        )
        block = finalizer.run(snapshot)
        block["required"] = True
        block["step_id"] = step.get("step_id")
        block["step_task_key"] = step.get("step_task_key")
        block["action_source"] = "execution_finalizer" if block.get("attempted") else "none"
        block["policy_owned"] = False
        block["policy_actions_used"] = int(record.get("sim_actions") or 0)
        obs = finalizer.last_obs if finalizer.last_obs is not None else (record.get("obs") or {})
        snap = finalizer.last_snapshot if finalizer.last_snapshot is not None else snapshot
        return block, obs, snap

    # ---- transport readiness (is the placed object safe to carry?) --------

    def _target_region_diagnostic(
        self,
        *,
        step: dict[str, Any],
        catalog_by_key: dict[str, dict[str, Any]],
        snapshot: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Where the step's object sits against the official target region.

        Read-only and ``diagnostic_only``: the official relation is evaluated
        for the record, not for the verdict. LIBERO's own site check also
        requires contact between the fixture and the object, which snapshots do
        not carry; the geometric bounds here are the part that can be measured
        from a pose read, and the note says so instead of implying more.
        """
        row = catalog_by_key.get(str(step.get("step_task_key"))) or {}
        goal = composite.step_goal(step) or {}
        objects = row.get("objects") or []
        targets = row.get("targets") or []
        object_n = str(goal.get("object_n") or (objects[0] if objects else ""))
        target_n = str(goal.get("target_n") or (targets[0] if targets else ""))
        if not object_n or not target_n:
            return None
        relation = str(goal.get("relation") or (row.get("semantics") or {}).get("relation") or "on")
        try:
            check = geometry.relation_check(snapshot, object_n, target_n, relation)
        except Exception as exc:  # noqa: BLE001 - a read must never fail a step
            return {"diagnostic_only": True, "error": f"{type(exc).__name__}: {exc}"}
        block = dict(check)
        block.update(
            {
                "diagnostic_only": True,
                "object_n": object_n,
                "target_n": target_n,
                "relation": relation,
                "step_id": step.get("step_id"),
                "step_task_key": step.get("step_task_key"),
                "measure": (
                    "official relation geometry on the step's final snapshot; "
                    "LIBERO's fixture-contact term is not evaluated here"
                ),
            }
        )
        if isinstance(check.get("delta_site"), list) and isinstance(check.get("site_size"), list):
            dx, dy = float(check["delta_site"][0]), float(check["delta_site"][1])
            sx, sy = float(check["site_size"][0]), float(check["site_size"][1])
            outside = max(abs(dx) - sx, abs(dy) - sy)
            block["outside_xy_m"] = round(outside, 4)
            block["near_miss"] = bool(0.0 < outside <= 0.03)
        return block

    def _transport_readiness(
        self,
        *,
        env: EnvClient,
        episode_id: str,
        steps: list[dict[str, Any]],
        index: int,
        step: dict[str, Any],
        step_task: dict[str, Any],
        catalog_by_key: dict[str, dict[str, Any]],
        task: dict[str, Any],
        record: dict[str, Any],
        obs: dict[str, Any],
        snapshot: dict[str, Any],
        token: str,
        stop_event: threading.Event,
        composite_state: dict[str, Any],
        timing: dict[str, Any],
        records,
    ) -> tuple[dict[str, Any], dict[str, Any], str]:
        """Judge whether the object placed by this step can be carried later.

        The step's own goal stays exactly as it was: this runs *after* the step
        is complete and only decides whether to re-seat the object once, end the
        episode with an explicit not-ready failure, or carry on. Every number it
        reads comes from the step's own post-release samples.
        """
        opts = transport_ready.options(self.cfg)
        block = transport_ready.evaluate(
            step=step,
            relation=record.get("final_evidence") or record.get("evidence"),
            samples=(record.get("placement") or {}).get("samples"),
            evidence=record.get("final_evidence") or record.get("evidence"),
            cfg=self.cfg,
        )
        block["applicable"] = transport_ready.applies(steps, index, catalog_by_key)
        block["reseat"] = {
            "schema": "embodied_engine.composite.reseat.v1",
            "enabled": bool(opts["reseat"]["enabled"]),
            "attempted": False,
            "calls": 0,
            "max_calls": int(opts["reseat"]["max_calls"]),
            "max_steps": int(opts["reseat"]["max_steps"]),
            "task_key": None,
            "prompt": None,
            "step_id": None,
            "status": "not_attempted",
            "reason": None,
            "policy_actions": 0,
            "finalizer_actions": 0,
            "sim_action_counts": None,
            "after": None,
            "reseat_succeeded": False,
        }
        record["transport_ready"] = block
        composite_state["transport_ready"] = block
        if not block["applicable"] or not block.get("checked"):
            block["reseat"]["reason"] = "no later step carries this target; nothing to re-seat"
            return obs, snapshot, token
        if block.get("ok") is None:
            # Not measured is not a failure: with fewer samples than the rule
            # requires there is no verdict, so nothing is executed on it.
            block["reseat"]["reason"] = (
                f"not measured ({block.get('reason_code')}); no re-seat is attempted"
            )
            block["reseat"]["status"] = "not_measured"
            records.append_journal(
                {
                    "t": time.time(),
                    "stage": "composite_transport_not_measured",
                    "episode_id": episode_id,
                    "step_id": step.get("step_id"),
                    "reason_code": block.get("reason_code"),
                    "samples": (block.get("evidence") or {}).get("samples"),
                }
            )
            return obs, snapshot, token
        if block.get("ok"):
            block["reseat"]["reason"] = "transport readiness holds; no re-seat"
            return obs, snapshot, token

        records.append_journal(
            {
                "t": time.time(),
                "stage": "composite_transport_not_ready",
                "episode_id": episode_id,
                "step_id": step.get("step_id"),
                "reason_code": block.get("reason_code"),
                "risk": block.get("risk"),
                "features": block.get("features"),
            }
        )
        if not opts["reseat"]["enabled"]:
            block["reseat"]["reason"] = "re-seat is disabled in config"
        elif self._action_backend() != "pi05":
            block["reseat"]["reason"] = "re-seat requires the whole-task policy backend"
        else:
            obs, snapshot, token = self._transport_reseat(
                env=env,
                episode_id=episode_id,
                step=step,
                step_task=step_task,
                task=task,
                record=record,
                snapshot=snapshot,
                token=token,
                stop_event=stop_event,
                composite_state=composite_state,
                timing=timing,
                records=records,
                opts=opts,
            )
            after = (block["reseat"].get("after") or {}) if isinstance(block["reseat"], dict) else {}
            if after.get("checked"):
                # The trigger is kept verbatim; the top-level fields then
                # describe the state transport actually started from.
                block["before_reseat"] = {
                    "ok": block.get("ok"),
                    "risk": block.get("risk"),
                    "reason_code": block.get("reason_code"),
                    "reasons": block.get("reasons"),
                    "features": block.get("features"),
                }
                block["reasons"] = after.get("reasons")
                block["features"] = after.get("features")
                block["reason_code"] = after.get("reason_code")
                block["risk"] = after.get("risk")
                block["ok"] = bool(after.get("ok"))
                block["ready"] = bool(after.get("ok"))
                block["reseat"]["reseat_succeeded"] = bool(after.get("ok"))
        if not block.get("ok"):
            on_failure = str(opts["reseat"].get("on_failure") or "stop")
            block["on_reseat_failure"] = on_failure
            if on_failure == "stop":
                reseat_row = block.get("reseat") or {}
                change = reseat_row.get("state_change") or {}
                tail = ""
                if reseat_row.get("status") == "skipped":
                    tail = (
                        "; the re-seat was refused before acting because the placement predicate "
                        "already held (recorded as unsupported, no action executed)"
                    )
                elif change.get("changed") is False:
                    tail = "; the re-seat ran and the geometry did not move"
                elif change.get("changed") is True:
                    tail = (
                        f"; the re-seat moved the object by "
                        f"{change.get('seat_delta_change_m')} m in seat depth and still did not "
                        f"reach a ready state"
                    )
                raise SkillFailure(
                    "composite_transport_not_ready",
                    (
                        f"step {step.get('step_id')} placed the object but it is not safe to "
                        f"carry ({block.get('reason_code')}); one bounded re-seat did not "
                        f"reach a ready state{tail}"
                    ),
                    {
                        "transport_ready": transport_ready.summarise(block),
                        "features": block.get("features"),
                        "reseat": block.get("reseat"),
                    },
                )
            block["proceeded_not_ready"] = True
            records.append_journal(
                {
                    "t": time.time(),
                    "stage": "composite_transport_proceeded_not_ready",
                    "episode_id": episode_id,
                    "step_id": step.get("step_id"),
                    "reason_code": block.get("reason_code"),
                }
            )
        return obs, snapshot, token

    def _transport_reseat(
        self,
        *,
        env: EnvClient,
        episode_id: str,
        step: dict[str, Any],
        step_task: dict[str, Any],
        task: dict[str, Any],
        record: dict[str, Any],
        snapshot: dict[str, Any],
        token: str,
        stop_event: threading.Event,
        composite_state: dict[str, Any],
        timing: dict[str, Any],
        records,
        opts: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any], str]:
        """One bounded re-run of the same catalog capability, from live state.

        No reset, no reload, no teleport, no object or physics write: the policy
        receives the same official instruction again and the executed actions
        are ordinary ``env.step`` calls counted in this episode.
        """
        reseat = composite_state["transport_ready"]["reseat"]
        reseat.update(
            {
                "attempted": True,
                "calls": int(reseat.get("calls") or 0) + 1,
                "task_key": step.get("step_task_key"),
                "prompt": step.get("prompt"),
                "step_id": f"r{int(step.get('step_index') or 1)}",
                "status": "executing",
            }
        )
        reseat_step = dict(step)
        reseat_step["step_id"] = str(reseat["step_id"])
        reseat_step["action_source"] = "policy_reseat"
        token = self.inference.reset(f"reseat:{episode_id}")
        obs, snapshot = env.raw_obs(), env.state_snapshot()
        reseat["start"] = {
            "observation_id": snapshot.get("observation_id"),
            "sim_action_count": snapshot.get("sim_action_count"),
            "scene_epoch": snapshot.get("scene_epoch"),
            "token": token,
        }
        records.append_journal(
            {
                "t": time.time(),
                "stage": "composite_transport_reseat_start",
                "episode_id": episode_id,
                "step_id": step.get("step_id"),
                "reseat_step_id": reseat["step_id"],
                "task_key": step.get("step_task_key"),
                "observation_id": snapshot.get("observation_id"),
                "sim_action_count": snapshot.get("sim_action_count"),
                "scene_epoch": snapshot.get("scene_epoch"),
            }
        )
        # Pre-check: the capability this re-seat re-runs has its own goal, and
        # that goal is usually *already true* when readiness refuses the carry
        # (readiness measures seat quality, not the placement predicate). Asking
        # the policy again would emit one action and stop on the first rollout
        # tick, which is motion without a re-seat. So the attempt is recorded as
        # skipped and named unsupported instead of being counted as a repair.
        precheck = composite.state_check(snapshot, reseat_step, task)
        reseat["precheck"] = {
            "predicate": precheck.get("predicate"),
            "ok": precheck.get("ok"),
            "observation_id": precheck.get("observation_id"),
            "sim_action_count": precheck.get("sim_action_count"),
            "note": "this is the re-run capability's own goal, not the transport-readiness verdict",
        }
        if precheck.get("ok"):
            reseat["status"] = "skipped"
            reseat["policy_actions"] = 0
            reseat["finalizer_actions"] = 0
            reseat["reseat_succeeded"] = False
            reseat["skip_reason"] = "goal_already_satisfied"
            reseat["after"] = transport_ready.evaluate(
                step=step,
                relation=record.get("final_evidence") or record.get("evidence"),
                samples=(record.get("placement") or {}).get("samples"),
                evidence=record.get("final_evidence") or record.get("evidence"),
                cfg=self.cfg,
            )
            reseat["state_change"] = transport_ready.change(
                composite_state["transport_ready"].get("features"),
                reseat["after"].get("features"),
            )
            reseat["state_change"]["note"] = (
                "no action was executed: the re-seat was refused by its own pre-check, so the geometry "
                "being unchanged says nothing about what a re-seat would have done"
            )
            reseat["classification"] = transport_ready.classify_reseat(
                skipped=True,
                stop_reason=None,
                policy_actions=0,
                state_change=reseat["state_change"],
                before_ok=composite_state["transport_ready"].get("ok"),
                after_ok=reseat["after"].get("ok"),
            )
            record["reseat"] = reseat
            records.append_journal(
                {
                    "t": time.time(),
                    "stage": "composite_transport_reseat_skipped",
                    "episode_id": episode_id,
                    "step_id": step.get("step_id"),
                    "reseat_step_id": reseat["step_id"],
                    "task_key": step.get("step_task_key"),
                    "skip_reason": reseat["skip_reason"],
                    "precheck_ok": True,
                }
            )
            return obs, snapshot, token
        started = time.perf_counter()
        try:
            reseat_record, obs, snapshot = self._run_whole_task(
                env=env,
                obs=obs,
                snapshot=snapshot,
                step=reseat_step,
                task=task,
                token=token,
                episode_id=episode_id,
                stop_event=stop_event,
                goal=composite.step_goal(reseat_step),
                max_steps=int(opts["reseat"]["max_steps"]),
                step_task=step_task,
            )
        except Stopped:
            reseat["status"] = "stopped"
            reseat["reason"] = "user_stop"
            composite_state["transport_ready"]["reason_code"] = "user_stop"
            raise
        except Exception as exc:  # noqa: BLE001 - report, never fake success
            reseat["status"] = "failed"
            reseat["reason"] = f"{type(exc).__name__}: {exc}"
            reseat["after"] = transport_ready.evaluate(
                step=step,
                relation=record.get("final_evidence") or record.get("evidence"),
                samples=(record.get("placement") or {}).get("samples"),
                evidence=record.get("final_evidence") or record.get("evidence"),
                cfg=self.cfg,
            )
            reseat["classification"] = transport_ready.classify_reseat(
                stop_reason=None,
                policy_actions=None,
                state_change=transport_ready.change(
                    composite_state["transport_ready"].get("features"),
                    reseat["after"].get("features"),
                ),
                before_ok=composite_state["transport_ready"].get("ok"),
                after_ok=reseat["after"].get("ok"),
            )
            return obs, snapshot, token
        reseat["s"] = round(time.perf_counter() - started, 3)
        timing["reseat_s"] = float(timing.get("reseat_s") or 0.0) + reseat["s"]
        placement = reseat_record.get("placement") or {}
        reseat["policy_actions"] = int(reseat_record.get("sim_actions") or 0)
        reseat["finalizer_actions"] = int(placement.get("extra_actions") or 0)
        reseat["sim_action_counts"] = {
            "start": (reseat.get("start") or {}).get("sim_action_count"),
            "end": snapshot.get("sim_action_count"),
        }
        reseat["result"] = {
            "ok": bool(reseat_record.get("ok")),
            "stop_reason": reseat_record.get("stop_reason"),
            "verification": reseat_record.get("verification"),
            "release_state": reseat_record.get("release_state"),
            "release_verified": bool(placement.get("release_verified")),
            "placement_verified": bool(placement.get("placement_verified")),
            "reason_code": placement.get("reason_code"),
        }
        reseat["status"] = "executed"
        reseat["after"] = transport_ready.evaluate(
            step=step,
            relation=reseat_record.get("final_evidence") or reseat_record.get("evidence"),
            samples=placement.get("samples"),
            evidence=reseat_record.get("final_evidence") or reseat_record.get("evidence"),
            cfg=self.cfg,
        )
        reseat["after_evidence"] = {
            "observation_ids": [row.get("observation_id") for row in (placement.get("samples") or [])],
            "sim_action_counts": [row.get("sim_action_count") for row in (placement.get("samples") or [])],
        }
        reseat["reseat_succeeded"] = bool(reseat["after"].get("ok"))
        reseat["state_change"] = transport_ready.change(
            composite_state["transport_ready"].get("features"),
            reseat["after"].get("features"),
        )
        before_ok = composite_state["transport_ready"].get("ok")
        reseat["classification"] = transport_ready.classify_reseat(
            stop_reason=reseat_record.get("stop_reason"),
            policy_actions=reseat["policy_actions"],
            state_change=reseat["state_change"],
            before_ok=before_ok,
            after_ok=reseat.get("after", {}).get("ok"),
        )
        record["reseat"] = reseat
        records.append_journal(
            {
                "t": time.time(),
                "stage": "composite_transport_reseat_end",
                "episode_id": episode_id,
                "step_id": step.get("step_id"),
                "reseat_step_id": reseat["step_id"],
                "status": reseat["status"],
                "policy_actions": reseat["policy_actions"],
                "succeeded": reseat["reseat_succeeded"],
                "after": transport_ready.summarise(reseat["after"]),
                "sim_action_count": snapshot.get("sim_action_count"),
            }
        )
        return obs, snapshot, token

    # ---- errors earlier steps' goals under a carried object ---------------

    def _carry_watch(
        self,
        *,
        steps: list[dict[str, Any]],
        index: int,
        catalog_by_key: dict[str, dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[str]]:
        """The earlier goals a later step must not destroy while it moves things.

        Only engine-predicate goals are watched: the anchor's own goal is the
        loaded scene's native flag and is already reported by the rollout. The
        goal is re-evaluated on every snapshot, so for "cheese in bowl" it is
        checked against the bowl's *current* pose while the bowl is carried.
        """
        watch: list[dict[str, Any]] = []
        for previous in steps[:index]:
            goal = composite.step_goal(previous)
            if not isinstance(goal, dict):
                continue
            watch.append(
                {
                    "step_id": previous.get("step_id"),
                    "key": previous.get("step_task_key"),
                    "goal_kind": previous.get("goal_kind"),
                    "object_n": goal.get("object_n"),
                    "target_n": goal.get("target_n"),
                    "relation": goal.get("relation"),
                    "goal": goal,
                }
            )
        current = catalog_by_key.get(str(steps[index].get("step_task_key"))) if index < len(steps) else {}
        carry = [str(name) for name in ((current or {}).get("objects") or []) if isinstance(name, str)]
        return watch, carry

    def _composite_handoff(
        self,
        *,
        env: EnvClient,
        episode_id: str,
        step: dict[str, Any],
        next_step: dict[str, Any],
        snapshot: dict[str, Any],
        catalog_by_key: dict[str, dict[str, Any]],
        task: dict[str, Any],
        home_eef: list[float] | None,
        composite_state: dict[str, Any],
        step_results: list[dict[str, Any]],
        timing: dict[str, Any],
        stop_event: threading.Event,
        records,
    ) -> tuple[dict[str, Any], dict[str, Any], str]:
        """Bounded transition between two composite steps.

        Shared by the executed step and the *skipped* step (one whose goal was
        already true before it started): both must leave the arm in a known
        pose, clear the policy buffer, and hand step 2 a fresh observation.
        """
        handoff_started = time.perf_counter()
        # Named phase so an external observer (and the stop test) can
        # synchronise on the handoff instead of guessing timing with sleeps.
        self._set_phase(
            "handoff",
            f"step {step.get('step_id')}->{next_step.get('step_id')} "
            f"{step.get('step_task_key')}->{next_step.get('step_task_key')}",
        )
        handoff = composite.Handoff(
            env=env,
            inference=self.inference,
            stop_event=stop_event,
            cfg=self.cfg,
            episode_id=episode_id,
            step=step,
            step_task=catalog_by_key[step.get("step_task_key")],
            task=task,
            home_eef=home_eef,
            home_quat=composite_state.get("home_quat"),
            robot_state=composite_state.get("robot_state_capture") or {},
            prior_step=(step_results[-1] if step_results else None),
            journal=lambda stage, row: records.append_journal(
                {"t": time.time(), "stage": stage, "episode_id": episode_id, "row": row}
            ),
            frame_sink=self._set_frames,
        )
        try:
            handoff_block, handoff_obs, handoff_snapshot, token = handoff.run(snapshot)
        except Stopped:
            raise
        except Exception as exc:  # noqa: BLE001 - report the stage, do not fake success
            raise SkillFailure(
                "composite_handoff_error",
                f"handoff after step {step.get('step_index')} raised {type(exc).__name__}: {exc}",
                {"traceback": traceback.format_exc()[-2000:]},
            ) from exc
        timing["handoff_s"] += time.perf_counter() - handoff_started
        composite_state["handoff_records"].append(handoff_block)
        composite_state["handoff"] = handoff_block
        records.append_journal(
            {
                "t": time.time(),
                "stage": "composite_handoff",
                "episode_id": episode_id,
                "status": handoff_block.get("status"),
                "from_step": step.get("step_id"),
                "to_step": next_step.get("step_id"),
                "sim_actions": handoff_block.get("sim_actions"),
            }
        )
        with self.lock:
            self.steps_done = [s for s in step_results]
        if handoff_block.get("status") != "ready":
            raise SkillFailure(
                "composite_handoff_failed",
                (
                    f"handoff after step {step.get('step_index')} is {handoff_block.get('status')}: "
                    f"{handoff_block.get('reason')}"
                ),
                {"handoff": handoff_block, "next_step": next_step.get("step_id")},
            )
        records.append_journal(
            {
                "t": time.time(),
                "stage": "composite_step_start",
                "episode_id": episode_id,
                "step_id": next_step.get("step_id"),
                "task_id": next_step.get("step_task_id"),
                "prompt": next_step.get("prompt"),
                "observation_id": handoff_snapshot.get("observation_id"),
                "scene_epoch": handoff_snapshot.get("scene_epoch"),
                "token": token,
            }
        )
        return handoff_obs, handoff_snapshot, token

    # ---- bounded final-goal repair ---------------------------------------

    def _composite_repair(
        self,
        *,
        env: EnvClient,
        task: dict[str, Any],
        catalog_by_key: dict[str, dict[str, Any]],
        step_results: list[dict[str, Any]],
        final_steps: list[dict[str, Any]],
        snapshot: dict[str, Any],
        baseline: dict[str, Any],
        composite_state: dict[str, Any],
        episode_id: str,
        stop_event: threading.Event,
        timing: dict[str, Any],
        token: str,
    ) -> dict[str, Any]:
        """At most one Qwen-chosen catalog task to restore a lost earlier goal.

        Preconditions are checked in code, not asked of the model: this path is
        only reachable when the *anchor* step (the loaded scene's own task) has
        already succeeded and released, and an earlier step's predicate was
        lost. The model then picks one catalog repair key or aborts; the engine
        looks up that key's official instruction and executes it with the same
        whole-task policy on the current state.
        """
        opts = composite.repair_options(self.cfg)
        block: dict[str, Any] = {
            "schema": "embodied_engine.composite.repair.v1",
            "enabled": bool(opts["enabled"]),
            "attempted": False,
            "triggered_by": None,
            "task_key": None,
            "task_id": None,
            "prompt": None,
            "goal_kind": None,
            "decision": None,
            "decision_source": None,
            "reason": None,
            "status": "not_attempted",
            "first_pass_success": False,
            "repair_succeeded": False,
            "final_all_goals_met": None,
            "action_source": "none",
            "policy_actions": 0,
            "finalizer_actions": 0,
            "budget": {
                "max_calls": int(opts["max_calls"]),
                "max_steps": int(opts["max_steps"]),
                "timeout_s": float(opts["timeout_s"]),
            },
            "candidates": [],
            "steps_before": [],
        }
        if not block["enabled"]:
            block["reason"] = "repair is disabled in config"
            return block

        anchor = final_steps[-1] if final_steps else {}
        failed = [row for row in final_steps[:-1] if not row.get("ok")]
        block["steps_before"] = [
            {
                "step_id": row.get("step", {}).get("step_id"),
                "task_key": row.get("step", {}).get("step_task_key"),
                "goal_kind": row.get("step", {}).get("goal_kind"),
                "ok": bool(row.get("ok")),
                "reason_code": (row.get("verification") or {}).get("reason_code"),
                "qpos": (row.get("verification") or {}).get("qpos"),
            }
            for row in final_steps
        ]
        if not anchor.get("ok"):
            # The anchor failed: this is the policy's own trajectory, not a
            # lost earlier goal. A repair task would hide the real failure.
            block["reason"] = "the final step did not reach its own goal; repair is limited to lost earlier goals"
            block["status"] = "ineligible_anchor_failed"
            return block
        if not failed:
            block["reason"] = "no earlier step goal is lost"
            block["status"] = "ineligible_nothing_lost"
            return block
        block["triggered_by"] = [
            {
                "step_id": row.get("step", {}).get("step_id"),
                "task_key": row.get("step", {}).get("step_task_key"),
                "goal_kind": row.get("step", {}).get("goal_kind"),
                "reason_code": (row.get("verification") or {}).get("reason_code"),
            }
            for row in failed
        ]
        anchor_placement = dict((step_results[-1] or {}).get("placement") or {})
        if opts["require_final_step_released"] and anchor_placement.get("required"):
            if not anchor_placement.get("placement_verified"):
                block["reason"] = "the final step's release is not verified; moving the arm could drag it"
                block["status"] = "ineligible_release_unverified"
                return block
        block["anchor_release"] = {
            "required": bool(anchor_placement.get("required")),
            "placement_verified": bool(anchor_placement.get("placement_verified")),
            "evidence_source": anchor_placement.get("evidence_source"),
            "action_source": anchor_placement.get("action_source"),
        }

        # Structured candidate list: only catalog capabilities the engine can
        # judge on this scene, with their current state. The model chooses a key.
        candidates: list[dict[str, Any]] = []
        for key, row in catalog_by_key.items():
            goal_desc = composite.completion_for(row, is_anchor=False)
            if goal_desc.get("kind") == "unsupported":
                continue
            support = task_catalog.scene_support(row, snapshot)
            if not support.get("ok"):
                continue
            goal = composite.step_goal({"completion": goal_desc})
            now = completion.goal_state(snapshot, goal or {}, row)
            candidates.append(
                {
                    "key": key,
                    "task_id": int(row["task_id"]),
                    "kind": str(row.get("kind") or ""),
                    "official_instruction": str(
                        row.get("official_instruction") or row.get("instruction") or ""
                    ),
                    "goal_kind": goal_desc.get("kind"),
                    "goal_met_now": bool(now.get("ok")),
                    "scene_support": {"ok": True, "missing": []},
                }
            )
        block["candidates"] = candidates
        repairable = [row for row in candidates if not row["goal_met_now"]]
        if not repairable:
            block["reason"] = "no catalog capability can change the current state"
            block["status"] = "ineligible_no_candidate"
            return block

        self._set_phase("repairing", "Qwen chooses one repair task for a lost earlier goal")
        request_started = time.perf_counter()
        try:
            reply = _post_json(
                f"{self.agent_url}/repair",
                {
                    "instruction": str(task.get("instruction") or ""),
                    "task_id": int(task["task_id"]),
                    "snapshot": snapshot,
                    "composite": {"keys": composite_state.get("keys")},
                    "steps": block["steps_before"],
                    "failed": block["triggered_by"],
                    "candidates": candidates,
                    "backend": self._action_backend(),
                },
                timeout_s=90,
            )
        except Exception as exc:  # noqa: BLE001 - a failed decision is not a success
            block["status"] = "decision_failed"
            block["reason"] = f"{type(exc).__name__}: {exc}"
            return block
        block["decision_s"] = round(time.perf_counter() - request_started, 3)
        block["decision_source"] = reply.get("policy") or "qwen"
        block["decision"] = reply.get("decision")
        block["reason"] = reply.get("reason")
        block["decision_agent_s"] = reply.get("total_s")
        records.append_journal(
            {
                "t": time.time(),
                "stage": "composite_repair_decision",
                "episode_id": episode_id,
                "decision": block["decision"],
                "task_key": reply.get("key"),
                "reason": block["reason"],
            }
        )
        if reply.get("status") != "ok" or block["decision"] != "repair":
            block["status"] = "aborted_by_agent"
            return block

        key = str(reply.get("key") or "")
        by_key = {row["key"]: row for row in candidates}
        candidate = by_key.get(key)
        if candidate is None:
            block["status"] = "decision_invalid"
            block["reason"] = f"the chosen key {key!r} is not an executable repair candidate"
            return block
        if candidate["goal_met_now"]:
            block["status"] = "decision_invalid"
            block["reason"] = f"{key} is already satisfied; it would not be a repair"
            return block
        repair_task = catalog_by_key[key]
        goal_desc = composite.completion_for(repair_task, is_anchor=False)
        block.update(
            {
                "task_key": key,
                "task_id": candidate["task_id"],
                "prompt": candidate["official_instruction"],
                "goal_kind": candidate["goal_kind"],
                "attempted": True,
                "status": "executing",
            }
        )
        repair_step = {
            "skill": "whole_task",
            "object_n": None,
            "target_n": None,
            "relation": "task",
            "prompt": candidate["official_instruction"],
            "step_id": "r1",
            "step_index": 1,
            "step_task_id": candidate["task_id"],
            "step_task_key": key,
            "goal_kind": candidate["goal_kind"],
            "completion": goal_desc,
            "action_source": "policy_repair",
        }
        # A repair is a new command on the current state: the buffer from the
        # anchor's last chunk must not be consumed, and step 1 of the repair
        # starts from a fresh observation.
        token = self.inference.reset(f"repair:{episode_id}")
        obs, snapshot = env.raw_obs(), env.state_snapshot()
        self._set_frames(obs, snapshot)
        block["start"] = {
            "observation_id": snapshot.get("observation_id"),
            "sim_action_count": snapshot.get("sim_action_count"),
            "scene_epoch": snapshot.get("scene_epoch"),
            "token": token,
        }
        records.append_journal(
            {
                "t": time.time(),
                "stage": "composite_repair_start",
                "episode_id": episode_id,
                "task_key": key,
                "prompt": candidate["official_instruction"],
                "observation_id": snapshot.get("observation_id"),
                "sim_action_count": snapshot.get("sim_action_count"),
                "scene_epoch": snapshot.get("scene_epoch"),
                "token": token,
            }
        )
        repair_started = time.perf_counter()
        try:
            record, obs, snapshot = self._run_whole_task(
                env=env,
                obs=obs,
                snapshot=snapshot,
                step=repair_step,
                task=task,
                token=token,
                episode_id=episode_id,
                stop_event=stop_event,
                goal=composite.step_goal(repair_step),
                max_steps=int(opts["max_steps"]),
                step_task=repair_task,
            )
        except Stopped:
            raise
        except Exception as exc:  # noqa: BLE001 - report, never fake success
            block["status"] = "failed"
            block["reason"] = f"{type(exc).__name__}: {exc}"
            block["repair_s"] = round(time.perf_counter() - repair_started, 3)
            return block
        block["repair_s"] = round(time.perf_counter() - repair_started, 3)
        timing["repair_s"] = float(timing.get("repair_s") or 0.0) + block["repair_s"]
        placement = record.get("placement") or {}
        block["policy_actions"] = int(record.get("sim_actions") or 0)
        block["finalizer_actions"] = int(placement.get("extra_actions") or 0)
        block["action_source"] = {
            "policy": "policy_repair",
            "finalizer": "execution_finalizer" if block["finalizer_actions"] else "none",
        }
        block["result"] = {
            "ok": bool(record.get("ok")),
            "verification": record.get("verification"),
            "sim_actions": record.get("sim_actions"),
            "chunks_used": record.get("chunks_used"),
            "stop_reason": record.get("stop_reason"),
            "placement": {
                key_: placement.get(key_)
                for key_ in (
                    "required",
                    "attempted",
                    "release_executed",
                    "release_verified",
                    "placement_verified",
                    "execution_status",
                    "reason_code",
                    "evidence_source",
                )
            },
        }
        block["status"] = "executed" if record.get("ok") else "repair_did_not_reach_goal"
        block["repair_succeeded"] = bool(record.get("ok"))
        block["episode_record"] = record
        block["_obs"] = obs
        block["_snapshot"] = snapshot
        block["_token"] = token
        return block

    # ---- placement completion -------------------------------------------

    @staticmethod
    def _placement_step(task: dict[str, Any]) -> dict[str, Any] | None:
        """Relation binding for a whole-task pick/place, from the catalog."""
        objects = task.get("objects") or []
        targets = task.get("targets") or []
        if not objects or not targets:
            return None
        return {
            "skill": "pick_place",
            "object_n": objects[0],
            "target_n": targets[0],
            "relation": (task.get("semantics") or {}).get("relation") or "on",
        }

    def _finish_placement(
        self,
        *,
        env: EnvClient,
        task: dict[str, Any],
        record: dict[str, Any],
        obs: dict[str, Any],
        snapshot: dict[str, Any],
        token: str,
        episode_id: str,
        stop_event: threading.Event,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], str]:
        """Decide how a whole-task episode's placement ends.

        The policy gets one bounded window to release on its own. If that ends
        without a measured release, the execution layer finishes the placement
        (open in place, bounded retreat, fresh samples) and every action it
        issues is marked ``evidence_source="execution_finalizer"``.
        """
        kind = str(task.get("kind") or "")
        required = kind in {"pick_place", "composite"}
        post_cfg = self._post_goal_cfg()
        fin_cfg = dict(post_cfg.get("finalizer") or {})
        post = dict(record.get("post_goal") or {})
        native = bool((record.get("verification") or {}).get("ok"))
        block = finalize.policy_release_block(
            required=required,
            native_goal=native,
            post_goal=post,
            snapshot=snapshot,
            object_name=(task.get("objects") or [None])[0],
            open_min_m=float(post_cfg.get("release_open_m") or 0.05),
        )
        block["policy"] = {
            "actions_after_goal": int(post.get("actions_used") or 0),
            "stop_reason": post.get("stop_reason"),
            "window_actions": int(post.get("window_actions") or 0),
            "release": post.get("release") or {},
            "release_verified": bool(block["release_verified"]),
        }
        if not required:
            block["execution_status"] = "not_applicable"
            block["reason_code"] = None
            return block, obs, snapshot, token
        if block["release_verified"]:
            return block, obs, snapshot, token
        relation_step = self._placement_step(task)
        if not fin_cfg.get("enabled"):
            block["reason_code"] = "release_not_verified"
            return block, obs, snapshot, token
        if relation_step is None:
            block["reason_code"] = "no_relation_binding"
            return block, obs, snapshot, token
        finalizer = finalize.PlacementFinalizer(
            env=env,
            task=task,
            relation_step=relation_step,
            execution_cfg=self.cfg["execution"],
            post_goal_cfg=post_cfg,
            stop_event=stop_event,
            episode_id=episode_id,
            journal=lambda stage, row: records.append_journal(
                {"t": time.time(), "stage": stage, "episode_id": episode_id, "row": row}
            ),
            frame_sink=self._set_frames,
            reset_policy=lambda reason: self.inference.reset(reason),
        )
        result = finalizer.run(snapshot)
        if isinstance(result.get("token"), str):
            token = result["token"]
        if getattr(finalizer, "_last_snapshot", None):
            snapshot = finalizer._last_snapshot
        if getattr(finalizer, "_last_obs", None) is not None:
            obs = finalizer._last_obs
        base = {
            key: block.get(key)
            for key in ("schema", "required", "native_goal", "release", "detach")
        }
        merged = dict(result)
        merged["schema"] = finalize.SCHEMA
        merged["required"] = True
        merged["native_goal"] = bool(block["native_goal"])
        merged["policy_release"] = block.get("release")
        merged["policy"] = block.get("policy")
        merged["policy_release_verified"] = bool(block["release_verified"])
        merged["policy_owned"] = False
        merged["release_executed"] = bool(
            block.get("release_executed") or result.get("release_executed")
        )
        merged["evidence_source"] = (
            "execution_finalizer" if result.get("release_executed") else "none"
        )
        if merged.get("execution_status") == "completed" and not bool(block["native_goal"]):
            merged["execution_status"] = "unknown"
        if base.get("detach") and not merged.get("policy_detach"):
            merged["policy_detach"] = base.get("detach")
        return merged, obs, snapshot, token


    def _run_episode(
        self,
        task: dict[str, Any],
        instruction: str,
        episode_id: str,
        run_id: int,
        stop_event: threading.Event,
        planner_generation: int,
        plan_keys: list[str] | None = None,
    ) -> None:
        started = time.time()
        started_iso = _now_iso()
        recovery_log: list[dict[str, Any]] = []
        init_started = time.perf_counter()
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
        # Scene load/reset is *inside* the episode wall clock (the engine resets
        # the env before every episode even when the scene was already loaded),
        # but it used to be outside every timing bucket. Measure it so the
        # decomposition adds up instead of leaving a silent gap.
        episode_init_s = time.perf_counter() - init_started
        initial_scene = _scene_fingerprint(snapshot)
        initial_state = _state_fingerprint(snapshot)
        initial_fixtures = _fixture_fingerprint(snapshot)
        handoff_mode = composite.handoff_mode(self.cfg)
        handoff_options = composite.handoff_options(self.cfg)
        robot_state_capture: dict = {}
        robot_state_capture_error = ""
        if handoff_mode == "robot_state_restore":
            # The episode's own initial robot state, not ``robot.init_qpos``: this is
            # read after the settle steps of reset, immediately before the first
            # task action, and stays inside the env service process.
            try:
                robot_state_capture = env.robot_state_capture(
                    f"episode_init:{episode_id}"
                )
            except Exception as exc:  # noqa: BLE001 - captured in the record
                robot_state_capture_error = f"{type(exc).__name__}: {exc}"
                robot_state_capture = {
                    "ok": False,
                    "reason_code": "capture_rpc_failed",
                    "reason": robot_state_capture_error,
                }
            records.append_journal(
                {
                    "event": "robot_state_capture",
                    "episode_id": episode_id,
                    "init_seed": self.seed,
                    "handoff_mode": handoff_mode,
                    "ok": bool(robot_state_capture.get("ok")),
                    "capture_id": robot_state_capture.get("capture_id"),
                    "reason_code": robot_state_capture.get("reason_code"),
                    "reason": robot_state_capture.get("reason"),
                    "observation_id": snapshot.get("observation_id"),
                    "sim_action_count": snapshot.get("sim_action_count"),
                    "robot": robot_state_capture.get("robot"),
                    "nonrobot_digest": robot_state_capture.get("nonrobot_digest"),
                }
            )
        token = self.inference.reset(f"episode:{episode_id}")
        with self.lock:
            self.scene_epoch = snapshot.get("scene_epoch")
            self.sim_action_count = int(snapshot.get("sim_action_count") or 0)
        self._set_frames(obs, snapshot)
        timing = {
            "episode_init_s": round(episode_init_s, 3),
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
            "handoff_s": 0.0,
            "repair_s": 0.0,
        }
        step_results: list[dict[str, Any]] = []
        attempt_digests: list[dict[str, Any]] = []
        step_attempts: dict[tuple[Any, ...], int] = {}
        home_eef = (baseline.get("sim") or {}).get("eef_pos")
        plan = None
        attempts: list[dict[str, Any]] = []
        failure: dict[str, Any] | None = None
        execution_status = "failed"
        native_ever = False
        # Episode scope, not verification scope: a SkillFailure raised by step 2
        # (budget exhausted) jumps straight to the record assembly, which still
        # reads these rows. They were first declared next to the verification
        # loop and that raised UnboundLocalError, leaving the episode thread dead
        # in phase "running" (observed on 12-step-2 budget failure).
        final_steps: list[dict[str, Any]] = []
        verification_attempts: list[dict[str, Any]] = []
        plan_completed_at: float | None = None
        openwam_stats: dict[str, Any] | None = None
        composite_state: dict[str, Any] = {"enabled": False}
        # Kept at episode scope: a stop raised from inside the placement
        # finalizer aborts the step before its record row exists, so the only
        # surviving evidence of "release was interrupted mid-way" is this block.
        last_placement: dict[str, Any] = {}
        try:
            self._set_phase("planning", "Qwen semantic routing + whole_task plan")
            plan_started = time.perf_counter()
            plan_response = _post_json(
                f"{self.agent_url}/plan",
                {
                    "instruction": instruction,
                    "task_id": task["task_id"],
                    "snapshot": snapshot,
                    "attempts": [],
                    "backend": self._action_backend(),
                    # Screening entry: an explicit ordered key list bypasses the
                    # language router. It is validated by the same catalog and
                    # composite contract, and recorded as a structured plan.
                    "structured_keys": plan_keys,
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
                    "understanding": plan_response.get("understanding"),
                    "reason": plan_response.get("reason"),
                    "reason_code": plan_response.get("reason_code"),
                    "planner_vision": False,
                }
            if plan_response.get("status") != "ok":
                raise SkillFailure(
                    "plan_invalid",
                    str(plan_response.get("reason") or "Agent plan was not valid"),
                    {"response": plan_response},
                )
            plan = plan_response["plan"]
            plan_completed_at = time.perf_counter()
            steps = list(plan["steps"])
            composite_plan = isinstance(plan, dict) and plan.get("task_scope") == "combo"
            catalog_by_key = {str(row["key"]): row for row in task_catalog.tasks()}
            # A plan is validated against the catalog the planner was given;
            # this is the second gate, because a wrong executor would run the
            # right skill against the wrong action contract (8-D proprio into a
            # 10-D EEF10 backend) and fail halfway through an episode.
            backend = self._action_backend()
            planned = [step.get("skill") for step in steps]
            if composite_plan:
                if backend not in ("openwam", "pi05"):
                    raise SkillFailure(
                        "plan_backend_mismatch",
                        "composite plans require a whole-task backend (openwam or pi05)",
                        {"backend": backend},
                    )
                composite_errors = composite.validate_plan(
                    plan, catalog=catalog_by_key, loaded_task=task, cfg=self.cfg
                )
                if composite_errors:
                    raise SkillFailure(
                        "composite_plan_invalid",
                        "the composite plan failed its own contract",
                        {"errors": composite_errors, "plan": plan},
                    )
            elif backend == "openwam" and planned != ["whole_task"]:
                raise SkillFailure(
                    "plan_backend_mismatch",
                    "openwam backend requires exactly one whole_task step",
                    {"backend": backend, "planned_skills": planned},
                )
            if not composite_plan and backend == "pi05" and planned != ["whole_task"]:
                raise SkillFailure(
                    "plan_backend_mismatch",
                    "pi05 backend requires exactly one whole_task step",
                    {"backend": backend, "planned_skills": planned},
                )
            if (not composite_plan and backend not in ("openwam", "pi05")
                    and "whole_task" in planned):
                raise SkillFailure(
                    "plan_backend_mismatch",
                    f"{backend} backend cannot execute whole_task",
                    {"backend": backend, "planned_skills": planned},
                )
            if composite_plan:
                composite_state = {
                    "enabled": True,
                    "keys": list((plan.get("composite") or {}).get("keys") or []),
                    "plan_source": plan.get("plan_source"),
                    "flow": (plan.get("composite") or {}).get("flow"),
                    "flow_expansion": plan.get("flow_expansion"),
                    "flow_guard": plan.get("flow_guard"),
                    "structured_entry": bool(plan_keys),
                    "anchor": {
                        "task_id": int(task["task_id"]),
                        "key": str(task.get("key")),
                        "bddl": task.get("bddl"),
                    },
                    "scene_epoch_start": snapshot.get("scene_epoch"),
                    "sim_action_count_start": snapshot.get("sim_action_count"),
                    "step_ids": [str(step.get("step_id")) for step in steps],
                    "first_pass_success": True,
                    "repair": None,
                    "repair_attempted": False,
                    "budgets": {
                        str(step.get("step_index")): composite.step_budget(
                            int(step.get("step_index") or 1), self.cfg
                        )
                        for step in steps
                    },
                    "handoff_records": [],
                    "handoff_mode": handoff_mode,
                    "handoff_options": handoff_options,
                    "robot_state_capture": robot_state_capture,
                    "robot_state_capture_error": robot_state_capture_error,
                }
                # The episode's initial wrist orientation, read before any
                # action: the handoff restores it so step 2 starts from a pose
                # close to the one the action model was trained to start in.
                home_quat = (baseline.get("sim") or {}).get("eef_quat")
                if home_quat is None:
                    try:
                        start_obs = env.raw_obs()
                        home_quat = _quat_of(start_obs)
                    except Exception as exc:  # noqa: BLE001 - orientation is optional evidence
                        home_quat = None
                        records.append_journal(
                            {"t": time.time(), "stage": "composite_home_quat_failed",
                             "episode_id": episode_id, "detail": f"{type(exc).__name__}: {exc}"}
                        )
                composite_state["home_quat"] = home_quat
            # The user-visible goal is the final step of the initial plan.
            # A replan may reorder or add steps, but dropping this step would
            # silently solve a different task.
            goal_step = dict(steps[-1]) if steps else None
            # Recovery must ask about the same official prompt the policy was
            # given, not the user's paraphrase.
            goal_prompt = str((goal_step or {}).get("prompt") or instruction)
            index = 0
            retries = 0
            replans = 0
            max_recoveries = int(self.cfg["execution"]["max_recoveries"])
            while index < len(steps):
                if stop_event.is_set():
                    raise Stopped("stop requested")
                step = steps[index]
                if composite_state.get("enabled"):
                    # Precondition check on the *fresh* state before the step's
                    # first action. A goal that already holds is recorded as
                    # skipped, never as a capability the policy demonstrated.
                    step_task_row = catalog_by_key.get(str(step.get("step_task_key"))) or {}
                    try:
                        pre_goal_state = composite.state_check(snapshot, step, task)
                    except Exception as exc:  # noqa: BLE001 - never skip on a read error
                        pre_goal_state = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                    composite_state.setdefault("preconditions", {})[str(step.get("step_id"))] = pre_goal_state
                    if pre_goal_state.get("ok"):
                        skipped = {
                            "step": step,
                            "ok": True,
                            "skipped": "goal_already_met_before_step",
                            "verification": {
                                "ok": True,
                                "method": "engine_predicate_precondition",
                                "predicate": pre_goal_state.get("predicate"),
                                "note": "the step goal already held before this step issued an action",
                            },
                            "sim_actions": 0,
                            "chunks_used": 0,
                            "native_events": [],
                            "timing": {"total_s": 0.0},
                            "stop_reason": "goal_already_met",
                            "placement": {
                                "schema": "embodied_engine.composite.step.v1",
                                "required": False,
                                "attempted": False,
                                "release_executed": False,
                                "release_verified": None,
                                "placement_verified": bool(pre_goal_state.get("ok")),
                                "retreat_status": "not_applicable",
                                "execution_status": "not_applicable",
                                "evidence_source": "precondition_check",
                                "policy_owned": False,
                                "action_source": "none",
                                "reason_code": None,
                                "note": "goal was already true before the step; no action was issued",
                            },
                        }
                        step_results.append(skipped)
                        with self.lock:
                            self.steps_done = [s for s in step_results]
                        records.append_journal(
                            {
                                "t": time.time(),
                                "stage": "composite_step_skipped",
                                "episode_id": episode_id,
                                "step_id": step.get("step_id"),
                                "step_task_key": step.get("step_task_key"),
                                "reason": "goal_already_met_before_step",
                                "observation_id": snapshot.get("observation_id"),
                                "sim_action_count": snapshot.get("sim_action_count"),
                            }
                        )
                        if index + 1 < len(steps):
                            obs, snapshot, token = self._composite_handoff(
                                env=env,
                                episode_id=episode_id,
                                step=step,
                                next_step=steps[index + 1],
                                snapshot=snapshot,
                                catalog_by_key=catalog_by_key,
                                task=task,
                                home_eef=home_eef,
                                composite_state=composite_state,
                                step_results=step_results,
                                timing=timing,
                                stop_event=stop_event,
                                records=records,
                            )
                            self._set_frames(obs, snapshot)
                        index += 1
                        continue
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
                binding = (
                    f" {step.get('object_n')} -> {step.get('target_n')}"
                    if step.get("object_n") or step.get("target_n")
                    else ""
                )
                label = step.get("step_task_key") or step.get("skill")
                self._set_phase(
                    "running",
                    f"step {index + 1}/{len(steps)} {label}{binding}".strip(),
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
                    # The OpenWAM server keeps the rest of the last generated
                    # chunk. Without this reset a retry would consume actions
                    # planned from the *previous* attempt's observation.
                    token = self.inference.reset(f"retry:{episode_id}:{attempt_no}")
                    records.append_journal(
                        {
                            "t": time.time(),
                            "stage": "policy_buffer_reset",
                            "episode_id": episode_id,
                            "attempt": attempt_no,
                            "reason": "retry must not reuse a stale action chunk",
                        }
                    )
                if step.get("skill") == "whole_task":
                    step_goal = composite.step_goal(step) if composite_state.get("enabled") else None
                    carry_watch, carry_objects = (
                        self._carry_watch(steps=steps, index=index, catalog_by_key=catalog_by_key)
                        if composite_state.get("enabled")
                        else ([], [])
                    )
                    record, obs, snapshot = self._run_whole_task(
                        env=env,
                        obs=obs,
                        snapshot=snapshot,
                        step=runner_step,
                        task=task,
                        token=token,
                        episode_id=episode_id,
                        stop_event=stop_event,
                        goal=step_goal,
                        watch_joints=_watched_joints(steps, index, snapshot, task),
                        max_steps=(
                            composite.step_budget(int(step.get("step_index") or 1), self.cfg)
                            if composite_state.get("enabled")
                            else None
                        ),
                        step_task=(
                            catalog_by_key.get(str(step.get("step_task_key")))
                            if composite_state.get("enabled")
                            else None
                        ),
                        watch_goals=carry_watch,
                        carry_objects=carry_objects,
                    )
                    timing["execution_s"] += time.perf_counter() - skill_started
                    timing["inference_s_sum"] += record["timing"]["inference_s_sum"]
                    timing["env_step_s_sum"] += record["timing"]["env_step_s_sum"]
                    timing["snapshot_s_sum"] += record["timing"]["snapshot_s_sum"]
                    timing["observe_s_sum"] += record["timing"].get("observe_s_sum", 0.0)
                    timing["convert_s_sum"] += record["timing"].get("convert_s_sum", 0.0)
                    timing["frame_s_sum"] += record["timing"].get("frame_s_sum", 0.0)
                    if step_goal is not None:
                        # An engine-predicate step has no held object and no
                        # placement to finalize; forcing the placement finalizer
                        # on it would be a different task. (A predicate step
                        # that *places* an object already ran its own bounded
                        # release finalizer and brought its block with it.)
                        if not record.get("placement"):
                            record["placement"] = {
                                "schema": "embodied_engine.composite.step.v1",
                                "required": False,
                                "attempted": False,
                                "release_executed": False,
                                "release_verified": False,
                                "placement_verified": False,
                                "retreat_status": "handled_by_handoff",
                                "execution_status": "not_applicable",
                                "evidence_source": "engine_predicate",
                                "policy_owned": True,
                                "reason_code": None,
                                "note": "this step's goal is an engine predicate; the placement finalizer is not applied",
                            }
                        placement = record["placement"]
                        if stop_event.is_set() or (placement or {}).get("reason_code") == "user_stop":
                            raise Stopped("stop requested during step placement finalizer")
                    else:
                        placement_started = time.perf_counter()
                        placement, obs, snapshot, token = self._finish_placement(
                            env=env,
                            task=task,
                            record=record,
                            obs=obs,
                            snapshot=snapshot,
                            token=token,
                            episode_id=episode_id,
                            stop_event=stop_event,
                        )
                        timing["release_s"] += time.perf_counter() - placement_started
                        record["placement"] = placement
                        if isinstance(placement, dict):
                            last_placement = placement
                        if stop_event.is_set() or placement.get("reason_code") == "user_stop":
                            # The finalizer swallows Stopped internally and
                            # reports it through the placement block. Without
                            # this re-raise the episode would continue and be
                            # recorded as ``completed`` / ``succeeded`` even
                            # though a stop truncated the release mid-way.
                            raise Stopped("stop requested during placement finalizer")
                    self._set_frames(obs, snapshot)
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
                    if composite_state.get("enabled"):
                        # Diagnostic only: where the placed object sits relative
                        # to the *official* target region, on the snapshot this
                        # step ended with. It never changes a judgement, but an
                        # anchor judged by the loaded scene's native flag has no
                        # other way to say "3 mm outside the region" vs "never
                        # carried it there".
                        record["target_region"] = self._target_region_diagnostic(
                            step=step,
                            catalog_by_key=catalog_by_key,
                            snapshot=snapshot,
                        )
                    step_results.append(record)
                    with self.lock:
                        self.steps_done = [s for s in step_results]
                    native_ever = native_ever or bool(record.get("native_events"))
                    self._set_frames(obs, snapshot)
                    if not record.get("ok"):
                        if composite_state.get("enabled"):
                            # A composite never retries a failed step inside the
                            # episode: the failure is reported at the stage it
                            # happened, with the actions it actually spent.
                            # The ordered plan did not complete on its first
                            # pass, so this is not a first-pass success even
                            # though no repair was attempted.
                            composite_state["first_pass_success"] = False
                            evidence = record.get("verification") or {}
                            # Which earlier steps' goals still hold on the very
                            # sample where this step gave up. Without it a
                            # composite failure cannot be told apart from "step 2
                            # destroyed step 1's state", because step 1 was
                            # verified minutes earlier. Read-only.
                            prior: list[dict[str, Any]] = []
                            for done in step_results:
                                prior_step = done.get("step") or {}
                                if prior_step.get("step_id") == step.get("step_id"):
                                    continue
                                if prior_step.get("skill") != "whole_task" or not prior_step.get("step_id"):
                                    continue
                                try:
                                    goal_step = composite.step_goal(prior_step) if composite_state.get(
                                        "enabled"
                                    ) else prior_step
                                    state = completion.goal_state(snapshot, goal_step, task, baseline)
                                except Exception as exc:  # noqa: BLE001
                                    state = {"ok": None, "error": f"{type(exc).__name__}: {exc}"}
                                prior.append({"step_id": prior_step.get("step_id"), "state": state})
                            raise SkillFailure(
                                f"composite_step_{step.get('step_index')}_failed",
                                (
                                    f"composite step {step.get('step_index')} "
                                    f"({step.get('step_task_key')}) did not reach its goal after "
                                    f"{record.get('sim_actions')} actions"
                                ),
                                {
                                    "step": {
                                        "step_id": step.get("step_id"),
                                        "task_id": step.get("step_task_id"),
                                        "goal_kind": step.get("goal_kind"),
                                        "prompt": step.get("prompt"),
                                    },
                                    "verification": evidence,
                                    "prior_steps_at_failure": prior,
                                    "sim_actions": record.get("sim_actions"),
                                    "budget": (composite_state.get("budgets") or {}).get(
                                        str(step.get("step_index"))
                                    ),
                                    "retries": 0,
                                },
                            )
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
                                "instruction": goal_prompt,
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
                            attempt_digests.append(_step_digest(step_results.pop()))
                            continue
                        raise SkillFailure(
                            failure["reason_code"],
                            failure["message"],
                            {"verification": evidence, "recovery": recovery},
                        )
                    if composite_state.get("enabled") and index + 1 < len(steps):
                        # Transport readiness: the step's own relation is met,
                        # but the object may be perched on the rim. Evaluated on
                        # the step's own post-release samples; may run exactly
                        # one bounded re-seat of the same catalog capability.
                        obs, snapshot, token = self._transport_readiness(
                            env=env,
                            episode_id=episode_id,
                            steps=steps,
                            index=index,
                            step=step,
                            step_task=catalog_by_key.get(str(step.get("step_task_key"))) or {},
                            catalog_by_key=catalog_by_key,
                            task=task,
                            record=record,
                            obs=obs,
                            snapshot=snapshot,
                            token=token,
                            stop_event=stop_event,
                            composite_state=composite_state,
                            timing=timing,
                            records=records,
                        )
                        if record.get("transport_ready", {}).get("reason_code") == "user_stop":
                            raise Stopped("stop requested during transport re-seat")
                        # Step handoff: verify, (release only if the step's own
                        # kind needs it), bounded retreat, fresh observation,
                        # policy buffer reset, then start the next step from
                        # that observation. No env reset, no scene reload.
                        next_step = steps[index + 1]
                        obs, snapshot, token = self._composite_handoff(
                            env=env,
                            episode_id=episode_id,
                            step=step,
                            next_step=next_step,
                            snapshot=snapshot,
                            catalog_by_key=catalog_by_key,
                            task=task,
                            home_eef=home_eef,
                            composite_state=composite_state,
                            step_results=step_results,
                            timing=timing,
                            stop_event=stop_event,
                            records=records,
                        )
                        self._set_frames(obs, snapshot)
                    index += 1
                    retries = 0
                    continue
                watch_joints = _watched_joints(steps, index, snapshot, task)
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
                    watch_joints=watch_joints,
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
                            "instruction": goal_prompt,
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
                        attempt_digests.append(_step_digest(step_results.pop()))
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
                            attempt_digests.append(_step_digest(step_results.pop()))
                            continue
                        replans += 1
                        attempt_digests.append(_step_digest(step_results.pop()))
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
                                "instruction": goal_prompt,
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
                            attempt_digests.append(_step_digest(step_results.pop()))
                            continue
                        if recovery.get("decision") == "replan" and recovery.get("steps"):
                            new_steps = list(recovery["steps"])
                            if not plan_contract.preserves_goal(new_steps, goal_step):
                                records.append_journal(
                                    {"t": time.time(), "stage": "replan_rejected", "episode_id": episode_id,
                                     "reason": "does not preserve the initial goal step", "steps": new_steps}
                                )
                                retries += 1
                                attempt_digests.append(_step_digest(step_results.pop()))
                                continue
                            replans += 1
                            attempt_digests.append(_step_digest(step_results.pop()))
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
            # A verification round may loop to look at a repaired episode or a
            # re-opened drawer. Without a hard cap an ineligible repair used to
            # `continue` forever (repair disabled -> status "not_attempted" ->
            # attempted stays False -> same branch again), which kept stepping
            # the environment: one episode reached 9895 sim actions and then
            # lost the env connection. Bounded here, and the repair branch is
            # entered at most once.
            verify_rounds = 0
            repair_considered = False
            final_steps = []
            verification_attempts = []
            while True:
                verify_rounds += 1
                if verify_rounds > 6:
                    raise SkillFailure(
                        "composite_verification_did_not_converge",
                        "final verification kept failing without a repair that issued actions",
                        {
                            "rounds": verify_rounds - 1,
                            "steps": final_steps,
                            "repair": composite_state.get("repair"),
                        },
                    )
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
                    completion_kind = str((step.get("completion") or {}).get("kind") or "")
                    if step.get("skill") == "whole_task" and completion_kind not in {"", "native_goal"}:
                        # An early composite step is judged by its engine
                        # predicate on the same fresh samples the final step
                        # is judged on, not by the loaded scene's native flag.
                        states = [composite.state_check(s, step, task) for s in samples]
                        predicate_ok = all(bool(state.get("ok")) for state in states)
                        first_state = states[0] if states else {}
                        check = {
                            "ok": predicate_ok,
                            "method": f"engine_predicate:{first_state.get('predicate')}",
                            "predicate": first_state.get("predicate"),
                            "samples": [bool(state.get("ok")) for state in states],
                            "qpos": [state.get("qpos") for state in states],
                            "threshold": first_state.get("threshold"),
                            "relation": first_state.get("relation"),
                            "height": first_state.get("height"),
                            "object_pose": first_state.get("object_pose"),
                            "relation_samples": [
                                {
                                    "ok": bool(state.get("ok")),
                                    "predicate": (state.get("relation") or {}).get("predicate"),
                                    "xy_m": (state.get("relation") or {}).get("xy_m"),
                                    "object_bottom_z": (state.get("relation") or {}).get("object_bottom_z"),
                                    "target_pose": (state.get("relation") or {}).get("target_pose"),
                                }
                                for state in states
                            ],
                            "native_goal_samples": [bool(s.get("native_goal")) for s in samples],
                            "sim_action_counts": [s.get("sim_action_count") for s in samples],
                            "observation_ids": [s.get("observation_id") for s in samples],
                            "reason_code": None if predicate_ok else "step_goal_regressed",
                            "note": (
                                "engine predicate on fresh post-episode samples; the loaded scene's native "
                                "flag is not this step's goal"
                            ),
                        }
                    elif step.get("skill") == "whole_task":
                        # Whole-task policies own the whole trajectory: the
                        # official LIBERO goal is the relation check, but a
                        # pick/place is only *complete* once a release was
                        # measured and the relation survived fresh samples.
                        flags = [bool(s.get("native_goal")) for s in samples]
                        native_ok = all(flags)
                        ids = [s.get("observation_id") for s in samples]
                        counts = [s.get("sim_action_count") for s in samples]
                        fresh = bool(
                            None not in ids
                            and len(set(ids)) == len(ids)
                            and all(isinstance(c, int) for c in counts)
                            and all(counts[i] < counts[i + 1] for i in range(len(counts) - 1))
                        )
                        placement = dict(result.get("placement") or {})
                        required = bool(placement.get("required"))
                        release = placement.get("release") or {}
                        release_count = release.get("sim_action_count")
                        if isinstance(release_count, list):
                            release_count = release_count[-1] if release_count else None
                        after_release = bool(
                            isinstance(release_count, int)
                            and all(isinstance(c, int) and c > release_count for c in counts)
                        )
                        if placement.get("evidence_source") == "execution_finalizer":
                            base_ok = bool(placement.get("placement_verified"))
                        else:
                            base_ok = bool(placement.get("release_verified"))
                        placement_ok = bool(native_ok and fresh and after_release and base_ok)
                        if not required:
                            placement["placement_verified"] = bool(native_ok)
                            placement["execution_status"] = "not_applicable"
                        else:
                            placement["placement_verified"] = placement_ok
                            if placement_ok:
                                placement["execution_status"] = "completed"
                                placement["reason_code"] = None
                            elif not native_ok:
                                placement["execution_status"] = "failed"
                                placement["reason_code"] = "native_goal_not_reached"
                            else:
                                placement["execution_status"] = "unknown"
                                placement["reason_code"] = (
                                    placement.get("reason_code") or "release_not_verified"
                                )
                        placement["post_release_samples"] = {
                            "observation_ids": ids,
                            "sim_action_counts": counts,
                            "fresh": fresh,
                            "after_release": after_release,
                            "release_sim_action_count": release_count,
                        }
                        result["placement"] = placement
                        check = {
                            "ok": native_ok,
                            "method": "native_libero_goal",
                            "samples": flags,
                            "reason_code": None if native_ok else "native_goal_not_reached",
                            "gripper_opening_m": [
                                round(float(v), 4)
                                for v in ((s.get("sim") or {}).get("gripper_opening_m") for s in samples)
                                if isinstance(v, (int, float))
                            ],
                            "placement_verified": bool(placement.get("placement_verified")),
                            "release_verified": bool(placement.get("release_verified")),
                            "release_executed": bool(placement.get("release_executed")),
                            "retreat_status": placement.get("retreat_status"),
                            "evidence_source": placement.get("evidence_source"),
                            "policy_owned": placement.get("policy_owned"),
                            "placement_status": placement.get("execution_status"),
                            "placement_reason": placement.get("reason_code"),
                            "note": (
                                "official LIBERO goal atoms on fresh samples; the placement verdict "
                                "is the measured release + post-release relation, not the goal alone"
                            ),
                        }
                        if required and not placement_ok:
                            check["reason_code"] = placement.get("reason_code") or "release_not_verified"
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
                # Every verification round is kept, not just the decisive one.
                # A first round that failed on 3 fresh samples and a later round
                # that held are different outcomes; losing the failed triple
                # would leave "first_pass_success=false" without evidence.
                verification_attempts.append(
                    {
                        "round": verify_rounds,
                        "ok": all_ok,
                        "observation_ids": [s.get("observation_id") for s in samples],
                        "sim_action_counts": [s.get("sim_action_count") for s in samples],
                        "steps": [
                            {
                                "step_id": (row.get("step") or {}).get("step_id"),
                                "task_key": (row.get("step") or {}).get("step_task_key"),
                                "ok": bool(row.get("ok")),
                                "reason_code": (row.get("verification") or {}).get("reason_code"),
                                "native_samples": (row.get("verification") or {}).get("samples"),
                                "relation_samples": [
                                    {
                                        "ok": item.get("ok"),
                                        "predicate": item.get("predicate"),
                                        "xy_m": item.get("xy_m"),
                                        "object_bottom_z": item.get("object_bottom_z"),
                                    }
                                    for item in ((row.get("verification") or {}).get("relation_samples") or [])
                                ],
                            }
                            for row in final_steps
                        ],
                    }
                )
                if all_ok:
                    if composite_state.get("repair"):
                        composite_state["repair"]["final_all_goals_met"] = True
                    break
                if composite_state.get("enabled") and not composite_state.get("repair_attempted") and not repair_considered:
                    # One bounded repair, only when the anchor step succeeded
                    # and an *earlier* goal was lost. The eligibility rules are
                    # checked inside the engine; the model only picks the task.
                    repair_considered = True
                    composite_state["first_pass_success"] = False
                    try:
                        repair_raw = self._composite_repair(
                            env=env,
                            task=task,
                            catalog_by_key=catalog_by_key,
                            step_results=step_results,
                            final_steps=final_steps,
                            snapshot=snapshot,
                            baseline=baseline,
                            composite_state=composite_state,
                            episode_id=episode_id,
                            stop_event=stop_event,
                            timing=timing,
                            token=token,
                        )
                    except Stopped:
                        raise
                    except Exception as exc:  # noqa: BLE001 - a repair bug is a failed repair
                        repair_raw = {
                            "status": "error",
                            "attempted": False,
                            "reason": f"{type(exc).__name__}: {exc}",
                            "repair_succeeded": False,
                        }
                    composite_state["repair"] = {
                        key: value for key, value in repair_raw.items() if not key.startswith("_")
                    }
                    repair_block = composite_state["repair"]
                    # Only a repair that actually issued actions counts as an
                    # attempt: a disabled/ineligible repair must not turn the
                    # no-repair baseline into "repair attempted".
                    composite_state["repair_attempted"] = bool(repair_block.get("attempted"))
                    # The repair record is evidence whether it succeeded or not:
                    # its action trace is what makes "the repair policy failed to
                    # reach the handle" distinguishable from "the repair was
                    # never really attempted". Archived before any failure raise.
                    repair_record = repair_block.get("episode_record")
                    if repair_record:
                        composite_state["repair"]["record_summary"] = {
                            "sim_actions": repair_record.get("sim_actions"),
                            "chunks_used": repair_record.get("chunks_used"),
                            "stop_reason": repair_record.get("stop_reason"),
                            "ok": repair_record.get("ok"),
                            "grasp": repair_record.get("grasp"),
                            "watch_events": repair_record.get("watch_events"),
                            "final_state": repair_record.get("final_state"),
                        }
                        composite_state["repair"]["episode"] = repair_record
                        composite_state["repair"].pop("episode_record", None)
                    if repair_block.get("status") != "executed":
                        composite_state["repair"]["final_all_goals_met"] = False
                        # The repair issued real actions even when it did not
                        # reach its goal. Take its last observation, then go
                        # round the verification loop once more: the archived
                        # record must carry the post-repair end state and the
                        # true total action count, not the pre-repair snapshot.
                        snapshot = repair_raw.get("_snapshot") or snapshot
                        obs = repair_raw.get("_obs") or obs
                        token = repair_raw.get("_token") or token
                        self._set_frames(obs, snapshot)
                        continue
                    obs = repair_raw.get("_obs")
                    snapshot = repair_raw.get("_snapshot")
                    token = repair_raw.get("_token") or token
                    # The repair is not another plan step: the ordered key list
                    # is unchanged and the repair row stays a separate event.
                    self._set_frames(obs, snapshot)
                    continue
                repairable = [
                    (position, step)
                    for position, step in failed
                    if step.get("skill") == "open_drawer" and not composite_state.get("enabled")
                ]
                if verify_repairs >= 1 or not repairable or len(repairable) != len(failed):
                    raise SkillFailure(
                        (
                            "composite_final_verification_failed"
                            if composite_state.get("enabled")
                            else "final_verification_failed"
                        ),
                        (
                            "the composite's end state does not satisfy every step goal on fresh samples"
                            if composite_state.get("enabled")
                            else "final cross-skill verification failed"
                        ),
                        {
                            "steps": final_steps,
                            "snapshot": snapshot,
                            "repair": composite_state.get("repair"),
                            "first_pass_success": composite_state.get("first_pass_success", True),
                        },
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
            # A stopped episode must still report what the simulator actually
            # did before the stop; the loop's ``snapshot`` is the one taken at
            # prepare time because the interrupted step never returned. Reading
            # it here is safe (the env server is still alive and the scene has
            # not changed), and without it the record said "0 actions".
            try:
                client = self.env.client
                if client is not None:
                    snapshot = client.state_snapshot()
                    self.sim_action_count = int(
                        snapshot.get("sim_action_count") or self.sim_action_count
                    )
            except Exception as exc:  # noqa: BLE001 - reporting must not mask the stop
                records.append_journal(
                    {"t": time.time(), "stage": "stop_snapshot_failed", "episode_id": episode_id,
                     "detail": f"{type(exc).__name__}: {exc}"}
                )
        except SkillFailure as exc:
            execution_status = "failed"
            failure = {
                **(failure or {}),
                "reason_code": exc.reason_code,
                "message": exc.message,
                "evidence": exc.evidence,
            }
            # A failed *final* verification still measured every step: keep
            # those samples so the record shows which goal regressed instead of
            # reporting "never checked".
            failed_steps = (exc.evidence or {}).get("steps")
            final_verification = (
                {"ok": False, "samples": len(failed_steps), "steps": failed_steps}
                if isinstance(failed_steps, list) and failed_steps
                else {}
            )
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
        placement: dict[str, Any] = {}
        for row in reversed(step_results):
            if isinstance(row, dict) and isinstance(row.get("placement"), dict):
                placement = row["placement"]
                break
        if not placement and last_placement:
            placement = last_placement
        release_required = str(task.get("kind") or "") in {"pick_place", "composite"}
        composite_record: dict[str, Any] | None = None
        if composite_state.get("enabled"):
            # Whether the arm is still holding anything it placed. Native goals
            # say where an object is, not that it was let go; a closed gripper
            # on the last object would satisfy some goals and still be a
            # different outcome. Measured on the final fresh sample.
            hold_rows: list[dict[str, Any]] = []
            opening_final = grasp_geometry.opening_m(final_snapshot)
            seen_objects: list[str] = []
            for result in step_results:
                step_row = result.get("step") or {}
                step_task_row = catalog_by_key.get(str(step_row.get("step_task_key"))) or {}
                if not composite.release_required(step_task_row):
                    continue
                for name in list(step_task_row.get("objects") or [])[:1]:
                    if not isinstance(name, str) or name in seen_objects:
                        continue
                    seen_objects.append(name)
                    frame = grasp_geometry.finger_frame(final_snapshot, name)
                    gap = frame.get("object_gap_m")
                    separation = frame.get("separation_m")
                    outside = bool(
                        isinstance(gap, (int, float))
                        and isinstance(separation, (int, float))
                        and gap > max(float(separation) / 2.0, 0.02)
                    )
                    open_wide = isinstance(opening_final, (int, float)) and float(opening_final) > 0.05
                    hold_rows.append(
                        {
                            "step_id": step_row.get("step_id"),
                            "step_task_key": step_row.get("step_task_key"),
                            "object_n": name,
                            "eef_object_m": grasp_geometry.eef_object_m(final_snapshot, name),
                            "object_gap_m": gap,
                            "separation_m": separation,
                            "outside_finger_span": outside,
                            "gripper_opening_m": opening_final,
                            "not_held": bool(outside or open_wide),
                        }
                    )
            hold_check = {
                "ok": (all(row["not_held"] for row in hold_rows) if hold_rows else None),
                "objects": hold_rows,
                "observation_id": final_snapshot.get("observation_id"),
                "note": (
                    "measured on the final fresh sample: finger opening plus each placed object's distance "
                    "from the pad line; a native goal alone does not prove the object was let go"
                ),
            }
            # Every step of the composite is reported separately: its own goal,
            # its own native observations, its own action count. The composite
            # verdict is the conjunction, never the last step alone.
            final_by_step = {
                int((row.get("step") or {}).get("step_index") or 0): row
                for row in (final_verification.get("steps") or [])
                if isinstance(row, dict)
            }
            step_rows = []
            for position, result in enumerate(step_results):
                step = result.get("step") or {}
                index = int(step.get("step_index") or position + 1)
                final_row = final_by_step.get(index) or {}
                verification = (final_row.get("verification") or {}) if final_row else {}
                step_task = catalog_by_key.get(str(step.get("step_task_key"))) or {}
                needs_release = composite.release_required(step_task)
                step_placement = result.get("placement") or {}
                if needs_release:
                    release_state = "verified" if step_placement.get("placement_verified") else "unknown"
                    release_source = step_placement.get("evidence_source") or "none"
                else:
                    release_state = "not_applicable"
                    release_source = "not_applicable"
                step_rows.append(
                    composite.step_summary(
                        step=step,
                        record=result,
                        snapshot=final_snapshot,
                        verification=(result.get("verification") or {}),
                        release_state=release_state,
                        release_source=release_source,
                        final_verification=verification,
                    )
                )
            handoff = composite_state.get("handoff") or {}
            scene_epochs = [composite_state.get("scene_epoch_start"), final_snapshot.get("scene_epoch")]
            step_count = len(composite_state.get("keys") or [])
            step_rows_ok = [
                {
                    "step_id": row.get("step_id"),
                    "key": row.get("task_key"),
                    "goal_kind": row.get("goal_kind"),
                    "ok": bool(row.get("ok")),
                    "ok_final": row.get("ok_final"),
                    "release_state": row.get("release_state"),
                    "sim_actions": row.get("sim_actions"),
                }
                for row in step_rows
            ]
            step_1_ok = bool(step_rows[0]["ok"]) if len(step_rows) > 0 else False
            step_1_final_ok = step_rows[0]["ok_final"] if len(step_rows) > 0 else None
            # The top-level readiness fields describe the step that was
            # actually judged. With three steps the last step has no later
            # carrier, so reading ``composite_state`` (which holds the *last*
            # step's block) reported None on every 3-step round.
            readiness_row = next(
                (
                    row
                    for row in step_rows
                    if isinstance(row.get("transport_ready"), dict)
                    and row["transport_ready"].get("checked")
                ),
                {},
            )
            readiness_block = readiness_row.get("transport_ready") or {}
            readiness_reseat = readiness_row.get("reseat") or {}
            anchor_row = step_rows[-1] if step_rows else {}
            anchor_native = bool((anchor_row.get("evidence") or {}).get("ok")) and len(step_rows) > 1
            anchor_native_final = anchor_row.get("ok_final") if step_rows else None
            # A handoff happens after step n (0-based) and before step n+1, so
            # its labels are step_ids[n] -> step_ids[n+1]. Deriving them from
            # the enumerate index labelled the first handoff "c0 -> c1", which
            # reads as "before step 1" and is wrong for 2- and 3-step plans.
            handoff_step_ids = list(composite_state.get("step_ids") or [])
            handoff_rows = [
                {
                    "index": n + 1,
                    "from_step": (
                        handoff_step_ids[n] if n < len(handoff_step_ids) else f"c{n + 1}"
                    ),
                    "to_step": (
                        handoff_step_ids[n + 1]
                        if n + 1 < len(handoff_step_ids)
                        else f"c{n + 2}"
                    ),
                    "mode": row.get("mode"),
                    "status": row.get("status"),
                    "sim_actions": row.get("sim_actions"),
                    "physical_continuity": bool(row.get("physical_continuity", False)),
                    "robot_teleport": bool(row.get("robot_teleport")),
                    "scene_epoch": row.get("scene_epoch"),
                    "object_moved_m": row.get("object_moved_m"),
                    "reason": row.get("reason"),
                    "counts": row.get("counts"),
                }
                for n, row in enumerate(composite_state.get("handoff_records") or [])
            ]
            composite_record = {
                "schema": composite.SCHEMA,
                "keys": composite_state.get("keys"),
                "flow": composite_state.get("flow"),
                "flow_expansion": composite_state.get("flow_expansion"),
                "flow_guard": composite_state.get("flow_guard"),
                "step_ids": composite_state.get("step_ids"),
                "anchor": composite_state.get("anchor"),
                "instruction": instruction,
                "bddl_mutation": False,
                "env_reset_between_steps": 0,
                "env_reload_between_steps": 0,
                "object_teleport": False,
                "scene_epoch_start": composite_state.get("scene_epoch_start"),
                "scene_epoch_end": final_snapshot.get("scene_epoch"),
                "scene_epoch_continuous": bool(
                    len(set(scene_epochs)) == 1 and scene_epochs[0] is not None
                ),
                "sim_action_count_start": composite_state.get("sim_action_count_start"),
                "sim_action_count_end": final_snapshot.get("sim_action_count"),
                "sim_actions_total": (
                    int(final_snapshot.get("sim_action_count") or 0)
                    - int(composite_state.get("sim_action_count_start") or 0)
                ),
                "budgets": composite_state.get("budgets"),
                "handoff_mode": composite_state.get("handoff_mode"),
                "handoff_options": composite_state.get("handoff_options"),
                "handoff_mode_default": composite.DEFAULT_HANDOFF["mode"],
                "robot_state_restore": bool(handoff.get("robot_state_restore")),
                "robot_teleport": bool(handoff.get("robot_teleport")),
                "physical_continuity": bool(handoff.get("physical_continuity", False)),
                "robot_state_capture": (
                    {
                        key: (composite_state.get("robot_state_capture") or {}).get(key)
                        for key in (
                            "ok",
                            "reason_code",
                            "reason",
                            "capture_id",
                            "observation_id",
                            "sim_action_count",
                            "robot",
                            "controller",
                            "nonrobot_digest",
                        )
                    }
                    if composite_state.get("robot_state_capture")
                    else None
                ),
                "robot_state_capture_error": composite_state.get("robot_state_capture_error") or None,
                "action_accounting": {
                    "policy_actions": sum(int(row.get("sim_actions") or 0) for row in step_results),
                    "handoff_retreat_actions": sum(
                        int(((row.get("counts") or {}).get("retreat_actions")) or 0)
                        for row in (composite_state.get("handoff_records") or [])
                    ),
                    "handoff_detach_actions": sum(
                        int(((row.get("counts") or {}).get("detach_actions")) or 0)
                        for row in (composite_state.get("handoff_records") or [])
                    ),
                    "handoff_settle_actions": sum(
                        int(((row.get("counts") or {}).get("settle_actions")) or 0)
                        for row in (composite_state.get("handoff_records") or [])
                    ),
                    "handoff_actions": sum(
                        int(row.get("sim_actions") or 0)
                        for row in (composite_state.get("handoff_records") or [])
                    ),
                    "finalizer_actions": sum(
                        int(((row.get("placement") or {}).get("extra_actions")) or 0)
                        for row in step_results
                    ),
                    "repair_policy_actions": int(
                        ((composite_state.get("repair") or {}).get("policy_actions")) or 0
                    ),
                    "repair_finalizer_actions": int(
                        ((composite_state.get("repair") or {}).get("finalizer_actions")) or 0
                    ),
                    "reseat_policy_actions": int(
                        (((composite_state.get("transport_ready") or {}).get("reseat") or {}).get(
                            "policy_actions"
                        ))
                        or 0
                    ),
                    "reseat_finalizer_actions": int(
                        (((composite_state.get("transport_ready") or {}).get("reseat") or {}).get(
                            "finalizer_actions"
                        ))
                        or 0
                    ),
                    "sim_actions_total": (
                        int(final_snapshot.get("sim_action_count") or 0)
                        - int(composite_state.get("sim_action_count_start") or 0)
                    ),
                },
                "handoff": handoff,
                "handoffs": handoff_rows,
                "handoff_count": len(handoff_rows),
                "steps": step_rows,
                "step_count": step_count,
                "steps_ok": step_rows_ok,
                "anchor_step_key": (composite_state.get("keys") or [None])[-1],
                "anchor_native": bool(anchor_native),
                "anchor_native_samples": (anchor_row.get("evidence") or {}).get("samples"),
                "anchor_release_verified": bool(
                    anchor_native and anchor_row.get("release_state") == "verified"
                ),
                "hold_check": hold_check,
                "final_verification_attempts": verification_attempts,
                # Legacy names kept for the existing manifest/report readers:
                # step_1_* is the first step, step_2_* is the anchor (last) step.
                "step_1_ok": step_1_ok,
                # Transport readiness (independent of every step goal).
                "step_1_met": step_1_ok,
                "ready_to_transport": readiness_block.get("ok") if readiness_row else None,
                "ready_to_transport_step": readiness_row.get("step_id") if readiness_row else None,
                "transport_ready": readiness_block or composite_state.get("transport_ready"),
                "reseat_attempted": bool(readiness_reseat.get("attempted")),
                "reseat_succeeded": bool(readiness_reseat.get("reseat_succeeded")),
                "reseat_actions": int(readiness_reseat.get("policy_actions") or 0)
                + int(readiness_reseat.get("finalizer_actions") or 0),
                "reseat_state_changed": bool(
                    (readiness_reseat.get("state_change") or {}).get("changed")
                ),
                "reseat": readiness_reseat or None,
                "repair_actions": int(
                    ((composite_state.get("repair") or {}).get("policy_actions")) or 0
                )
                + int(((composite_state.get("repair") or {}).get("finalizer_actions")) or 0),
                "step_1_final_ok": step_1_final_ok,
                "step_2_native": bool(anchor_native),
                "step_2_final_native": anchor_native_final,
                "final_verification_read": bool(final_verification),
                "first_pass_success": bool(composite_state.get("first_pass_success", True)),
                "repair_attempted": bool(composite_state.get("repair_attempted")),
                "repair_task_key": ((composite_state.get("repair") or {}).get("task_key")),
                "repair_succeeded": bool(((composite_state.get("repair") or {}).get("repair_succeeded"))),
                "final_all_goals_met": (
                    (composite_state.get("repair") or {}).get("final_all_goals_met")
                    if composite_state.get("repair")
                    # Strict: a step that was never verified (ok_final None,
                    # e.g. the episode failed before the final check) is not
                    # a met goal.
                    else bool(step_rows) and all(row.get("ok_final") is True for row in step_rows)
                ),
                "repair": composite_state.get("repair"),
                "step_2_native_samples": (anchor_row.get("evidence") or {}).get("samples"),
                "step_2_release_verified": bool((anchor_native and anchor_row.get("release_state") == "verified")),
                "combo_verified": bool(
                    execution_status == "completed"
                    and len(step_rows) == step_count
                    and all(row.get("ok") for row in step_rows)
                    and all(row.get("ok_final") is not False for row in step_rows)
                    and hold_check.get("ok") is not False
                ),
                "native_goal_current": native_final,
                "native_goal_ever": bool(native_ever),
                "step_1_goal_ever": any(
                    (row.get("verification") or {}).get("ok") for row in step_results[:1]
                ),
                "note": (
                    f"{step_count} ordered steps in one episode. Every step before the last is judged by its "
                    "engine predicate on live/fresh samples (a carried object's goal is re-checked against its "
                    "current pose), the last by the loaded scene's native goal; each placing step also needs a "
                    "measured release, and the final sample must show nothing still gripped"
                ),
            }
        if composite_record is not None:
            user_goal = bool(composite_record["combo_verified"])
            user_status = "met" if user_goal else "not_met"
            release_required = bool(composite_record.get("step_2_release_verified") or user_goal)
        elif release_required and placement:
            placement_ok = bool(placement.get("placement_verified"))
            user_goal = bool(execution_status == "completed" and placement_ok)
            if user_goal:
                user_status = "met"
            elif placement.get("execution_status") == "failed" and not native_final:
                user_status = "not_met"
            else:
                user_status = "unknown"
        else:
            user_goal = execution_status == "completed"
            user_status = "met" if user_goal else "not_met"
        if not placement:
            placement = {
                "required": release_required,
                "attempted": False,
                "release_executed": False,
                "release_verified": False,
                "placement_verified": False,
                "retreat_status": "skipped",
                "execution_status": "not_attempted",
                "evidence_source": "none",
                "reason_code": "no_whole_task_step" if release_required else None,
            }
        release_evidence = _release_evidence(task, step_results, final_snapshot, placement)
        record = {
            "schema": "embodied_engine.run.v1",
            "task_id": task["task_id"],
            "task_bddl": task["bddl"],
            "instruction": instruction,
            "semantic": (self.plan or {}).get("understanding") or {},
            "episode_id": episode_id,
            "run_id": run_id,
            "scene_epoch": final_snapshot.get("scene_epoch"),
            "initial_scene_fingerprint": initial_scene,
            "initial_state_fingerprint": initial_state,
            # Static props are placed by LIBERO's unseeded sampler on every
            # reset, so they are NOT covered by the state fingerprint. Kept
            # separate so old records stay comparable.
            "initial_fixture_fingerprint": initial_fixtures,
            # Sampling protocol of *this* episode. Old records (before this
            # field existed) are the legacy protocol by definition and must not
            # be re-read as if they had been seeded.
            "sampling": {
                "policy_seed": self.active_policy_seed(),
                "policy_seed_source": self.policy_seed_source,
                "scene_seed": int(self.seed),
                "protocol": "policy_seed_v1"
                if self.active_policy_seed() is not None
                else "legacy_prompt_index",
                "accelerators_changed": False,
            },
            "started_at": started_iso,
            "ended_at": _now_iso(),
            "duration_s": round(ended - started, 3),
            "phase": execution_status,
            "execution_status": execution_status,
            "native_task_success": bool(native_ever or native_final),
            "native_goal_final": native_final,
            "user_goal_success": bool(user_goal),
            "user_goal_status": user_status,
            "task_scope": "combo" if composite_record is not None else "single",
            "placement": placement,
            "composite": composite_record,
            "failure_code": (failure or {}).get("reason_code"),
            "failure_reason": (failure or {}).get("message"),
            "failure_evidence": (failure or {}).get("evidence"),
            "plan": {
                "interpretation": (plan or {}).get("interpretation"),
                "success_condition": (plan or {}).get("success_condition"),
                "task_scope": (plan or {}).get("task_scope") or "single",
                "composite": (plan or {}).get("composite"),
                "steps": (plan or {}).get("steps"),
                "source": (self.plan or {}).get("source") or "qwen3.5-9b-semantic-router",
                "agent_mode": (self.plan or {}).get("agent_mode") or "qwen",
                "official_instruction": ((self.plan or {}).get("understanding") or {}).get("official_instruction"),
                "planner_vision": False,
                "latency_s": round(timing["planner_s"], 3),
                "agent_response": self.plan,
            },
            "steps": step_results,
            "openwam": openwam_stats,
            "final_verification": final_verification,
            "release_evidence": release_evidence,
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
            "attempt_digests": attempt_digests,
            "human_interventions": 0,
            "inference_mode": (self._health().get("inference") or {}).get("mode") or self.cfg["inference"]["mode"],
            "action_backend": self._action_backend(),
            "model": self._health().get("action_model"),
            "agent_model": self._health().get("agent", {}).get("model"),
            "env": self.env.view(),
            "page": {"ui": "page_ui (restored old Page, backend bridge)", "human_confirmation": False},
        }
        # Record assembly (including the cached health/identity read) is part of
        # the episode wall clock; it is measured here so the decomposition can
        # name it instead of calling it "unexplained". The JSON write itself is
        # reported in the journal row below.
        record_timing = record.get("timing")
        if isinstance(record_timing, dict):
            record_timing["record_assemble_s"] = round(time.time() - ended, 3)
        write_started = time.perf_counter()
        path = records.write_run(record)
        write_s = round(time.perf_counter() - write_started, 4)
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
                "record_write_s": write_s,
                "record": str(path),
            }
        )
        with self.lock:
            self.metrics["episodes_succeeded" if execution_status == "completed" else "episodes_stopped" if execution_status == "stopped" else "episodes_failed"] += 1
            self.metrics["last_episode_s"] = record["duration_s"]
            self.last_result = {
                "episode_id": episode_id,
                "task_id": task["task_id"],
                "success": bool(user_goal),
                "execution_status": execution_status,
                "native_task_success": record["native_task_success"],
                "user_goal_status": user_status,
                "placement": {
                    key: placement.get(key)
                    for key in (
                        "required",
                        "release_executed",
                        "release_verified",
                        "placement_verified",
                        "retreat_status",
                        "execution_status",
                        "evidence_source",
                        "policy_owned",
                        "extra_actions",
                        "extra_s",
                    )
                },
                "reason": (failure or {}).get("message")
                or (
                    "all skills and final verification passed"
                    if user_goal
                    else (
                        "native goal reached; placement evidence insufficient"
                        if native_final
                        else "all skills and final verification passed"
                    )
                ),
                "failure_code": (failure or {}).get("reason_code"),
                "duration_s": record["duration_s"],
                "sim_action_count": record["sim_action_count"],
                "initial_scene_fingerprint": record.get("initial_scene_fingerprint"),
                "initial_state_fingerprint": record.get("initial_state_fingerprint"),
                "timing": record["timing"],
                # Generation statistics of *this* episode (client latency and
                # server latency per chunk) so the page can show real numbers
                # instead of a remembered average.
                "action_backend": record.get("action_backend"),
                "generation": (
                    (((record.get("steps") or [{}])[0].get("chunk_stats") or {}).get("generation")) or None
                ),
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
                "key": row["key"],
                "label": f"{row['task_id']}: {row['instruction']}",
                "instruction": row["instruction"],
                "official_instruction": row.get("official_instruction"),
                "kind": row["kind"],
                "bddl": row["bddl"],
            }
        )
    return web.json_response(
        {
            "status": "ok",
            "catalog": task_catalog.catalog_evidence(),
            "scenes": rows,
        }
    )


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
        plan_keys = body.get("plan_keys")
        if plan_keys is not None and not isinstance(plan_keys, list):
            return web.json_response(
                {"status": "bad_request", "error": "plan_keys must be a list of catalog keys"},
                status=400,
            )
        out = _engine().start_run(
            int(body.get("task_id")),
            body.get("instruction"),
            None if seed is None else int(seed),
            [str(key) for key in plan_keys] if plan_keys else None,
            None if body.get("policy_seed") is None else int(body["policy_seed"]),
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


async def handoff(request: web.Request) -> web.Response:
    body = await _json_body(request)
    try:
        out = await asyncio.to_thread(
            _engine().set_handoff_mode,
            str(body.get("mode") or ""),
            body.get("options") or {},
        )
        return web.json_response(out)
    except ValueError as exc:
        return web.json_response({"status": "bad_request", "error": str(exc)}, status=400)
    except RuntimeError as exc:
        return web.json_response({"status": "conflict", "error": str(exc)}, status=409)


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
    app.router.add_post("/api/load", load)
    app.router.add_post("/api/run", run)
    app.router.add_post("/api/stop", stop)
    app.router.add_post("/api/reset", reset)
    app.router.add_post("/api/handoff", handoff)
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
