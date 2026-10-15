"""Thin WebSocket adapter. It speaks the existing Page UI frame protocol.

HARNESS_LIVE=1 uses the LIBERO env. HARNESS_LIVE=0 stays on the dry path and
refuses to pretend that placeholder frames are a real episode.
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import threading
import time
import urllib.request
from pathlib import Path
from queue import Empty, Queue

from aiohttp import web, WSMsgType

from executor import Session, ToolOutcome, run_plan
from planner import SHORT_MEMORY, load_human_plan, make_openai_client, request_plan

ROOT = Path(__file__).resolve().parent
PLAN_PATH = ROOT / "plans" / "two_object.json"
HOST = os.environ.get("SEQUENTIAL_HOST", "127.0.0.1")
PORT = int(os.environ.get("SEQUENTIAL_PORT", "8766"))
LIVE = os.environ.get("HARNESS_LIVE", "0") == "1"
DIAGNOSTIC_S = float(os.environ.get("SEQUENTIAL_DIAGNOSTIC_S", "0"))
LOAD_TIMEOUT_S = float(os.environ.get("SEQUENTIAL_LOAD_TIMEOUT_S", "300"))
PLANNER_URL = os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")
PLANNER_MODEL = os.environ.get("PLANNER_MODEL", "Qwen3.5-9B")

# Kept only so a unit test can recognize the old placeholder. Live mode never sends it.
TINY_JPEG = bytes.fromhex(
    "ffd8ffe000104a46494600010100000100010000ffdb004300080606070605080707070909080a0c140d0c0b0b0c1912130f141d1a1f1e1d1a1c1c20242e2720222c231c1c2837292c30313434341f27393d38323c2e333432ffc0000b080001000101011100ffc4001f0000010501010101010100000000000000000102030405060708090a0bffc400b5100002010303020403050504040000017d01020300041105122131410613516107227114328191a1082342b1c11552d1f02433627282090a161718191a25262728292a3435363738393a434445464748494a535455565758595a636465666768696a737475767778797a838485868788898a92939495969798999aa2a3a4a5a6a7a8a9aab2b3b4b5b6b7b8b9bac2c3c4c5c6c7c8c9cad2d3d4d5d6d7d8d9dae1e2e3e4e5e6e7e8e9eaf1f2f3f4f5f6f7f8f9faffda00080001000100003f00fbffd9"
)


def frame_bytes(header: dict, agent: bytes, wrist: bytes) -> bytes:
    header = dict(header)
    header["a_len"] = len(agent)
    header["w_len"] = len(wrist)
    payload = json.dumps(header, ensure_ascii=False).encode("utf-8")
    return len(payload).to_bytes(4, "big") + payload + agent + wrist


def _image_size(raw: bytes) -> tuple[int, int]:
    from PIL import Image

    image = Image.open(io.BytesIO(raw))
    return int(image.size[0]), int(image.size[1])


UI_MAX_EDGE = 512
UI_JPEG_QUALITY = 70


def _ui_jpeg(raw: bytes) -> tuple[bytes, tuple[int, int], tuple[int, int]]:
    """Downscale a grounding PNG for the page. Pi0.5 still reads the sim observation."""
    from PIL import Image

    image = Image.open(io.BytesIO(raw)).convert("RGB")
    source = (int(image.size[0]), int(image.size[1]))
    image.thumbnail((UI_MAX_EDGE, UI_MAX_EDGE), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=UI_JPEG_QUALITY)
    return buf.getvalue(), source, (int(image.size[0]), int(image.size[1]))


class DryBackend:
    """CPU stand-in used by tests. The page path does not treat it as a real episode."""

    def __init__(self) -> None:
        self.calls = []
        self._native = False
        self.fail_ids = set()

    def execute(self, action: str, args: dict):
        self.calls.append((action, args))
        if action in self.fail_ids:
            return ToolOutcome(executed=False, error=f"{action} failed", real=False, error_class="physical")
        xyz = [0.05, 0.22, 0.48] if action == "segment" else None
        grasp = True if action == "pi0_pick" else None
        return ToolOutcome(
            executed=True,
            grasp_signal=grasp,
            native_terminated=self._native,
            xyz=xyz,
            eef_pos=[0.0, 0.0, 0.62],
            real=False,
        )

    def observe(self) -> dict:
        return {"fresh": True, "llm": False, "observation_id": "dry", "cameras": {}}

    def native_success(self) -> bool:
        return self._native

    def frames(self):
        return None, None

    def close(self) -> None:
        return None


def _component_row(components: list, *names: str):
    wanted = set(names)
    for item in components or []:
        if item.get("name") in wanted:
            return item
    return None


class App:
    def __init__(self) -> None:
        self.human_plans = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
        self.catalog = self.human_plans
        self.suites = None
        self.suite_error = ""
        self.scene_spec = None
        self.loading = False
        self.session = Session("none")
        self.backend = None
        self.confirm_q: Queue = Queue()
        self.clients = set()
        self.loop = None
        self.lock = threading.Lock()
        self.running = False
        self.run_id = 0
        self.awaiting_confirm = False
        self.logs: list[str] = []
        self.mode = "live" if LIVE else "dry"
        self.load_token = 0
        self.recovery_attempts = 0
        self.binding = None
        self.display_seq = 0
        self.client_frames = None
        self.planner = make_openai_client(PLANNER_URL, PLANNER_MODEL)

    def scenes(self) -> list[dict]:
        return [
            {"id": key, "label": value["goal"], "instruction": value["goal"]}
            for key, value in self.catalog.items()
        ]

    def health(self) -> dict:
        """Adapter status plus a cached, time-bounded probe. Does not start models or reset."""
        checkpoints = {"pi05": False, "sam3": False, "env": False}
        error = ""
        try:
            from live import require_resources

            paths = require_resources()
            checkpoints = {"pi05": True, "sam3": True, "env": True, "paths": paths}
        except Exception as exc:
            error = str(exc)
        components = []
        if self.backend is not None and hasattr(self.backend, "component_health"):
            components = self.backend.component_health()
        env_row = _component_row(components, "env", "env_server")
        vla_row = _component_row(components, "vla", "vla_server")
        if self.loading:
            env_state = "loading"
        elif env_row is None:
            env_state = "not_loaded"
        elif env_row["alive"]:
            env_state = "ready"
        else:
            env_state = "down"
        if vla_row is None:
            vla_state = "not_loaded"
        elif vla_row["alive"]:
            vla_state = "ready"
        else:
            vla_state = "down"
        planner = PROBE["planner"]
        return {
            "type": "health",
            "adapter": "online",
            "mode": self.mode,
            "harness_live": LIVE,
            "port": PORT,
            "pid": os.getpid(),
            "planner": planner,
            "planner_ok": planner == "ok",
            "planner_detail": PROBE["detail"],
            "planner_url": PLANNER_URL,
            "checkpoints": checkpoints,
            "components": components,
            "env": env_state,
            "vla": vla_state,
            "suites": "ready" if self.suites else ("failed" if self.suite_error else "loading"),
            "suite_error": self.suite_error,
            "can_select_scene": True,
            "can_manual_plan": True,
            "can_model_plan": planner == "ok",
            "silent_dry_fallback": False,
            "error": error,
        }

    def broadcast(self, message: dict) -> None:
        if self.loop is None:
            return
        data = json.dumps(message, ensure_ascii=False)
        for ws in list(self.clients):
            asyncio.run_coroutine_threadsafe(ws.send_str(data), self.loop)

    def log(self, text: str) -> None:
        self.logs.append(text)
        self.logs = self.logs[-200:]
        self.broadcast({"type": "log", "text": text})

    def broadcast_frame(self, header: dict) -> None:
        header = dict(header)
        header["mode"] = self.mode
        header["plan_source"] = header.get("plan_source") or ""
        header["recovery_attempts"] = self.recovery_attempts
        if not LIVE or self.backend is None or not hasattr(self.backend, "capture_display"):
            self.broadcast(
                {
                    "type": "state",
                    "phase": "failed",
                    "detail": "没有真实相机帧",
                    "reason": header.get("reason") or "no real frame",
                }
            )
            return
        try:
            snap = self.backend.capture_display()
        except Exception:
            snap = None
        if not snap:
            self.broadcast(
                {
                    "type": "state",
                    "phase": "failed",
                    "detail": "动作后没有新的相机帧",
                    "reason": "missing camera frame",
                }
            )
            return
        agent = snap["agent"]
        wrist = snap["wrist"]
        agent_meta = snap["agent_meta"]
        wrist_meta = snap["wrist_meta"]
        self.last_encode_s = round((snap.get("encode_ms") or 0) / 1000, 3)
        header.update({
            "aw": agent_meta["out_w"],
            "ah": agent_meta["out_h"],
            "ww": wrist_meta["out_w"],
            "wh": wrist_meta["out_h"],
            "src_aw": agent_meta["src_w"],
            "src_ah": agent_meta["src_h"],
            "src_ww": wrist_meta["src_w"],
            "src_wh": wrist_meta["src_h"],
            "source": agent_meta["source"],
            "flip": agent_meta["flip"],
            "jpeg_quality": agent_meta["jpeg_quality"],
            "capture_ms": snap.get("capture_ms"),
            "encode_ms": snap.get("encode_ms"),
            "stage": "tool_end",
            "placeholder": False,
        })
        payload = frame_bytes(header, agent, wrist)
        if self.loop is None:
            return
        for ws in list(self.clients):
            asyncio.run_coroutine_threadsafe(ws.send_bytes(payload), self.loop)

    def note_display(self, header: dict, agent: bytes, wrist: bytes) -> None:
        """One 512 frame pushed by the env owner. Encoding already happened off that step."""
        meta = {
            "primitive": header.get("primitive") or "env_step",
            "observation_id": f"display-{header.get('frame_id')}",
            "timestamp": header.get("timestamp"),
            "stage": header.get("stage") or "motion",
            "capture_ms": header.get("render_ms"),
            "render_ms": header.get("render_ms"),
            "action_ms": header.get("action_ms"),
            "encode_ms": header.get("encode_ms"),
            "env_action_count": header.get("env_action_count"),
            "agent": {
                "source": header.get("source"),
                "flip": header.get("flip"),
                "jpeg_quality": header.get("jpeg_quality"),
                "out_w": header.get("aw"),
                "out_h": header.get("ah"),
                "src_w": header.get("src_aw"),
                "src_h": header.get("src_ah"),
            },
            "wrist": {
                "source": header.get("source"),
                "flip": header.get("flip"),
                "jpeg_quality": header.get("jpeg_quality"),
                "out_w": header.get("ww"),
                "out_h": header.get("wh"),
                "src_w": header.get("src_ww"),
                "src_h": header.get("src_wh"),
            },
        }
        self.send_display(agent, wrist, meta)
        self._save_display_jpeg(header, agent)

    def _save_display_jpeg(self, header: dict, agent: bytes) -> None:
        out = getattr(self.backend, "output_dir", None)
        if out is None or not agent:
            return
        folder = Path(out) / "display"
        folder.mkdir(parents=True, exist_ok=True)
        fid = int(header.get("frame_id") or 0)
        (folder / f"{fid:04d}.jpg").write_bytes(agent)

    def send_display(self, agent: bytes, wrist: bytes, meta: dict) -> None:
        """Send one already-encoded snapshot. Called off the environment thread."""
        self.display_seq += 1
        stamp = meta.get("timestamp") or time.time()
        header = {
            "type": "frame",
            "eid": self.session.episode_id,
            "fid": self.display_seq,
            "frame_id": self.display_seq,
            "primitive": meta.get("primitive") or "",
            "observation_id": meta.get("observation_id"),
            "timestamp": stamp,
            "frame_time": time.strftime("%H:%M:%S", time.localtime(stamp)),
            "env_action_count": meta.get("env_action_count"),
            "stage": meta.get("stage"),
            "capture_ms": meta.get("capture_ms"),
            "render_ms": meta.get("render_ms"),
            "action_ms": meta.get("action_ms"),
            "encode_ms": meta.get("encode_ms"),
            "source": (meta.get("agent") or {}).get("source"),
            "flip": (meta.get("agent") or {}).get("flip"),
            "jpeg_quality": (meta.get("agent") or {}).get("jpeg_quality"),
            "aw": (meta.get("agent") or {}).get("out_w"),
            "ah": (meta.get("agent") or {}).get("out_h"),
            "ww": (meta.get("wrist") or {}).get("out_w"),
            "wh": (meta.get("wrist") or {}).get("out_h"),
            "src_aw": (meta.get("agent") or {}).get("src_w"),
            "src_ah": (meta.get("agent") or {}).get("src_h"),
            "src_ww": (meta.get("wrist") or {}).get("src_w"),
            "src_wh": (meta.get("wrist") or {}).get("src_h"),
            "phase": "executing",
            "detail": meta.get("primitive") or "",
            "term": False,
            "success": False,
            "placeholder": False,
            "live_chunk": True,
            "mode": self.mode,
        }
        payload = frame_bytes(header, agent, wrist)
        if self.loop is None:
            return
        for ws in list(self.clients):
            asyncio.run_coroutine_threadsafe(ws.send_bytes(payload), self.loop)

    def load(self, message) -> None:
        with self.lock:
            if self.loading:
                self.broadcast({
                    "type": "load_failed",
                    "detail": "已有加载在进行，不会再创建环境。",
                })
                return
            self.loading = True
            self.load_token += 1
            token = self.load_token
            self.session.cancel()
            self._wake()
            self.running = False
            self.awaiting_confirm = False
        try:
            self._load_locked(message, token)
        finally:
            with self.lock:
                self.loading = False

    def _load_locked(self, message, token: int) -> None:
        try:
            spec = self._scene_from_message(message)
        except Exception as exc:
            self.broadcast({"type": "load_failed", "detail": str(exc)})
            self.broadcast({"type": "state", "phase": "failed", "detail": str(exc)})
            return
        scene_id = spec["scene_id"]
        self.broadcast({"type": "state", "phase": "loading", "detail": scene_id, "mode": self.mode})
        self.log(f"load {scene_id} mode={self.mode}")
        if not LIVE:
            episode = self._begin_session(scene_id)
            self.scene_spec = spec
            self.broadcast(
                {
                    "type": "loaded",
                    "instruction": spec["goal"],
                    "env_id": scene_id,
                    "episode_id": episode,
                    "scene_epoch": self.session.scene_epoch,
                    "mode": "dry",
                    "placeholder": True,
                    "scene": {
                        "task_language": spec["goal"],
                        "object_names": [],
                        "summary_zh": spec.get("summary_zh"),
                        "placeholder": True,
                        "note": "干跑未创建 LIBERO，没有新图像",
                    },
                }
            )
            self.broadcast(
                {
                    "type": "state",
                    "phase": "failed",
                    "detail": "干跑未创建环境，没有新图像。真实模式请用 HARNESS_LIVE=1 重启本进程。",
                    "mode": "dry",
                }
            )
            return

        with self.lock:
            episode = self._begin_session(scene_id)
            self.scene_spec = spec
            self.binding = None
        out = ROOT.parents[0] / "runs" / "sequential" / episode
        box: dict = {}
        abandoned = threading.Event()
        same = (
            self.backend is not None
            and hasattr(self.backend, "scene_key")
            and self.backend.scene_key() == (spec["suite"], int(spec["task_id"]), int(spec["seed"]))
        )

        def work() -> None:
            backend = self.backend
            try:
                from live import LiveBackend

                def on_phase(name: str) -> None:
                    self.broadcast({"type": "state", "phase": "loading", "detail": name, "mode": "live"})

                if same:
                    on_phase("reset")
                    backend.warm_reset()
                elif backend is not None:
                    on_phase("env_create")
                    backend.retarget(spec, out)
                else:
                    backend = LiveBackend(spec, out, on_phase=on_phase)
                if abandoned.is_set() or token != self.load_token:
                    if not same:
                        backend.close()
                    return
                box["backend"] = backend
                box["info"] = backend.scene_info()
                box["timings"] = dict(backend.timings)
            except Exception as exc:
                if backend is not None and not same:
                    backend.close()
                box["error"] = exc

        thread = threading.Thread(target=work, daemon=True)
        thread.start()
        thread.join(LOAD_TIMEOUT_S)
        if thread.is_alive():
            abandoned.set()
            self.broadcast(
                {
                    "type": "load_failed",
                    "detail": f"加载超时（{int(LOAD_TIMEOUT_S)}s），已停止等待。不会继续显示加载中。",
                }
            )
            self.broadcast({"type": "state", "phase": "failed", "detail": "load_timeout"})
            return
        if token != self.load_token:
            if box.get("backend") is not None:
                box["backend"].close()
            return
        if box.get("error") is not None:
            err = box["error"]
            detail = str(err)
            log = getattr(err, "log", "")
            self.log(detail)
            if log:
                self.log(log[-1000:])
            self.broadcast({"type": "load_failed", "detail": detail})
            self.broadcast({"type": "state", "phase": "failed", "detail": detail})
            return
        with self.lock:
            self.backend = box["backend"]
            self.backend.scene_epoch = self.session.scene_epoch
            info = box["info"]
        bus = getattr(self.backend, "frame_bus", None)
        if bus is not None:
            bus.set_sender(self.send_display)
        self.broadcast(
            {
                "type": "loaded",
                "instruction": info.get("task_language") or spec["goal"],
                "env_id": scene_id,
                "episode_id": episode,
                "scene_epoch": self.session.scene_epoch,
                "plan_version": self.session.plan_version,
                "mode": "live",
                "placeholder": False,
                "summary_zh": spec.get("summary_zh"),
                "scene": info,
                "timings": box.get("timings") or {},
            }
        )
        self.broadcast_frame(
            {
                "eid": episode,
                "fid": int(info.get("env_step") or 0),
                "phase": "loaded",
                "detail": scene_id,
                "term": False,
                "success": False,
                "reason": "",
            }
        )
        timings = dict(box.get("timings") or {})
        timings["encode_transfer_s"] = getattr(self, "last_encode_s", None)
        self.broadcast({"type": "state", "phase": "loaded", "detail": scene_id, "timings": timings, "mode": "live"})
        self.log("load timings " + json.dumps(timings, ensure_ascii=False))

    def _begin_session(self, scene_id: str) -> str:
        previous = 0 if self.session.scene_id == "none" else self.session.scene_epoch
        self.session = Session(scene_id)
        self.session.scene_epoch = previous
        return self.session.begin_episode()

    def _scene_from_message(self, message) -> dict:
        import re

        from suites import goal_relation, load_suites, task_record

        if self.suites is None:
            self.suites = load_suites()
        if isinstance(message, str) or (isinstance(message, dict) and message.get("scene") and not message.get("suite")):
            scene_id = message if isinstance(message, str) else str(message.get("scene"))
            matched = re.fullmatch(r"(libero_spatial|libero_object|libero_goal|libero_10)_t(\d+)(?:_s(\d+))?", scene_id)
            if not matched:
                raise ValueError(f"unknown scene {scene_id}")
            suite, task_id, seed = matched.group(1), int(matched.group(2)), int(matched.group(3) or 0)
        else:
            suite = str(message.get("suite") or "")
            task_id = int(message.get("task_id"))
            seed = int(message.get("seed") or 0)
        task = task_record(self.suites, suite, task_id)
        if seed < 0 or seed >= int(task["init_states"]):
            raise ValueError(f"seed {seed} 超出该任务初始状态数量 {task['init_states']}")
        return {
            "scene_id": f"{suite}_t{task_id}_s{seed}",
            "suite": suite,
            "task_id": task_id,
            "seed": seed,
            "goal": task["language"],
            "task": task["language"],
            "summary_zh": task["summary_zh"],
            "task_key": task.get("task_key"),
            "instances": task.get("instances") or [],
            "fixtures": task.get("fixtures") or [],
            "goal_relation": goal_relation(task["language"]),
            "long_candidate": task["long_candidate"],
            "init_states": task["init_states"],
        }

    def start(self, kind: str, message: dict, execute: bool = True) -> None:
        if self.session.scene_id == "none" or (LIVE and self.backend is None):
            self.broadcast({"type": "load_failed", "detail": "load a scene first"})
            return
        if self.running and kind != "restart":
            self.broadcast({"type": "state", "phase": "failed", "detail": "already running"})
            return
        plan_source = message.get("plan_source") or "model"
        if plan_source not in {"model", "manual", "replay"}:
            self.broadcast({"type": "state", "phase": "failed", "detail": f"unknown plan_source {plan_source}"})
            return
        self.run_id += 1
        run_id = self.run_id
        self.running = True
        if kind == "restart":
            self.session.cancel()
            self.awaiting_confirm = False
            self._wake()
        threading.Thread(
            target=self._run,
            args=(
                kind,
                plan_source,
                run_id,
                message.get("manual_scope") or "single",
                execute,
                message.get("stop_after"),
                message.get("deadline_s"),
            ),
            daemon=True,
        ).start()

    def stop(self) -> None:
        cancel = getattr(self.planner, "cancel", None)
        if cancel is not None:
            cancel()
        self.session.cancel()
        self.awaiting_confirm = False
        self._wake()
        self.running = False
        self.broadcast(
            {
                "type": "ended",
                "episode_id": self.session.episode_id,
                "success": False,
                "reason": "stopped",
            }
        )

    def confirm(self, message: dict) -> None:
        if not self.awaiting_confirm:
            self.broadcast({"type": "state", "phase": "failed", "detail": "当前没有等待确认"})
            return
        epoch = message.get("scene_epoch")
        version = message.get("plan_version")
        episode = message.get("episode_id")
        if episode and episode != self.session.episode_id:
            self.log("discard confirm: episode mismatch")
            return
        if epoch is not None and int(epoch) != self.session.scene_epoch:
            self.log("discard confirm: scene_epoch mismatch")
            return
        if version is not None and int(version) != self.session.plan_version:
            self.log("discard confirm: plan_version mismatch")
            return
        self.confirm_q.put(message.get("verdict"))

    def choose_binding(self, message: dict) -> None:
        current = self.binding or {}
        ref = message.get("object_ref")
        if ref not in (current.get("candidates") or []):
            self.broadcast({"type": "state", "phase": "failed", "detail": "所选对象不在待确认列表中，没有开始动作"})
            return
        chosen = dict(current)
        chosen["status"] = "resolved"
        chosen["object_ref"] = ref
        chosen["evidence_source"] = "user_selected"
        chosen["bound_to_object_ref"] = True
        chosen["region_seen"] = False
        chosen["note"] = "用户显式选择，不是视觉判断，也不会自动开始动作"
        self.binding = chosen
        self.broadcast({"type": "binding", "binding": _public_binding(chosen)})

    def _run(self, kind: str, plan_source: str, run_id: int, manual_scope: str = "single", execute: bool = True, stop_after: str | None = None, deadline_s: float | None = None) -> None:
        try:
            self._run_inner(kind, plan_source, run_id, manual_scope, execute, stop_after, deadline_s)
        except Exception as exc:
            self.log(f"run failed: {exc}")
            self.broadcast({"type": "state", "phase": "failed", "detail": str(exc)})
            self.broadcast(
                {
                    "type": "ended",
                    "episode_id": self.session.episode_id,
                    "success": False,
                    "reason": str(exc),
                }
            )
        finally:
            if self.run_id == run_id:
                self.running = False
                self.awaiting_confirm = False

    def _run_inner(self, kind: str, plan_source: str, run_id: int, manual_scope: str = "single", execute: bool = True, stop_after: str | None = None, deadline_s: float | None = None) -> None:
        if kind == "restart":
            self.session.cancel()
            self._wake()
            if not LIVE:
                self._refuse_dry()
                return
            self.binding = None
            self.session.begin_episode()
            if self.backend is None:
                self._fail_run(self.session.episode_id, "load a scene first")
                return
            self.broadcast({"type": "state", "phase": "loading", "detail": "reset", "mode": "live"})
            self.backend.warm_reset()
            self.backend.scene_epoch = self.session.scene_epoch
        if not LIVE:
            self._refuse_dry()
            return
        episode = self.session.episode_id
        epoch = self.session.scene_epoch
        spec = self.scene_spec or {}
        self.broadcast({"type": "started", "episode_id": episode, "mode": self.mode, "plan_source": plan_source})
        scene_objects, binding, info = self._scene_binding(spec)
        self.broadcast({"type": "binding", "binding": _public_binding(binding)})
        if binding.get("status") != "resolved":
            self._emit_planner(
                plan_source=None,
                stage="binding_unresolved",
                summary=spec.get("goal") or "",
                output="",
                errors=[{
                    "path": "binding.object_ref",
                    "code": "binding_unresolved",
                    "received": binding.get("candidates"),
                    "expected": "one explicit object_ref",
                }],
                repairs=0,
                attempts=[],
                binding=binding,
                vision=False,
                planning_input="scene_metadata",
                elapsed_s=0,
            )
            self._fail_run(episode, "BINDING_UNRESOLVED")
            return
        if plan_source == "manual":
            try:
                plan = self._manual_plan(spec, manual_scope, binding)
            except Exception as exc:
                self._emit_planner(
                    plan_source=None,
                    stage="validation_failed",
                    summary=str(exc),
                    output="",
                    errors=[{"path": "plan", "code": "manual_rejected", "received": str(exc), "expected": "one bound transfer"}],
                    repairs=0,
                    attempts=[],
                    binding=binding,
                    vision=False,
                    planning_input="scene_metadata",
                    elapsed_s=0,
                )
                self._fail_run(episode, f"PLANNING_FAILED {exc}")
                return
            self._emit_planner(
                plan_source="manual",
                stage="ok",
                summary=f"用户选择人工计划 scope={manual_scope}",
                output=json.dumps({"plan_id": plan.get("plan_id"), "semantic_step_ids": plan.get("semantic_step_ids"), "steps": [step["step_id"] for step in plan["steps"]]}, ensure_ascii=False),
                errors=[],
                repairs=0,
                attempts=[],
                binding=binding,
                vision=False,
                planning_input="scene_metadata",
                elapsed_s=0,
            )
        elif plan_source == "replay":
            try:
                plan = self._saved_plan(spec, binding, scene_objects)
            except Exception as exc:
                self._emit_planner(
                    plan_source="replay",
                    stage="validation_failed",
                    summary=str(exc),
                    output="",
                    errors=[{"path": "plan", "code": "replay_rejected", "received": str(exc), "expected": "the saved object_ref still bound"}],
                    repairs=0,
                    attempts=[],
                    binding=binding,
                    vision=False,
                    planning_input="scene_metadata",
                    elapsed_s=0,
                )
                self._fail_run(episode, str(exc))
                return
            self._emit_planner(
                plan_source="replay",
                stage="ok",
                summary=plan.get("replay_note") or "",
                output=json.dumps({"request_id": plan.get("request_id"), "steps": [step["step_id"] for step in plan["steps"]]}, ensure_ascii=False),
                errors=[],
                repairs=0,
                attempts=[{"role": "saved", "request_id": plan.get("request_id"), "finish_reason": "reused", "usage": {}}],
                binding=binding,
                vision=False,
                planning_input="scene_metadata",
                elapsed_s=0,
                plan_from="saved",
            )
        else:
            if PROBE["planner"] != "ok":
                self._fail_run(episode, f"planner {PROBE['planner']}: {PROBE['detail'] or 'unavailable'}")
                return
            self.broadcast({"type": "state", "phase": "planning", "detail": PLANNER_MODEL, "mode": "live", "plan_source": "model"})
            image_ids = [f"agentview:{info.get('env_step')}", f"wrist:{info.get('env_step')}"]
            from schema import transfer_slots

            slots = transfer_slots(spec.get("goal") or "") or []
            required_ref = binding.get("object_ref") if len(slots) == 1 else None
            scene_text = json.dumps(
                {
                    "planning_input": "scene_metadata",
                    "vision": False,
                    "goal_relation": spec.get("goal_relation"),
                    "allowed_refs": sorted(scene_objects),
                    "required_object_ref": required_ref,
                    "binding": _public_binding(binding),
                    "capabilities": {
                        ref: {
                            "semantic_type": item.get("semantic_type"),
                            "movable": item.get("movable"),
                            "can_be_target": item.get("can_be_target"),
                            "target_relations": item.get("target_relations"),
                            "capability_source": item.get("capability_source"),
                        }
                        for ref, item in scene_objects.items()
                    },
                    "record_index": info.get("env_step"),
                    "image_ids": image_ids,
                    "note": "image bytes are not in this request; ids only. sim_state binding is not vision.",
                },
                ensure_ascii=False,
            )
            started = time.monotonic()

            def on_planner(phase: str, detail: str) -> None:
                self.broadcast({
                    "type": "state",
                    "phase": phase,
                    "detail": detail,
                    "mode": "live",
                    "plan_source": "model",
                })

            try:
                result = request_plan(
                    task=spec["goal"],
                    scene_observation=scene_text,
                    client=self.planner,
                    short_memory=SHORT_MEMORY,
                    image_ids=image_ids,
                    scene_objects=scene_objects,
                    required_object_ref=required_ref,
                    on_event=on_planner,
                )
            except Exception as exc:
                self._emit_planner(
                    plan_source=None,
                    stage="request_failed",
                    summary=scene_text[:500],
                    output="",
                    errors=[{"path": "planner.request", "code": "planner_request_failed", "received": str(exc), "expected": "one plan response"}],
                    repairs=0,
                    attempts=[],
                    binding=binding,
                    vision=False,
                    planning_input="scene_metadata",
                    elapsed_s=time.monotonic() - started,
                )
                self._fail_run(episode, f"PLANNING_FAILED {exc}")
                return
            elapsed = time.monotonic() - started
            attempts = _public_attempts(result.get("attempts") or [])
            self._store_attempts(attempts, result)
            self._emit_planner(
                plan_source=result.get("plan_source"),
                stage=result.get("stage") or "validation_failed",
                summary=scene_text[:500],
                output="",
                errors=result.get("errors") or [],
                repairs=result.get("repairs") or 0,
                attempts=attempts,
                binding=binding,
                vision=False,
                planning_input="scene_metadata",
                elapsed_s=elapsed,
                response_kind=(attempts[-1].get("response_kind") if attempts else None),
                image_ids=image_ids,
                plan_from=result.get("plan_from"),
            )
            if result["status"] != "ok" or result["plan"] is None:
                self._fail_run(episode, "PLANNING_FAILED")
                return
            plan = result["plan"]
            plan["planner_stage"] = "ok"
            plan["request_id"] = attempts[-1].get("request_id") if attempts else None
        plan["episode_id"] = episode
        if plan_source in {"model", "replay"}:
            plan["planner_stage"] = "ok"
        self.log(f"plan_source={plan.get('plan_source')} steps={len(plan['steps'])}")
        if not execute:
            self.broadcast({"type": "state", "phase": "plan_ready", "detail": "只规划，不执行", "mode": "live", "plan_source": plan.get("plan_source")})
            self.broadcast({"type": "ended", "episode_id": episode, "success": False, "reason": "plan_only_not_executed", "native_success": False})
            return

        def confirm(subtask: str):
            self.awaiting_confirm = True
            self.broadcast_frame(
                {
                    "eid": episode,
                    "fid": self._fid(),
                    "phase": "waiting_confirmation",
                    "detail": subtask,
                    "term": False,
                    "success": False,
                    "reason": "subtask unknown until confirmed",
                    "plan_source": plan.get("plan_source"),
                }
            )
            with self.lock:
                queued = self.confirm_q
            verdict = queued.get()
            self.awaiting_confirm = False
            if self.session.scene_epoch != epoch or self.session.cancelled:
                return None
            return verdict

        def on_event(event: dict) -> None:
            if event.get("skill") in {"pi0_pick", "move_to", "set_gripper"} or event.get("detail") == "local replan":
                self.binding = None
            self.broadcast({"type": "state", **event, "mode": "live"})
            self.log(json.dumps(event, ensure_ascii=False)[:500])
            if event.get("phase") in {"executing", "failed"}:
                self.broadcast_frame(
                    {
                        "eid": episode,
                        "fid": self._fid(),
                        "phase": event["phase"],
                        "detail": event.get("detail") or event.get("step_id") or "",
                        "term": event.get("phase") == "failed",
                        "success": False,
                        "reason": event.get("reason") or "",
                        "plan_source": plan.get("plan_source"),
                    }
                )

        deadline = None
        if isinstance(deadline_s, (int, float)) and not isinstance(deadline_s, bool) and float(deadline_s) > 0:
            deadline = time.monotonic() + min(float(deadline_s), 240.0)
        elif DIAGNOSTIC_S > 0:
            deadline = time.monotonic() + DIAGNOSTIC_S
        result = run_plan(
            plan,
            self.backend,
            self.session,
            confirm,
            epoch=epoch,
            deadline_s=deadline,
            on_event=on_event,
            task=spec["goal"],
            stop_after=stop_after if plan_source == "replay" else None,
        )
        if result["episode_id"] != episode or result["scene_epoch"] != epoch:
            return
        if result["status"] == "discarded":
            return
        native = bool(result["native_success"])
        reason = result["reason"] or result["status"]
        status = result["status"]
        code = (result.get("execution") or {}).get("reason_code")
        if status == "stopped_at_check":
            phase = "stopped"
            reason = "诊断停止 · " + (result["reason"] or "")
        elif status == "diagnostic_saved":
            phase = "stopped"
            reason = "预算到时 · " + (result["reason"] or "")
        elif native or (status == "completed" and result.get("reason") == "NATIVE_SUCCESS"):
            phase = "completed"
            status = "completed"
            reason = "任务完成"
        elif status == "completed":
            phase = "incomplete"
            reason = "计划已执行，环境未判定成功"
        elif code == "manual_unknown" or str(result.get("reason") or "").startswith("manual_unknown"):
            phase = "waiting_confirmation"
            status = "insufficient_evidence"
            reason = "证据不足 · " + (result["reason"] or "")
        else:
            phase = "failed"
        self.broadcast_frame(
            {
                "eid": episode,
                "fid": self._fid(),
                "phase": phase,
                "status": status,
                "detail": result["status"],
                "term": True,
                "success": native,
                "reason": reason,
                "plan_source": plan.get("plan_source"),
            }
        )
        self.broadcast(
            {
                "type": "ended",
                "episode_id": episode,
                "success": native,
                "reason": reason,
                "native_success": native,
                "plan_source": plan.get("plan_source"),
                "status": result["status"],
                "planner_stage": plan.get("planner_stage"),
                "execution": result.get("execution"),
                "frames": getattr(getattr(self.backend, "frame_bus", None), "stats", lambda: {})(),
                "client_frames": self.client_frames,
            }
        )

    def _saved_plan(self, spec: dict, binding: dict, scene_objects: dict) -> dict:
        path = ROOT / "records" / "planner_last.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("stage") != "ok":
            raise RuntimeError("已保存的计划没有通过校验，不会执行")
        attempt = (payload.get("attempts") or [None])[0] or {}
        calls = attempt.get("tool_calls") or []
        if not calls:
            raise RuntimeError("已保存的计划没有 tool call，不会执行")
        request_id = attempt.get("request_id")
        arguments = json.loads(calls[0]["function"]["arguments"])
        saved_ref = ((arguments.get("steps") or [{}])[0]).get("object_ref")
        bound = binding.get("object_ref")
        if binding.get("status") != "resolved" or bound != saved_ref:
            raise RuntimeError(
                f"当前绑定是 {bound}（{binding.get('evidence_source')}），已校验计划的 object_ref 是 {saved_ref}。不更换对象，也不再次请求 Planner。"
            )
        from schema import prepare_plan

        plan = prepare_plan(
            arguments,
            task=spec.get("goal") or arguments.get("goal"),
            scene_objects=scene_objects,
            required_object_ref=bound,
        )
        plan["plan_source"] = "replay"
        plan["request_id"] = request_id
        plan["planner_stage"] = "ok"
        plan["replay_note"] = (
            f"复用 {request_id}。该请求没有保存 episode_id、scene_epoch、plan_version，"
            "不能当成当前会话。已按当前场景重新绑定，object_ref 仍一致，所以没有再次请求 Planner。"
        )
        return plan

    def _scene_binding(self, spec: dict) -> tuple[dict, dict, dict]:
        from binding import bind_task, positions_moved
        from suites import objects_for_names

        info = self.backend.scene_info()
        sim = self.backend.sim_state()
        poses = {}
        for name, item in (sim.get("objects") or {}).items():
            if isinstance(item, dict) and isinstance(item.get("pos"), list):
                poses[name] = item["pos"]
        names = list(info.get("object_names") or [])
        for name in poses:
            if name not in names:
                names.append(name)
        objects = objects_for_names(names, spec)
        observation_id = f"{spec.get('scene_id')}:epoch{self.session.scene_epoch}:step{info.get('env_step')}"
        cached = self.binding
        if (
            cached
            and cached.get("scene_epoch") == self.session.scene_epoch
            and cached.get("status") == "resolved"
            and not positions_moved(cached.get("positions"), poses)
        ):
            return objects, cached, info
        binding = bind_task(
            spec.get("goal") or "",
            objects,
            poses,
            observation_id=observation_id,
            scene_epoch=self.session.scene_epoch,
        )
        binding["positions"] = poses
        self.binding = binding
        return objects, binding, info

    def _store_attempts(self, attempts: list[dict], result: dict) -> None:
        path = ROOT / "records" / "planner_last.json"
        payload = {
            "stage": result.get("stage"),
            "repairs": result.get("repairs"),
            "plan_from": result.get("plan_from"),
            "executed": False,
            "attempts": attempts,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _manual_plan(self, spec: dict, scope: str, binding: dict | None = None) -> dict:
        if scope == "long":
            key = f"{spec.get('suite')}_t{spec.get('task_id')}"
            if key not in self.human_plans or int(spec.get("seed") or 0) != 0:
                raise RuntimeError(f"没有 {key} seed 0 的长任务人工计划，不会改用模型计划。")
            plan = load_human_plan(PLAN_PATH, key)
            plan["plan_source"] = "manual"
            return plan
        from suites import single_manual_plan

        info = self.backend.scene_info()
        poses = (binding or {}).get("positions") or {}
        return single_manual_plan(
            spec["goal"],
            info.get("object_names") or [],
            suite=spec["suite"],
            task_id=int(spec["task_id"]),
            seed=int(spec["seed"]),
            task_record=spec,
            poses=poses,
            observation_id=(binding or {}).get("observation_id"),
            scene_epoch=self.session.scene_epoch,
            binding=binding,
        )

    def _emit_planner(self, **payload) -> None:
        if isinstance(payload.get("binding"), dict):
            payload["binding"] = _public_binding(payload["binding"])
        text = json.dumps(payload, ensure_ascii=False, default=str)
        text = text.replace("Bearer EMPTY", "Bearer [redacted]")
        self.broadcast({"type": "planner_io", **json.loads(text), "model": PLANNER_MODEL})

    def _refuse_dry(self) -> None:
        self.broadcast({"type": "started", "episode_id": self.session.episode_id, "mode": "dry"})
        self.broadcast(
            {
                "type": "state",
                "phase": "failed",
                "detail": "HARNESS_LIVE=0，干跑不会执行动作，也不会用占位帧进入确认",
                "mode": "dry",
                "plan_source": None,
            }
        )
        self.broadcast(
            {
                "type": "ended",
                "episode_id": self.session.episode_id,
                "success": False,
                "reason": "dry_run_refused",
            }
        )

    def _fail_run(self, episode: str, reason: str) -> None:
        self.broadcast({"type": "state", "phase": "failed", "detail": reason, "mode": "live"})
        self.broadcast(
            {
                "type": "ended",
                "episode_id": episode,
                "success": False,
                "reason": reason,
                "native_success": False,
            }
        )

    def _fid(self) -> int:
        if self.backend is None or not hasattr(self.backend, "env_step"):
            return 0
        return int(self.backend.env_step() or 0)

    def _wake(self) -> None:
        queued = self.confirm_q
        self.confirm_q = Queue()
        queued.put(None)

    def _close_backend(self) -> None:
        if self.backend is not None:
            self.backend.close()
            self.backend = None


def _public_binding(binding: dict | None) -> dict:
    if not binding:
        return {}
    return {key: value for key, value in binding.items() if key != "positions"}


def _public_attempts(attempts: list[dict]) -> list[dict]:
    rows = []
    for item in attempts:
        rows.append(
            {
                "index": item.get("index"),
                "role": item.get("role"),
                "request_id": item.get("request_id"),
                "usage": item.get("usage") or {},
                "response_kind": item.get("response_kind"),
                "finish_reason": item.get("finish_reason"),
                "errors": item.get("errors") or [],
                "tool_calls": item.get("tool_calls"),
                "content": None if item.get("tool_calls") else item.get("content"),
            }
        )
    return rows


APP = App()
PROBE = {"planner": "unknown", "detail": "", "checked_at": 0.0}
PROBE_TIMEOUT_S = 1.5


def _probe_planner() -> None:
    try:
        with urllib.request.urlopen(PLANNER_URL.rstrip("/") + "/models", timeout=PROBE_TIMEOUT_S) as response:
            PROBE["planner"] = "ok" if response.status == 200 else "down"
            PROBE["detail"] = f"HTTP {response.status}"
    except TimeoutError:
        PROBE["planner"] = "timeout"
        PROBE["detail"] = f"planner probe exceeded {PROBE_TIMEOUT_S}s"
    except Exception as exc:
        PROBE["planner"] = "down" if not isinstance(exc, TimeoutError) else "timeout"
        PROBE["detail"] = str(exc)
        if "timed out" in str(exc).lower():
            PROBE["planner"] = "timeout"
    PROBE["checked_at"] = time.time()


def _planner_probe_loop() -> None:
    while True:
        _probe_planner()
        time.sleep(5)


def _suite_loader() -> None:
    try:
        from suites import load_suites

        APP.suites = load_suites()
        APP.suite_error = ""
    except Exception as exc:
        APP.suite_error = str(exc)


def _display_inlet() -> None:
    """Accept 512 display frames pushed by the env process. One connection, latest only."""
    import socket

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    os.environ["SEQUENTIAL_DISPLAY_PORT"] = str(server.getsockname()[1])
    while True:
        conn, _addr = server.accept()
        conn.settimeout(2.0)
        try:
            while True:
                payload = _read_exact(conn, 4)
                if not payload:
                    break
                hlen = int.from_bytes(payload, "big")
                blob = _read_exact(conn, hlen)
                if blob is None:
                    break
                header = json.loads(blob)
                agent = _read_exact(conn, int(header.get("a_len") or 0))
                wrist = _read_exact(conn, int(header.get("w_len") or 0))
                if agent is None or wrist is None:
                    break
                APP.note_display(header, agent, wrist)
        except Exception:
            pass
        finally:
            conn.close()


def _read_exact(conn, size: int) -> bytes | None:
    if size < 0:
        return None
    buf = b""
    while len(buf) < size:
        chunk = conn.recv(size - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


def start_background() -> None:
    threading.Thread(target=_planner_probe_loop, name="planner-probe", daemon=True).start()
    threading.Thread(target=_suite_loader, name="suite-loader", daemon=True).start()
    threading.Thread(target=_display_inlet, name="display-inlet", daemon=True).start()


_NO_STORE = {"Cache-Control": "no-store"}


async def index(_request):
    return web.FileResponse(ROOT / "static" / "index.html", headers=_NO_STORE)


async def app_js(_request):
    return web.FileResponse(ROOT / "static" / "app.js", headers=_NO_STORE)


async def hook_js(_request):
    return web.FileResponse(ROOT / "static" / "hook.js", headers=_NO_STORE)


async def health(_request):
    return web.json_response(APP.health(), headers=_NO_STORE)


async def suites(_request):
    if APP.suites is None and not APP.suite_error:
        return web.json_response({"type": "suites", "status": "loading", "suites": []}, headers=_NO_STORE)
    if APP.suite_error:
        return web.json_response(
            {"type": "suites", "status": "failed", "detail": APP.suite_error, "suites": []},
            headers=_NO_STORE,
        )
    return web.json_response({"type": "suites", "status": "ready", "suites": APP.suites}, headers=_NO_STORE)


async def websocket(request):
    ws = web.WebSocketResponse(max_msg_size=0)
    await ws.prepare(request)
    APP.clients.add(ws)
    APP.loop = asyncio.get_running_loop()
    await ws.send_str(json.dumps(APP.health(), ensure_ascii=False))
    try:
        async for msg in ws:
            if msg.type != WSMsgType.TEXT:
                continue
            data = json.loads(msg.data)
            kind = data.get("type")
            if kind == "get_scenes":
                if APP.suites is None and not APP.suite_error:
                    await ws.send_str(json.dumps({"type": "suites", "status": "loading", "suites": []}, ensure_ascii=False))
                elif APP.suite_error:
                    await ws.send_str(json.dumps({"type": "suites", "status": "failed", "detail": APP.suite_error, "suites": []}, ensure_ascii=False))
                else:
                    await ws.send_str(json.dumps({"type": "suites", "status": "ready", "suites": APP.suites}, ensure_ascii=False))
            elif kind == "get_health":
                await ws.send_str(json.dumps(APP.health(), ensure_ascii=False))
            elif kind == "ping":
                await ws.send_str(json.dumps({"type": "pong"}))
            elif kind == "load":
                threading.Thread(target=APP.load, args=(data,), daemon=True).start()
            elif kind in {"start", "restart"}:
                APP.start(kind, data, execute=True)
            elif kind == "plan_only":
                APP.start("start", data, execute=False)
            elif kind == "replay":
                data["plan_source"] = "replay"
                if "stop_after" not in data:
                    data["stop_after"] = "grasp_held"
                APP.start("start", data, execute=True)
            elif kind == "bind":
                APP.choose_binding(data)
            elif kind == "stop":
                APP.stop()
            elif kind == "confirm":
                APP.confirm(data)
            elif kind == "client_perf":
                APP.client_frames = {
                    "frames_received": data.get("frames_received"),
                    "frames_drawn": data.get("frames_drawn"),
                    "distinct_drawn": data.get("distinct_drawn"),
                    "max_gap_ms": data.get("max_gap_ms"),
                    "max_age_ms": data.get("max_age_ms"),
                    "decode_ms": data.get("decode_ms"),
                    "draw_ms": data.get("draw_ms"),
                    "gap_ms": data.get("gap_ms"),
                }
    finally:
        APP.clients.discard(ws)
    return ws


def main() -> None:
    try:
        os.setpgrp()
    except OSError:
        pass
    app = web.Application()
    app.router.add_get("/", index)
    app.router.add_get("/health", health)
    app.router.add_get("/suites", suites)
    app.router.add_get("/static/app.js", app_js)
    app.router.add_get("/static/hook.js", hook_js)
    app.router.add_get("/ws", websocket)
    print(f"sequential adapter mode={'live' if LIVE else 'dry'} pid={os.getpid()} port={PORT}", flush=True)
    start_background()
    web.run_app(app, host=HOST, port=PORT)


if __name__ == "__main__":
    main()
