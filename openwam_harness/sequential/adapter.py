"""Thin WebSocket adapter. It speaks the existing Page UI frame protocol.

HARNESS_LIVE=1 uses the LIBERO env. HARNESS_LIVE=0 stays on the dry path and
refuses to pretend that placeholder frames are a real episode.
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import subprocess
import threading
import time
import traceback
import urllib.request
from pathlib import Path
from aiohttp import web, WSMsgType

from executor import Session, ToolOutcome, run_plan
from planner import SHORT_MEMORY, load_human_plan, make_openai_client, request_plan
from planner_profiles import comparison_record, profiles, public_profile
from run_context import RunContext

ROOT = Path(__file__).resolve().parent
PLAN_PATH = ROOT / "plans" / "two_object.json"
HOST = os.environ.get("SEQUENTIAL_HOST", "127.0.0.1")
PORT = int(os.environ.get("SEQUENTIAL_PORT", "8766"))
LIVE = os.environ.get("HARNESS_LIVE", "0") == "1"
DIAGNOSTIC_S = float(os.environ.get("SEQUENTIAL_DIAGNOSTIC_S", "0"))
LOAD_TIMEOUT_S = float(os.environ.get("SEQUENTIAL_LOAD_TIMEOUT_S", "300"))
PLANNER_URL = os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")
PLANNER_MODEL = os.environ.get("PLANNER_MODEL", "Qwen3.5-9B")
STATIC_VERSION = "view8"
RUN_LIMIT_S = 180.0
EXECUTION_LABEL = "同步"


def _startup_commit() -> str:
    """Commit of this process at import. Later disk HEAD is not substituted."""
    repo = Path(__file__).resolve().parents[2]
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            text=True,
            timeout=5,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unavailable"


STARTUP_COMMIT = _startup_commit()

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
        self.runs = RunContext()
        self.clients = set()
        self.loop = None
        self.lock = threading.Lock()
        self.run_id = 0
        self.last_payload: bytes | None = None
        self.logs: list[str] = []
        self.mode = "live" if LIVE else "dry"
        self.execution_mode = "benchmark"
        self.load_token = 0
        self.recovery_attempts = 0
        self.binding = None
        self.display_seq = 0
        self.client_frames = None
        self.run_journal: list[dict] = []
        self.request_seq = 0
        self.active_request_id = ""
        self.planner_profile = os.environ.get("PLANNER_PROFILE", "edge")
        if self.planner_profile not in {"edge", "cloud"}:
            self.planner_profile = "edge"
        self.planner = self._make_planner()

    @property
    def running(self) -> bool:
        return self.runs.active() or self.runs.producer_alive()

    @running.setter
    def running(self, value: bool) -> None:
        if value:
            if not self.runs.active():
                self.runs.begin(self.run_id, self.session.episode_id, self.session.scene_epoch, "", self.execution_mode)
            return
        if self.runs.active():
            self.runs.finish("cancelled", self.runs.root_error or "cleared")
        self.runs.thread = None

    @property
    def awaiting_confirm(self) -> bool:
        return self.runs.phase == "waiting_confirmation"

    @awaiting_confirm.setter
    def awaiting_confirm(self, value: bool) -> None:
        if value and self.runs.active() and self.runs.phase != "cancelling":
            self.runs.phase = "waiting_confirmation"
        elif not value and self.runs.phase == "waiting_confirmation":
            self.runs.phase = "executing"

    @property
    def confirm_q(self):
        return self.runs.confirm_q

    def active_profile(self) -> dict:
        catalog = profiles()
        profile = catalog.get(self.planner_profile) or catalog["edge"]
        return profile

    def _make_planner(self):
        profile = self.active_profile()
        if profile.get("id") != self.planner_profile and profile.get("id"):
            self.planner_profile = profile["id"]
        return make_openai_client(
            str(profile.get("base_url") or PLANNER_URL),
            str(profile.get("model") or PLANNER_MODEL),
            api_key=str(profile.get("api_key") or "EMPTY"),
            input_mode=str(profile.get("input_mode") or "scene_metadata"),
        )

    def select_profile(self, name: str) -> None:
        if name not in profiles():
            self.broadcast({"type": "state", "phase": "failed", "detail": f"未知 Planner {name}"})
            return
        self.planner_profile = name
        self.planner = self._make_planner()
        _probe_planner()
        self.broadcast({"type": "planner_profile", "profile": public_profile(self.active_profile())})
        self.broadcast(self.health())

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
        profile = self.active_profile()
        probe = PROBE.get(self.planner_profile) or {}
        planner = probe.get("status") or "unknown"
        shown = {}
        for key, value in profiles().items():
            public = public_profile(value)
            row = PROBE.get(key) or {}
            public["reachable"] = row.get("status") == "ok"
            public["probe"] = row.get("status") or "unknown"
            public["model_ids"] = list(row.get("model_ids") or [])
            shown[key] = public
        return {
            "type": "health",
            "adapter": "online",
            "mode": self.mode,
            "harness_live": LIVE,
            "port": PORT,
            "pid": os.getpid(),
            "commit": STARTUP_COMMIT,
            "static_version": STATIC_VERSION,
            "execution": EXECUTION_LABEL,
            "planner": planner,
            "planner_ok": planner == "ok",
            "planner_detail": probe.get("detail") or "",
            "planner_url": profile.get("base_url") or PLANNER_URL,
            "planner_model": profile.get("model") or PLANNER_MODEL,
            "planner_profile": self.planner_profile,
            "planner_profiles": shown,
            "loading": self.loading,
            "checkpoints": checkpoints,
            "components": components,
            "env": env_state,
            "vla": vla_state,
            "suites": "ready" if self.suites else ("failed" if self.suite_error else "loading"),
            "suite_error": self.suite_error,
            "can_select_scene": True,
            "can_manual_plan": False,
            "saved_plan_fallback": False,
            "can_model_plan": bool(profile.get("available")) and planner == "ok",
            "silent_dry_fallback": False,
            "error": error,
        }

    def scene_context(self) -> dict:
        """Authoritative loaded scene. Seed is the init-state index; missing stays missing."""
        spec = self.scene_spec or {}
        loaded = self.session.scene_id != "none" and bool(spec)
        seed = spec.get("seed") if "seed" in spec else None
        init_state = spec.get("init_state_id") if "init_state_id" in spec else seed
        return {
            "loaded": loaded,
            "suite": spec.get("suite"),
            "task_id": spec.get("task_id"),
            "seed": seed,
            "init_state_id": init_state,
            "episode_id": self.session.episode_id if loaded else None,
            "scene_epoch": self.session.scene_epoch if loaded else None,
            "plan_version": self.session.plan_version if loaded else None,
            "goal": spec.get("goal") or "",
            "summary_zh": spec.get("summary_zh") or "",
            "user_text": spec.get("user_text") or "",
            "ended": bool(loaded and self.session.cancelled),
        }

    def start_block_reason(self) -> str:
        if self.loading:
            return "正在加载"
        if self.runs.phase == "cancelling" or (self.runs.producer_alive() and self.runs.finished):
            return "正在取消"
        if self.runs.phase == "waiting_confirmation":
            return "等待确认"
        if self.running:
            return "正在执行"
        scene = self.scene_context()
        if not scene["loaded"] or (LIVE and self.backend is None):
            return "场景未加载"
        if scene["ended"]:
            return "回合已结束"
        profile = self.active_profile()
        if not profile.get("available"):
            return "云端未配置" if profile.get("id") == "cloud" else "模型未就绪"
        if (PROBE.get(self.planner_profile) or {}).get("status") != "ok":
            return "模型未就绪"
        return ""

    def _note(self, run_id: int, stage: str, status: str, code: str = "", detail: str = "", request_id: str | None = None) -> None:
        if self.runs.run_id == run_id:
            self.runs.add_event(stage, status, detail, code)
        row = {
            "run_id": run_id,
            "client_request_id": self.active_request_id if request_id is None else request_id,
            "t": round(time.time(), 3),
            "stage": stage,
            "status": status,
            "code": code,
            "detail": str(detail)[:500],
        }
        self.run_journal.append(row)
        self.run_journal = self.run_journal[-200:]
        try:
            path = ROOT / "records" / "run_journal.jsonl"
            path.parent.mkdir(exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        except OSError:
            pass
        self.broadcast({"type": "run_stage", **row})

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
        self.last_payload = payload
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
        self.last_payload = payload
        if self.loop is None:
            return
        for ws in list(self.clients):
            asyncio.run_coroutine_threadsafe(ws.send_bytes(payload), self.loop)

    def load(self, message) -> None:
        with self.lock:
            if self.running:
                cancelling = self.runs.phase == "cancelling" or (self.runs.producer_alive() and self.runs.finished)
                self.broadcast({
                    "type": "load_rejected",
                    "code": "busy_cancelling" if cancelling else "busy_running",
                    "detail": "正在取消，动作结束后才能加载。" if cancelling else "正在执行，不能同时加载。",
                    "client_request_id": message.get("client_request_id") if isinstance(message, dict) else None,
                    "run_phase": self.runs.phase,
                    "root_error": self.runs.root_error,
                })
                return
            if self.loading:
                self.broadcast({
                    "type": "load_rejected",
                    "code": "busy_loading",
                    "detail": "已有加载在进行，不会再创建环境。",
                    "client_request_id": message.get("client_request_id") if isinstance(message, dict) else None,
                })
                return
            self.loading = True
            self.load_token += 1
            token = self.load_token
            self.session.cancel()
            self._wake()
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
            self._broadcast_load_error(exc)
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
            and self.backend.scene_key() == (spec["suite"], spec["task_id"], spec["seed"])
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
            self.log("".join(traceback.format_exception(type(err), err, err.__traceback__)))
            log = getattr(err, "log", "")
            if log:
                self.log(log[-1000:])
            self._broadcast_load_error(err)
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
                "suite": spec["suite"],
                "task_id": spec["task_id"],
                "seed": spec["seed"],
                "task_source": spec.get("task_source"),
                "execution": EXECUTION_LABEL,
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
        if self.runs.finished and not self.runs.producer_alive():
            self.runs = RunContext()
        self.broadcast(self.snapshot())
        self.log("load timings " + json.dumps(timings, ensure_ascii=False))

    def _begin_session(self, scene_id: str) -> str:
        previous = 0 if self.session.scene_id == "none" else self.session.scene_epoch
        self.session = Session(scene_id)
        self.session.scene_epoch = previous
        return self.session.begin_episode()

    def _broadcast_load_error(self, exc: BaseException) -> None:
        detail = str(exc)
        field = getattr(exc, "field", None)
        cause = exc.__cause__ or exc.__context__
        frames = traceback.extract_tb((cause or exc).__traceback__)
        if field is None and frames:
            last = frames[-1]
            detail = f"{detail} @ {last.filename}:{last.lineno} {last.name}"
        self.log(detail)
        payload = {"type": "load_failed", "detail": detail}
        state = {"type": "state", "phase": "failed", "detail": detail}
        if field is not None:
            payload["field"] = field
            state["field"] = field
        self.broadcast(payload)
        self.broadcast(state)

    def _scene_from_message(self, message) -> dict:
        from scene_request import resolve_load
        from suites import load_suites

        if self.suites is None:
            self.suites = load_suites()
        spec = resolve_load(message, self.suites)
        self.log(
            f"seed fixed={spec['seed']} suite={spec['suite']} task={spec['task_id']} source={spec['task_source']}"
        )
        return spec

    def start(self, kind: str, message: dict, execute: bool = True) -> None:
        plan_source = message.get("plan_source") if "plan_source" in message else "model"
        with self.lock:
            self.request_seq += 1
            attempt = self.request_seq
            ack = self._ack_start(kind, message, plan_source)
            ack["run_id"] = attempt
            request_id = ack.get("client_request_id") or ""
            self.active_request_id = request_id if ack["status"] == "accepted" else self.active_request_id
            if ack["status"] != "accepted":
                self._note(attempt, "received", "ok", detail=kind, request_id=request_id)
                self._note(attempt, "rejected", "rejected", code=ack.get("code") or "", detail=ack.get("detail") or "", request_id=request_id)
                self.broadcast(ack)
                return
            self.run_id = attempt
            self.active_request_id = request_id
            self.planner_profile = ack["profile"]
            self.execution_mode = ack.get("execution_mode") or "benchmark"
            raw_task = message.get("task") if isinstance(message.get("task"), str) else ""
            self.runs.begin(attempt, self.session.episode_id, self.session.scene_epoch, raw_task, self.execution_mode)
            self.planner = self._make_planner()
            if kind == "restart":
                self.session.cancel()
                self._wake()
        self._note(attempt, "received", "ok", detail=kind)
        self._note(attempt, "accepted", "ok", detail=ack.get("detail") or "")
        self.broadcast(ack)
        raw_task = message.get("task") if isinstance(message.get("task"), str) else ""
        thread = threading.Thread(
            target=self._run,
            args=(
                kind,
                "model",
                attempt,
                message.get("manual_scope") or "single",
                execute,
                message.get("stop_after"),
                message.get("deadline_s"),
                raw_task,
            ),
            daemon=True,
        )
        self.runs.attach(thread)
        thread.start()

    def _ack_start(self, kind: str, message: dict, plan_source) -> dict:
        request_id = message.get("client_request_id") if isinstance(message, dict) else None
        base = {"type": "start_ack", "client_request_id": request_id if isinstance(request_id, str) else None}
        if plan_source != "model":
            return {**base, "status": "rejected", "code": "plan_source", "detail": "开始只发起新的模型规划，不使用保存计划、人工计划或隐藏回退"}
        if not isinstance(request_id, str) or not request_id.strip():
            return {**base, "status": "rejected", "code": "request_id_missing", "detail": "开始请求缺少 client_request_id"}
        mode = message.get("execution_mode", "benchmark") if isinstance(message, dict) else "benchmark"
        if mode not in {"benchmark", "free"}:
            return {**base, "status": "rejected", "code": "execution_mode", "detail": f"未知执行模式 {mode}"}
        if self.loading:
            return {**base, "status": "rejected", "code": "busy_loading", "detail": "正在加载"}
        if self.running:
            return {**base, "status": "rejected", "code": "busy_running", "detail": "正在执行"}
        spec = self.scene_spec or {}
        if self.session.scene_id == "none" or not spec or (LIVE and self.backend is None):
            return {**base, "status": "rejected", "code": "scene_missing", "detail": "场景未加载"}
        if kind != "restart" and self.session.cancelled:
            return {**base, "status": "rejected", "code": "episode_ended", "detail": "回合已结束"}
        if not isinstance(message, dict) or "scene_epoch" not in message or message.get("scene_epoch") is None:
            return {**base, "status": "rejected", "code": "epoch_missing", "detail": "scene_epoch 缺失，不会把 null 当成 0"}
        epoch = message.get("scene_epoch")
        if isinstance(epoch, bool) or not isinstance(epoch, int):
            return {**base, "status": "rejected", "code": "epoch_type", "detail": f"scene_epoch 不是整数（收到 {epoch!r}）"}
        if epoch != self.session.scene_epoch:
            return {**base, "status": "rejected", "code": "epoch_stale", "detail": f"场景版本已过期：客户端 {epoch}，服务端 {self.session.scene_epoch}"}
        if isinstance(message, dict) and "task" in message and message.get("task") is None:
            return {**base, "status": "rejected", "code": "task_null", "detail": "task 为 null。留空表示使用当前目标，不能传 null"}
        raw = message.get("task", "") if isinstance(message, dict) else ""
        if not isinstance(raw, str):
            return {**base, "status": "rejected", "code": "task_type", "detail": f"task 不是字符串（收到 {type(raw).__name__}）"}
        profile_name = message.get("planner_profile") or self.planner_profile
        if profile_name not in {"edge", "cloud"}:
            return {**base, "status": "rejected", "code": "profile_unknown", "detail": f"未知模型 {profile_name}，没有改用另一端"}
        profile = profiles().get(profile_name) or {}
        if not profile.get("available"):
            detail = "云端未配置，没有改用边缘或其他模型" if profile_name == "cloud" else "模型未配置"
            return {**base, "status": "rejected", "code": "profile_unconfigured", "detail": detail}
        probe = PROBE.get(profile_name) or {}
        if probe.get("status") != "ok":
            return {
                **base,
                "status": "rejected",
                "code": "profile_offline",
                "detail": f"{profile.get('label') or profile_name} Planner 未就绪：{probe.get('detail') or probe.get('status') or 'unknown'}",
            }
        text = raw.strip()
        return {
            **base,
            "status": "accepted",
            "code": "accepted",
            "detail": spec.get("summary_zh") or spec.get("goal") or "",
            "scene_epoch": epoch,
            "user_text": raw if text else "",
            "summary_zh": spec.get("summary_zh") or "",
            "goal": spec.get("goal") or "",
            "used_loaded_goal": text == "",
            "profile": profile_name,
            "execution_mode": mode,
        }

    def stop(self) -> None:
        from vla_async import get_controller

        if not self.running:
            self.broadcast(self.snapshot())
            return
        get_controller().clear("cancel")
        cancel = getattr(self.planner, "cancel", None)
        if cancel is not None:
            cancel()
        self.session.cancel()
        self.runs.cancel()
        self.broadcast({"type": "state", "phase": "cancelling", "detail": "正在取消", "mode": "live", "root_error": self.runs.root_error})
        self.broadcast(self.snapshot())

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
        if not self.runs.submit_confirm(message.get("verdict")):
            self.broadcast({"type": "state", "phase": self.runs.phase, "detail": "确认已失效", "root_error": self.runs.root_error})
            return

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

    def _run(self, kind: str, plan_source: str, run_id: int, manual_scope: str = "single", execute: bool = True, stop_after: str | None = None, deadline_s: float | None = None, user_text: str = "") -> None:
        try:
            self._run_inner(kind, plan_source, run_id, manual_scope, execute, stop_after, deadline_s, user_text)
        except Exception as exc:
            self.log(f"run failed: {exc}")
            self.runs.note_error(str(exc), "exception")
            self._close_run("failed", str(exc), code="exception")
        finally:
            if self.runs.run_id == run_id and not self.runs.finished:
                phase = "cancelled" if self.session.cancelled or self.runs.phase == "cancelling" else "failed"
                self._close_run(phase, self.runs.root_error or phase)
            self.runs.release(threading.current_thread())
            self._persist_run()
            self.broadcast(self.snapshot())

    def _run_inner(self, kind: str, plan_source: str, run_id: int, manual_scope: str = "single", execute: bool = True, stop_after: str | None = None, deadline_s: float | None = None, user_text: str = "") -> None:
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
        spec = dict(self.scene_spec or {})
        typed = user_text.strip()
        if not typed:
            spec["user_text"] = ""
            self.broadcast({
                "type": "task_bound",
                "used_loaded_goal": True,
                "user_text": "",
                "summary_zh": spec.get("summary_zh") or "",
                "goal": spec.get("goal") or "",
                "run_id": run_id,
            })
        else:
            spec["user_text"] = user_text
        self.scene_spec = spec
        free = self.execution_mode == "free"
        predicates = []
        ordered = False
        alias_hints = []
        if self.backend is not None and hasattr(self.backend, "set_execution_mode"):
            self.backend.set_execution_mode(self.execution_mode)
        self.runs.set_phase("preparing")
        self._note(run_id, "task_check", "running", detail=typed or spec.get("summary_zh") or "")
        self.broadcast({"type": "state", "phase": "checking_task", "detail": spec.get("summary_zh") or "", "mode": "live", "run_id": run_id})
        if free:
            scene_objects, binding, info = self._scene_objects(spec)
            from free_task import prepare_free_request

            prepared = prepare_free_request(
                typed or spec.get("summary_zh") or "",
                scene_objects,
                binding.get("positions") or {},
            )
            predicates = []
            ordered = bool(prepared.get("ordered"))
            alias_hints = prepared.get("alias_hints") or []
            canonical = typed or spec.get("summary_zh") or ""
            self._note(run_id, "task_check", "ok", detail=canonical)
            self.broadcast({
                "type": "task_bound",
                "used_loaded_goal": not bool(typed),
                "user_text": user_text if typed else "",
                "summary_zh": canonical,
                "goal": canonical,
                "run_id": run_id,
                "execution_mode": "free",
            })
            spec = dict(spec)
            spec["native_goal"] = spec.get("goal") or ""
            spec["user_goal"] = canonical
            binding = {
                "status": "unresolved",
                "evidence_source": "sim_state",
                "vision": False,
                "object_ref": None,
                "positions": binding.get("positions") or {},
                "alias_hints": alias_hints,
                "note": "自由模式不在进 Planner 前绑定名称。仿真位姿不是视觉判断，也不切换任务。",
            }
        else:
            decision = self._instruction_decision(typed, spec)
            if decision["status"] == "ambiguous":
                self._note(run_id, "task_check", "rejected", code="ambiguous", detail=decision.get("question") or "")
                self.broadcast({"type": "clarify", "question": decision["question"], "candidates": decision.get("candidates") or []})
                self._fail_run(episode, decision["question"])
                return
            if decision["status"] == "unsupported":
                self._note(run_id, "task_check", "rejected", code="unsupported", detail=decision.get("reason") or "")
                self._fail_run(episode, decision["reason"])
                return
            self._note(run_id, "task_check", "ok", detail=decision.get("language") or "")
            if typed:
                self.broadcast({
                    "type": "task_bound",
                    "used_loaded_goal": False,
                    "user_text": user_text,
                    "summary_zh": decision.get("summary_zh") or spec.get("summary_zh") or "",
                    "goal": decision.get("language") or "",
                    "run_id": run_id,
                })
            if decision.get("changed"):
                task_id = decision.get("task_id")
                if isinstance(task_id, bool) or not isinstance(task_id, int):
                    self._note(run_id, "task_switch", "rejected", code="task_id", detail=repr(task_id))
                    self._fail_run(episode, f"task_id 无效（收到 {task_id!r}），不会猜测或切换任务。")
                    return
                self._note(run_id, "task_switch", "running", detail=decision.get("summary_zh") or "")
                self.broadcast({
                    "type": "state",
                    "phase": "switching",
                    "detail": decision.get("summary_zh") or "",
                    "mode": "live",
                    "run_id": run_id,
                })
                spec = self._adopt_task(task_id)
                spec = dict(spec)
                spec["user_text"] = user_text
                episode = self.session.episode_id
                epoch = self.session.scene_epoch
            canonical = decision.get("language") or spec.get("goal") or ""
            row = decision.get("capability") or {}
            from capabilities import reject_reason

            reason = reject_reason(row) if row else None
            if reason:
                self._note(run_id, "capability", "rejected", code="unsupported_skill", detail=reason)
                self._fail_run(episode, reason)
                return
            spec = dict(spec)
            spec["goal"] = canonical
            spec["task"] = canonical
            scene_objects, binding, info = self._scene_binding(spec)
        self.scene_spec = spec
        self._note(run_id, "binding", binding.get("status") or "unknown", detail=binding.get("object_ref") or binding.get("evidence_source") or "")
        self.broadcast({"type": "state", "phase": "binding", "detail": binding.get("evidence_source") or "", "mode": "live", "run_id": run_id})
        self.broadcast({"type": "binding", "binding": _public_binding(binding)})
        if not free and binding.get("status") != "resolved":
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
            from candidates import binding_question, number_candidates

            question = binding_question(binding, number_candidates(scene_objects))
            self.broadcast({"type": "clarify", "question": question, "candidates": binding.get("candidates") or []})
            self._fail_run(episode, question)
            return
        if plan_source != "model":
            self._fail_run(episode, "开始只发起新的模型规划，不使用保存计划、人工计划或隐藏回退")
            return
        probe = PROBE.get(self.planner_profile) or {}
        if probe.get("status") != "ok":
            self._note(run_id, "planner_request", "rejected", code="profile_offline", detail=probe.get("detail") or probe.get("status") or "")
            self._fail_run(episode, f"planner {probe.get('status')}: {probe.get('detail') or 'unavailable'}")
            return
        profile = self.active_profile()
        self.planner = self._make_planner()
        self.runs.set_phase("planning")
        self._note(run_id, "planner_request", "running", detail=profile.get("model") or "")
        self.broadcast({"type": "state", "phase": "planning", "detail": profile.get("model") or PLANNER_MODEL, "mode": "live", "plan_source": "model", "run_id": run_id})
        image_ids = [f"agentview:{info.get('env_step')}", f"wrist:{info.get('env_step')}"]
        from candidates import number_candidates, scene_graph
        from schema import transfer_slots

        from skills import implemented_skills, unimplemented_skills

        slots = [] if free else (transfer_slots(spec.get("goal") or "") or [])
        required_ref = None if free else (binding.get("object_ref") if len(slots) == 1 else None)
        numbered = number_candidates(scene_objects)
        graph = scene_graph(scene_objects, (binding or {}).get("positions"))
        scene_text = json.dumps(
            {
                "planning_input": "scene_metadata",
                "vision": False,
                "evidence_source": "sim_state",
                "execution_mode": self.execution_mode,
                "goal_relation": None if free else spec.get("goal_relation"),
                "canonical_goal": canonical,
                "native_goal": spec.get("native_goal") or spec.get("goal"),
                "user_text": user_text,
                "predicates": predicates,
                "alias_hints": alias_hints if free else [],
                "alias_hints_note": "提示不是绑定。未出现在别名表里的措辞仍交给 Planner。",
                "implemented_skills": implemented_skills(),
                "unimplemented_skills": unimplemented_skills(),
                "left_right_frame": "仿真世界坐标 x，较小为左，较大为右。不是相机画面的左右。",
                "scene_graph": graph,
                "candidates": numbered,
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
                task=canonical if free else spec["goal"],
                scene_observation=scene_text,
                client=self.planner,
                short_memory=SHORT_MEMORY,
                image_ids=image_ids,
                scene_objects=scene_objects,
                required_object_ref=required_ref,
                numbered_candidates=numbered,
                on_event=on_planner,
                execution_mode=self.execution_mode,
                predicates=predicates,
                ordered=ordered,
            )
        except Exception as exc:
            self._note(run_id, "planner_response", "failed", code="planner_request_failed", detail=str(exc))
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
        self._note(
            run_id,
            "planner_response",
            "ok" if result.get("status") == "ok" else "failed",
            code=result.get("stage") or "",
            detail=(result.get("attempts") or [{}])[-1].get("request_id") or "",
        )
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
            comparison=comparison_record(self.active_profile(), binding, scene_text),
            telemetry=(attempts[-1].get("telemetry") if attempts else {}) or {},
            candidate_resolution=result.get("candidate_resolution") or [],
        )
        if result.get("status") == "clarify":
            question = result.get("question") or "请说明要哪一个物体"
            self._note(run_id, "validate", "rejected", code="clarify", detail=question)
            self.runs.note_error(question, "clarify")
            self.broadcast({"type": "clarify", "question": question, "candidates": []})
            self._close_run("failed", question, code="clarify")
            return
        if result["status"] != "ok" or result["plan"] is None:
            self._note(run_id, "validate", "failed", code=result.get("stage") or "validation_failed", detail="PLANNING_FAILED")
            self._fail_run(episode, "PLANNING_FAILED")
            return
        self._note(run_id, "validate", "ok", detail=result.get("stage") or "ok")
        plan = result["plan"]
        plan["planner_stage"] = "ok"
        plan["request_id"] = attempts[-1].get("request_id") if attempts else None
        plan["episode_id"] = episode
        if plan_source in {"model", "replay"}:
            plan["planner_stage"] = "ok"
        self._note(run_id, "compile", "ok", detail=str(len(plan.get("steps") or [])))
        self.broadcast({"type": "state", "phase": "validating", "detail": f"{len(plan.get('steps') or [])} steps", "mode": "live", "run_id": run_id})
        self.log(f"plan_source={plan.get('plan_source')} steps={len(plan['steps'])}")
        if not execute:
            self.broadcast({"type": "state", "phase": "plan_ready", "detail": "只规划，不执行", "mode": "live", "plan_source": plan.get("plan_source")})
            self.broadcast({"type": "ended", "episode_id": episode, "success": False, "reason": "plan_only_not_executed", "native_success": False})
            return

        def confirm(subtask: str):
            self.runs.set_phase("waiting_confirmation")
            self.broadcast(self.snapshot())
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
            verdict = self.runs.wait_confirm()
            if verdict in {None, "timeout"} or self.session.scene_epoch != epoch or self.session.cancelled:
                return None
            return verdict

        deadline = None
        run_cap = 300.0 if free else RUN_LIMIT_S
        client_cap = 360.0 if free else 240.0
        if isinstance(deadline_s, (int, float)) and not isinstance(deadline_s, bool) and float(deadline_s) > 0:
            deadline = time.monotonic() + min(float(deadline_s), client_cap)
        elif DIAGNOSTIC_S > 0:
            deadline = time.monotonic() + DIAGNOSTIC_S
        else:
            deadline = time.monotonic() + run_cap
        seen_tool = {"done": False}

        def on_event(event: dict) -> None:
            if event.get("phase") == "executing":
                self.runs.set_phase("executing")
            if event.get("phase") == "waiting_confirmation":
                self.runs.set_phase("waiting_confirmation")
            if event.get("execution"):
                self.runs.execution = event.get("execution")
            if event.get("phase") in {"failed", "waiting_confirmation"}:
                self.runs.note_error(str(event.get("reason") or event.get("detail") or ""), str(event.get("reason_code") or ""))
            if not seen_tool["done"] and event.get("skill"):
                seen_tool["done"] = True
                self._note(run_id, "first_tool", "ok", detail=str(event.get("skill")))
            if event.get("skill") in {"pi0_pick", "move_to", "set_gripper"} or event.get("detail") == "local replan":
                self.binding = None
            self.broadcast({"type": "state", **event, "mode": "live", "run_id": run_id})
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
            phase = "cancelled" if self.session.cancelled or self.runs.phase == "cancelling" else "failed"
            reason = "已取消" if phase == "cancelled" else (self.runs.root_error or result.get("reason") or "确认结束")
            self._close_run(phase, reason, code="confirm_timeout" if self.runs.root_code == "confirm_timeout" else phase)
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
        elif self.execution_mode == "free":
            user_ok = result.get("user_goal_success") is True
            native_ok = bool(result.get("native_task_success"))
            if user_ok:
                phase = "completed"
                status = "completed"
                reason = "用户目标完成"
            elif native_ok:
                phase = "incomplete"
                reason = "原生目标已达成，用户组合目标未完成"
            elif status == "completed":
                phase = "incomplete"
                reason = "计划已执行，用户目标未完成"
            else:
                phase = "failed"
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
        self.runs.native_task_success = bool(result.get("native_task_success"))
        self.runs.user_goal_success = result.get("user_goal_success")
        self.runs.subgoal_results = result.get("subgoal_results")
        self.runs.execution = result.get("execution")
        self.runs.plan_summary = str(plan.get("goal") or "")[:200]
        terminal = "succeeded" if phase == "completed" else "cancelled" if phase == "stopped" else "failed"
        if terminal != "succeeded":
            self.runs.note_error(reason, code or "")
        self.broadcast_frame(
            {
                "eid": episode,
                "fid": self._fid(),
                "phase": phase,
                "status": status,
                "detail": result["status"],
                "term": True,
                "success": bool(result.get("user_goal_success")) if self.execution_mode == "free" else native,
                "native_task_success": bool(result.get("native_task_success")),
                "user_goal_success": result.get("user_goal_success"),
                "reason": reason,
                "plan_source": plan.get("plan_source"),
            }
        )
        self._close_run(
            terminal,
            reason,
            code=code or "",
            ended={
                "episode_id": episode,
                "success": bool(result.get("user_goal_success")) if self.execution_mode == "free" else native,
                "reason": reason,
                "native_success": native,
                "native_task_success": bool(result.get("native_task_success")),
                "user_goal_success": result.get("user_goal_success"),
                "subgoal_results": result.get("subgoal_results"),
                "execution_mode": self.execution_mode,
                "plan_source": plan.get("plan_source"),
                "status": status,
                "planner_stage": plan.get("planner_stage"),
                "execution": result.get("execution"),
                "frames": getattr(getattr(self.backend, "frame_bus", None), "stats", lambda: {})(),
                "client_frames": self.client_frames,
            },
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

    def _instruction_decision(self, user_text: str, spec: dict) -> dict:
        from instruction import resolve_instruction

        tasks = None
        if self.suites:
            tasks = [
                {
                    "suite": suite["id"],
                    "id": task["id"],
                    "language": task["language"],
                    "summary_zh": task["summary_zh"],
                    "task_key": task.get("task_key"),
                }
                for suite in self.suites
                for task in suite.get("tasks") or []
            ]
        return resolve_instruction(user_text, spec, tasks)

    def _adopt_task(self, task_id: int) -> dict:
        """Load the matched task and drop the previous native success criterion."""
        current = self.scene_spec or {}
        from scene_request import FieldError

        if "seed" not in current or current.get("seed") is None:
            raise FieldError("seed", "missing", current.get("seed"), "已加载场景的 seed 0")
        spec = self._scene_from_message(
            {"suite": current.get("suite"), "task_id": task_id, "seed": current.get("seed")}
        )
        self.log(
            f"显式加载 {spec['scene_id']} seed={spec['seed']}，成功标准改为该任务的 env.terminated，不沿用上一个任务"
        )
        if LIVE and self.backend is not None:
            out = ROOT.parents[0] / "runs" / "sequential" / self.session.episode_id
            same = self.backend.scene_key() == (spec["suite"], spec["task_id"], spec["seed"])
            if same:
                self.backend.warm_reset()
            else:
                self.backend.retarget(spec, out)
            self._begin_session(spec["scene_id"])
            self.backend.scene_epoch = self.session.scene_epoch
        else:
            self._begin_session(spec["scene_id"])
        self.scene_spec = spec
        self.binding = None
        self.broadcast(
            {
                "type": "loaded",
                "instruction": spec["goal"],
                "summary_zh": spec.get("summary_zh"),
                "env_id": spec["scene_id"],
                "episode_id": self.session.episode_id,
                "scene_epoch": self.session.scene_epoch,
                "mode": self.mode,
                "explicit_task_switch": True,
                "suite": spec["suite"],
                "task_id": spec["task_id"],
                "seed": spec["seed"],
                "task_source": spec.get("task_source"),
                "execution": EXECUTION_LABEL,
            }
        )
        return spec

    def _scene_objects(self, spec: dict) -> tuple[dict, dict, dict]:
        from suites import objects_for_names

        info = self.backend.scene_info()
        sim = self.backend.sim_state()
        poses = {}
        for name, item in (sim.get("objects") or {}).items():
            if isinstance(item, dict) and isinstance(item.get("pos"), list):
                poses[name] = item["pos"]
        names = list(info.get("object_names") or [])
        for item in list(spec.get("instances") or []) + list(spec.get("fixtures") or []):
            name = item.get("name") if isinstance(item, dict) else None
            if name and name not in names:
                names.append(name)
        for name in poses:
            if name not in names:
                names.append(name)
        objects = objects_for_names(names, spec)
        return objects, {"status": "scene", "positions": poses, "evidence_source": "sim_state", "vision": False}, info

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
        self.runs.note_error(reason)
        self._close_run(
            "failed",
            reason,
            ended={
                "episode_id": episode,
                "success": False,
                "reason": reason,
                "native_success": False,
                "native_task_success": False,
                "user_goal_success": False,
            },
        )

    def _close_run(self, phase: str, reason: str, code: str = "", ended: dict | None = None) -> None:
        first = self.runs.finish(phase, reason, code)
        self._persist_run()
        if not first:
            self.broadcast(self.snapshot())
            return
        self.broadcast({"type": "state", "phase": phase, "detail": self.runs.root_error or reason, "mode": "live", "root_error": self.runs.root_error})
        payload = {
            "type": "ended",
            "episode_id": self.session.episode_id,
            "success": phase == "succeeded",
            "reason": self.runs.root_error or reason,
            "root_error": self.runs.root_error,
            "run_phase": phase,
        }
        if ended:
            payload.update(ended)
            payload["type"] = "ended"
            payload["reason"] = self.runs.root_error or ended.get("reason") or reason
        self.broadcast(payload)
        self.broadcast(self.snapshot())

    def _persist_run(self) -> None:
        try:
            self.runs.save(ROOT / "records" / "runs")
        except Exception as exc:
            self.log(f"run record failed: {exc}")

    def snapshot(self) -> dict:
        return {
            "type": "snapshot",
            "scene": self.scene_context(),
            "run": self.runs.public(),
            "loading": self.loading,
            "start_block": self.start_block_reason(),
            "execution_mode": self.execution_mode,
            "has_frame": self.last_payload is not None,
        }

    def _fid(self) -> int:
        if self.backend is None or not hasattr(self.backend, "env_step"):
            return 0
        return int(self.backend.env_step() or 0)

    def _wake(self) -> None:
        self.runs._wake()

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
                "telemetry": item.get("telemetry") or {},
                "errors": item.get("errors") or [],
                "tool_calls": item.get("tool_calls"),
                "content": None if item.get("tool_calls") else item.get("content"),
            }
        )
    return rows


APP = App()
PROBE = {
    "edge": {"status": "unknown", "detail": "", "model_ids": []},
    "cloud": {"status": "unknown", "detail": "", "model_ids": []},
}
PROBE_TIMEOUT_S = 1.5


def _probe_planner() -> None:
    """GET /models only. This does not send a generation request."""
    for key, profile in profiles().items():
        if not profile.get("available") or not profile.get("base_url"):
            PROBE[key] = {"status": "unconfigured", "detail": "未配置", "model_ids": []}
            continue
        url = str(profile["base_url"]).rstrip("/") + "/models"
        request = urllib.request.Request(url, headers={"Authorization": f"Bearer {profile.get('api_key') or 'EMPTY'}"})
        try:
            with urllib.request.urlopen(request, timeout=PROBE_TIMEOUT_S) as response:
                payload = json.loads(response.read().decode("utf-8"))
                ids = [
                    item.get("id")
                    for item in (payload.get("data") or [])
                    if isinstance(item, dict) and item.get("id")
                ]
                PROBE[key] = {
                    "status": "ok" if response.status == 200 else "down",
                    "detail": f"HTTP {response.status}",
                    "model_ids": ids,
                }
        except TimeoutError:
            PROBE[key] = {"status": "timeout", "detail": f"probe exceeded {PROBE_TIMEOUT_S}s", "model_ids": []}
        except Exception as exc:
            status = "timeout" if "timed out" in str(exc).lower() else "down"
            PROBE[key] = {"status": status, "detail": str(exc), "model_ids": []}


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
    await ws.send_str(json.dumps(APP.snapshot(), ensure_ascii=False))
    if APP.last_payload:
        await ws.send_bytes(APP.last_payload)
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
            elif kind == "get_snapshot":
                await ws.send_str(json.dumps(APP.snapshot(), ensure_ascii=False))
                if APP.last_payload:
                    await ws.send_bytes(APP.last_payload)
            elif kind == "ping":
                await ws.send_str(json.dumps({"type": "pong"}))
            elif kind == "load":
                threading.Thread(target=APP.load, args=(data,), daemon=True).start()
            elif kind in {"start", "restart"}:
                APP.start(kind, data, execute=True)
            elif kind == "plan_only":
                APP.start("start", data, execute=False)
            elif kind == "replay":
                await ws.send_str(json.dumps({
                    "type": "state",
                    "phase": "failed",
                    "detail": "开始只发起新的模型规划，不使用保存计划、人工计划或隐藏回退",
                }, ensure_ascii=False))
            elif kind == "set_planner":
                APP.select_profile(str(data.get("profile") or "edge"))
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
    print(
        f"sequential adapter mode={'live' if LIVE else 'dry'} pid={os.getpid()} port={PORT} "
        f"commit={STARTUP_COMMIT} static={STATIC_VERSION} execution=sync",
        flush=True,
    )
    start_background()
    web.run_app(app, host=HOST, port=PORT)


if __name__ == "__main__":
    main()
