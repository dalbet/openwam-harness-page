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


class App:
    def __init__(self) -> None:
        self.human_plans = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
        self.catalog = self.human_plans
        self.suites = None
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
        self.planner = make_openai_client(PLANNER_URL, PLANNER_MODEL)

    def scenes(self) -> list[dict]:
        return [
            {"id": key, "label": value["goal"], "instruction": value["goal"]}
            for key, value in self.catalog.items()
        ]

    def health(self) -> dict:
        checkpoints = {"pi05": False, "sam3": False, "env": False}
        error = ""
        try:
            from live import require_resources

            checkpoints = require_resources()
            checkpoints = {"pi05": True, "sam3": True, "env": True, "paths": checkpoints}
        except Exception as exc:
            error = str(exc)
        components = []
        if self.backend is not None and hasattr(self.backend, "component_health"):
            components = self.backend.component_health()
        return {
            "type": "health",
            "mode": self.mode,
            "harness_live": LIVE,
            "port": PORT,
            "pid": os.getpid(),
            "planner_ok": _planner_ok(),
            "planner_url": PLANNER_URL,
            "checkpoints": checkpoints,
            "components": components,
            "env": "ready" if components and all(item["alive"] for item in components if item["name"] == "env") else "not_loaded",
            "vla": "ready" if components and all(item["alive"] for item in components if item["name"] == "vla") else "not_loaded",
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
        if not LIVE or self.backend is None or not hasattr(self.backend, "frames"):
            self.broadcast(
                {
                    "type": "state",
                    "phase": "failed",
                    "detail": "没有真实相机帧",
                    "reason": header.get("reason") or "no real frame",
                }
            )
            return
        agent, wrist = self.backend.frames()
        if not agent or not wrist:
            self.broadcast(
                {
                    "type": "state",
                    "phase": "failed",
                    "detail": "动作后没有新的相机帧",
                    "reason": "missing camera frame",
                }
            )
            return
        started = time.perf_counter()
        agent, src_a, (aw, ah) = _ui_jpeg(agent)
        wrist, src_w, (ww, wh) = _ui_jpeg(wrist)
        self.last_encode_s = round(time.perf_counter() - started, 3)
        header.update({
            "aw": aw,
            "ah": ah,
            "ww": ww,
            "wh": wh,
            "src_aw": src_a[0],
            "src_ah": src_a[1],
            "src_ww": src_w[0],
            "src_wh": src_w[1],
            "placeholder": False,
        })
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
            "goal_relation": goal_relation(task["language"]),
            "long_candidate": task["long_candidate"],
            "init_states": task["init_states"],
        }

    def start(self, kind: str, message: dict) -> None:
        if self.session.scene_id == "none" or (LIVE and self.backend is None):
            self.broadcast({"type": "load_failed", "detail": "load a scene first"})
            return
        if self.running and kind != "restart":
            self.broadcast({"type": "state", "phase": "failed", "detail": "already running"})
            return
        plan_source = message.get("plan_source") or "model"
        if plan_source not in {"model", "manual"}:
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
            args=(kind, plan_source, run_id, message.get("manual_scope") or "single"),
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

    def _run(self, kind: str, plan_source: str, run_id: int, manual_scope: str = "single") -> None:
        try:
            self._run_inner(kind, plan_source, run_id, manual_scope)
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

    def _run_inner(self, kind: str, plan_source: str, run_id: int, manual_scope: str = "single") -> None:
        if kind == "restart":
            self.session.cancel()
            self._wake()
            if not LIVE:
                self._refuse_dry()
                return
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
        if plan_source == "manual":
            plan = self._manual_plan(spec, manual_scope)
            self._emit_planner(
                plan_source="manual",
                summary=f"用户选择人工计划 scope={manual_scope}",
                output=json.dumps({"plan_id": plan.get("plan_id"), "semantic_step_ids": plan.get("semantic_step_ids"), "steps": [step["step_id"] for step in plan["steps"]]}, ensure_ascii=False),
                errors=[],
                repairs=0,
                usage={},
                elapsed_s=0,
            )
        else:
            self.broadcast({"type": "state", "phase": "planning", "detail": PLANNER_MODEL, "mode": "live", "plan_source": "model"})
            info = self.backend.scene_info()
            image_ids = [f"agentview:{info.get('env_step')}", f"wrist:{info.get('env_step')}"]
            from suites import _objects_from_names

            scene_text = json.dumps(
                {
                    "goal_relation": spec.get("goal_relation"),
                    "objects": _objects_from_names(info.get("object_names") or []),
                    "sim_object_names": info.get("object_names"),
                    "env_step": info.get("env_step"),
                    "image_ids": image_ids,
                    "planner_image_max_edge": 384,
                    "note": "image bytes are not inlined; ids only",
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
                    on_event=on_planner,
                )
            except Exception as exc:
                self._emit_planner(
                    plan_source=None,
                    summary=scene_text[:500],
                    output="",
                    errors=[{"code": "planner_request_failed", "received": str(exc), "expected": "one plan response"}],
                    repairs=0,
                    usage={},
                    elapsed_s=time.monotonic() - started,
                )
                self._fail_run(episode, f"PLANNING_FAILED {exc}")
                return
            elapsed = time.monotonic() - started
            usage = {}
            for attempt in result.get("attempts") or []:
                for key, value in (attempt.get("usage") or {}).items():
                    if isinstance(value, int):
                        usage[key] = usage.get(key, 0) + value
            output = ""
            attempts = result.get("attempts") or []
            if attempts:
                output = attempts[-1].get("content") or json.dumps(attempts[-1].get("tool_calls"), ensure_ascii=False)
            self._emit_planner(
                plan_source=result.get("plan_source"),
                summary=scene_text[:500],
                output=output,
                errors=result.get("errors") or [],
                repairs=result.get("repairs") or 0,
                usage=usage,
                elapsed_s=elapsed,
                response_kind=(attempts[-1].get("response_kind") if attempts else None),
                image_ids=image_ids,
            )
            if result["status"] != "ok" or result["plan"] is None:
                self._fail_run(episode, "PLANNING_FAILED")
                return
            plan = result["plan"]
        self.log(f"plan_source={plan.get('plan_source')} steps={len(plan['steps'])}")

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
        if DIAGNOSTIC_S > 0:
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
        )
        if result["episode_id"] != episode or result["scene_epoch"] != epoch:
            return
        if result["status"] == "discarded":
            return
        native = bool(result["native_success"]) if result["status"] == "completed" else False
        reason = result["reason"] or result["status"]
        phase = "failed" if not native or result["status"] != "completed" else "completed"
        self.broadcast_frame(
            {
                "eid": episode,
                "fid": self._fid(),
                "phase": phase,
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
            }
        )

    def _manual_plan(self, spec: dict, scope: str) -> dict:
        if scope == "long":
            key = f"{spec.get('suite')}_t{spec.get('task_id')}"
            if key not in self.human_plans or int(spec.get("seed") or 0) != 0:
                raise RuntimeError(f"没有 {key} seed 0 的长任务人工计划，不会改用模型计划。")
            plan = load_human_plan(PLAN_PATH, key)
            plan["plan_source"] = "manual"
            return plan
        from suites import single_manual_plan

        info = self.backend.scene_info()
        return single_manual_plan(
            spec["goal"],
            info.get("object_names") or [],
            suite=spec["suite"],
            task_id=int(spec["task_id"]),
            seed=int(spec["seed"]),
        )

    def _emit_planner(self, **payload) -> None:
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


def _planner_ok() -> bool:
    try:
        with urllib.request.urlopen(PLANNER_URL.rstrip("/") + "/models", timeout=2) as response:
            return response.status == 200
    except Exception:
        return False


APP = App()


async def index(_request):
    return web.FileResponse(ROOT / "static" / "index.html")


async def app_js(_request):
    return web.FileResponse(ROOT / "static" / "app.js")


async def hook_js(_request):
    return web.FileResponse(ROOT / "static" / "hook.js")


async def health(_request):
    return web.json_response(APP.health())


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
                try:
                    if APP.suites is None:
                        from suites import load_suites

                        APP.suites = load_suites()
                    await ws.send_str(json.dumps({"type": "suites", "suites": APP.suites}, ensure_ascii=False))
                except Exception as exc:
                    await ws.send_str(json.dumps({"type": "load_failed", "detail": f"读取 LIBERO 套件失败: {exc}"}, ensure_ascii=False))
            elif kind == "get_health":
                await ws.send_str(json.dumps(APP.health(), ensure_ascii=False))
            elif kind == "ping":
                await ws.send_str(json.dumps({"type": "pong"}))
            elif kind == "load":
                threading.Thread(target=APP.load, args=(data,), daemon=True).start()
            elif kind in {"start", "restart"}:
                APP.start(kind, data)
            elif kind == "stop":
                APP.stop()
            elif kind == "confirm":
                APP.confirm(data)
            elif kind == "client_perf":
                continue
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
    app.router.add_get("/static/app.js", app_js)
    app.router.add_get("/static/hook.js", hook_js)
    app.router.add_get("/ws", websocket)
    print(f"sequential adapter mode={'live' if LIVE else 'dry'} pid={os.getpid()} port={PORT}", flush=True)
    web.run_app(app, host=HOST, port=PORT)


if __name__ == "__main__":
    main()
