"""Live LIBERO backend. Imported only when the adapter is started with HARNESS_LIVE=1.

It reuses the existing env, Pi0.5, and SAM3 processes. It does not reset between
subtasks and does not call the planner. A missing env or VLA raises; it does not
switch to the dry backend.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from executor import ToolOutcome
from frame_bus import FrameBus

def _install_vla_experiment(primitives, controller) -> None:
    """Do not issue a prefetch that is discarded.

    The previous hook predicted once in the background and then called
    ``finish_predict(..., apply=False)``. The next chunk still called
    ``model.predict`` on the env thread. That extra prediction is not sent.
    Synchronous ``_vlm_chunk`` remains the only inference. Async stays closed
    until a chunk can be consumed with a matching observation and action index.
    """
    del primitives, controller


HARNESS = Path(__file__).resolve().parents[1]
RPENT = HARNESS / "RPent"


class LiveStartupError(RuntimeError):
    def __init__(self, message: str, *, log: str = "") -> None:
        super().__init__(message)
        self.log = log


def _latest_png(output_dir: Path, name: str) -> bytes | None:
    folder = output_dir / name
    if folder.is_dir():
        frames = sorted(folder.glob("*.png"))
        if frames:
            return frames[-1].read_bytes()
    if folder.is_file():
        return folder.read_bytes()
    return None


def require_resources() -> dict[str, str]:
    pi05 = Path(
        os.environ.get(
            "PI05_CHECKPOINT_PATH",
            str(HARNESS / "checkpoints" / "RLinf-Pi05-LIBERO-130-fullshot-SFT"),
        )
    )
    sam3 = Path(
        os.environ.get(
            "SAM3_CHECKPOINT_PATH",
            str(HARNESS / "checkpoints" / "sam3" / "sam3.pt"),
        )
    )
    env_code = RPENT / "robots" / "libero" / "robot_spec.py"
    missing = []
    if not pi05.exists():
        missing.append(f"VLA checkpoint missing: {pi05}")
    if not sam3.exists():
        missing.append(f"SAM3 checkpoint missing: {sam3}")
    if not env_code.is_file():
        missing.append(f"LIBERO env code missing: {env_code}")
    if missing:
        raise LiveStartupError(
            "真实模式缺少环境或 VLA，已停止，不会静默改跑干跑。" + " ".join(missing)
        )
    return {"pi05": str(pi05), "sam3": str(sam3), "env": str(env_code)}


class LiveBackend:
    def __init__(self, scene: dict, output_dir: Path, on_phase=None) -> None:
        self.scene = scene
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.bindings: dict[str, list[float]] = {}
        self._llm_calls = 0
        self._daemons = []
        self.toolkit = None
        self.frame_bus = None
        self.scene_epoch = 0
        self.resources = require_resources()
        self._model = None
        self._sam3 = None
        self.timings: dict = {}
        self._on_phase = on_phase
        self.execution_mode = "benchmark"
        self._flags = None
        self._open()

    def set_execution_mode(self, mode: str) -> None:
        from episode_service import publish_mode
        from free_task import EpisodeFlags

        if mode not in {"benchmark", "free"}:
            raise ValueError(mode)
        self.execution_mode = mode
        self._flags = EpisodeFlags(mode)
        publish_mode(mode)

    def _episode_flags(self):
        if self._flags is None or self._flags.mode != self.execution_mode:
            from free_task import EpisodeFlags

            self._flags = EpisodeFlags(self.execution_mode)
        return self._flags

    def _absorb_terminal(self) -> None:
        """Record the service's native-goal event. Do not clear the client latch.

        Benchmark mode still stops when the client has latched ``terminated``.
        Free mode reads ``env.native_goal`` from the env service. Truncation
        stays final. This method does not assign ``env.terminated`` and does
        not reset.
        """
        if self.toolkit is None:
            return
        env = self.toolkit.primitives.env
        truncated = bool(env.truncated)
        if self.execution_mode == "free":
            self._episode_flags().absorb(self._query_native_goal(), truncated)
            return
        self._episode_flags().absorb(bool(env.terminated), truncated)

    def _query_native_goal(self) -> bool:
        env = self.toolkit.primitives.env if self.toolkit is not None else None
        client = getattr(env, "_client", None)
        if client is None:
            return False
        try:
            payload = client.call("env.native_goal", timeout_s=2.0)
        except Exception:
            return False
        return bool(isinstance(payload, dict) and payload.get("native_goal"))

    def _ended_reason(self) -> str | None:
        env = self.toolkit.primitives.env if self.toolkit is not None else None
        if env is not None and bool(getattr(env, "truncated", False)):
            return "环境已到时间上限，不再 step"
        if self.execution_mode != "free" and env is not None and bool(getattr(env, "terminated", False)):
            return "环境已按原生规则结束，不再 step"
        if not self.episode_continuable():
            return "环境已结束，不再 step"
        return None

    def episode_continuable(self) -> bool:
        return self._episode_flags().continuable()

    def project_instance(self, object_ref: str, camera: str = "agentview") -> dict | None:
        """One pixel on the visible surface of this instance. Not a visual label."""
        if self.toolkit is None or camera != "agentview":
            return None
        sim = self.sim_state()
        objects = sim.get("objects") or {}
        target = (objects.get(object_ref) or {}).get("pos")
        if not isinstance(target, list) or len(target) < 3:
            return None
        others = []
        for name, item in objects.items():
            if name == object_ref or not isinstance(item, dict):
                continue
            pos = item.get("pos")
            if isinstance(pos, list) and len(pos) >= 3:
                others.append(pos)
        try:
            record = self.toolkit.state.latest_record()
            step = None if record is None else record.step_idx
        except Exception:
            return None
        world = None
        artifact = None
        for name in ("agentview_world_high.npz", "agentview_world.npz"):
            try:
                world = self.toolkit.state.load(name, step=step)
                artifact = name
                break
            except Exception:
                continue
        if world is None:
            return None
        from localize import pixel_for_instance

        found = pixel_for_instance(world, target, others)
        if not found:
            return None
        found["artifact"] = artifact
        found["object_ref"] = object_ref
        found["camera"] = camera
        return found

    def scene_key(self) -> tuple:
        from scene_request import require_int

        return (
            self.scene["suite"],
            require_int(self.scene.get("task_id"), "task_id"),
            require_int(self.scene.get("seed"), "seed"),
        )

    def _open(self) -> None:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("MUJOCO_GL", "egl")
        os.environ.setdefault("LIBERO_TYPE", "standard")
        os.environ.setdefault("PI05_CHECKPOINT_PATH", self.resources["pi05"])
        os.environ.setdefault("SAM3_CHECKPOINT_PATH", self.resources["sam3"])
        sys.path.insert(0, str(RPENT))
        try:
            from rpent.dashboard.events import NullDashboardEventSink
            from robots.libero import robot_spec
            self._robot_spec = robot_spec
        except Exception as exc:
            raise LiveStartupError(
                f"真实模式无法导入 LIBERO 环境: {exc}",
                log=self._log_tail(),
            ) from exc

        args = self._namespace()
        self._args = args
        events = NullDashboardEventSink()
        self._events = events
        timings: dict[str, float] = {}
        try:
            config = robot_spec._parse_config(args)
            self._config = config
            self._phase("service_start")
            started = time.perf_counter()
            model_daemons, model_kwargs = robot_spec._init_runtime(
                args, self.output_dir, events, {"vla", "sam3"}
            )
            timings["service_start_s"] = round(time.perf_counter() - started, 3)
            self._phase("env_create")
            started = time.perf_counter()
            env_daemons, env_kwargs = robot_spec._init_runtime(
                args, self.output_dir, events, {"env"}
            )
            timings["env_create_s"] = round(time.perf_counter() - started, 3)
            runtime_kwargs = {**model_kwargs, **env_kwargs}
            self._daemons = [*model_daemons, *env_daemons]
            self._phase("reset")
            self.toolkit, marks = _timed_toolkit(
                robot_spec, runtime_kwargs, events, config, self.output_dir
            )
            self.frame_bus = getattr(self.toolkit.primitives, "_display_bus", None)
            timings.update(marks)
            self._model = self.toolkit.primitives.model
            self._sam3 = self.toolkit.primitives._sam3_client
            timings["kind"] = "cold"
            self.timings = timings
            self._phase("first_frame")
        except Exception as exc:
            self.close()
            raise LiveStartupError(
                f"真实环境或 VLA 启动失败: {exc}",
                log=self._log_tail(),
            ) from exc
        dead = [item["name"] for item in self.component_health() if not item["alive"]]
        if dead:
            log = self._log_tail()
            self.close()
            raise LiveStartupError(f"组件已退出: {', '.join(dead)}", log=log)
        if not all(self.frames()):
            log = self._log_tail()
            self.close()
            raise LiveStartupError("reset 之后没有新的双相机帧", log=log)

    def execute(self, action: str, args: dict) -> ToolOutcome:
        if action == "pi0_doubled":
            return ToolOutcome(
                executed=False,
                error="pi0_doubled is not a sequential action",
                real=False,
                error_class="plan",
            )
        if action == "reset":
            return ToolOutcome(
                executed=False,
                error="reset between subtasks is rejected",
                real=False,
                error_class="plan",
            )
        ended = self._ended_reason()
        if ended:
            return ToolOutcome(
                executed=False,
                error=ended,
                real=False,
                error_class="plan",
            )
        try:
            tool_args = tool_arguments(args)
            result = self.toolkit.execute_tool(action, tool_args)
        except Exception as exc:
            parsed = interpret_exception(exc)
            return ToolOutcome(
                executed=parsed["executed"],
                error=parsed["error"],
                real=True,
                error_class=parsed["error_class"],
                category=parsed["category"],
                env_step=self.env_step(),
            )
        data = result.data if isinstance(result.data, dict) else {"result": result.data}
        if action == "pi0_pick":
            from chunk_trace import take_chunk_trace, take_grasp_samples

            data = dict(data)
            data["chunks"] = take_chunk_trace(self.toolkit.primitives)
            data["samples"] = take_grasp_samples(self.toolkit.primitives)
            data["env_action_count"] = int(getattr(self.toolkit.primitives, "_env_action_count", 0))
            data["record_note"] = "states.json step_idx counts tool records, not env actions inside a chunk"
        if action == "pi0_pick":
            data = attach_saved_pick_result(data, _saved_record_result(self.toolkit))
        parsed = interpret_tool_data(action, data)
        self._absorb_terminal()
        env = self.toolkit.primitives.env
        eef = None
        state = data.get("state") if isinstance(data.get("state"), dict) else {}
        if isinstance(state.get("robot0_eef_pos"), list):
            eef = [float(v) for v in state["robot0_eef_pos"][:3]]
        return ToolOutcome(
            executed=parsed["executed"],
            error=parsed["error"],
            grasp_signal=parsed["grasp_signal"],
            native_terminated=bool(env.terminated),
            native_truncated=bool(env.truncated),
            xyz=parsed["xyz"],
            eef_pos=eef,
            raw={k: v for k, v in data.items() if k != "images"},
            real=True,
            error_class=parsed["error_class"],
            category=parsed["category"],
            env_step=self.env_step(),
        )

    def observe(self) -> dict:
        agent, wrist = self.frames()
        record = self.toolkit.state.latest_record() if self.toolkit is not None else None
        step = record.step_idx if record is not None else None
        return {
            "observation_id": f"live-{step}-{time.time_ns()}",
            "scene_epoch": self.scene_epoch,
            "env_step": step,
            "timestamp": time.time(),
            "cameras": {
                "agentview": {"id": f"agentview:{step}", "bytes": len(agent or b"")},
                "wrist": {"id": f"wrist:{step}", "bytes": len(wrist or b"")},
            },
            "sim": self.sim_state(),
            "named_poses": self.named_poses(),
            "llm": False,
            "fresh": True,
        }

    def named_poses(self) -> dict:
        """Read-only site and body snapshot. A missing service is an error, not a pose."""
        env = self.toolkit.primitives.env if self.toolkit is not None else None
        client = getattr(env, "_client", None)
        if client is None:
            return {"bodies": {}, "sites": {}, "error": "state_service_unavailable"}
        try:
            payload = client.call("env.named_poses", timeout_s=2.0)
        except Exception as exc:
            return {"bodies": {}, "sites": {}, "error": str(exc)}
        if not isinstance(payload, dict):
            return {"bodies": {}, "sites": {}, "error": "named pose payload is not an object"}
        return payload

    def sim_state(self) -> dict:
        env = self.toolkit.primitives.env
        sim = dict(sim_sample(env)["sim"])
        sim["terminated"] = bool(env.terminated)
        sim["truncated"] = bool(env.truncated)
        return sim

    def env_step(self) -> int | None:
        if self.toolkit is None:
            return None
        record = self.toolkit.state.latest_record()
        return None if record is None else int(record.step_idx)

    def native_success(self) -> bool:
        if self.toolkit is None:
            return False
        flags = self._episode_flags()
        if flags.truncated:
            return False
        if self.execution_mode == "free":
            return bool(flags.native_goal_met)
        env = self.toolkit.primitives.env
        # env.terminated sticks on the official LIBERO success check.
        # truncated is a horizon/budget stop and is not success.
        return bool(env.terminated) and not bool(env.truncated) and bool(self.toolkit.solved())

    def frames(self) -> tuple[bytes | None, bytes | None]:
        return (
            _latest_png(self.output_dir, "agentview_high.png"),
            _latest_png(self.output_dir, "wrist_high.png"),
        )

    def capture_display(self) -> dict | None:
        """Same raw-camera view used between chunks. No extra render."""
        if self.toolkit is None:
            return None
        started = time.perf_counter()
        from display_view import encode_display, prepare_camera, prepare_display_pair

        env = self.toolkit.primitives.env
        try:
            agent = prepare_camera(
                env.render_camera(camera_name="agentview", height=512, width=512, depth=False),
                source="render_camera.agentview",
            )
            wrist = prepare_camera(
                env.render_camera(camera_name="robot0_eye_in_hand", height=512, width=512, depth=False),
                source="render_camera.robot0_eye_in_hand",
            )
        except Exception:
            agent, wrist = prepare_display_pair(env.raw_obs())
        capture_ms = round((time.perf_counter() - started) * 1000, 2)
        encode_started = time.perf_counter()
        agent_jpg, agent_meta = encode_display(agent)
        wrist_jpg, wrist_meta = encode_display(wrist)
        return {
            "agent": agent_jpg,
            "wrist": wrist_jpg,
            "capture_ms": capture_ms,
            "encode_ms": round((time.perf_counter() - encode_started) * 1000, 2),
            "agent_meta": agent_meta,
            "wrist_meta": wrist_meta,
            "stage": "tool_end",
        }

    def hold_for_grasp(self, steps: int = 4) -> list[dict]:
        """Advance physics while holding the eef and the current gripper command.

        ``+1`` keeps a closed gripper. This is not a repeated ``set_gripper`` tool call.
        """
        if self.toolkit is None or not self.episode_continuable():
            return []
        import numpy as np

        env = self.toolkit.primitives.env
        prim = self.toolkit.primitives
        rows = []
        for index in range(max(0, int(steps))):
            raw = env.raw_obs()
            grip = raw.get("robot0_gripper_qpos")
            opening = None if grip is None else abs(float(grip[0])) + abs(float(grip[1]))
            command = 1.0 if opening is None or opening < 0.05 else -1.0
            action = np.zeros(7, dtype=np.float32)
            action[6] = command
            env.step(action)
            version = int(getattr(prim, "_env_action_count", 0)) + 1
            prim._env_action_count = version
            sample = sim_sample(env)
            sample["observation_id"] = f"hold-{version}"
            sample["timestamp"] = time.time()
            sample["env_action_count"] = version
            sample["hold_step"] = index + 1
            sample["hold_gripper_command"] = command
            rows.append(sample)
            publish_raw_display(prim, {
                "primitive": "hold",
                "observation_id": sample["observation_id"],
                "timestamp": sample["timestamp"],
                "env_action_count": version,
                "stage": "hold",
            })
            self._absorb_terminal()
            if not self.episode_continuable():
                break
        return rows

    def scene_info(self) -> dict:
        agent, wrist = self.frames()
        if not agent or not wrist:
            raise LiveStartupError("加载后没有新的相机帧", log=self._log_tail())
        record = self.toolkit.state.latest_record()
        state = getattr(record, "state", {}) or {}
        extras = getattr(record, "extras", {}) or {}
        env = self.toolkit.primitives.env
        return {
            "suite": self.scene.get("suite"),
            "task_id": self.scene.get("task_id"),
            "seed": self.scene.get("seed"),
            "task_language": extras.get("task_language") or self.scene.get("goal"),
            "object_names": state.get("object_names") or [],
            "eef_pos": state.get("robot0_eef_pos"),
            "gripper_qpos": state.get("robot0_gripper_qpos"),
            "env_step": None if record is None else record.step_idx,
            "terminated": bool(env.terminated),
            "truncated": bool(env.truncated),
            "agent_bytes": len(agent),
            "wrist_bytes": len(wrist),
            "placeholder": False,
            "components": self.component_health(),
            "timings": self.timings,
            "image_role": {
                "sim_camera": "256x256 observation used by Pi0.5; this path is unchanged",
                "grounding_png": "1024x1024 agentview_high/wrist_high for segmentation and back-projection",
                "ui_jpeg_max_edge": 512,
                "planner_image_max_edge": 384,
            },
        }

    def _phase(self, name: str) -> None:
        if self._on_phase is not None:
            self._on_phase(name)

    def component_health(self) -> list[dict]:
        rows = []
        for daemon in self._daemons or []:
            code = daemon.poll()
            rows.append({"name": daemon.name, "alive": code is None, "exit_code": code})
        return rows

    def _namespace(self):
        return SimpleNamespace(
            suite=self.scene["suite"],
            task=self.scene["task_id"],
            seed=self.scene["seed"],
            libero_type="standard",
            max_episode_steps=10000,
            cuda_device=int(os.environ.get("SEQUENTIAL_CUDA_DEVICE", os.environ.get("CUDA_VISIBLE_DEVICES", "0")).split(",")[0]),
            planner="api",
            env_endpoint=None,
            vla_endpoint=None,
            sam3_endpoint=None,
            molmo_endpoint=None,
            collect_flywheel_data=False,
            enable_direct_action=False,
            memory_dir=None,
            memory_profile="hf",
            output_dir=self.output_dir,
            explore=False,
            explore_sessions=1,
        )

    def warm_reset(self) -> dict:
        """Reset the same suite/task/seed without restarting env, VLA, or SAM3."""
        from robots.libero.tools import dump_state

        started = time.perf_counter()
        from vla_async import get_controller

        get_controller().clear("reset")
        self.toolkit._solved = False
        self._flags = None
        self.toolkit.primitives.reset()
        reset_s = round(time.perf_counter() - started, 3)
        started = time.perf_counter()
        dump_state(self.toolkit.primitives, self.toolkit.state, log=None)
        frame_s = round(time.perf_counter() - started, 3)
        self.bindings = {}
        self.timings = {
            "kind": "warm",
            "service_start_s": 0.0,
            "env_create_s": 0.0,
            "reset_s": reset_s,
            "first_frame_s": frame_s,
        }
        return self.timings

    def retarget(self, scene: dict, output_dir: Path) -> dict:
        """Replace the env for a different task. VLA and SAM3 stay up."""
        self._stop_named("env_server")
        bus = getattr(self, "frame_bus", None)
        if bus is not None:
            bus.close()
            self.frame_bus = None
        try:
            if self.toolkit is not None:
                self.toolkit.close()
        except Exception:
            pass
        self.scene = scene
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.bindings = {}
        args = self._namespace()
        self._args = args
        started = time.perf_counter()
        env_daemons, env_kwargs = self._robot_spec._init_runtime(
            args, self.output_dir, self._events, {"env"}
        )
        env_s = round(time.perf_counter() - started, 3)
        env_kwargs["model"] = self._model
        env_kwargs["sam3_client"] = self._sam3
        self.toolkit, marks = _timed_toolkit(
            self._robot_spec, env_kwargs, self._events, self._config, self.output_dir
        )
        self.frame_bus = getattr(self.toolkit.primitives, "_display_bus", None)
        self._daemons.extend(env_daemons)
        self.timings = {"kind": "retarget", "service_start_s": 0.0, "env_create_s": env_s, **marks}
        if not all(self.frames()):
            raise LiveStartupError("切换任务后没有新的双相机帧", log=self._log_tail())
        return self.timings

    def _stop_named(self, name: str) -> None:
        kept = []
        for daemon in self._daemons or []:
            if daemon.name == name:
                try:
                    daemon.stop()
                except Exception:
                    pass
            else:
                kept.append(daemon)
        self._daemons = kept

    def close(self) -> None:
        bus = getattr(self, "frame_bus", None)
        if bus is not None:
            bus.close()
            self.frame_bus = None
        for daemon in self._daemons or []:
            try:
                daemon.stop()
            except Exception:
                pass
        self._daemons = []

    def _log_tail(self) -> str:
        parts = []
        for name in ("env_server.log", "vla_server.log", "sam3_server.log"):
            path = self.output_dir / name
            if not path.is_file():
                continue
            text = path.read_text(encoding="utf-8", errors="replace")[-1500:]
            parts.append(f"--- {name} ---\n{text}")
        return "\n".join(parts)


def _timed_toolkit(robot_spec, runtime_kwargs, events, config, output_dir):
    from robots.libero.tools import LiberoPrimitives
    import robots.libero.tools as tools

    marks: dict[str, float] = {}
    orig_reset = LiberoPrimitives.reset
    orig_dump = tools.dump_state

    def reset(prim):
        started = time.perf_counter()
        try:
            return orig_reset(prim)
        finally:
            marks["reset_s"] = round(time.perf_counter() - started, 3)

    def dump(*args, **kwargs):
        started = time.perf_counter()
        try:
            return orig_dump(*args, **kwargs)
        finally:
            marks["first_frame_s"] = round(time.perf_counter() - started, 3)

    LiberoPrimitives.reset = reset
    tools.dump_state = dump
    try:
        toolkit = robot_spec.get_toolkit(
            runtime_kwargs=runtime_kwargs,
            dashboard_events=events,
            config=config,
            mode="evaluation",
            state_output_dir=output_dir,
        )
    finally:
        LiberoPrimitives.reset = orig_reset
        tools.dump_state = orig_dump
    from chunk_trace import attach_chunk_trace, set_chunk_hook

    attach_chunk_trace(toolkit.primitives)
    # Discard-only prefetch is not installed. Execution stays synchronous.
    bus = FrameBus()
    toolkit.primitives._display_bus = bus

    def on_chunk(primitives, obs, row) -> None:
        sample = sim_sample(primitives.env)
        sample["observation_id"] = row.get("observation_id")
        sample["timestamp"] = time.time()
        sample["env_action_count"] = row.get("env_actions_after")
        primitives._grasp_samples = list(getattr(primitives, "_grasp_samples", []))
        primitives._grasp_samples.append(sample)
        _note_vla_image(primitives, obs)
        publish_raw_display(
            primitives,
            {
                "primitive": "pi0_pick",
                "observation_id": row.get("observation_id"),
                "timestamp": sample["timestamp"],
                "env_action_count": row.get("env_actions_after"),
                "stage": "chunk",
            },
        )

    set_chunk_hook(on_chunk)
    return toolkit, marks


def sim_sample(env) -> dict:
    """Read the current sim truth on the thread that just stepped. No render."""
    raw = env.raw_obs()
    objects = {}
    contact = {}
    for key, value in raw.items():
        if not isinstance(key, str):
            continue
        if "contact" in key or "touch" in key:
            contact[key] = _plain(value)
            continue
        if not key.endswith("_pos") or "robot0" in key or "to_robot" in key:
            continue
        try:
            pos = [float(item) for item in list(value)[:3]]
        except (TypeError, ValueError):
            continue
        if len(pos) == 3:
            objects[key[:-4]] = {"pos": pos, "identified": True}
    grip = raw.get("robot0_gripper_qpos")
    opening = None
    if grip is not None:
        opening = abs(float(grip[0])) + abs(float(grip[1]))
    eef = raw.get("robot0_eef_pos")
    return {
        "sim": {
            "objects": objects,
            "eef_pos": None if eef is None else [float(item) for item in list(eef)[:3]],
            "gripper_opening_m": opening,
            "contact": contact or None,
            "evidence_source": "sim_state",
        }
    }


def _plain(value):
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return value.reshape(-1)[:8].astype(float).tolist()
    except Exception:
        pass
    if isinstance(value, (int, float, str, bool)):
        return value
    return None


def publish_raw_display(primitives, meta: dict) -> None:
    """Publish the current raw cameras. Caller must already own the env thread."""
    bus = getattr(primitives, "_display_bus", None)
    if bus is None:
        return
    started = time.perf_counter()
    try:
        from display_view import prepare_display_pair

        agent, wrist = prepare_display_pair(primitives.env.raw_obs())
    except Exception:
        bus.skipped = getattr(bus, "skipped", 0) + 1
        return
    meta = dict(meta)
    meta["capture_ms"] = round((time.perf_counter() - started) * 1000, 2)
    meta["stage"] = meta.get("stage") or "raw"
    bus.publish(agent, wrist, meta)


def _note_vla_image(primitives, obs) -> None:
    """Record that the chunk obs is the VLA view, and do not publish it."""
    if getattr(primitives, "_vla_image_noted", False) or not isinstance(obs, dict):
        return
    image = obs.get("main_images")
    if image is None:
        return
    try:
        import numpy as np

        array = np.asarray(image)
    except Exception:
        return
    primitives._vla_image_noted = True
    primitives._vla_image_shape = [int(value) for value in array.shape]


def _find_xyz(data: dict):
    found = []

    def walk(value) -> None:
        if isinstance(value, dict):
            if isinstance(value.get("world_xyz"), list) and len(value["world_xyz"]) >= 3:
                found.append([float(v) for v in value["world_xyz"][:3]])
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(data)
    return found[0] if found else None


def tool_arguments(args: dict) -> dict:
    """Instance ids and prompt labels stay off the tool call. The tool receives one prompt."""
    return {
        key: value
        for key, value in args.items()
        if key not in {"object_ref", "grounding_query", "vla_prompt"}
    }


def attach_saved_pick_result(data: dict, saved: dict | None) -> dict:
    """Keep the tool body and the post-tool observation as separate fields.

    ``success`` is also copied to the top level so grasp_signal can read it.
    ``terminated`` stays inside ``tool_result`` and is not the call status.
    """
    tool_keys = ("success", "chunks_used", "peak_lift_m", "min_gripper_opening", "final_gripper_opening", "terminated", "truncated")
    source = saved if isinstance(saved, dict) and "success" in saved else data
    tool = {key: source[key] for key in tool_keys if isinstance(source, dict) and key in source}
    observation = {
        key: value
        for key, value in data.items()
        if key not in tool_keys and key not in {"tool_result", "observation"}
    }
    merged = dict(observation)
    if tool:
        merged["tool_result"] = tool
        if "success" in tool:
            merged["success"] = tool["success"]
    merged["observation"] = observation
    return merged


def _saved_record_result(toolkit) -> dict | None:
    try:
        record = toolkit.state.latest_record()
    except Exception:
        return None
    saved = getattr(record, "result", None)
    return saved if isinstance(saved, dict) else None


def interpret_exception(exc: BaseException) -> dict:
    """Keep the exception text. Do not rename it and do not report success."""
    return {
        "executed": False,
        "error": str(exc),
        "error_class": "service",
        "category": "tool_exception",
        "xyz": None,
        "grasp_signal": None,
    }


def interpret_tool_data(action: str, data: dict) -> dict:
    """Split a finished tool payload from a failed call.

    ``success`` on pi0_pick is grasp evidence. A segment ``found: false`` is a
    localization miss. Neither one is a failed tool call, and neither one is
    read from ``env.terminated``.
    """
    error = data.get("error")
    error_text = None if error in (None, "") else str(error)
    if data.get("code") == "segment_artifact_save_failed" or (
        error_text is not None and error_text.startswith("failed to persist")
    ):
        return _call_failure(error_text or "segment artifact was not saved", "tool_exception", "physical")
    if action == "segment" and data.get("found") is False and error_text and _error_class(error_text) != "service":
        return {
            "executed": True,
            "error": error_text,
            "error_class": None,
            "category": "subtask_incomplete",
            "xyz": None,
            "grasp_signal": None,
        }
    if error_text:
        kind = _error_class(error_text)
        if kind == "service":
            return _call_failure(error_text, "service", "service")
        if _parameter_error(error_text):
            return _call_failure(error_text, "parameter", "plan")
        return _call_failure(error_text, "tool_exception", "physical")
    grasp = None
    if action == "pi0_pick" and "success" in data:
        grasp = bool(data.get("success"))
    return {
        "executed": True,
        "error": None,
        "error_class": None,
        "category": None,
        "xyz": _find_xyz(data),
        "grasp_signal": grasp,
    }


def _call_failure(error: str, category: str, error_class: str) -> dict:
    return {
        "executed": False,
        "error": error,
        "error_class": error_class,
        "category": category,
        "xyz": None,
        "grasp_signal": None,
    }


def _parameter_error(error: str) -> bool:
    lowered = error.lower()
    markers = ("unknown", "invalid", "needs exactly", "unexpected keyword", "missing")
    return any(marker in lowered for marker in markers)


def _error_class(error: str) -> str:
    text = error.lower()
    markers = ("connection", "timeout", "timed out", "unavailable", "rpc", "cuda", "out of memory", "refused")
    if any(marker in text for marker in markers):
        return "service"
    return "physical"
