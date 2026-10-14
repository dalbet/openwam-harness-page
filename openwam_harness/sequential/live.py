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
        self.scene_epoch = 0
        self.resources = require_resources()
        self._model = None
        self._sam3 = None
        self.timings: dict = {}
        self._on_phase = on_phase
        self._open()

    def scene_key(self) -> tuple:
        return (self.scene["suite"], int(self.scene["task_id"]), int(self.scene["seed"]))

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
        try:
            tool_args = {key: value for key, value in args.items() if key != "object_ref"}
            result = self.toolkit.execute_tool(action, tool_args)
        except Exception as exc:
            return ToolOutcome(
                executed=False,
                error=str(exc),
                real=True,
                error_class="service",
                env_step=self.env_step(),
            )
        data = result.data if isinstance(result.data, dict) else {"result": result.data}
        if action == "pi0_pick":
            from chunk_trace import take_chunk_trace

            data = dict(data)
            data["chunks"] = take_chunk_trace(self.toolkit.primitives)
            data["env_action_count"] = int(getattr(self.toolkit.primitives, "_env_action_count", 0))
            data["record_note"] = "states.json step_idx counts tool records, not env actions inside a chunk"
        if data.get("error"):
            error = str(data.get("error"))
            return ToolOutcome(
                executed=False,
                error=error,
                real=True,
                error_class=_error_class(error),
                env_step=self.env_step(),
                raw={k: v for k, v in data.items() if k != "images"},
            )
        env = self.toolkit.primitives.env
        xyz = _find_xyz(data)
        grasp = data.get("success") if action == "pi0_pick" else None
        eef = None
        state = data.get("state") if isinstance(data.get("state"), dict) else {}
        if isinstance(state.get("robot0_eef_pos"), list):
            eef = [float(v) for v in state["robot0_eef_pos"][:3]]
        return ToolOutcome(
            executed=True,
            grasp_signal=None if grasp is None else bool(grasp),
            native_terminated=bool(env.terminated),
            native_truncated=bool(env.truncated),
            xyz=xyz,
            eef_pos=eef,
            raw={k: v for k, v in data.items() if k != "images"},
            real=True,
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
            "llm": False,
            "fresh": True,
        }

    def sim_state(self) -> dict:
        env = self.toolkit.primitives.env
        raw = env.raw_obs()
        objects = {}
        for key, value in raw.items():
            if not (isinstance(key, str) and key.endswith("_pos")):
                continue
            if "robot0" in key or "to_robot" in key:
                continue
            try:
                pos = [float(item) for item in list(value)[:3]]
            except (TypeError, ValueError):
                continue
            if len(pos) != 3:
                continue
            objects[key[:-4]] = {"pos": pos, "identified": True}
        grip = raw.get("robot0_gripper_qpos")
        opening = None
        if grip is not None:
            opening = abs(float(grip[0])) + abs(float(grip[1]))
        eef = raw.get("robot0_eef_pos")
        return {
            "objects": objects,
            "eef_pos": None if eef is None else [float(item) for item in list(eef)[:3]],
            "gripper_opening_m": opening,
            "terminated": bool(env.terminated),
            "truncated": bool(env.truncated),
            "evidence_source": "sim_state_assisted",
        }

    def env_step(self) -> int | None:
        if self.toolkit is None:
            return None
        record = self.toolkit.state.latest_record()
        return None if record is None else int(record.step_idx)

    def native_success(self) -> bool:
        if self.toolkit is None:
            return False
        env = self.toolkit.primitives.env
        # env.terminated sticks on the official LIBERO success check.
        # truncated is a horizon/budget stop and is not success.
        return bool(env.terminated) and not bool(env.truncated) and bool(self.toolkit.solved())

    def frames(self) -> tuple[bytes | None, bytes | None]:
        return (
            _latest_png(self.output_dir, "agentview_high.png"),
            _latest_png(self.output_dir, "wrist_high.png"),
        )

    def scene_info(self) -> dict:
        agent, wrist = self.frames()
        if not agent or not wrist:
            raise LiveStartupError("加载后没有新的相机帧", log=self._log_tail())
        record = self.toolkit.state.latest_record()
        state = getattr(record, "state", {}) or {}
        extras = getattr(record, "extras", {}) or {}
        env = self.toolkit.primitives.env
        return {
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
        self.toolkit._solved = False
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
        self._stop_named("env")
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
    from chunk_trace import attach_chunk_trace

    attach_chunk_trace(toolkit.primitives)
    return toolkit, marks


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


def _error_class(error: str) -> str:
    text = error.lower()
    markers = ("connection", "timeout", "timed out", "unavailable", "rpc", "cuda", "out of memory", "refused")
    if any(marker in text for marker in markers):
        return "service"
    return "physical"
