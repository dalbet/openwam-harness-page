"""Live LIBERO backend. Imported only when the adapter is started with HARNESS_LIVE=1.

It reuses the existing env, Pi0.5, and SAM3 processes. It does not reset between
subtasks and does not call the planner.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

from executor import ToolOutcome

HARNESS = Path(__file__).resolve().parents[1]
RPENT = HARNESS / "RPent"


def _latest_png(output_dir: Path, name: str) -> bytes | None:
    folder = output_dir / name
    if folder.is_dir():
        frames = sorted(folder.glob("*.png"))
        if frames:
            return frames[-1].read_bytes()
    if folder.is_file():
        return folder.read_bytes()
    return None


class LiveBackend:
    def __init__(self, scene: dict, output_dir: Path) -> None:
        self.scene = scene
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.bindings: dict[str, list[float]] = {}
        self._llm_calls = 0
        self._daemons = []
        self.toolkit = None
        self._open()

    def _open(self) -> None:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("MUJOCO_GL", "egl")
        os.environ.setdefault("LIBERO_TYPE", "standard")
        os.environ.setdefault(
            "PI05_CHECKPOINT_PATH",
            str(HARNESS / "checkpoints" / "RLinf-Pi05-LIBERO-130-fullshot-SFT"),
        )
        os.environ.setdefault(
            "SAM3_CHECKPOINT_PATH",
            str(HARNESS / "checkpoints" / "sam3" / "sam3.pt"),
        )
        sys.path.insert(0, str(RPENT))
        from rpent.dashboard.events import NullDashboardEventSink
        from robots.libero import robot_spec

        args = SimpleNamespace(
            suite=self.scene["suite"],
            task=self.scene["task_id"],
            seed=self.scene["seed"],
            libero_type="standard",
            max_episode_steps=10000,
            cuda_device=int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]),
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
        events = NullDashboardEventSink()
        config = robot_spec._parse_config(args)
        self._daemons, runtime_kwargs = robot_spec._init_runtime(
            args, self.output_dir, events, {"env", "vla", "sam3"}
        )
        self.toolkit = robot_spec.get_toolkit(
            runtime_kwargs=runtime_kwargs,
            dashboard_events=events,
            config=config,
            mode="evaluation",
            state_output_dir=self.output_dir,
        )

    def execute(self, action: str, args: dict) -> ToolOutcome:
        if action == "pi0_doubled":
            return ToolOutcome(executed=False, error="pi0_doubled is not a sequential action")
        result = self.toolkit.execute_tool(action, args)
        data = result.data if isinstance(result.data, dict) else {"result": result.data}
        if data.get("error"):
            return ToolOutcome(executed=False, error=str(data.get("error")), raw=data)
        xyz = _find_xyz(data)
        grasp = data.get("success") if action == "pi0_pick" else None
        native = bool(self.toolkit.solved())
        eef = None
        state = data.get("state") if isinstance(data.get("state"), dict) else {}
        if isinstance(state.get("robot0_eef_pos"), list):
            eef = [float(v) for v in state["robot0_eef_pos"][:3]]
        return ToolOutcome(
            executed=True,
            grasp_signal=None if grasp is None else bool(grasp),
            native_terminated=native,
            xyz=xyz,
            eef_pos=eef,
            raw={k: v for k, v in data.items() if k != "images"},
        )

    def observe(self) -> dict:
        return {"agent": _latest_png(self.output_dir, "agentview_high.png"), "llm": False}

    def native_success(self) -> bool:
        return bool(self.toolkit and self.toolkit.solved())

    def frames(self) -> tuple[bytes | None, bytes | None]:
        return (
            _latest_png(self.output_dir, "agentview_high.png"),
            _latest_png(self.output_dir, "wrist_high.png"),
        )

    def close(self) -> None:
        for daemon in self._daemons or []:
            try:
                daemon.stop()
            except Exception:
                pass
        self._daemons = []


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
