"""Project paths and configuration for the embodied engine."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "config"
RECORDS_DIR = PROJECT_ROOT / "records"
RUNS_DIR = RECORDS_DIR / "runs"
LOGS_DIR = RECORDS_DIR / "logs"

AGENT_MODES = ("qwen", "d1", "d1_with_qwen_fallback")


def load_json(name: str) -> dict[str, Any]:
    path = CONFIG_DIR / name
    return json.loads(path.read_text(encoding="utf-8"))


def engine_config() -> dict[str, Any]:
    return load_json("engine.json")


def agent_mode() -> str:
    mode = str(engine_config().get("agent", {}).get("mode") or "qwen")
    return mode if mode in AGENT_MODES else "qwen"


def set_agent_mode(mode: str) -> None:
    """Persist the Agent mode atomically; callers gate this on an idle engine."""
    if mode not in AGENT_MODES:
        raise ValueError(f"unknown agent mode {mode!r}; expected one of {AGENT_MODES}")
    path = CONFIG_DIR / "engine.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data.setdefault("agent", {})["mode"] = mode
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def task_config() -> dict[str, Any]:
    return load_json("tasks.json")


def skill_config() -> dict[str, Any]:
    return load_json("skills.json")


def tasks_by_id() -> dict[int, dict[str, Any]]:
    return {int(row["task_id"]): row for row in task_config()["tasks"]}


def task_for(task_id: int) -> dict[str, Any]:
    row = tasks_by_id().get(int(task_id))
    if row is None:
        raise KeyError(f"unknown libero_goal task_id={task_id}")
    return row


def service_url(name: str) -> str:
    services = engine_config()["services"]
    row = services[name]
    return f"http://{row['host']}:{row['port']}"


def ensure_dirs() -> None:
    for path in (RECORDS_DIR, RUNS_DIR, LOGS_DIR):
        path.mkdir(parents=True, exist_ok=True)
