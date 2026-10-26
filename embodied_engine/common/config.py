"""Project paths and configuration for the embodied engine."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "config"
RECORDS_DIR = PROJECT_ROOT / "records"
RUNS_DIR = RECORDS_DIR / "runs"
LOGS_DIR = RECORDS_DIR / "logs"


def load_json(name: str) -> dict[str, Any]:
    path = CONFIG_DIR / name
    return json.loads(path.read_text(encoding="utf-8"))


def engine_config() -> dict[str, Any]:
    return load_json("engine.json")


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
