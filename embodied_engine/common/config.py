"""Project paths and configuration for the embodied engine.

The task table lives in ``config/tasks.json`` and is read through
``common.task_catalog`` so the UI, Agent and engine cannot drift apart.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from common import task_catalog

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
    return task_catalog.load()


def skill_config() -> dict[str, Any]:
    return load_json("skills.json")


def tasks_by_id() -> dict[int, dict[str, Any]]:
    return task_catalog.task_by_id()


def task_for(task_id: int) -> dict[str, Any]:
    return task_catalog.task_for(task_id)


def service_url(name: str) -> str:
    services = engine_config()["services"]
    row = services[name]
    return f"http://{row['host']}:{row['port']}"


def ensure_dirs() -> None:
    for path in (RECORDS_DIR, RUNS_DIR, LOGS_DIR):
        path.mkdir(parents=True, exist_ok=True)
