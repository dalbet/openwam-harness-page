"""Episode records with small, bounded retention.

One JSON document per episode under ``records/runs`` (last 8 kept) and a
1 MiB journal with one backup. No image or full-observation dumps.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from common import config


MAX_RUNS = 200
JOURNAL_LIMIT = 1024 * 1024


def _rotate(path: Path, limit: int) -> None:
    try:
        if path.exists() and path.stat().st_size >= limit:
            backup = path.with_suffix(path.suffix + ".1")
            os.replace(path, backup)
    except OSError:
        pass


def append_journal(row: dict[str, Any]) -> None:
    config.ensure_dirs()
    path = config.RECORDS_DIR / "run_journal.jsonl"
    _rotate(path, JOURNAL_LIMIT)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, default=_default) + "\n")


def write_run(row: dict[str, Any]) -> Path:
    config.ensure_dirs()
    episode = str(row.get("episode_id") or f"ep-{int(time.time() * 1000)}")
    path = config.RUNS_DIR / f"{episode}.json"
    path.write_text(json.dumps(row, ensure_ascii=False, indent=2, default=_default), encoding="utf-8")
    _prune_runs()
    return path


def _prune_runs() -> None:
    rows = sorted(config.RUNS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for old in rows[MAX_RUNS:]:
        try:
            old.unlink()
        except OSError:
            pass


def _default(value: Any):
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
    except Exception:
        pass
    return str(value)
