"""Single retention policy for regenerable runtime output.

Normal mode keeps short text and the latest observation steps.
SEQUENTIAL_DEBUG=1 also stores display JPEG and more observation steps.
This module is the only cleaner. It does not delete checkpoints or virtualenvs.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

JOURNAL_MAX_BYTES = 1_000_000
SERVICE_LOG_MAX_BYTES = 2_000_000
RUN_SUMMARIES = 8
EPISODE_DIRS = 3
OBSERVATION_STEPS = 2
DEBUG_OBSERVATION_STEPS = 20


def debug_enabled() -> bool:
    return os.environ.get("SEQUENTIAL_DEBUG", "0") == "1"


def retention_enabled() -> bool:
    """The serving process turns this on. Imported unit tests must not."""
    return os.environ.get("SEQUENTIAL_PRUNE_EPISODES", "0") == "1"


def observation_steps() -> int:
    return DEBUG_OBSERVATION_STEPS if debug_enabled() else OBSERVATION_STEPS


def append_jsonl(path: Path, row: dict, max_bytes: int = JOURNAL_MAX_BYTES) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(row, ensure_ascii=False) + "\n"
    rotate_file(path, max_bytes)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line)


def rotate_file(path: Path, max_bytes: int = SERVICE_LOG_MAX_BYTES) -> None:
    """Keep one backup when the current file is already at the cap."""
    if not path.is_file():
        return
    if path.stat().st_size < max_bytes:
        return
    backup = path.with_name(path.name + ".1")
    if backup.exists():
        backup.unlink()
    path.replace(backup)


def prune_summaries(directory: Path, keep: int = RUN_SUMMARIES) -> None:
    if not directory.is_dir():
        return
    files = [item for item in directory.glob("*.json") if item.is_file()]
    files.sort(key=lambda item: item.stat().st_mtime, reverse=True)
    for old in files[keep:]:
        old.unlink()


def save_summary(directory: Path, name: str, payload: dict, keep: int = RUN_SUMMARIES) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    prune_summaries(directory, keep)


def prune_episode_dirs(parent: Path, keep: int = EPISODE_DIRS, protect: list[Path] | None = None) -> None:
    if not parent.is_dir():
        return
    protected = {path.resolve() for path in (protect or [])}
    dirs = [item for item in parent.iterdir() if item.is_dir()]
    dirs.sort(key=lambda item: item.stat().st_mtime, reverse=True)
    for old in dirs[keep:]:
        if old.resolve() in protected:
            continue
        _remove_tree(old)


def prune_observation_steps(episode_dir: Path, keep: int | None = None) -> None:
    """Drop old numbered artifact files. states.json and server logs stay."""
    if not episode_dir.is_dir():
        return
    keep_n = observation_steps() if keep is None else keep
    for child in episode_dir.iterdir():
        if not child.is_dir():
            continue
        if child.name == "display" and not debug_enabled():
            for item in child.iterdir():
                if item.is_file():
                    item.unlink()
            continue
        numbered = []
        for item in child.iterdir():
            if item.is_file() and item.stem.isdigit():
                numbered.append((int(item.stem), item))
        if len(numbered) <= keep_n:
            continue
        numbered.sort()
        for _index, path in numbered[:-keep_n]:
            path.unlink()


def _remove_tree(path: Path) -> None:
    if path.is_symlink():
        path.unlink()
        return
    if path.is_file():
        path.unlink()
        return
    for child in path.iterdir():
        _remove_tree(child)
    path.rmdir()


def main() -> None:
    import sys

    if len(sys.argv) == 4 and sys.argv[1] == "rotate":
        rotate_file(Path(sys.argv[2]), int(sys.argv[3]))
        return
    raise SystemExit("usage: log_budget.py rotate FILE MAX_BYTES")


if __name__ == "__main__":
    main()
