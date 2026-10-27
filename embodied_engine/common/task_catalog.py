"""Versioned LIBERO-Goal task catalog.

There is exactly one task table in the project: ``config/tasks.json``. The
UI, the Agent router, the engine and the tests all read it through this
module, so a task number cannot mean different things in different layers.

The catalog records the official LIBERO instruction (Task.language, which this
LIBERO fork derives from the task filename), the BDDL ``:language`` line, the
BDDL and init-file SHA256 fingerprints, the native goal atoms and the complete
semantic binding. BDDL is used here for audit and offline scoring only; no
planner ever receives a BDDL goal as the answer.

The catalog is *not* a synonym dictionary. It describes the ten supported
capabilities; the language model decides which capability a user utterance
means.
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = PROJECT_ROOT / "config" / "tasks.json"


@lru_cache(maxsize=1)
def _raw_bytes() -> bytes:
    return CATALOG_PATH.read_bytes()


@lru_cache(maxsize=1)
def load() -> dict[str, Any]:
    return json.loads(_raw_bytes().decode("utf-8"))


def fingerprint() -> str:
    """SHA256 of the exact catalog file used by this process."""
    return hashlib.sha256(_raw_bytes()).hexdigest()


def version() -> str:
    return str(load().get("version") or "unknown")


def schema() -> str:
    return str(load().get("schema") or "unknown")


def libero_identity() -> dict[str, Any]:
    return dict(load().get("libero") or {})


def tasks() -> list[dict[str, Any]]:
    return list(load().get("tasks") or [])


def task_by_id() -> dict[int, dict[str, Any]]:
    return {int(row["task_id"]): row for row in tasks()}


def task_for(task_id: int) -> dict[str, Any]:
    row = task_by_id().get(int(task_id))
    if row is None:
        raise KeyError(f"unknown libero_goal task_id={task_id}")
    return row


def task_by_key() -> dict[str, dict[str, Any]]:
    return {str(row["key"]): row for row in tasks()}


def capability_rows() -> list[dict[str, Any]]:
    """The full ten-capability candidate list sent to the semantic router.

    Deliberately excludes task_id and the official instruction string: the
    model must choose from semantic descriptions, not from the currently
    loaded task or from a literal answer key.
    """
    return [
        {
            "key": str(row["key"]),
            "operation": str((row.get("semantics") or {}).get("operation") or row.get("kind")),
            "capability": str(row.get("capability") or ""),
        }
        for row in tasks()
    ]


def _scene_objects(snapshot: dict[str, Any]) -> set[str]:
    return set(((snapshot.get("sim") or {}).get("objects") or {}).keys())


def _scene_sites(snapshot: dict[str, Any]) -> set[str]:
    named = snapshot.get("named_poses") or {}
    return set((named.get("sites") or {}).keys()) | set((named.get("bodies") or {}).keys())


def _present(name: str, pool: set[str]) -> bool:
    """A fixture exists if it is named exactly or by any of its sites.

    LIBERO-Goal exposes movable objects as ``sim.objects`` and fixed fixtures
    (stove, cabinet, rack) only through their sites, e.g. ``flat_stove_1``
    appears as ``flat_stove_1_default_site`` / ``flat_stove_1_burner``. This
    is an entity-presence test, not a semantic match.
    """
    if name in pool:
        return True
    prefix = f"{name}_"
    return any(entry.startswith(prefix) for entry in pool)


def _required_entities(task: dict[str, Any]) -> list[tuple[str, str]]:
    semantics = task.get("semantics") or {}
    rows: list[tuple[str, str]] = []
    if semantics.get("source"):
        rows.append(("object", str(semantics["source"])))
    if semantics.get("target"):
        rows.append(("target", str(semantics["target"])))
    for subgoal in semantics.get("subgoals") or []:
        if subgoal.get("source"):
            rows.append(("object", str(subgoal["source"])))
        if subgoal.get("target"):
            rows.append(("target", str(subgoal["target"])))
    # De-duplicate while preserving order.
    seen: set[tuple[str, str]] = set()
    return [row for row in rows if not (row in seen or seen.add(row))]


def scene_support(task: dict[str, Any], snapshot: dict[str, Any]) -> dict[str, Any]:
    """Check the chosen catalog task against the live scene.

    This is a scene/entity check only. It does not decide what the user meant
    and it does not read the loaded task or a BDDL goal.
    """
    objects = _scene_objects(snapshot)
    sites = _scene_sites(snapshot)
    checked: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for kind, name in _required_entities(task):
        if kind == "object":
            present = _present(name, objects)
            where = "sim.objects"
        else:
            present = _present(name, sites) or _present(name, objects)
            where = "named_poses.sites|named_poses.bodies|sim.objects"
        row = {"kind": kind, "name": name, "present": present, "where": where}
        checked.append(row)
        if not present:
            missing.append(row)
    return {
        "ok": not missing,
        "checked": checked,
        "missing": missing,
        "scene_objects": sorted(objects),
        "scene_sites": sorted(sites),
    }


def catalog_evidence() -> dict[str, Any]:
    libero = libero_identity()
    return {
        "schema": schema(),
        "version": version(),
        "sha256": fingerprint(),
        "path": str(CATALOG_PATH),
        "suite": load().get("suite"),
        "libero_repo": libero.get("repo"),
        "libero_commit": libero.get("commit"),
        "libero_package_path": libero.get("package_path"),
        "tasks_info_sha256": libero.get("tasks_info_sha256"),
    }


__all__ = [
    "CATALOG_PATH",
    "capability_rows",
    "catalog_evidence",
    "fingerprint",
    "libero_identity",
    "load",
    "scene_support",
    "task_by_id",
    "task_by_key",
    "task_for",
    "tasks",
    "version",
]
