"""Edge and cloud planner endpoints.

Secrets stay in the environment or in planner_profiles.local.json, which is
gitignored. This module never writes a credential into the repository.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent
LOCAL_PATH = ROOT / "planner_profiles.local.json"

HELD_CONSTANT = ("vla", "perception", "executor", "planning_input")


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _local_file() -> dict[str, Any]:
    path = Path(os.environ.get("PLANNER_PROFILES_FILE", str(LOCAL_PATH)))
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _pick(file_value: Any, env_name: str, default: str = "") -> str:
    env_value = _env(env_name)
    if env_value:
        return env_value
    if isinstance(file_value, str) and file_value.strip():
        return file_value.strip()
    return default


def profiles() -> dict[str, dict[str, Any]]:
    """Return edge and cloud descriptions. Missing cloud credentials stay unavailable."""
    local = _local_file()
    edge_file = local.get("edge") if isinstance(local.get("edge"), dict) else {}
    cloud_file = local.get("cloud") if isinstance(local.get("cloud"), dict) else {}
    edge_key = _pick(edge_file.get("api_key"), "EDGE_PLANNER_API_KEY", _env("OPENAI_API_KEY", "EMPTY"))
    cloud_key = _pick(cloud_file.get("api_key"), "CLOUD_PLANNER_API_KEY")
    cloud_url = _pick(cloud_file.get("base_url"), "CLOUD_PLANNER_BASE_URL")
    edge = {
        "id": "edge",
        "label": "边缘",
        "base_url": _pick(edge_file.get("base_url"), "EDGE_PLANNER_BASE_URL", _env("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")),
        "model": _pick(edge_file.get("model"), "EDGE_PLANNER_MODEL", _env("PLANNER_MODEL", "Qwen3.5-9B")),
        "api_key": edge_key or "EMPTY",
        "input_mode": "scene_metadata",
        "available": True,
        "credential": "env",
    }
    cloud = {
        "id": "cloud",
        "label": "云端",
        "base_url": cloud_url,
        "model": _pick(cloud_file.get("model"), "CLOUD_PLANNER_MODEL"),
        "api_key": cloud_key,
        "input_mode": "scene_metadata",
        "available": bool(cloud_url and cloud_key and _pick(cloud_file.get("model"), "CLOUD_PLANNER_MODEL")),
        "credential": "configured" if cloud_key else "missing",
    }
    return {"edge": edge, "cloud": cloud}


def public_profile(profile: dict[str, Any]) -> dict[str, Any]:
    """Drop the API key. Keep the endpoint and model that were actually selected."""
    return {
        "id": profile.get("id"),
        "label": profile.get("label"),
        "endpoint": profile.get("base_url") or "",
        "model": profile.get("model") or "",
        "input_mode": profile.get("input_mode") or "scene_metadata",
        "available": bool(profile.get("available")),
        "credential": profile.get("credential") or "missing",
        "host": urlsplit(str(profile.get("base_url") or "")).netloc,
    }


def comparison_record(profile: dict[str, Any], binding: dict[str, Any] | None, planning_input: str) -> dict[str, Any]:
    """The only swapped module is the planner. Binding rules are not profile-specific."""
    text = planning_input.encode("utf-8")
    return {
        "swapped_modules": ["planner"],
        "held_constant": list(HELD_CONSTANT),
        "binding_rule": "sim_state_assisted",
        "binding_same_for_profiles": True,
        "binding_evidence_source": (binding or {}).get("evidence_source"),
        "input_mode": profile.get("input_mode") or "scene_metadata",
        "vision": False,
        "input_sha256": hashlib.sha256(text).hexdigest(),
        "endpoint": profile.get("base_url") or "",
        "model": profile.get("model") or "",
        "profile": profile.get("id"),
    }


def build_telemetry(
    *,
    endpoint: str,
    model: str,
    input_mode: str,
    usage: dict[str, Any] | None,
    sent_at: float,
    response_at: float | None,
    first_token_at: float | None,
    done_at: float,
) -> dict[str, Any]:
    """Client-side queue, first token, and total time. Server queue fields are copied when present."""
    usage = dict(usage or {})
    queue_s = None if response_at is None else round(max(0.0, response_at - sent_at), 4)
    ttft_s = None if first_token_at is None else round(max(0.0, first_token_at - sent_at), 4)
    total_s = round(max(0.0, done_at - sent_at), 4)
    server_queue = usage.get("queue_time")
    if server_queue is None:
        server_queue = usage.get("queue_time_s")
    return {
        "endpoint": endpoint,
        "model": model,
        "input_mode": input_mode,
        "usage": usage,
        "queue_s": queue_s,
        "server_queue_s": server_queue,
        "ttft_s": ttft_s,
        "total_s": total_s,
    }
