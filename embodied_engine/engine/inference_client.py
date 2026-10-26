"""HTTP client for the inference service (port 14600)."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any

import numpy as np

from common import numpy_json


class InferenceClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def _post(self, path: str, body: dict[str, Any], timeout_s: float) -> dict[str, Any]:
        data = json.dumps(body).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}{path}", data=data, headers={"Content-Type": "application/json"}, method="POST"
        )
        try:
            with self._opener.open(request, timeout=timeout_s) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:800]
            try:
                return json.loads(detail)
            except json.JSONDecodeError:
                return {"status": "error", "error": f"HTTP {exc.code}: {detail}"}

    def _get(self, path: str, timeout_s: float = 10.0) -> dict[str, Any]:
        request = urllib.request.Request(f"{self.base_url}{path}")
        with self._opener.open(request, timeout=timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))

    def health(self) -> dict[str, Any]:
        try:
            return self._get("/health", timeout_s=15)
        except Exception as exc:
            return {"status": "unreachable", "error": f"{type(exc).__name__}: {exc}"}

    def reset(self, reason: str) -> str:
        out = self._post("/reset", {"reason": reason}, timeout_s=15)
        return str(out.get("token") or "")

    def cancel(self, reason: str) -> dict[str, Any]:
        return self._post("/cancel", {"reason": reason}, timeout_s=15)

    def infer(self, request: dict[str, Any], obs: dict[str, Any], timeout_s: float = 180.0) -> dict[str, Any]:
        body = {"request": request, "obs": numpy_json.encode(obs)}
        out = self._post("/infer", body, timeout_s=timeout_s)
        if out.get("status") == "ok" and "actions" in out:
            out["actions"] = np.asarray(numpy_json.decode(out["actions"]), dtype=np.float32)
        return out

    def prefetch(self, request: dict[str, Any], obs: dict[str, Any]) -> dict[str, Any]:
        body = {"request": request, "obs": numpy_json.encode(obs)}
        return self._post("/prefetch", body, timeout_s=30)

    def result(self, request_id: str, wait_s: float) -> dict[str, Any]:
        out = self._post("/result", {"request_id": request_id, "wait_s": wait_s}, timeout_s=wait_s + 30)
        if out.get("status") == "ready" and "actions" in out:
            actions = out["actions"]
            if isinstance(actions, dict):
                actions = numpy_json.decode(actions)
            out["actions"] = np.asarray(actions, dtype=np.float32)
        return out

    def metrics(self) -> dict[str, Any]:
        try:
            return self._get("/metrics")
        except Exception:
            return {}


def model_view(obs: dict[str, Any]) -> dict[str, Any]:
    """Only the fields the policy consumes; keeps the request small."""
    return {
        "main_images": obs["main_images"],
        "wrist_images": obs.get("wrist_images"),
        "states": obs["states"],
        "task_descriptions": obs.get("task_descriptions"),
    }


def request_row(
    *,
    token: str,
    episode_id: str,
    scene_epoch: int | None,
    snapshot: dict[str, Any],
    skill: str,
    step_id: int,
    prompt: str,
) -> dict[str, Any]:
    return {
        "token": token,
        "episode_id": episode_id,
        "scene_epoch": scene_epoch,
        "observation_id": snapshot.get("observation_id"),
        "sim_action_count": snapshot.get("sim_action_count"),
        "skill": skill,
        "step_id": step_id,
        "prompt": prompt,
        "created_at": time.time(),
    }
