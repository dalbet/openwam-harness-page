"""Arm (model/backend) table and verified switching for the Page UI.

Three arms are exposed to the page:

    S0  OpenWAM + candidate C        backend=openwam   policy server :14541
    S1  pi0.5 original backend       backend=pi05      generation :14552 (openpi)
    S2  pi0.5 + FlashRT BF16         backend=pi05      generation :14551 (flashrt)

Two rules make this safe and honest:

* The page never talks to a GPU process. It posts ``{"type": "set_arm"}``; this
  module runs the configured lifecycle command (``tools/arm_switch.sh``, which
  is ledger-checked and never uses pkill/killall) and then verifies the result
  against the engine's *live* identity.
* "Selected" and "actually serving" are separate fields, always both sent to
  the page. A switch is only reported as done when the live identity matches
  the arm that was asked for; a mismatch stays an error, so the page cannot
  show S2 while S1 is executing.

The feature is off unless ``PAGE_ARMS`` names a config file, so the production
page keeps its old behaviour (no arm selector, no lifecycle commands).
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any, Awaitable, Callable

# The engine only accepts a switch in these phases; a switch during an episode
# would leave a rollout running against a router that is being replaced.
IDLE_PHASES = {"ready", "succeeded", "failed", "stopped"}
BUSY_PHASES = {"loading", "planning", "running", "recovering", "verifying", "stopping"}


def _port(url: str | None) -> str:
    if not url:
        return ""
    tail = str(url).rstrip("/").rsplit(":", 1)
    return tail[-1] if len(tail) == 2 else ""


class ArmTable:
    """Arm definitions plus the lifecycle command, from ``PAGE_ARMS``."""

    def __init__(self, path: str | None = None):
        self.path = Path(path) if path else None
        self.rows: dict[str, dict[str, Any]] = {}
        self.switch_cmd: list[str] = []
        self.verify_timeout_s = 240.0
        self.error = ""
        self._mtime = 0.0
        if self.path is None:
            self.error = "PAGE_ARMS is not set: arm switching is disabled"
            return
        self.reload()

    @property
    def enabled(self) -> bool:
        return bool(self.rows) and bool(self.switch_cmd)

    def reload(self) -> None:
        if self.path is None:
            return
        try:
            mtime = self.path.stat().st_mtime
            if mtime == self._mtime and self.rows:
                return
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            self.rows = {str(k): dict(v) for k, v in (raw.get("arms") or {}).items()}
            self.switch_cmd = [str(x) for x in (raw.get("switch_cmd") or [])]
            self.verify_timeout_s = float(raw.get("verify_timeout_s") or 240.0)
            self._mtime = mtime
            self.error = ""
        except Exception as exc:  # noqa: BLE001 - a bad file must not break the page
            self.rows = {}
            self.switch_cmd = []
            self.error = f"{type(exc).__name__}: {exc}"

    def public(self) -> list[dict[str, Any]]:
        return [
            {
                "id": key,
                "label": row.get("label") or key,
                "policy": row.get("policy") or "",
                "backend": row.get("backend") or "",
                "server_url": row.get("server_url") or "",
                "precision": row.get("precision") or "",
            }
            for key, row in self.rows.items()
        ]

    def match(self, status: dict[str, Any]) -> dict[str, Any]:
        """Which arm is *actually* serving, according to the engine identity."""
        identity = status.get("identity") or {}
        action = identity.get("action_model") or {}
        backend = str(action.get("backend") or "").lower()
        url = str(action.get("server_url") or "")
        serving_arm = str(action.get("serving_arm") or "")
        precision = str(action.get("precision") or "")
        mode = str((action.get("server") or {}).get("mode") or "")
        actual = None
        if serving_arm in self.rows:
            actual = serving_arm
        elif backend == "openwam":
            actual = "S0"
        elif backend == "pi05":
            want = _port(url)
            if want:
                for key, row in self.rows.items():
                    if _port(row.get("server_url")) == want:
                        actual = key
                        break
            elif mode:
                for key, row in self.rows.items():
                    if row.get("mode") == mode:
                        actual = key
                        break
        row = self.rows.get(actual or "") or {}
        verified = bool(actual) and (
            not row.get("backend") or str(row["backend"]).lower() == backend
        ) and (
            not row.get("server_url") or _port(row["server_url"]) == _port(url)
        )
        if actual == "S2":
            verified = verified and "bf16" in precision.lower()
        return {
            "actual": actual or "unknown",
            "actual_label": (row.get("label") if actual else "") or f"未知（backend={backend or '?'}）",
            "actual_backend": backend,
            "actual_server_url": url,
            "actual_precision": precision,
            "actual_mode": mode,
            "verified": verified,
        }


class ArmSwitcher:
    """Serialize arm switches and report every stage to the page."""

    def __init__(
        self,
        table: ArmTable,
        *,
        engine_get: Callable[[str, float], Awaitable[dict]],
        broadcast: Callable[[dict], Awaitable[None]],
        log: Callable[[str], None],
    ):
        self.table = table
        self.engine_get = engine_get
        self.broadcast = broadcast
        self.log = log
        self.lock = asyncio.Lock()
        self.switching: str | None = None
        self.last: dict[str, Any] = {}

    def state(self, status: dict[str, Any], selected: str | None = None) -> dict[str, Any]:
        match = self.table.match(status)
        phase = str(status.get("phase") or "")
        arm_row = self.table.rows.get(match["actual"]) or {}
        return {
            "type": "arm_state",
            "enabled": self.table.enabled,
            "error": self.table.error,
            "arms": self.table.public(),
            # Before the user picks anything the selector shows what is
            # actually serving, so "selected" never starts out blank.
            "selected": selected or self.last.get("selected")
            or (match["actual"] if match["actual"] in self.table.rows else None),
            "actual": match["actual"],
            "actual_label": match["actual_label"],
            "actual_backend": match["actual_backend"],
            "actual_server_url": match["actual_server_url"],
            "actual_precision": match["actual_precision"],
            "verified": match["verified"],
            "checkpoint": arm_row.get("checkpoint") or "",
            "policy": arm_row.get("policy") or "",
            "phase": phase,
            "busy": phase in BUSY_PHASES or self.switching is not None,
            "switching": self.switching,
            "steps": self.last.get("steps") or [],
            "detail": self.last.get("detail") or "",
            "status": self.last.get("status") or ("idle" if self.table.enabled else "disabled"),
        }

    async def _emit(self, stage: str, detail: str = "") -> None:
        steps = self.last.setdefault("steps", [])
        steps.append({"at": round(time.time(), 3), "stage": stage, "detail": detail})
        del steps[:-40]
        self.log(f"switch {stage}: {detail}")
        await self.broadcast(
            {
                "type": "arm_progress",
                "stage": stage,
                "detail": detail,
                "switching": self.switching,
                "steps": steps,
            }
        )

    async def switch(self, arm: str, status: dict[str, Any]) -> dict[str, Any]:
        """Run the lifecycle command, then verify against the live identity."""
        if not self.table.enabled:
            return {"status": "error", "error": self.table.error or "arm switching is disabled"}
        if arm not in self.table.rows:
            return {"status": "error", "error": f"unknown arm {arm}"}
        if self.switching is not None:
            return {"status": "error", "error": f"a switch to {self.switching} is already running"}
        phase = str(status.get("phase") or "")
        if phase in BUSY_PHASES:
            return {"status": "error", "error": f"engine is {phase}; stop or wait before switching"}
        current = self.table.match(status)
        if current["actual"] == arm and current["verified"]:
            self.last = {"selected": arm, "status": "ok", "detail": "already serving", "steps": []}
            return {"status": "ok", "arm": arm, "detail": "already serving", "state": self.state(status, arm)}

        async with self.lock:
            self.switching = arm
            self.last = {"selected": arm, "status": "switching", "steps": [], "detail": ""}
            await self._emit("requested", f"{arm} {self.table.rows[arm].get('label') or ''}")
            await self.broadcast(self.state(status, arm))
            cmd = [part.replace("{arm}", arm) for part in self.table.switch_cmd]
            lines: list[str] = []
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                    env=dict(os.environ),
                )
                assert proc.stdout is not None
                async for raw in proc.stdout:
                    line = raw.decode("utf-8", "replace").strip()
                    if not line:
                        continue
                    lines.append(line)
                    del lines[:-200]
                    if line.startswith("STEP "):
                        _, stage, detail = (line.split(" ", 2) + ["", ""])[:3]
                        await self._emit(stage, detail)
                    elif line.startswith("RESULT "):
                        await self._emit("lifecycle", line[7:][:400])
                code = await proc.wait()
            except Exception as exc:  # noqa: BLE001 - report, never crash the page
                code = -1
                await self._emit("lifecycle_error", f"{type(exc).__name__}: {exc}")
            if code != 0:
                self.switching = None
                detail = next((ln for ln in reversed(lines) if ln.startswith("RESULT ")), "") or f"exit {code}"
                self.last.update(status="error", detail=detail)
                st = self.state(status, arm)
                st["error"] = detail
                await self.broadcast(st)
                return {"status": "error", "error": detail, "state": st}

            # The lifecycle command is not trusted on its own: the engine's
            # identity is what the page shows and what decides "verified".
            deadline = time.time() + self.table.verify_timeout_s
            await self._emit("verify", "waiting for the engine to report the new backend")
            verified = None
            while time.time() < deadline:
                try:
                    live = await self.engine_get("/api/status", 10.0)
                except Exception:  # noqa: BLE001 - engine may be mid-restart
                    await asyncio.sleep(1.0)
                    continue
                match = self.table.match(live)
                if match["actual"] == arm and match["verified"]:
                    verified = match
                    break
                await asyncio.sleep(1.0)
            self.switching = None
            if verified is None:
                try:
                    live = await self.engine_get("/api/status", 10.0)
                except Exception:  # noqa: BLE001
                    live = status
                match = self.table.match(live)
                detail = (
                    f"selected {arm} but the engine reports {match['actual']} "
                    f"({match['actual_backend']} {match['actual_server_url']} {match['actual_precision']})"
                )
                self.last.update(status="error", detail=detail)
                st = self.state(live, arm)
                st["error"] = detail
                await self.broadcast(st)
                return {"status": "error", "error": detail, "state": st}

            self.last.update(status="ok", detail="identity verified")
            await self._emit("ready", f"{arm} verified: {verified['actual_label']}")
            live = await self.engine_get("/api/status", 10.0)
            st = self.state(live, arm)
            await self.broadcast(st)
            return {"status": "ok", "arm": arm, "state": st}
