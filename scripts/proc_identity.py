"""Identify and stop only this project's planner and adapter.

A listener is eligible when its command line and working directory match.
A pid file also stores /proc starttime so a recycled pid is not signalled.
Foreign listeners are reported and left running.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "openwam_harness" / "sequential"))

from process_guard import is_project_adapter, listener_pids, stop_project_adapter  # noqa: E402

PID_DIR = ROOT / "runtime" / "pids"
PLANNER_MARKS = ("vllm serve", "Qwen3.5-9B", str(ROOT / "openwam_harness" / "checkpoints"))
ADAPTER_MARKS = ("adapter.py",)


def starttime(pid: int) -> str | None:
    try:
        text = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except OSError:
        return None
    tail = text.rsplit(")", 1)[-1].split()
    if len(tail) < 20:
        return None
    return tail[19]


def cmdline(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
    except OSError:
        return ""


def cwd(pid: int) -> str:
    try:
        return os.readlink(f"/proc/{pid}/cwd")
    except OSError:
        return ""


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _marks(name: str) -> tuple[str, ...]:
    if name == "planner":
        return PLANNER_MARKS
    if name == "adapter":
        return ADAPTER_MARKS
    raise SystemExit(f"unknown service {name}")


def is_project_service(name: str, pid: int) -> bool:
    command = cmdline(pid)
    if not command:
        return False
    if name == "adapter":
        return is_project_adapter(pid)
    return all(mark in command for mark in _marks(name))


def classify(name: str, port: int) -> str:
    pids = listener_pids(port)
    if not pids:
        return "free"
    pid = pids[0]
    if is_project_service(name, pid):
        return f"project {pid}"
    return f"foreign {pid}"


def record_path(name: str) -> Path:
    return PID_DIR / f"{name}.json"


def write_record(name: str, pid: int) -> None:
    PID_DIR.mkdir(parents=True, exist_ok=True)
    token = starttime(pid)
    if token is None:
        raise SystemExit(f"{name} pid {pid} exited before it could be recorded")
    payload = {
        "name": name,
        "pid": pid,
        "starttime": token,
        "cwd": cwd(pid),
        "marks": list(_marks(name)),
    }
    record_path(name).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def read_record(name: str) -> dict | None:
    path = record_path(name)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def record_matches(name: str) -> tuple[bool, dict | None, str]:
    payload = read_record(name)
    if payload is None:
        return False, None, "no pid file"
    try:
        pid = int(payload.get("pid"))
    except (TypeError, ValueError):
        return False, payload, "pid file is unreadable"
    if not alive(pid):
        return False, payload, "pid is not alive"
    if starttime(pid) != payload.get("starttime"):
        return False, payload, "starttime does not match; pid was reused"
    if not is_project_service(name, pid):
        return False, payload, "command line is not this service"
    return True, payload, "ok"


def descendants(pid: int) -> list[int]:
    found = []
    for text in os.listdir("/proc"):
        if not text.isdigit():
            continue
        child = int(text)
        try:
            status = Path(f"/proc/{child}/status").read_text(encoding="utf-8")
        except OSError:
            continue
        for line in status.splitlines():
            if line.startswith("PPid:") and int(line.split()[1]) == pid:
                found.append(child)
                found.extend(descendants(child))
                break
    return found


def _signal_matching(pid: int, token: str, marks: tuple[str, ...]) -> None:
    if not alive(pid) or starttime(pid) != token:
        return
    command = cmdline(pid)
    if marks and not any(mark in command for mark in marks):
        return
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        return
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline and alive(pid) and starttime(pid) == token:
        time.sleep(0.2)
    if alive(pid) and starttime(pid) == token:
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            return


def stop_named(name: str, port: int) -> str:
    matched, payload, detail = record_matches(name)
    if matched and payload is not None:
        pid = int(payload["pid"])
        if name == "adapter":
            stop_project_adapter(pid)
            record_path(name).unlink(missing_ok=True)
            return f"stopped {pid}"
        children = []
        for child in descendants(pid):
            token = starttime(child)
            if token is not None:
                children.append((child, token, cmdline(child)))
        _signal_matching(pid, str(payload["starttime"]), _marks(name))
        for child, token, _command in children:
            if token is None:
                continue
            _signal_matching(child, token, ())
        record_path(name).unlink(missing_ok=True)
        return f"stopped {pid}"

    decision = classify(name, port)
    if decision.startswith("foreign"):
        return decision
    if decision == "free":
        record_path(name).unlink(missing_ok=True)
        return f"not running ({detail})"
    pid = int(decision.split()[1])
    if name == "adapter":
        stop_project_adapter(pid)
        record_path(name).unlink(missing_ok=True)
        return f"stopped {pid}"
    token = starttime(pid)
    if token is None:
        return f"not running ({detail})"
    children = [(child, starttime(child)) for child in descendants(pid)]
    _signal_matching(pid, token, _marks(name))
    for child, child_token in children:
        if child_token is None:
            continue
        _signal_matching(child, child_token, ())
    record_path(name).unlink(missing_ok=True)
    return f"stopped {pid}"


def status_line(name: str, port: int) -> str:
    matched, payload, detail = record_matches(name)
    decision = classify(name, port)
    pid_text = "none"
    if payload is not None:
        pid_text = f"{payload.get('pid')} identity={'ok' if matched else detail}"
    return f"{name} port {port} pidfile={pid_text} listen={decision}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("classify", "write", "stop", "status"))
    parser.add_argument("--name", choices=("planner", "adapter"))
    parser.add_argument("--port", type=int)
    parser.add_argument("--pid", type=int)
    args = parser.parse_args()
    if args.action == "classify":
        if args.name is None or args.port is None:
            raise SystemExit("classify needs --name and --port")
        print(classify(args.name, args.port))
        return 0
    if args.action == "write":
        if args.name is None or args.pid is None:
            raise SystemExit("write needs --name and --pid")
        write_record(args.name, args.pid)
        print(f"recorded {args.name} {args.pid}")
        return 0
    if args.action == "stop":
        if args.name is None or args.port is None:
            raise SystemExit("stop needs --name and --port")
        print(stop_named(args.name, args.port))
        return 0
    if args.name is None or args.port is None:
        raise SystemExit("status needs --name and --port")
    print(status_line(args.name, args.port))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
