"""Decide whether this project's adapter may replace the process on a port.

Foreign listeners are left alone. This never scans or signals unrelated processes.
"""

from __future__ import annotations

import argparse
import os
import signal
import time
from pathlib import Path


def listener_pids(port: int) -> list[int]:
    inodes = set()
    for path in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            lines = Path(path).read_text(encoding="utf-8").splitlines()[1:]
        except OSError:
            continue
        for line in lines:
            parts = line.split()
            local, state, inode = parts[1], parts[3], parts[9]
            ip_hex, port_hex = local.split(":")
            if int(port_hex, 16) != port or state != "0A":
                continue
            if ip_hex not in {"0100007F", "00000000000000000000000001000000"}:
                continue
            inodes.add(inode)
    found = []
    for pid_text in os.listdir("/proc"):
        if not pid_text.isdigit():
            continue
        fd_dir = Path("/proc") / pid_text / "fd"
        try:
            fds = list(fd_dir.iterdir())
        except OSError:
            continue
        for fd in fds:
            try:
                target = os.readlink(fd)
            except OSError:
                continue
            if target.startswith("socket:[") and target[8:-1] in inodes:
                found.append(int(pid_text))
                break
    return found


def is_project_adapter(pid: int) -> bool:
    try:
        command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
        cwd = os.readlink(f"/proc/{pid}/cwd")
    except OSError:
        return False
    return "adapter.py" in command and cwd.rstrip("/").endswith("openwam_harness/sequential")


def child_pids(pid: int) -> list[int]:
    children = []
    for pid_text in os.listdir("/proc"):
        if not pid_text.isdigit():
            continue
        try:
            text = Path(f"/proc/{pid_text}/status").read_text(encoding="utf-8")
        except OSError:
            continue
        for line in text.splitlines():
            if line.startswith("PPid:") and int(line.split()[1]) == pid:
                children.append(int(pid_text))
    return children


def stop_project_adapter(pid: int) -> None:
    """Stop one project adapter and the env/VLA/SAM3 children it spawned."""
    targets = [pid, *child_pids(pid)]
    for target in targets:
        if target != pid and not _child_allowed(target):
            continue
        try:
            os.kill(target, signal.SIGTERM)
        except OSError:
            pass
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline and _alive(pid):
        time.sleep(0.2)
    if _alive(pid):
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass


def _child_allowed(pid: int) -> bool:
    try:
        command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
    except OSError:
        return False
    markers = ("env_server.py", "pi05_vla_server.py", "sam3_server.py", "adapter.py")
    return any(marker in command for marker in markers)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--stop", action="store_true")
    args = parser.parse_args()
    pids = listener_pids(args.port)
    if not pids:
        print("free")
        return 0
    pid = pids[0]
    if not is_project_adapter(pid):
        print(f"foreign {pid}")
        return 2
    if args.stop:
        stop_project_adapter(pid)
        print(f"stopped {pid}")
        return 0
    print(f"project {pid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
