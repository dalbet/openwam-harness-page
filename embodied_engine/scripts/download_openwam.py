#!/usr/bin/env python3
"""Resume a large Hugging Face file with parallel HTTP ranges.

The single-connection mirror link used for the first OpenWAM attempt stalled
around 1.5 MB/s. The mirror does support ``Range``; this script splits only the
missing tail into independent ranges and writes each range at its absolute
offset, so an interrupted download keeps every completed byte.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

import requests

CHUNK = 4 * 1024 * 1024


def range_total(session: requests.Session, url: str) -> int | None:
    response = session.get(url, headers={"Range": "bytes=0-0"}, stream=True, timeout=60)
    try:
        if response.status_code != 206:
            return None
        content_range = response.headers.get("Content-Range") or ""
        _, _, total = content_range.partition("/")
        return int(total) if total.isdigit() else None
    finally:
        response.close()


def load(url: str, path: Path, total: int, connections: int) -> None:
    session = requests.Session()
    session.trust_env = False
    current = path.stat().st_size if path.exists() else 0
    if current > total:
        raise SystemExit(f"existing file is larger than expected ({current} > {total})")
    if current == total:
        print(f"already complete: {total} bytes", flush=True)
        return
    print(f"resume at {current} / {total} ({total - current} bytes missing)", flush=True)
    with path.open("ab") as handle:
        handle.truncate(total)

    lock = threading.Lock()
    counters = {"done": 0, "errors": 0}
    stop = threading.Event()
    span = total - current
    segment = (span + connections - 1) // connections
    threads: list[threading.Thread] = []

    def worker(index: int, start: int, end: int) -> None:
        local = requests.Session()
        local.trust_env = False
        position = start
        while position <= end and not stop.is_set():
            finish = min(end, position + CHUNK - 1)
            headers = {
                "Range": f"bytes={position}-{finish}",
                "Accept-Encoding": "identity",
                "User-Agent": "embodied-engine-downloader/1",
            }
            try:
                with local.get(url, headers=headers, stream=True, timeout=(30, 180)) as response:
                    if response.status_code != 206:
                        raise RuntimeError(f"range {position}-{finish}: HTTP {response.status_code}")
                    expected = finish - position + 1
                    got = 0
                    with path.open("r+b") as target:
                        target.seek(position)
                        for chunk in response.iter_content(1024 * 256):
                            if not chunk:
                                continue
                            target.write(chunk)
                            got += len(chunk)
                            with lock:
                                counters["done"] += len(chunk)
                            if got > expected:
                                raise RuntimeError(f"range {position}-{finish} overran")
                    if got != expected:
                        raise RuntimeError(f"range {position}-{finish}: short {got}/{expected}")
                position = finish + 1
            except Exception as exc:  # noqa: BLE001 - retry any transport error
                with lock:
                    counters["errors"] += 1
                print(f"[seg {index}] retry at {position}: {type(exc).__name__}: {exc}", flush=True)
                time.sleep(2.0)

    for index in range(connections):
        start = current + index * segment
        if start >= total:
            break
        end = min(total - 1, start + segment - 1)
        thread = threading.Thread(target=worker, args=(index, start, end), daemon=True)
        thread.start()
        threads.append(thread)

    started = time.time()
    total_missing = span
    while any(thread.is_alive() for thread in threads):
        time.sleep(5)
        with lock:
            done = counters["done"]
        now = time.time() - started
        speed = done / now if now > 0 else 0.0
        remaining = max(0, total_missing - done)
        eta = remaining / speed if speed > 0 else 0
        print(
            f"progress {done / 1e9:.2f}/{total_missing / 1e9:.2f} GB  "
            f"{speed / 1e6:.1f} MB/s  ETA {eta / 60:.1f} min",
            flush=True,
        )
    for thread in threads:
        thread.join()
    size = path.stat().st_size
    print(f"finished: size={size} expected={total} errors={counters['errors']}", flush=True)
    if size != total:
        raise SystemExit(f"incomplete file: {size} != {total}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--total-size", type=int, default=None)
    parser.add_argument("--connections", type=int, default=8)
    args = parser.parse_args()
    path = Path(args.output)
    total = args.total_size
    if total is None:
        session = requests.Session()
        session.trust_env = False
        total = range_total(session, args.url)
    if not total:
        print("cannot determine total size", file=sys.stderr)
        return 2
    load(args.url, path, total, max(1, args.connections))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
