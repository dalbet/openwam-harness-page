#!/usr/bin/env python3
"""Build the round's master manifest: one file that points at every record.

The manifest never copies evidence, it lists it: each group is a directory of
raw records with bytes and SHA256, so a reader can verify the files the report
quotes without a second copy of the data. Code identity (branch, commits, tags,
source bundle) is read from git so it cannot drift from the tree that produced
the rounds.

Usage:
    python tools/make_manifest.py \
        --round pi05-household-ready-transport-20261105 \
        --out /path/MANIFEST_20261105_household_ready.json \
        --group formal=/path/raw/composite/formal_3step_20261105 \
        --group dev=/path/raw/composite/dev_transport_20261105 \
        --notes notes.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]


def git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(PROJECT), *args],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
    except OSError:
        return ""


def digest(path: Path) -> dict:
    data = path.read_bytes()
    return {
        "path": str(path),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def group(root: Path) -> list[dict]:
    if root.is_file():
        return [digest(root)]
    return [
        digest(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and not path.name.startswith(".")
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--group", action="append", default=[], help="name=path")
    parser.add_argument("--bundle", action="append", default=[], help="name=path (source bundles)")
    parser.add_argument("--notes", type=Path)
    args = parser.parse_args()
    groups: dict[str, list[dict]] = {}
    for spec in args.group:
        name, _, path = spec.partition("=")
        groups[name] = group(Path(path))
    bundles = {}
    for spec in args.bundle:
        name, _, path = spec.partition("=")
        bundles[name] = digest(Path(path)) if Path(path).exists() else {"path": path, "missing": True}
    notes = json.loads(args.notes.read_text(encoding="utf-8")) if args.notes else {}
    manifest = {
        "schema": "embodied_engine.composite_manifest.v3",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "round": args.round,
        "scope": notes.get("scope"),
        "results": notes.get("results"),
        "findings": notes.get("findings"),
        "code": {
            "repo": "/home/host/page-dmh",
            "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "head": git("rev-parse", "HEAD"),
            "head_subject": git("log", "-1", "--pretty=%s"),
            "tags": [tag for tag in git("tag", "--points-at", "HEAD").splitlines() if tag],
            "baseline_tags": [tag for tag in git("tag", "-l", "backup/pi05*").splitlines() if tag],
            "dirty": bool(git("status", "--porcelain")),
            "bundles": bundles,
        },
        "groups": groups,
        "notes": notes.get("note"),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {args.out} ({sum(len(v) for v in groups.values())} files in {len(groups)} groups)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
