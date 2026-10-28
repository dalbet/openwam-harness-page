#!/usr/bin/env python
"""Copy acceptance evidence out of the rotating record directory.

``records/runs`` is pruned to the newest ``MAX_RUNS`` episodes, so an
acceptance table can silently lose the raw episodes its rows point at. This
script freezes one baseline snapshot:

* every ``records/acceptance/*.json`` manifest,
* every raw episode record still on disk that a manifest row references,
* a manifest with sha256, the git HEAD and an explicit list of rows whose raw
  record is already gone (reported as missing, never recreated).

It is read-only with respect to the source files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parents[1]
ACCEPTANCE = PROJECT / "records" / "acceptance"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*args: str) -> str:
    try:
        out = subprocess.run(
            ["git", *args], cwd=PROJECT.parent, capture_output=True, text=True, timeout=20, check=False
        )
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        for key in ("rows", "rounds", "episodes"):
            value = payload.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    return []


def _already_archived(name: str) -> Path | None:
    """A previous run may have copied the episode next to its manifest."""
    for candidate in sorted(ACCEPTANCE.glob(f"*_runs/{name}")):
        if candidate.is_file():
            return candidate
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="")
    parser.add_argument("--label", default="baseline")
    args = parser.parse_args()
    out = Path(args.out) if args.out else ACCEPTANCE / f"archive_{datetime.now():%Y%m%d}_{args.label}"
    out.mkdir(parents=True, exist_ok=True)

    manifests = sorted(ACCEPTANCE.glob("*.json"))
    copied_manifests: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    copied_records = 0
    for manifest in manifests:
        target = out / manifest.name
        if not target.exists():
            shutil.copy2(manifest, target)
        copied_manifests.append(
            {"name": manifest.name, "bytes": manifest.stat().st_size, "sha256": _sha256(manifest)}
        )
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            missing.append({"manifest": manifest.name, "reason": f"unreadable: {exc}"})
            continue
        for index, row in enumerate(_rows(payload)):
            record = row.get("record") or row.get("record_path")
            if not isinstance(record, str) or not record:
                continue
            source = Path(record)
            if not source.is_absolute():
                source = PROJECT / source
            if not source.exists():
                archived = _already_archived(source.name)
                if archived is not None:
                    source = archived
                else:
                    missing.append(
                        {
                            "manifest": manifest.name,
                            "row": index,
                            "task_id": row.get("task_id"),
                            "episode_id": row.get("episode_id"),
                            "record": str(record),
                            "reason": "rotated_out_of_records_runs",
                        }
                    )
                    continue
            runs_dir = out / f"{manifest.stem}_runs"
            runs_dir.mkdir(exist_ok=True)
            destination = runs_dir / source.name
            if not destination.exists():
                shutil.copy2(source, destination)
                copied_records += 1

    manifest_out = {
        "schema": "embodied_engine.evidence_archive.v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "label": args.label,
        "git_head": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "source_dir": str(ACCEPTANCE),
        "manifests": copied_manifests,
        "manifest_count": len(copied_manifests),
        "raw_records_copied": copied_records,
        "missing_raw_records": missing,
        "missing_count": len(missing),
        "note": (
            "Missing raw episodes were already deleted by records/runs rotation; they are listed, "
            "not recreated or back-filled."
        ),
    }
    (out / "manifest.json").write_text(json.dumps(manifest_out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "out": str(out),
                "manifests": len(copied_manifests),
                "raw_records_copied": copied_records,
                "missing_raw_records": len(missing),
                "git_head": manifest_out["git_head"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
