"""Paths inside this checkout. No machine-specific prefix."""

from __future__ import annotations

from pathlib import Path


def harness_root() -> Path:
    return Path(__file__).resolve().parents[1]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def rpent_site_packages() -> Path:
    return harness_root() / "envs" / "rpent" / "lib" / "python3.10" / "site-packages"


def site(*parts: str) -> Path:
    return rpent_site_packages().joinpath(*parts)
