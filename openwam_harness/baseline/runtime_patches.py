"""Replayable fixes for third-party packages that are not in this git tree.

robosuite 1.5.2's ``robot_base_factory`` rejects ``default_base is None``.
``rpent-libero`` 0.2.0's ``OnTheGroundPanda`` uses that value for the
living-room tasks (no Rethink pedestal). ``NullMount`` is robosuite's own
"no mount" and does not change the Pi0.5 action size or control rate.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

OLD = """\
    from robosuite.models.bases import BASE_MAPPING

    return BASE_MAPPING.get(name, "Unknown base name: {}".format(name))(idn=idn)
"""

NEW = """\
    from robosuite.models.bases import BASE_MAPPING

    # rpent-libero OnTheGroundPanda.default_base is None (no pedestal).
    # robosuite 1.5.2 has no None entry, so the lookup returns an error
    # string and then crashes. NullMount is the published "no mount".
    if name is None:
        name = "NullMount"

    return BASE_MAPPING.get(name, "Unknown base name: {}".format(name))(idn=idn)
"""


def factory_path() -> Path:
    """Locate robosuite inside the rpent env, not the system interpreter."""
    candidate = Path(
        "/root/Page_dmh/openwam_harness/envs/rpent/lib/python3.10/site-packages"
        "/robosuite/models/bases/robot_base_factory.py"
    )
    if candidate.is_file():
        return candidate
    spec = importlib.util.find_spec("robosuite.models.bases.robot_base_factory")
    if spec is None or not spec.origin:
        raise FileNotFoundError("robosuite robot_base_factory is not installed")
    return Path(spec.origin)


def apply_null_mount_patch(path: Path | None = None) -> str:
    """Insert the NullMount fallback. A second call leaves the file unchanged."""
    target = path or factory_path()
    text = target.read_text(encoding="utf-8")
    if NEW in text:
        return "already-applied"
    if OLD not in text:
        raise RuntimeError(f"unexpected robot_base_factory contents: {target}")
    target.write_text(text.replace(OLD, NEW, 1), encoding="utf-8")
    return "applied"


def link_libero_assets(site_assets: Path, downloaded: Path) -> str:
    """Point the installed LIBERO package at the downloaded mesh tree."""
    if site_assets.is_symlink():
        return "already-linked"
    if site_assets.exists():
        return "present"
    if not downloaded.is_dir():
        raise FileNotFoundError(f"LIBERO assets missing: {downloaded}")
    site_assets.symlink_to(downloaded)
    return "linked"
