"""Loaded by Python at startup when this directory is on PYTHONPATH.

Only the LIBERO env server installs the display hook. The adapter process
uses the same PYTHONPATH and must not wrap itself.
"""

import os
import sys

try:
    import episode_service

    episode_service.install()
except Exception:
    episode_service = None

if os.environ.get("SEQUENTIAL_DISPLAY_HOOK") == "1" and any(
    str(arg).endswith("env_server.py") for arg in sys.argv
):
    try:
        import env_display_hook

        env_display_hook.install()
    except Exception:
        pass
