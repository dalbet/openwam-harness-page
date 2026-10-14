"""The NullMount fallback must be idempotent and must not invent a new base."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from runtime_patches import NEW, OLD, apply_null_mount_patch


class NullMountPatchTest(unittest.TestCase):
    def test_apply_once_and_again(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "robot_base_factory.py"
            path.write_text("header\n" + OLD + "\n", encoding="utf-8")
            self.assertEqual(apply_null_mount_patch(path), "applied")
            self.assertIn('name = "NullMount"', path.read_text(encoding="utf-8"))
            self.assertNotIn("action_dim", path.read_text(encoding="utf-8"))
            self.assertEqual(apply_null_mount_patch(path), "already-applied")
            self.assertEqual(path.read_text(encoding="utf-8"), "header\n" + NEW + "\n")

    def test_rejects_unknown_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "robot_base_factory.py"
            path.write_text("something else\n", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                apply_null_mount_patch(path)


if __name__ == "__main__":
    unittest.main()
