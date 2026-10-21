"""Load contract: missing task_id is the suite default, null is an error."""

from __future__ import annotations

import os
import unittest
from pathlib import Path

from capabilities import DEFAULT_TASK_ID, GROUPS, SUITE_IDS, builtin_matrix
from instruction import resolve_instruction
from scene_request import FieldError, resolve_load
from task_zh import TASK_ZH

ROOT = Path(__file__).resolve().parent
BUTTON = {"type": "load", "suite": "libero_spatial"}


def catalog() -> list[dict]:
    suites = []
    for suite, languages in GROUPS:
        tasks = []
        for index, language in enumerate(languages):
            tasks.append(
                {
                    "id": index,
                    "task_key": f"{suite}:t{index}",
                    "language": language,
                    "summary_zh": TASK_ZH[language],
                    "init_states": 50,
                    "long_candidate": suite == "libero_10" and index in {0, 1},
                    "instances": [],
                    "fixtures": [],
                }
            )
        suites.append({"id": suite, "tasks": tasks})
    return suites


class LoadContractTest(unittest.TestCase):
    def test_historical_omitted_task_id_is_int_none(self):
        # Former adapter._scene_from_message line, before the controls were removed:
        #     task_id = int(message.get("task_id"))
        with self.assertRaises(TypeError) as caught:
            int(BUTTON.get("task_id"))
        self.assertIn("NoneType", str(caught.exception))

    def test_button_payload_uses_matrix_default_and_fixed_seed(self):
        spec = resolve_load(BUTTON, catalog())
        self.assertEqual(spec["suite"], "libero_spatial")
        self.assertEqual(spec["task_id"], 0)
        self.assertEqual(spec["seed"], 0)
        self.assertEqual(spec["task_source"], "default")
        self.assertEqual(spec["execution"], "同步")
        self.assertTrue(spec["summary_zh"])

    def test_missing_task_is_not_null(self):
        omitted = resolve_load({"suite": "libero_goal"}, catalog())
        self.assertEqual(omitted["task_id"], 1)
        self.assertEqual(omitted["task_source"], "default")
        self.assertEqual(omitted["seed"], 0)
        with self.assertRaises(FieldError) as caught:
            resolve_load({"suite": "libero_goal", "task_id": None}, catalog())
        self.assertEqual(caught.exception.field["path"], "task_id")
        self.assertEqual(caught.exception.field["code"], "null")
        self.assertIsNone(caught.exception.field["received"])

    def test_explicit_zero_seed_and_rejected_seed(self):
        spec = resolve_load({"suite": "libero_object", "seed": 0}, catalog())
        self.assertEqual(spec["seed"], 0)
        self.assertEqual(spec["task_id"], 0)
        for payload, path in (
            ({"suite": "libero_object", "seed": None}, "seed"),
            ({"suite": "libero_object", "seed": 3}, "seed"),
            ({"suite": "libero_object", "seed": "0"}, "seed"),
            ({"suite": "libero_object", "init_state_id": None}, "init_state_id"),
            ({"suite": "libero_object", "init_state_id": 4}, "init_state_id"),
            ({"suite": "nope"}, "suite"),
            ({"suite": "libero_spatial", "task_id": 40}, "task_id"),
            ({"suite": "libero_spatial", "task_id": True}, "task_id"),
        ):
            with self.assertRaises(FieldError) as caught:
                resolve_load(payload, catalog())
            self.assertEqual(caught.exception.field["path"], path, payload)

    def test_four_defaults_are_open_and_goal_zero_is_not(self):
        rows = {(item["suite"], item["task_id"]): item for item in builtin_matrix()}
        self.assertEqual(set(DEFAULT_TASK_ID), set(SUITE_IDS))
        for suite, task_id in DEFAULT_TASK_ID.items():
            row = rows[(suite, task_id)]
            self.assertTrue(row["demo_open"], suite)
            spec = resolve_load({"suite": suite}, catalog())
            self.assertEqual(spec["task_id"], task_id)
            self.assertEqual(spec["seed"], 0)
            self.assertEqual(spec["goal"], row["language"])
        self.assertFalse(rows[("libero_goal", 0)]["demo_open"])
        self.assertIn("drawer", rows[("libero_goal", 0)]["language"])

    def test_failed_load_releases_the_lock(self):
        import adapter

        app = adapter.App()
        messages = []
        app.broadcast = messages.append
        app.suites = catalog()
        app.load({"type": "load", "suite": "nope"})
        app.load({"type": "load", "suite": "libero_goal", "task_text": "没有这句标准任务"})
        self.assertFalse(app.loading)
        details = [item.get("detail", "") for item in messages if item.get("type") == "load_failed"]
        self.assertEqual(len(details), 2)
        self.assertFalse(any("已有加载在进行" in item for item in details))
        self.assertTrue(any(item.get("field", {}).get("path") == "suite" for item in messages))
        self.assertTrue(any(item.get("field", {}).get("code") == "unknown" for item in messages))

    def test_missing_loaded_task_does_not_guess(self):
        with self.assertRaises(TypeError):
            int({}.get("task_id"))
        decision = resolve_instruction("把碗放到炉灶上", {"suite": "libero_goal", "goal": "put the bowl on the stove"})
        self.assertEqual(decision["status"], "unsupported")
        self.assertIn("task_id", decision["reason"])
        self.assertFalse(decision["changed"])

    def test_health_freezes_startup_commit_and_static_version(self):
        import adapter

        page = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
        script = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
        self.assertIn(f"static/app.js?v={adapter.STATIC_VERSION}", page)
        self.assertNotIn('id="task-id"', page)
        self.assertNotIn('id="seed"', page)
        self.assertIn("执行：同步", script)
        health = adapter.APP.health()
        self.assertEqual(health["commit"], adapter.STARTUP_COMMIT)
        self.assertEqual(health["static_version"], adapter.STATIC_VERSION)
        self.assertEqual(health["execution"], "同步")
        self.assertEqual(health["pid"], os.getpid())
        self.assertNotEqual(health["commit"], "")


if __name__ == "__main__":
    unittest.main()
