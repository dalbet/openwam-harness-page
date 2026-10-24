"""Retention caps. No model, no environment, no network."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import log_budget


class LogBudgetTest(unittest.TestCase):
    def test_journal_rotates_once_and_observation_steps_stay_bounded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            journal = root / "run_journal.jsonl"
            log_budget.append_jsonl(journal, {"n": 1}, max_bytes=1)
            log_budget.append_jsonl(journal, {"n": 2}, max_bytes=1)
            self.assertTrue((root / "run_journal.jsonl.1").is_file())
            self.assertEqual(json.loads(journal.read_text(encoding="utf-8")), {"n": 2})

            episode = root / "episodes" / "aaa"
            art = episode / "agentview_world.npz"
            art.mkdir(parents=True)
            for index in range(5):
                (art / f"{index:02d}.npz").write_bytes(b"x" * 10)
            (episode / "states.json").write_text("{}", encoding="utf-8")
            log_budget.prune_observation_steps(episode, keep=2)
            left = sorted(path.name for path in art.iterdir())
            self.assertEqual(left, ["03.npz", "04.npz"])
            self.assertTrue((episode / "states.json").is_file())

            parent = root / "episodes"
            for index, name in enumerate(("aaa", "bbb", "ccc", "ddd")):
                directory = parent / name
                directory.mkdir(exist_ok=True)
                (directory / "marker").write_text(name, encoding="utf-8")
                os.utime(directory, (index + 1, index + 1))
            log_budget.prune_episode_dirs(parent, keep=3)
            self.assertEqual(sorted(path.name for path in parent.iterdir()), ["bbb", "ccc", "ddd"])

            summary = root / "summaries"
            summary.mkdir()
            for index in range(10):
                path = summary / f"{index}.json"
                path.write_text(json.dumps({"i": index}), encoding="utf-8")
                os.utime(path, (index + 1, index + 1))
            log_budget.prune_summaries(summary, keep=8)
            names = sorted(path.name for path in summary.glob("*.json"))
            self.assertEqual(len(names), 8)
            self.assertIn("9.json", names)
            self.assertNotIn("0.json", names)

    def test_retention_stays_off_unless_the_server_enables_it(self):
        previous = os.environ.pop("SEQUENTIAL_PRUNE_EPISODES", None)
        try:
            self.assertFalse(log_budget.retention_enabled())
            os.environ["SEQUENTIAL_PRUNE_EPISODES"] = "1"
            self.assertTrue(log_budget.retention_enabled())
        finally:
            if previous is None:
                os.environ.pop("SEQUENTIAL_PRUNE_EPISODES", None)
            else:
                os.environ["SEQUENTIAL_PRUNE_EPISODES"] = previous


if __name__ == "__main__":
    unittest.main()
