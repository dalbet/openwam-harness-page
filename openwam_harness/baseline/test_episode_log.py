"""Contract tests for the unified episode log. No simulator and no weights."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from episode_log import (
    LIBERO10_TASKS,
    NATIVE_SUCCESS_SIGNAL,
    SCHEMA,
    _load_json,
    build_episode,
    validate_episode,
    write_episode,
)


def _write_run(tmp: Path, *, success: bool, error: str | None) -> None:
    (tmp / "run.log").write_text(
        "\n".join(
            [
                "2026-10-14 00:00:01 INFO [api] [text] pick the soup",
                "2026-10-14 00:00:20 INFO [cli] elapsed: 19.0s",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp / "states.json").write_text(
        json.dumps(
            {
                "steps": [
                    {
                        "step_idx": 0,
                        "command": {"action": "pi0_pick"},
                        "elapsed_s": 8.5,
                        "terminated": False,
                        "truncated": False,
                    },
                    {
                        "step_idx": 1,
                        "command": {"action": "pi0_place"},
                        "elapsed_s": 4.0,
                        "terminated": success,
                        "truncated": False,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    transcript = {
        "environment_success": success,
        "elapsed_s": 19.0,
        "error": error,
        "stats": {
            "turns_used": 3,
            "tool_calls": 2,
            "total_input_tokens": 100,
            "total_output_tokens": 40,
        },
        "messages": [{"role": "assistant", "content": "x"}],
        "finish": {"summary": "done" if success else "timeout"},
    }
    (tmp / "transcript_10_t0_s0.json").write_text(json.dumps(transcript), encoding="utf-8")


def _record(tmp: Path) -> dict:
    return build_episode(
        mode="original",
        harness={"name": "rpent", "commit": "a8c5311"},
        models={
            "planner": {"id": "Qwen3.5-9B"},
            "vla": {"id": "pi05", "checkpoint": "RLinf-Pi05-LIBERO-130-fullshot-SFT"},
            "sam3": {"checkpoint": "sam3.pt"},
        },
        task={
            "suite": "libero_10",
            "task_id": 0,
            "task_name": LIBERO10_TASKS[0],
            "seed": 0,
            "libero_type": "standard",
        },
        budget={"max_turns": 100, "max_episode_steps": 10000, "planner_timeout_s": 1200},
        output_dir=tmp,
        repeat_index=0,
        wall_started="2026-10-14T00:00:00Z",
        wall_ended="2026-10-14T00:00:19Z",
        process_returncode=0,
    )


class EpisodeLogTest(unittest.TestCase):
    def test_concatenated_transcript_uses_the_last_object(self) -> None:
        path = self.tmp_path / "transcript_10_t0_s0.json"
        path.write_text('{"turns_used": 1}{"stats": {"turns_used": 4}}', encoding="utf-8")
        # setUp runs after this method is bound; write after tmp exists via the test body
        loaded = _load_json(path)
        self.assertEqual(loaded["stats"]["turns_used"], 4)

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_success_record_keeps_native_signal_and_splits_wait(self) -> None:
        tmp_path = self.tmp_path
        _write_run(tmp_path, success=True, error=None)
        record = _record(tmp_path)
        validate_episode(record)
        self.assertEqual(record["schema"], SCHEMA)
        self.assertEqual(record["mode"], "original")
        self.assertIs(record["result"]["environment_success"], True)
        self.assertEqual(record["result"]["native_signal"], NATIVE_SUCCESS_SIGNAL)
        self.assertEqual(record["result"]["planner_calls"], 3)
        self.assertEqual(record["result"]["tool_time_s"], 12.5)
        self.assertEqual(record["result"]["planner_wait_s"], 6.5)
        self.assertIsNone(record["result"]["failure_reason"])
        names = [event["name"] for event in record["events"]]
        self.assertEqual(names[0], "episode_start")
        self.assertIn("tool", names)
        self.assertEqual(names[-1], "episode_end")
        written = write_episode(tmp_path / "episode.json", record)
        loaded = json.loads(written.read_text(encoding="utf-8"))
        self.assertEqual(loaded["task"]["seed"], 0)
        self.assertEqual(loaded["models"]["vla"]["id"], "pi05")

    def test_failure_keeps_false_success_and_reason(self) -> None:
        _write_run(self.tmp_path, success=False, error="planner timed out")
        record = _record(self.tmp_path)
        validate_episode(record)
        self.assertIs(record["result"]["environment_success"], False)
        self.assertEqual(record["result"]["failure_reason"], "planner timed out")

    def test_missing_transcript_is_null_success_with_reason(self) -> None:
        record = _record(self.tmp_path)
        validate_episode(record)
        self.assertIsNone(record["result"]["environment_success"])
        self.assertIn("transcript missing", record["result"]["failure_reason"])

    def test_validate_rejects_replaced_native_signal(self) -> None:
        _write_run(self.tmp_path, success=True, error=None)
        record = _record(self.tmp_path)
        record["result"]["native_signal"] = "planner_said_done"
        with self.assertRaises(ValueError) as caught:
            validate_episode(record)
        self.assertIn("env.terminated", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
