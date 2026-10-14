"""Unified episode log for the original Harness/RPent baseline.

The native success signal is RPent ``environment_success``, which is
``LiberoToolkit.solved()``. That flag is the OR of ``env.terminated`` captured
after each stateful tool. LIBERO sets ``terminated`` only when the official
task predicate succeeds. ``truncated`` is a timeout and is not success.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = "page.harness.episode.v1"
NATIVE_SUCCESS_SIGNAL = "env.terminated"

# Canonical LIBERO-10 indices. Two-object sequential pick-and-place.
LIBERO10_TASKS = {
    0: "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket",
    1: "LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_json(path: Path) -> Any:
    """Load a JSON document. RPent can append a second object on rerun."""
    text = path.read_text(encoding="utf-8")
    decoder = json.JSONDecoder()
    index = 0
    last = None
    length = len(text)
    while index < length:
        while index < length and text[index].isspace():
            index += 1
        if index >= length:
            break
        value, end = decoder.raw_decode(text, index)
        last = value
        index = end
    if last is None:
        raise json.JSONDecodeError("empty JSON", text, 0)
    return last


def _find_transcript(output_dir: Path) -> Path | None:
    matches = sorted(output_dir.glob("transcript_*.json"))
    return matches[-1] if matches else None


def _planner_calls(transcript: dict[str, Any]) -> int:
    stats = transcript.get("stats") or {}
    turns = stats.get("turns_used")
    if isinstance(turns, int):
        return turns
    messages = transcript.get("messages") or []
    return sum(1 for message in messages if message.get("role") == "assistant")


def _tool_events(states: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not states:
        return []
    events: list[dict[str, Any]] = []
    for step in states.get("steps") or []:
        command = step.get("command") or {}
        action = command.get("action")
        if not action:
            continue
        events.append(
            {
                "name": "tool",
                "step_idx": step.get("step_idx"),
                "tool": action,
                "elapsed_s": step.get("elapsed_s"),
                "terminated": bool(step.get("terminated")),
                "truncated": bool(step.get("truncated")),
            }
        )
    return events


def _parse_run_log(path: Path | None) -> list[dict[str, Any]]:
    """Pull timestamped planner text and the final elapsed line from run.log."""
    if path is None or not path.is_file():
        return []
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if len(line) < 20 or line[4] != "-" or line[10] != " ":
            continue
        stamp = line[:19]
        if "[api]" in line and "[text]" in line:
            events.append({"name": "planner_text", "time": stamp})
        elif "[api]" in line and "[thinking]" in line:
            events.append({"name": "planner_thinking", "time": stamp})
        elif " elapsed:" in line or line.rstrip().endswith("s") and "elapsed:" in line:
            events.append({"name": "runner_elapsed", "time": stamp, "line": line.strip()})
    return events


def _planner_wait_s(log_events: list[dict[str, Any]], tool_events: list[dict[str, Any]]) -> float | None:
    """Wall time of planner text turns is not a wait. Wait is total minus tools.

    Callers pass ``total_elapsed_s`` separately. This helper only sums tool time.
    """
    del log_events
    total = 0.0
    seen = False
    for event in tool_events:
        elapsed = event.get("elapsed_s")
        if isinstance(elapsed, (int, float)):
            total += float(elapsed)
            seen = True
    return round(total, 3) if seen else None


def build_episode(
    *,
    mode: str,
    harness: dict[str, Any],
    models: dict[str, Any],
    task: dict[str, Any],
    budget: dict[str, Any],
    output_dir: Path | str,
    repeat_index: int,
    wall_started: str,
    wall_ended: str,
    process_returncode: int | None,
) -> dict[str, Any]:
    """Assemble one episode record from an RPent output directory."""
    output = Path(output_dir)
    transcript_path = _find_transcript(output)
    transcript: dict[str, Any] = {}
    if transcript_path is not None:
        loaded = _load_json(transcript_path)
        if isinstance(loaded, dict):
            transcript = loaded
    states_path = output / "states.json"
    states = _load_json(states_path) if states_path.is_file() else None
    log_events = _parse_run_log(output / "run.log")
    tool_events = _tool_events(states if isinstance(states, dict) else None)
    tool_time_s = _planner_wait_s(log_events, tool_events)

    environment_success = transcript.get("environment_success", None)
    if environment_success is not None:
        environment_success = bool(environment_success)
    elapsed_s = transcript.get("elapsed_s")
    planner_wait_s = None
    if isinstance(elapsed_s, (int, float)) and tool_time_s is not None:
        planner_wait_s = round(max(0.0, float(elapsed_s) - tool_time_s), 3)

    error = transcript.get("error")
    failure_reason = None if environment_success else (error or _infer_failure(transcript, process_returncode))

    events: list[dict[str, Any]] = [{"name": "episode_start", "time": wall_started}]
    events.extend(log_events)
    events.extend(tool_events)
    events.append({"name": "episode_end", "time": wall_ended})

    record = {
        "schema": SCHEMA,
        "mode": mode,
        "repeat_index": repeat_index,
        "harness": harness,
        "models": models,
        "task": task,
        "budget": budget,
        "events": events,
        "result": {
            "environment_success": environment_success,
            "native_signal": NATIVE_SUCCESS_SIGNAL,
            "native_signal_meaning": (
                "LiberoToolkit.solved() ORs env.terminated after each stateful tool. "
                "LIBERO sets terminated only for the official task predicate."
            ),
            "elapsed_s": elapsed_s,
            "planner_calls": _planner_calls(transcript) if transcript else None,
            "planner_tokens_in": (transcript.get("stats") or {}).get("total_input_tokens"),
            "planner_tokens_out": (transcript.get("stats") or {}).get("total_output_tokens"),
            "tool_calls": (transcript.get("stats") or {}).get("tool_calls"),
            "tool_time_s": tool_time_s,
            "planner_wait_s": planner_wait_s,
            "failure_reason": failure_reason,
            "process_returncode": process_returncode,
            "transcript": str(transcript_path) if transcript_path else None,
            "output_dir": str(output),
        },
    }
    return record


def _infer_failure(transcript: dict[str, Any], returncode: int | None) -> str:
    if not transcript:
        if returncode not in (None, 0):
            return f"rpent exited {returncode} before writing a transcript"
        return "transcript missing; no native success signal"
    finish = transcript.get("finish") or {}
    summary = finish.get("summary") if isinstance(finish, dict) else None
    if summary:
        return str(summary)
    if returncode not in (None, 0):
        return f"rpent exited {returncode}"
    return "environment_success is false"


def validate_episode(record: dict[str, Any]) -> None:
    """Raise ValueError when a record misses the phase-1 contract."""
    if record.get("schema") != SCHEMA:
        raise ValueError(f"schema must be {SCHEMA}")
    for key in ("mode", "harness", "models", "task", "budget", "events", "result"):
        if key not in record:
            raise ValueError(f"missing {key}")
    task = record["task"]
    for key in ("suite", "task_id", "task_name", "seed", "libero_type"):
        if key not in task:
            raise ValueError(f"task missing {key}")
    models = record["models"]
    for key in ("planner", "vla", "sam3"):
        if key not in models:
            raise ValueError(f"models missing {key}")
    result = record["result"]
    if "environment_success" not in result:
        raise ValueError("result missing environment_success")
    if result.get("native_signal") != NATIVE_SUCCESS_SIGNAL:
        raise ValueError("native_signal must stay env.terminated")
    success = result["environment_success"]
    if success is not None and not isinstance(success, bool):
        raise ValueError("environment_success must be bool or null")
    if success is None and not result.get("failure_reason"):
        raise ValueError("null success requires failure_reason")
    names = [event.get("name") for event in record["events"]]
    if "episode_start" not in names or "episode_end" not in names:
        raise ValueError("events must include episode_start and episode_end")


def write_episode(path: Path | str, record: dict[str, Any]) -> Path:
    validate_episode(record)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return destination
