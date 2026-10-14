"""Classify the saved original-mode episodes. Does not start a new run."""

from __future__ import annotations

import json
from pathlib import Path

RUNS = Path(__file__).resolve().parents[1] / "runs" / "baseline_original"
OUT = Path(__file__).resolve().parent / "records" / "failure_catalog.json"


def _tool_names(transcript: dict) -> list[str]:
    names = []
    for message in transcript.get("messages") or []:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") == "tool_use":
                names.append(part.get("name"))
    return names


def _load_last_json(path: Path):
    text = path.read_text(encoding="utf-8")
    decoder = json.JSONDecoder()
    index = 0
    last = None
    while index < len(text):
        while index < len(text) and text[index].isspace():
            index += 1
        if index >= len(text):
            break
        value, end = decoder.raw_decode(text, index)
        last = value
        index = end
    return last


def _class_for(reason: str) -> str:
    text = reason.lower()
    if "maximum context length" in text or "input_tokens" in text:
        return "context_error"
    if "timed out" in text or "timeout" in text:
        return "planner_timeout"
    if "interrupt" in text or "keyboard" in text or "stopped" in text:
        return "interrupted"
    if "manip" in text or "grasp failed" in text:
        return "manipulation_failure"
    return "unclassified"


def main() -> None:
    rows = []
    for episode in sorted(RUNS.glob("*.episode.json")):
        payload = json.loads(episode.read_text(encoding="utf-8"))
        result = payload.get("result") or payload
        reason = str(result.get("failure_reason") or "")
        run_dir = RUNS / episode.name.replace(".episode.json", "")
        transcripts = list(run_dir.glob("transcript_*.json"))
        names: list[str] = []
        if transcripts:
            names = _tool_names(_load_last_json(transcripts[0]) or {})
        rows.append(
            {
                "episode": episode.name.removesuffix(".episode.json"),
                "failure_class": _class_for(reason),
                "failure_reason": reason[:500],
                "environment_success": result.get("environment_success"),
                "elapsed_s": result.get("elapsed_s"),
                "planner_calls": result.get("planner_calls"),
                "tool_calls": result.get("tool_calls"),
                "tool_names": names,
                "pi05_grasp_called": "pi0_pick" in names,
                "pi0_doubled_called": "pi0_doubled" in names,
                "sam3_called": "segment" in names,
            }
        )
    interrupted = RUNS / "libero_10_t1_s0_r1" / "run.log"
    log_text = interrupted.read_text(encoding="utf-8") if interrupted.exists() else ""
    rows.append(
        {
            "episode": "libero_10_t1_s0_r1",
            "failure_class": "interrupted",
            "failure_reason": "operator SIGINT before the repeat finished; transcript was not written",
            "environment_success": None,
            "tool_names": [],
            "pi05_grasp_called": "pi0_pick" in log_text and "[tool]" in log_text,
            "pi0_doubled_called": False,
            "sam3_called": False,
            "log_tail": "\n".join(log_text.strip().splitlines()[-8:]),
        }
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
