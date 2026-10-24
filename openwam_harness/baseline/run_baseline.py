"""Run the original RPent Harness on two fixed LIBERO-10 tasks.

This does not add a sequential executor. It keeps upstream control frequency,
Pi0.5 sampling, and the default tool loop. Each repeat writes one episode
record whose success bit is the native ``env.terminated`` signal.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from episode_log import LIBERO10_TASKS, build_episode, utc_now, write_episode
from runtime_patches import apply_null_mount_patch, link_libero_assets

ROOT = Path(__file__).resolve().parents[1]
RPENT = ROOT / "envs" / "rpent" / "bin" / "rpent"
DEFAULT_RUNS = ROOT / "runs" / "baseline_original"

# Locked protocol for phase 1. Repeats share the seed so variance is the
# planner/policy, not a new initial state.
CANDIDATES = (0, 1)
SEED = 0
REPEATS = 3
MAX_TURNS = 100
MAX_EPISODE_STEPS = 10000
PLANNER_TIMEOUT_S = 1200
LIBERO_TYPE = "standard"
SUITE = "libero_10"


def _env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    env["PI05_CHECKPOINT_PATH"] = args.pi05
    env["SAM3_CHECKPOINT_PATH"] = args.sam3
    env["LIBERO_TYPE"] = LIBERO_TYPE
    env.setdefault("MUJOCO_GL", "egl")
    env.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    env.setdefault("OPENAI_API_KEY", "EMPTY")
    # Keep the planner off the proxy if one is set for downloads.
    no_proxy = env.get("NO_PROXY", "")
    extra = "127.0.0.1,localhost"
    env["NO_PROXY"] = extra if not no_proxy else f"{no_proxy},{extra}"
    env["no_proxy"] = env["NO_PROXY"]
    return env


def _command(args: argparse.Namespace, task_id: int, output_dir: Path) -> list[str]:
    return [
        str(RPENT),
        "--robot",
        "libero",
        "--suite",
        SUITE,
        "--task",
        str(task_id),
        "--seed",
        str(SEED),
        "--libero-type",
        LIBERO_TYPE,
        "--planner",
        "api",
        "--model",
        args.model,
        "--base-url",
        args.base_url,
        "--reasoning-effort",
        "none",
        "--max-turns",
        str(MAX_TURNS),
        "--planner-timeout-s",
        str(PLANNER_TIMEOUT_S),
        "--max-episode-steps",
        str(MAX_EPISODE_STEPS),
        "--memory-profile",
        args.memory_profile,
        "--cuda-device",
        str(args.cuda_device),
        "--output-dir",
        str(output_dir),
    ]


def run_one(args: argparse.Namespace, task_id: int, repeat_index: int) -> dict:
    name = f"{SUITE}_t{task_id}_s{SEED}_r{repeat_index}"
    output_dir = Path(args.runs) / name
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = utc_now()
    command = _command(args, task_id, output_dir)
    (output_dir / "command.txt").write_text(" ".join(command) + "\n", encoding="utf-8")
    print(f"RUN {name}", flush=True)
    completed = subprocess.run(
        command,
        env=_env(args),
        cwd=str(ROOT / "RPent"),
        check=False,
    )
    ended = utc_now()
    record = build_episode(
        mode="original",
        harness={
            "name": "rpent",
            "commit": args.rpent_commit,
            "entrypoint": "rpent.cli.main",
        },
        models={
            "planner": {
                "id": "Qwen3.5-9B",
                "api_model": args.model,
                "base_url": args.base_url,
            },
            "vla": {
                "id": "pi05",
                "checkpoint": args.pi05,
                "embodiment": "libero",
                "openpi_config": "pi05_libero",
            },
            "sam3": {"checkpoint": args.sam3},
        },
        task={
            "suite": SUITE,
            "task_id": task_id,
            "task_name": LIBERO10_TASKS[task_id],
            "seed": SEED,
            "libero_type": LIBERO_TYPE,
        },
        budget={
            "max_turns": MAX_TURNS,
            "max_episode_steps": MAX_EPISODE_STEPS,
            "planner_timeout_s": PLANNER_TIMEOUT_S,
            "control_note": "upstream Pi0.5 libero preset and env step rate unchanged",
        },
        output_dir=output_dir,
        repeat_index=repeat_index,
        wall_started=started,
        wall_ended=ended,
        process_returncode=completed.returncode,
    )
    episode_path = Path(args.runs) / f"{name}.episode.json"
    write_episode(episode_path, record)
    print(
        f"DONE {name} success={record['result']['environment_success']} "
        f"elapsed={record['result']['elapsed_s']} rc={completed.returncode}",
        flush=True,
    )
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default=str(DEFAULT_RUNS))
    parser.add_argument(
        "--pi05",
        default=str(ROOT / "checkpoints" / "RLinf-Pi05-LIBERO-130-fullshot-SFT"),
    )
    parser.add_argument(
        "--sam3",
        default=str(ROOT / "checkpoints" / "sam3" / "sam3.pt"),
    )
    parser.add_argument("--model", default="openai:Qwen3.5-9B")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--memory-profile", default="hf", choices=["hf", "local"])
    parser.add_argument("--rpent-commit", default="a8c5311")
    parser.add_argument("--tasks", default="0,1")
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--repeat-offset", type=int, default=0)
    args = parser.parse_args()
    if not RPENT.is_file():
        print(f"missing rpent entrypoint: {RPENT}", file=sys.stderr)
        return 2
    print(f"null-mount patch: {apply_null_mount_patch()}", flush=True)
    site = (
        ROOT
        / "envs"
        / "rpent"
        / "lib"
        / "python3.10"
        / "site-packages"
        / "libero"
        / "libero"
        / "assets"
    )
    print(
        "libero assets: "
        + link_libero_assets(site, ROOT / "deps" / "LIBERO-assets"),
        flush=True,
    )
    tasks = [int(item) for item in args.tasks.split(",") if item]
    records = []
    for task_id in tasks:
        if task_id not in LIBERO10_TASKS:
            print(f"task {task_id} is not a locked candidate", file=sys.stderr)
            return 2
        for repeat in range(args.repeat_offset, args.repeat_offset + args.repeats):
            records.append(run_one(args, task_id, repeat))
    summary = {
        "mode": "original",
        "suite": SUITE,
        "seed": SEED,
        "libero_type": LIBERO_TYPE,
        "episodes": [
            {
                "task_id": item["task"]["task_id"],
                "repeat_index": item["repeat_index"],
                "environment_success": item["result"]["environment_success"],
                "elapsed_s": item["result"]["elapsed_s"],
                "planner_calls": item["result"]["planner_calls"],
                "planner_wait_s": item["result"]["planner_wait_s"],
                "failure_reason": item["result"]["failure_reason"],
            }
            for item in records
        ],
    }
    summary_path = Path(args.runs) / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"summary {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
