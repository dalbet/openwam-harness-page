"""Is one policy request reproducible?

The engine never sends a seed, so the server draws its initial noise from
``torch.Generator().manual_seed(42)`` for every request. Two requests that
carry the *same* observation and prompt should therefore be bit-identical if
the whole path were deterministic. This probe sends the same observation twice
(with a buffer reset in between, so the second request regenerates instead of
dequeuing the first chunk) and reports the difference.

It talks to the resident OpenWAM server only; it never steps the simulator.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from models.openwam_backend import OpenWAMBackend  # noqa: E402


def _frame(engine: str, view: str) -> bytes:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(f"{engine}/api/frame?view={view}", timeout=20) as response:
        return response.read()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", default="http://127.0.0.1:14800")
    parser.add_argument("--url", default="ws://127.0.0.1:14520")
    parser.add_argument("--prompt", default="put the bowl on the stove")
    parser.add_argument("--state", default="", help="comma separated 10-D EEF10 proprio; zeros if omitted")
    parser.add_argument("--out", default=str(PROJECT / "records/acceptance/inference_determinism.json"))
    parser.add_argument("--save-npz", default="", help="store the exact observation so another process can replay it")
    parser.add_argument("--input-npz", default="", help="replay a stored observation instead of reading the live frames")
    args = parser.parse_args()

    if args.input_npz:
        saved = np.load(args.input_npz)
        obs = {
            "main_images": saved["main_images"],
            "wrist_images": saved["wrist_images"],
            "states": [float(v) for v in saved["states"]],
        }
        state = obs["states"]
    else:
        head = _frame(args.engine, "agentview")
        wrist = _frame(args.engine, "wrist")
        state = (
            [float(v) for v in args.state.split(",")]
            if args.state
            else [0.0] * 10
        )
        if len(state) != 10:
            raise SystemExit(f"--state needs 10 values, got {len(state)}")
        obs = {
            "main_images": np.asarray(Image.open(io.BytesIO(head)).convert("RGB")),
            "wrist_images": np.asarray(Image.open(io.BytesIO(wrist)).convert("RGB")),
            "states": state,
        }
    if args.save_npz:
        np.savez(
            args.save_npz,
            main_images=obs["main_images"],
            wrist_images=obs["wrist_images"],
            states=np.asarray(state, dtype=np.float32),
        )
    backend = OpenWAMBackend(url=args.url, generate_horizon=32, execute_horizon=32, timeout_s=300)
    runs = []
    for index in range(2):
        backend.reset()
        result = backend.predict(obs, args.prompt)
        runs.append(
            {
                "run": index + 1,
                "action": result["action_raw"],
                "server_step": result["server_step"],
                "server_latency_ms": result["server_latency_ms"],
            }
        )
    first = np.asarray(runs[0]["action"], dtype=np.float64)
    second = np.asarray(runs[1]["action"], dtype=np.float64)
    diff = np.abs(first - second)
    block = {
        "engine": args.engine,
        "server": args.url,
        "prompt": args.prompt,
        "state": state,
        "identical": bool(np.array_equal(first, second)),
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "runs": runs,
        "note": "same observation, same prompt, no seed sent by the client (server default 42), buffer reset before each request",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(block, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps({k: block[k] for k in ("identical", "max_abs_diff", "mean_abs_diff")}, ensure_ascii=False))
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
