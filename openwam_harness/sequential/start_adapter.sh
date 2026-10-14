#!/usr/bin/env bash
# Sequential Page adapter. Default is the CPU dry path.
# HARNESS_LIVE=1 loads the existing LIBERO env, Pi0.5, and SAM3 on the single GPU.
# Stop the planner first if that load runs out of memory. Do not start a second planner.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export LIBERO_TYPE=standard
export PYTHONPATH="${ROOT}/sequential${PYTHONPATH:+:$PYTHONPATH}"
export HARNESS_LIVE="${HARNESS_LIVE:-0}"
export SEQUENTIAL_HOST="${SEQUENTIAL_HOST:-127.0.0.1}"
export SEQUENTIAL_PORT="${SEQUENTIAL_PORT:-8766}"
cd "${ROOT}/sequential"
exec "${ROOT}/envs/rpent/bin/python" adapter.py
