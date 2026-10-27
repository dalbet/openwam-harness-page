#!/usr/bin/env bash
# Sequential Page adapter.
# HARNESS_LIVE=1 loads LIBERO, Pi0.5, and SAM3. It is passed to the child explicitly.
# If this project's adapter already holds the port, that pid is replaced.
# A foreign listener is left running and this script exits.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export LIBERO_TYPE="${LIBERO_TYPE:-standard}"
export PYTHONPATH="${ROOT}/sequential${PYTHONPATH:+:$PYTHONPATH}"
export HARNESS_LIVE="${HARNESS_LIVE:-0}"
export SEQUENTIAL_HOST="${SEQUENTIAL_HOST:-127.0.0.1}"
export SEQUENTIAL_PORT="${SEQUENTIAL_PORT:-8766}"
export SEQUENTIAL_LOAD_TIMEOUT_S="${SEQUENTIAL_LOAD_TIMEOUT_S:-300}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://127.0.0.1:8000/v1}"
export SEQUENTIAL_DISPLAY_HOOK="${SEQUENTIAL_DISPLAY_HOOK:-1}"
# Planner stays on GPU 0. LIBERO, Pi0.5, and SAM3 use the next GPU.
export SEQUENTIAL_CUDA_DEVICE="${SEQUENTIAL_CUDA_DEVICE:-1}"
_local_no_proxy="127.0.0.1,localhost,::1"
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}${_local_no_proxy}"
export no_proxy="${NO_PROXY}"
PY="${ROOT}/envs/rpent/bin/python"
GUARD="${ROOT}/sequential/process_guard.py"

decision="$("$PY" "$GUARD" --port "$SEQUENTIAL_PORT" || true)"
case "$decision" in
  free)
    ;;
  project\ *)
    echo "port ${SEQUENTIAL_PORT} is held by project adapter pid ${decision#project }. Replacing it. Not reusing the old process."
    "$PY" "$GUARD" --port "$SEQUENTIAL_PORT" --stop
    ;;
  foreign\ *)
    echo "port ${SEQUENTIAL_PORT} is held by pid ${decision#foreign }, which is not this project's adapter. Not stopping it." >&2
    exit 2
    ;;
  *)
    echo "could not classify port ${SEQUENTIAL_PORT}: ${decision}" >&2
    exit 2
    ;;
esac

cd "${ROOT}/sequential"
echo "starting sequential adapter HARNESS_LIVE=${HARNESS_LIVE} on ${SEQUENTIAL_HOST}:${SEQUENTIAL_PORT}"
exec env \
  HARNESS_LIVE="${HARNESS_LIVE}" \
  SEQUENTIAL_HOST="${SEQUENTIAL_HOST}" \
  SEQUENTIAL_PORT="${SEQUENTIAL_PORT}" \
  SEQUENTIAL_LOAD_TIMEOUT_S="${SEQUENTIAL_LOAD_TIMEOUT_S}" \
  OPENAI_BASE_URL="${OPENAI_BASE_URL}" \
  SEQUENTIAL_DISPLAY_HOOK="${SEQUENTIAL_DISPLAY_HOOK}" \
  SEQUENTIAL_CUDA_DEVICE="${SEQUENTIAL_CUDA_DEVICE}" \
  HF_HUB_OFFLINE="${HF_HUB_OFFLINE}" \
  MUJOCO_GL="${MUJOCO_GL}" \
  LIBERO_TYPE="${LIBERO_TYPE}" \
  PYTHONPATH="${PYTHONPATH}" \
  "$PY" adapter.py
