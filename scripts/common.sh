#!/usr/bin/env bash
# Shared paths for the LIBERO sequential entry points. Source this file.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HARNESS="${ROOT}/openwam_harness"
RUNTIME="${ROOT}/runtime"
LOG_DIR="${RUNTIME}/logs"
PID_DIR="${RUNTIME}/pids"
PY_GUARD="$(command -v python3)"

if [[ -f "${ROOT}/config/libero.env" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT}/config/libero.env"
fi

export PLANNER_HOST="${PLANNER_HOST:-127.0.0.1}"
export PLANNER_PORT="${PLANNER_PORT:-8000}"
export SEQUENTIAL_HOST="${SEQUENTIAL_HOST:-127.0.0.1}"
export SEQUENTIAL_PORT="${SEQUENTIAL_PORT:-8766}"
export SEQUENTIAL_CUDA_DEVICE="${SEQUENTIAL_CUDA_DEVICE:-1}"
export PLANNER_CUDA_DEVICE="${PLANNER_CUDA_DEVICE:-0}"
export PI05_CHECKPOINT_PATH="${PI05_CHECKPOINT_PATH:-${HARNESS}/checkpoints/RLinf-Pi05-LIBERO-130-fullshot-SFT}"
export SAM3_CHECKPOINT_PATH="${SAM3_CHECKPOINT_PATH:-${HARNESS}/checkpoints/sam3/sam3.pt}"
export PLANNER_MODEL_PATH="${PLANNER_MODEL_PATH:-${HARNESS}/checkpoints/Qwen3.5-9B}"
export PLANNER_MODEL="${PLANNER_MODEL:-Qwen3.5-9B}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://${PLANNER_HOST}:${PLANNER_PORT}/v1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export LIBERO_TYPE="${LIBERO_TYPE:-standard}"
export SEQUENTIAL_DEBUG="${SEQUENTIAL_DEBUG:-0}"

VLLM_PY="${HARNESS}/envs/vllm/bin/python"
VLLM_BIN="${HARNESS}/envs/vllm/bin/vllm"
ADAPTER_PY="${HARNESS}/envs/rpent/bin/python"
mkdir -p "${LOG_DIR}" "${PID_DIR}"

require_file() {
  local path="$1" label="$2"
  if [[ ! -e "${path}" ]]; then
    echo "missing ${label}: ${path}" >&2
    exit 1
  fi
}

rotate_log() {
  local path="$1"
  "${PY_GUARD}" "${HARNESS}/sequential/log_budget.py" rotate "${path}" 2000000
}

guard() {
  "${PY_GUARD}" "${ROOT}/scripts/proc_identity.py" "$@"
}
