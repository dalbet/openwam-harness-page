#!/usr/bin/env bash
# Local Qwen3.5-9B planner for the original Harness API loop.
# Uses this project's vLLM 0.24.0 environment.
# FlashInfer's sampler JIT does not see this Blackwell GPU (CUDA < 12.9),
# so sampling stays on the PyTorch path. That does not change Pi0.5.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export VLLM_USE_FLASHINFER_SAMPLER=0

# Planner binds localhost. Do not send that traffic through an outbound proxy.
_local_no_proxy="127.0.0.1,localhost,::1"
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}${_local_no_proxy}"
export no_proxy="${NO_PROXY}"

# Triton compiles driver.c, which includes Python.h. This venv is system
# CPython, and its include directory has no development headers. This account
# cannot install them. Use the uv-managed CPython of the same minor version.
_py_mm="$("${ROOT}/envs/vllm/bin/python" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')"
_uv_bin=""
if command -v uv >/dev/null 2>&1; then
  _uv_bin="$(command -v uv)"
elif [[ -x "${HOME}/.local/bin/uv" ]]; then
  _uv_bin="${HOME}/.local/bin/uv"
fi
_header_dir=""
if [[ -n "${_uv_bin}" ]]; then
  _uv_py="$("${_uv_bin}" python find "${_py_mm}" 2>/dev/null || true)"
  if [[ -n "${_uv_py}" && -e "${_uv_py}" ]]; then
    _uv_root="$(cd "$(dirname "${_uv_py}")/.." && pwd -P)"
    if [[ -f "${_uv_root}/include/python${_py_mm}/Python.h" ]]; then
      _header_dir="${_uv_root}/include/python${_py_mm}"
    fi
  fi
fi
if [[ -z "${_header_dir}" ]]; then
  echo "start_planner: Python.h for CPython ${_py_mm} was not found. Triton cannot compile cuda_utils." >&2
  exit 1
fi
case ":${C_INCLUDE_PATH:-}:" in
  *":${_header_dir}:"*) ;;
  *) export C_INCLUDE_PATH="${_header_dir}${C_INCLUDE_PATH:+:${C_INCLUDE_PATH}}" ;;
esac

exec "${ROOT}/envs/vllm/bin/vllm" serve \
  "${ROOT}/checkpoints/Qwen3.5-9B" \
  --served-model-name Qwen3.5-9B \
  --host 127.0.0.1 \
  --port 8000 \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --max-num-seqs 1 \
  --gpu-memory-utilization 0.48 \
  --cpu-offload-gb 6 \
  --enforce-eager \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --limit-mm-per-prompt '{"image":32}' \
  --trust-remote-code
