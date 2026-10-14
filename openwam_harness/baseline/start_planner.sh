#!/usr/bin/env bash
# Local Qwen3.5-9B planner for the original Harness API loop.
# Reuses the existing Show-Harness vLLM 0.24.0 environment.
# FlashInfer's sampler JIT does not see this Blackwell GPU (CUDA < 12.9),
# so sampling stays on the PyTorch path. That does not change Pi0.5.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export VLLM_USE_FLASHINFER_SAMPLER=0
exec "${ROOT}/../Show-Harness/.venv-vllm/bin/vllm" serve \
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
