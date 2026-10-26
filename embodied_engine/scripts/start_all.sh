#!/usr/bin/env bash
# Start the three-layer engine on its own ports. Does not touch the old
# sequential stack (adapter :8766, RPent Pi0.5 :44481, SAM3 :60071).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY_RPENT="/home/host/page-dmh/openwam_harness/envs/rpent/bin/python"
CKPT="/home/host/page-dmh/openwam_harness/checkpoints/RLinf-Pi05-LIBERO-130-fullshot-SFT"
LOG="$ROOT/records/logs"
PID="$ROOT/records"
mkdir -p "$LOG" "$PID"

wait_http() {
  local url="$1" name="$2" tries="${3:-120}"
  for _ in $(seq 1 "$tries"); do
    if curl -sf --max-time 2 "$url" >/dev/null 2>&1; then
      echo "[ok] $name"
      return 0
    fi
    sleep 1
  done
  echo "[fail] $name did not become ready: $url" >&2
  return 1
}

wait_rpc() {
  # RPent facades expose no GET routes: readiness is an RPC "healthz" call.
  local url="$1" name="$2" tries="${3:-120}"
  for _ in $(seq 1 "$tries"); do
    if curl -sf --max-time 2 -X POST "$url/call" \
        -H 'Content-Type: application/json' \
        -d '{"method":"healthz","args":[],"kwargs":{}}' >/dev/null 2>&1; then
      echo "[ok] $name"
      return 0
    fi
    sleep 1
  done
  echo "[fail] $name did not become ready: $url/call healthz" >&2
  return 1
}

# 0. Dedicated planner LLM instance on GPU3 (:8001). The shared :8000 instance
#    is heavily loaded by other users and measured ~9s per planning call; the
#    dedicated instance measured ~1.1s on the same prompt. If GPU3 is busy the
#    agent falls back to whatever base_url config/engine.json points at.
VLLM_PY="/home/host/page-dmh/openwam_harness/envs/vllm/bin/python"
if [ -x "$VLLM_PY" ] && ! curl -sf --max-time 2 http://127.0.0.1:8001/v1/models >/dev/null 2>&1; then
  echo "[start] planner vLLM :8001"
  CPATH=/home/host/.local/share/uv/python/cpython-3.10.22-linux-x86_64-gnu/include/python3.10 \
  VLLM_USE_FLASHINFER_SAMPLER=0 \
  CUDA_VISIBLE_DEVICES=3 \
  nohup "$VLLM_PY" -m vllm.entrypoints.openai.api_server \
    --model /home/host/page-dmh/openwam_harness/checkpoints/Qwen3.5-9B \
    --served-model-name Qwen3.5-9B --host 127.0.0.1 --port 8001 \
    --gpu-memory-utilization 0.72 --max-model-len 16384 --enforce-eager \
    --dtype bfloat16 --trust-remote-code --max-num-seqs 8 \
    > "$LOG/vllm_agent_gpu3.log" 2>&1 &
  echo $! > "$PID/planner_llm.pid"
  wait_http http://127.0.0.1:8001/v1/models "planner vLLM" 240
else
  echo "[ok] planner vLLM already listening on :8001 (or unavailable)"
fi

# 1. Resident Pi0.5 policy server (14510), reused by the inference layer.
if ! curl -sf --max-time 2 -X POST http://127.0.0.1:14510/call -H 'Content-Type: application/json' -d '{"method":"healthz","args":[],"kwargs":{}}' >/dev/null 2>&1; then
  echo "[start] Pi0.5 :14510"
  nohup "$PY_RPENT" /home/host/page-dmh/openwam_harness/RPent/rpent/robots/components/pi05_vla_server.py \
    --embodiment libero --transport http --host 127.0.0.1 --port 14510 \
    --model-path "$CKPT" \
    --cuda-device 1 \
    > "$LOG/pi05_server.log" 2>&1 < /dev/null &
  echo $! > "$PID/pi05_server.pid"
  wait_rpc http://127.0.0.1:14510 "Pi0.5" 180
else
  echo "[ok] Pi0.5 already listening on :14510"
fi

# 2. Inference service (queue, token isolation, bounded prefetch).
if ! curl -sf --max-time 2 http://127.0.0.1:14600/health >/dev/null 2>&1; then
  echo "[start] inference :14600"
  nohup "$PY_RPENT" "$ROOT/inference/inference_service.py" \
    > "$LOG/inference_service.log" 2>&1 &
  echo $! > "$PID/inference_service.pid"
  wait_http http://127.0.0.1:14600/health "inference" 60
else
  echo "[ok] inference already listening on :14600"
fi

# 3. Agent service (Qwen planner + structured recovery).
if ! curl -sf --max-time 2 http://127.0.0.1:14700/health >/dev/null 2>&1; then
  echo "[start] agent :14700"
  nohup "$PY_RPENT" "$ROOT/agent/agent_service.py" \
    > "$LOG/agent_service.log" 2>&1 &
  echo $! > "$PID/agent_service.pid"
  wait_http http://127.0.0.1:14700/health "agent" 30
else
  echo "[ok] agent already listening on :14700"
fi

# 4. Engine (owns the LIBERO env subprocess and episodes).
if ! curl -sf --max-time 2 http://127.0.0.1:14800/health >/dev/null 2>&1; then
  echo "[start] engine :14800"
  nohup "$PY_RPENT" "$ROOT/engine/engine_service.py" \
    > "$LOG/engine_service.log" 2>&1 &
  echo $! > "$PID/engine_service.pid"
  wait_http http://127.0.0.1:14800/health "engine" 30
else
  echo "[ok] engine already listening on :14800"
fi

# 5. Restored Page UI bridge (14900).
if ! curl -sf --max-time 2 http://127.0.0.1:14900/health >/dev/null 2>&1; then
  echo "[start] page_ui :14900"
  nohup "$PY_RPENT" "$ROOT/page_ui/server.py" \
    > "$LOG/page_ui.log" 2>&1 &
  echo $! > "$PID/page_ui.pid"
  wait_http http://127.0.0.1:14900/health "page_ui" 30
else
  echo "[ok] page_ui already listening on :14900"
fi

echo "[done] UI http://127.0.0.1:14900  engine http://127.0.0.1:14800"
