#!/usr/bin/env bash
# Start the local Qwen planner. A second copy is not started.
# Process start, HTTP ready, and weight load are reported separately.
set -euo pipefail
source "$(dirname "$0")/common.sh"

require_file "${PLANNER_MODEL_PATH}/config.json" "planner checkpoint"
require_file "${VLLM_BIN}" "vllm executable"
require_file "${VLLM_PY}" "planner interpreter"

decision="$(guard classify --name planner --port "${PLANNER_PORT}")"
case "${decision}" in
  free)
    ;;
  project\ *)
    echo "planner already running pid ${decision#project } on ${PLANNER_HOST}:${PLANNER_PORT}"
    exit 0
    ;;
  foreign\ *)
    echo "port ${PLANNER_PORT} is held by pid ${decision#foreign }, which is not this project's planner. Not stopping it." >&2
    exit 2
    ;;
  *)
    echo "could not classify planner port ${PLANNER_PORT}: ${decision}" >&2
    exit 2
    ;;
esac

log="${LOG_DIR}/planner.log"
rotate_log "${log}"
echo "starting planner process"
CUDA_VISIBLE_DEVICES="${PLANNER_CUDA_DEVICE}" \
  nohup bash "${HARNESS}/baseline/start_planner.sh" >>"${log}" 2>&1 &
pid=$!
echo "planner process started pid ${pid}. Weights are not ready yet."

deadline=$((SECONDS + 20))
while (( SECONDS < deadline )); do
  if ! kill -0 "${pid}" 2>/dev/null; then
    echo "planner process exited before vllm replaced it. See ${log}" >&2
    exit 1
  fi
  if tr '\0' ' ' < "/proc/${pid}/cmdline" 2>/dev/null | grep -F -q "vllm"; then
    break
  fi
  sleep 0.2
done
if ! tr '\0' ' ' < "/proc/${pid}/cmdline" 2>/dev/null | grep -F -q "vllm"; then
  echo "planner process did not become vllm. See ${log}" >&2
  exit 1
fi
guard write --name planner --pid "${pid}"

echo "waiting for planner ready at ${OPENAI_BASE_URL}/models"
ready_deadline=$((SECONDS + 300))
while (( SECONDS < ready_deadline )); do
  if ! kill -0 "${pid}" 2>/dev/null; then
    echo "planner process exited while loading. See ${log}" >&2
    exit 1
  fi
  if curl -sf --max-time 3 "${OPENAI_BASE_URL}/models" | grep -F -q "\"${PLANNER_MODEL}\""; then
    echo "planner ready pid ${pid}. Model weights are loaded: ${PLANNER_MODEL}"
    exit 0
  fi
  sleep 2
done
echo "planner process is still up pid ${pid}, but ${OPENAI_BASE_URL}/models did not report ${PLANNER_MODEL} within 300s. See ${log}" >&2
exit 1
