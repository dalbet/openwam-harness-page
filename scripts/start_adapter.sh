#!/usr/bin/env bash
# Start the live sequential adapter. Missing weights are an error.
# This does not load Pi0.5, SAM3, or LIBERO until the page loads a scene.
set -euo pipefail
source "$(dirname "$0")/common.sh"

if [[ "${HARNESS_LIVE:-1}" != "1" ]]; then
  echo "refusing to start. This entry only runs the live backend (HARNESS_LIVE=1), and it will not fall back to dry mode." >&2
  exit 1
fi
export HARNESS_LIVE=1

require_file "${PI05_CHECKPOINT_PATH}" "Pi0.5 checkpoint"
require_file "${SAM3_CHECKPOINT_PATH}" "SAM3 checkpoint"
require_file "${HARNESS}/RPent/robots/libero/robot_spec.py" "LIBERO robot spec"
require_file "${ADAPTER_PY}" "adapter interpreter"
require_file "${HARNESS}/sequential/adapter.py" "adapter.py"

decision="$(guard classify --name adapter --port "${SEQUENTIAL_PORT}")"
case "${decision}" in
  free)
    ;;
  project\ *)
    echo "adapter already running pid ${decision#project } on ${SEQUENTIAL_HOST}:${SEQUENTIAL_PORT}"
    echo "service ready can be checked with scripts/status.sh. Model load still waits for a scene."
    exit 0
    ;;
  foreign\ *)
    echo "port ${SEQUENTIAL_PORT} is held by pid ${decision#foreign }, which is not this project's adapter. Not stopping it." >&2
    exit 2
    ;;
  *)
    echo "could not classify adapter port ${SEQUENTIAL_PORT}: ${decision}" >&2
    exit 2
    ;;
esac

log="${LOG_DIR}/adapter.log"
rotate_log "${log}"
echo "starting adapter process HARNESS_LIVE=1"
HARNESS_LIVE=1 SEQUENTIAL_CUDA_DEVICE="${SEQUENTIAL_CUDA_DEVICE}" \
  nohup bash "${HARNESS}/sequential/start_adapter.sh" >>"${log}" 2>&1 &
pid=$!
echo "adapter process started pid ${pid}. HTTP ready and model load are separate."

deadline=$((SECONDS + 20))
while (( SECONDS < deadline )); do
  if ! kill -0 "${pid}" 2>/dev/null; then
    echo "adapter process exited before it served the port. See ${log}" >&2
    exit 1
  fi
  if tr '\0' ' ' < "/proc/${pid}/cmdline" 2>/dev/null | grep -F -q "adapter.py"; then
    break
  fi
  sleep 0.2
done
if ! tr '\0' ' ' < "/proc/${pid}/cmdline" 2>/dev/null | grep -F -q "adapter.py"; then
  echo "adapter process did not become adapter.py. See ${log}" >&2
  exit 1
fi
guard write --name adapter --pid "${pid}"

echo "waiting for adapter HTTP ready"
ready_deadline=$((SECONDS + 30))
while (( SECONDS < ready_deadline )); do
  if ! kill -0 "${pid}" 2>/dev/null; then
    echo "adapter process exited before HTTP ready. See ${log}" >&2
    exit 1
  fi
  if curl -sf --max-time 3 "http://${SEQUENTIAL_HOST}:${SEQUENTIAL_PORT}/health" | grep -F -q '"adapter": "online"'; then
    echo "adapter HTTP ready pid ${pid} on ${SEQUENTIAL_HOST}:${SEQUENTIAL_PORT}"
    echo "Pi0.5, SAM3, and LIBERO are not loaded yet. They start when a scene is loaded."
    exit 0
  fi
  sleep 0.5
done
echo "adapter process is still up pid ${pid}, but /health did not become ready within 30s. See ${log}" >&2
exit 1
