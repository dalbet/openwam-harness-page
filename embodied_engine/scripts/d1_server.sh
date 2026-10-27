#!/usr/bin/env bash
# Resident d1-3B decision server on its own GPU/port.
#
# Usage: d1_server.sh start|stop|restart|status
#
# The checkpoint, license and hash are read from config/engine.json
# agent.d1, so records/d1_server.json is the same identity the Agent
# service advertises on /health.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RECORD="$ROOT/records/d1_server.json"
PIDFILE="$ROOT/records/d1_server.pid"
LOG="$ROOT/records/logs"
mkdir -p "$LOG"

read_cfg() {
  python3 - "$1" <<'PY'
import json, sys
cfg = json.load(open('/home/host/page-dmh/embodied_engine/config/engine.json'))
key = sys.argv[1]
if key == 'port':
    value = cfg['services']['d1']['port']
else:
    value = cfg['agent']['d1'][key]
print(json.dumps(value) if isinstance(value, (dict, list)) else value)
PY
}

MODEL="$(read_cfg model)"
MODEL_PATH="$(read_cfg model_path)"
MODEL_SHA="$(read_cfg model_sha256)"
LICENSE="$(read_cfg license)"
PYTHON="$(read_cfg python)"
GPU="$(read_cfg gpu_index)"
PORT="$(read_cfg port)"

is_up() {
  python3 - "$PORT" <<'PY'
import json, sys, urllib.request
try:
    with urllib.request.urlopen(f"http://127.0.0.1:{sys.argv[1]}/health", timeout=2) as r:
        data = json.loads(r.read().decode("utf-8"))
    raise SystemExit(0 if data.get("status") == "ok" else 1)
except Exception:
    raise SystemExit(1)
PY
}

wait_up() {
  for _ in $(seq 1 "${1:-240}"); do
    if is_up; then return 0; fi
    sleep 2
  done
  return 1
}

cmd_start() {
  if is_up; then
    echo "[ok] d1 already listening on 127.0.0.1:$PORT"
    return 0
  fi
  echo "[start] d1 $MODEL on GPU$GPU port $PORT"
  cd "$ROOT"
  CUDA_VISIBLE_DEVICES="$GPU" nohup "$PYTHON" agent/d1_service.py \
    > "$LOG/d1_server.log" 2>&1 < /dev/null &
  local pid=$!
  echo "$pid" > "$PIDFILE"
  cat > "$RECORD" <<EOF
{
  "pid": $pid,
  "url": "http://127.0.0.1:$PORT",
  "port": $PORT,
  "gpu_index": $GPU,
  "python": "$PYTHON",
  "model": "$MODEL",
  "model_path": "$MODEL_PATH",
  "model_sha256": "$MODEL_SHA",
  "license": "$LICENSE",
  "dtype": "bfloat16",
  "output_tokens_per_decision": 0,
  "interface": "system_one (single forward pass, typed answers)",
  "started_at": "$(date -Iseconds)"
}
EOF
  if wait_up 240; then
    echo "[ok] d1 ready on http://127.0.0.1:$PORT"
  else
    echo "[fail] d1 did not become ready; see $LOG/d1_server.log" >&2
    return 1
  fi
}

cmd_stop() {
  local pid=""
  [ -f "$PIDFILE" ] && pid="$(cat "$PIDFILE")"
  if [ -z "$pid" ]; then
    pid="$(pgrep -f "agent/d1_service.py" | head -1 || true)"
  fi
  if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
    if ! tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | grep -q "agent/d1_service.py"; then
      echo "[warn] pid $pid is not the d1 service; refusing to stop it" >&2
      return 1
    fi
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 20); do
      kill -0 "$pid" 2>/dev/null || break
      sleep 0.5
    done
    kill -9 "$pid" 2>/dev/null || true
    echo "[stop] d1 pid $pid"
  else
    echo "[ok] d1 not running"
  fi
  rm -f "$PIDFILE"
}

cmd_status() {
  local state="down"
  if is_up; then state="listening"; fi
  echo "== d1 server =="
  echo "url          : http://127.0.0.1:$PORT ($state)"
  echo "gpu          : $GPU"
  echo "model        : $MODEL"
  echo "checkpoint   : $MODEL_PATH"
  echo "license      : $LICENSE"
  if [ -f "$RECORD" ]; then
    echo "-- launch record --"
    cat "$RECORD"
  fi
  curl -s --max-time 5 "http://127.0.0.1:$PORT/health" | python3 -m json.tool 2>/dev/null || echo "health unavailable"
}

case "${1:-status}" in
  start) cmd_start ;;
  stop) cmd_stop ;;
  restart) cmd_stop; cmd_start ;;
  status) cmd_status ;;
  *) echo "usage: $0 start|stop|restart|status" >&2; exit 2 ;;
esac
