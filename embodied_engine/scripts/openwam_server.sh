#!/usr/bin/env bash
# Resident OpenWAM policy server (WebSocket) on its own GPU/port.
#
# Usage: openwam_server.sh start|stop|status|restart
#
# Everything that defines the served contract is read from
# config/engine.json action_model.openwam, so the launch record written to
# records/openwam_server.json is the same values the engine advertises:
# checkpoint file, generate_horizon (model-side 32-step chunk),
# execute_horizon (server inference_horizon: actions executed before
# re-planning), denoise_steps and the GPU index.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RECORD="$ROOT/records/openwam_server.json"
LOG="$ROOT/records/logs"
mkdir -p "$LOG"

read_cfg() {
  /home/host/page-dmh/openwam_harness/envs/rpent/bin/python - "$1" <<'PY'
import json, sys
cfg = json.load(open('/home/host/page-dmh/embodied_engine/config/engine.json'))['action_model']['openwam']
key = sys.argv[1]
value = cfg[key]
print(json.dumps(value) if isinstance(value, (dict, list)) else value)
PY
}

CKPT_DIR="$(read_cfg checkpoint)"
CKPT_FILE="$(read_cfg checkpoint_file)"
REPO="$(read_cfg repo)"
PYTHON="$(read_cfg python)"
PORT="$(read_cfg server_port)"
GPU="$(read_cfg gpu_index)"
GEN_H="$(read_cfg generate_horizon)"
EXEC_H="$(read_cfg execute_horizon)"
STEPS="$(read_cfg denoise_steps)"

is_up() {
  "$PYTHON" - "$PORT" <<'PY'
import json, sys
from websockets.sync.client import connect
try:
    with connect(f"ws://127.0.0.1:{sys.argv[1]}", open_timeout=2, close_timeout=2, ping_interval=None) as sock:
        sock.send(json.dumps({"type": "ping"}))
        reply = json.loads(sock.recv(timeout=10) or "{}")
        raise SystemExit(0 if reply.get("type") == "pong" else 1)
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
    echo "[ok] OpenWAM already listening on 127.0.0.1:$PORT"
    return 0
  fi
  echo "[start] OpenWAM on GPU$GPU port $PORT (generate_horizon=$GEN_H execute_horizon=$EXEC_H steps=$STEPS)"
  cd "$REPO"
  CUDA_VISIBLE_DEVICES="$GPU" nohup "$PYTHON" scripts/deploy.py \
    --ckpt-dir "$CKPT_DIR" \
    --ckpt-name "$CKPT_FILE" \
    --device cuda:0 \
    --host 127.0.0.1 \
    --port "$PORT" \
    --compile-enabled false \
    --inference-mode sync \
    --denoise-steps "$STEPS" \
    --inference-horizon "$EXEC_H" \
    > "$LOG/openwam_server.log" 2>&1 < /dev/null &
  local pid=$!
  echo "$pid" > "$ROOT/records/openwam_server.pid"
  cat > "$RECORD" <<EOF
{
  "pid": $pid,
  "url": "ws://127.0.0.1:$PORT",
  "port": $PORT,
  "gpu_index": $GPU,
  "python": "$PYTHON",
  "repo": "$REPO",
  "repo_commit": "$(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo unknown)",
  "checkpoint_dir": "$CKPT_DIR",
  "checkpoint_file": "$CKPT_FILE",
  "generate_horizon": $GEN_H,
  "execute_horizon": $EXEC_H,
  "denoise_steps": $STEPS,
  "inference_mode": "sync",
  "compile_enabled": false,
  "started_at": "$(date -Iseconds)"
}
EOF
  if wait_up 300; then
    echo "[ok] OpenWAM ready on ws://127.0.0.1:$PORT"
  else
    echo "[fail] OpenWAM did not become ready; see $LOG/openwam_server.log" >&2
    return 1
  fi
}

cmd_stop() {
  local pid=""
  [ -f "$ROOT/records/openwam_server.pid" ] && pid="$(cat "$ROOT/records/openwam_server.pid")"
  if [ -z "$pid" ]; then
    pid="$(pgrep -f "scripts/deploy.py.*--port $PORT" | head -1 || true)"
  fi
  if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 20); do
      kill -0 "$pid" 2>/dev/null || break
      sleep 0.5
    done
    kill -9 "$pid" 2>/dev/null || true
    echo "[stop] OpenWAM pid $pid"
  else
    echo "[ok] OpenWAM not running"
  fi
  rm -f "$ROOT/records/openwam_server.pid"
}

cmd_status() {
  local state="down"
  if is_up; then state="listening"; fi
  echo "== OpenWAM server =="
  echo "url          : ws://127.0.0.1:$PORT ($state)"
  echo "gpu          : $GPU"
  echo "checkpoint   : $CKPT_DIR/$CKPT_FILE"
  echo "generate_h   : $GEN_H"
  echo "execute_h    : $EXEC_H"
  echo "denoise      : $STEPS"
  if [ -f "$RECORD" ]; then
    echo "-- launch record --"
    cat "$RECORD"
  fi
  curl -s --max-time 5 http://127.0.0.1:14600/health | /home/host/page-dmh/openwam_harness/envs/rpent/bin/python -c '
import json, sys
try:
    d = json.load(sys.stdin)
except Exception:
    print("inference service unreachable"); raise SystemExit(0)
ident = d.get("identity") or {}
print("-- inference identity --")
print("backend      :", ident.get("backend"))
print("model        :", ident.get("name"))
print("mode         :", d.get("mode"))
print("integration  :", (ident.get("openwam") or {}).get("integration"))
print("reason       :", (ident.get("openwam") or {}).get("reason"))
print("server       :", json.dumps(ident.get("server"), ensure_ascii=False))
'
}

case "${1:-status}" in
  start) cmd_start ;;
  stop) cmd_stop ;;
  restart) cmd_stop; cmd_start ;;
  status) cmd_status ;;
  *) echo "usage: $0 start|stop|restart|status" >&2; exit 2 ;;
esac
