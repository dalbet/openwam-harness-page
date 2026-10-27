#!/usr/bin/env bash
# Restart only the engine service (config changes are read at start-up).
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="/home/host/page-dmh/openwam_harness/envs/rpent/bin/python"
PORT=14800

pid="$(ss -ltnp "sport = :$PORT" 2>/dev/null | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2 || true)"
if [ -n "${pid:-}" ]; then
  echo "[stop] engine pid=$pid"
  kill "$pid" 2>/dev/null || true
  for _ in $(seq 1 50); do
    kill -0 "$pid" 2>/dev/null || break
    sleep 0.1
  done
  kill -9 "$pid" 2>/dev/null || true
fi
sleep 0.5
cd "$ROOT" || exit 1
nohup "$PY" engine/engine_service.py > records/logs/engine_service.log 2>&1 &
for _ in $(seq 1 60); do
  if curl -sf --max-time 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    new="$(ss -ltnp "sport = :$PORT" 2>/dev/null | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2 || true)"
    echo "$new" > records/engine_service.pid
    echo "[ok] engine pid=$new"
    exit 0
  fi
  sleep 0.5
done
echo "[fail] engine did not become ready" >&2
tail -20 records/logs/engine_service.log >&2
exit 1
