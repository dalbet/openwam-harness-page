#!/usr/bin/env bash
# Stop only the processes this project recorded. The old sequential stack is
# never matched by pattern.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PID="$ROOT/records"

stop_pidfile() {
  local name="$1" file="$PID/$2"
  if [ ! -f "$file" ]; then
    echo "[skip] $name (no pid file)"
    return
  fi
  local pid
  pid="$(cat "$file" 2>/dev/null || true)"
  if [ -z "$pid" ]; then
    rm -f "$file"
    return
  fi
  if kill -0 "$pid" 2>/dev/null; then
    echo "[stop] $name pid=$pid"
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 50); do
      kill -0 "$pid" 2>/dev/null || break
      sleep 0.1
    done
    kill -9 "$pid" 2>/dev/null || true
  else
    echo "[gone] $name pid=$pid"
  fi
  rm -f "$file"
}

stop_pidfile page_ui page_ui.pid
stop_pidfile engine engine_service.pid
stop_pidfile agent agent_service.pid
stop_pidfile inference inference_service.pid
stop_pidfile pi05 pi05_server.pid
stop_pidfile "planner vLLM" planner_llm.pid

# The env server is owned by the engine; remove a stale record if it is gone.
if [ -f "$PID/env_server.pid" ]; then
  env_pid="$(cat "$PID/env_server.pid" 2>/dev/null || true)"
  if [ -n "$env_pid" ] && kill -0 "$env_pid" 2>/dev/null; then
    echo "[note] env server pid=$env_pid still alive (engine stop is bounded); not killed blindly"
  else
    rm -f "$PID/env_server.pid"
  fi
fi
echo "[done]"
