#!/usr/bin/env bash
# 用法: bash /root/Page_dmh/scripts/page/status.sh
set -uo pipefail
source "$(dirname "$0")/common.sh"

show_one() {
  local name="$1" pidf="$2" mark="$3" port="$4"
  local pid lp
  pid="$(read_pid "$pidf" || true)"
  lp="$(listen_pid "$port" || true)"
  echo "== $name port $port =="
  if [[ -n "$pid" ]]; then
    if pid_alive "$pid" && cmdline_has "$pid" "$mark"; then
      echo "pidfile: $pid identity=ok"
    elif pid_alive "$pid"; then
      echo "pidfile: $pid identity=MISMATCH"
    else
      echo "pidfile: $pid not alive"
    fi
  else
    echo "pidfile: none"
  fi
  if [[ -n "$lp" ]]; then
    if cmdline_has "$lp" "$mark"; then
      echo "listen: $lp identity=ok"
    else
      echo "listen: $lp identity=OTHER"
      tr '\0' ' ' < "/proc/$lp/cmdline" 2>/dev/null || true
      echo
    fi
  else
    echo "listen: none"
  fi
}

show_one "Page UI" "$UI_PIDF" "$UI_MARK" 8765
if curl -sf --max-time 3 -o /dev/null "http://127.0.0.1:8765/"; then
  echo "http://127.0.0.1:8765/ : ok"
else
  echo "http://127.0.0.1:8765/ : down"
fi
show_one "VLM" "$VLM_PIDF" "$VLM_MARK" 8000
if model_healthy; then
  echo "http://127.0.0.1:8000/v1/models : $MODEL_NAME"
else
  echo "http://127.0.0.1:8000/v1/models : not our model or down"
fi
