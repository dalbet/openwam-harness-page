#!/usr/bin/env bash
# 幂等启动 Page。已有健康的本模型服务则复用，不重复占 GPU。
# 用法: bash /root/Page_dmh/scripts/page/start.sh
set -uo pipefail
source "$(dirname "$0")/common.sh"
page_env

start_model() {
  if model_healthy; then
    local lp
    lp="$(listen_pid 8000 || true)"
    if [[ -n "$lp" ]] && cmdline_has "$lp" "$VLM_MARK"; then
      echo "$lp" > "$VLM_PIDF"
      echo "VLM already healthy, reuse pid $lp"
      return 0
    fi
    echo "8000 already serves $MODEL_NAME (pid ${lp:-unknown}) but it is not $VLM_MARK."
    echo "Reuse it. Not starting a second model and not stopping it."
    return 0
  fi
  local lp
  lp="$(listen_pid 8000 || true)"
  if [[ -n "$lp" ]]; then
    echo "8000 is occupied by pid $lp and is not serving $MODEL_NAME. Not starting a second model."
    return 2
  fi
  local old
  old="$(read_pid "$VLM_PIDF" || true)"
  if [[ -n "$old" ]] && pid_alive "$old" && cmdline_has "$old" "$VLM_MARK"; then
    echo "VLM process $old is starting or unhealthy. Not starting a second copy."
    return 0
  fi
  echo "starting VLM (transformers+peft) on 127.0.0.1:8000"
  nohup "$VLM_PY" "$PUI/vlm_server.py" >> "$VLM_LOG" 2>&1 &
  echo $! > "$VLM_PIDF"
  echo "  pid $(cat "$VLM_PIDF") log $VLM_LOG"
}

start_ui() {
  local lp old
  lp="$(listen_pid 8765 || true)"
  if [[ -n "$lp" ]] && cmdline_has "$lp" "$UI_MARK"; then
    echo "$lp" > "$UI_PIDF"
    echo "Page UI already running, reuse pid $lp"
    return 0
  fi
  if [[ -n "$lp" ]]; then
    echo "8765 is occupied by pid $lp, which is not $UI_MARK. Not starting Page UI."
    return 2
  fi
  old="$(read_pid "$UI_PIDF" || true)"
  if [[ -n "$old" ]] && pid_alive "$old" && cmdline_has "$old" "$UI_MARK"; then
    echo "Page UI pid $old is alive but not listening yet. Not starting a second copy."
    return 0
  fi
  echo "starting Page UI on 127.0.0.1:8765"
  nohup "$UI_PY" "$PUI/server.py" >> "$UI_LOG" 2>&1 &
  echo $! > "$UI_PIDF"
  echo "  pid $(cat "$UI_PIDF") log $UI_LOG"
}

start_model || exit $?
start_ui
