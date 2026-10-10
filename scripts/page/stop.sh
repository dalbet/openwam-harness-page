#!/usr/bin/env bash
# 默认只停止本项目的 Page UI。
# --all 同时停止本脚本确认过身份的模型服务，不碰来源不明的 8000。
# 用法: bash /root/Page_dmh/scripts/page/stop.sh [--all]
set -uo pipefail
source "$(dirname "$0")/common.sh"

stop_one() {
  local name="$1" pidf="$2" mark="$3" port="$4"
  local pid lp
  pid="$(read_pid "$pidf" || true)"
  lp="$(listen_pid "$port" || true)"
  local targets=()
  [[ -n "$pid" ]] && targets+=("$pid")
  if [[ -n "$lp" && "$lp" != "$pid" ]]; then
    targets+=("$lp")
  fi
  if [[ ${#targets[@]} -eq 0 ]]; then
    rm -f "$pidf"
    echo "$name: not running"
    return 0
  fi
  local stopped=0
  for t in "${targets[@]}"; do
    if ! pid_alive "$t"; then
      continue
    fi
    if ! cmdline_has "$t" "$mark"; then
      echo "$name: skip pid $t (cmdline is not $mark)"
      continue
    fi
    kill "$t" 2>/dev/null || true
    for _ in 1 2 3 4 5 6 7 8 9 10; do
      pid_alive "$t" || break
      sleep 0.3
    done
    if pid_alive "$t"; then
      kill -9 "$t" 2>/dev/null || true
    fi
    echo "$name: stopped pid $t"
    stopped=1
  done
  if [[ "$stopped" -eq 1 ]]; then
    rm -f "$pidf"
  elif [[ -n "$pid" ]] && ! pid_alive "$pid"; then
    rm -f "$pidf"
    echo "$name: stale pid file removed"
  fi
}

stop_one "Page UI" "$UI_PIDF" "$UI_MARK" 8765
if [[ "${1:-}" == "--all" ]]; then
  stop_one "VLM" "$VLM_PIDF" "$VLM_MARK" 8000
else
  echo "VLM left running. Use --all to stop the model owned by this project."
fi
