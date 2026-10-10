#!/usr/bin/env bash
# 兼容入口。真正的逻辑在 /root/Page_dmh/scripts/page/。
# stop 默认只停 UI；要同时停本项目的模型请用 stop --all。
set -uo pipefail
DIR=/root/Page_dmh/scripts/page
case "${1:-status}" in
  start) exec bash "$DIR/start.sh" ;;
  stop) exec bash "$DIR/stop.sh" ${2:+$2} ;;
  status) exec bash "$DIR/status.sh" ;;
  restart)
    bash "$DIR/stop.sh" --all
    exec bash "$DIR/start.sh"
    ;;
  *) echo "usage: $0 {start|stop [--all]|restart|status}"; exit 1 ;;
esac
