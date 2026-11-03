#!/usr/bin/env bash
# Stop the integrated demo page — only the process this project started.
#
# Verification (all must hold): the pid in records/page_ui_demo.json is alive,
# its start time matches the recorded one, its cwd is the recorded project
# root, and it is the process listening on the recorded port. Anything else is
# refused. No pkill/killall and no pattern matching: the pid is the target.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REC="$ROOT/records/page_ui_demo.json"
PY="${PAGE_PY:-/home/host/page-dmh/openwam_harness/envs/rpent/bin/python}"
[[ -f $REC ]] || { echo "no demo page record at $REC"; exit 0; }
mapfile -t row < <("$PY" - "$REC" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
print(r["pid"]); print(r["start_epoch"]); print(r["port"]); print(r["project_root"])
PY
)
PID=${row[0]}; START=${row[1]}; PORT=${row[2]}; PROJ=${row[3]}
kill -0 "$PID" 2>/dev/null || { echo "pid $PID already gone"; exit 0; }
NOW_START=$($PY -c "import os;print(round(os.stat('/proc/$PID').st_mtime,3))" 2>/dev/null || true)
CWD=$(readlink "/proc/$PID/cwd" 2>/dev/null || true)
LISTEN=$(ss -ltnp 2>/dev/null | grep ":$PORT " | grep -c "pid=$PID," || true)
if [[ "$NOW_START" != "$START" || "$CWD" != "$PROJ" || "$LISTEN" != "1" ]]; then
  echo "REFUSING to stop pid $PID: start=$NOW_START(rec $START) cwd=$CWD(listen=$LISTEN)" >&2
  exit 7
fi
kill "$PID"
for _ in $(seq 1 50); do kill -0 "$PID" 2>/dev/null || { echo "stopped page pid $PID"; exit 0; }; sleep 0.2; done
echo "pid $PID did not exit" >&2; exit 8
