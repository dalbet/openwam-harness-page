#!/usr/bin/env bash
# Start the integrated demo page on its own port (default 14901).
#
# The page source is the project's own page_ui (not a copy and not the
# experimental console): what is started here is the thing a user clicks.
# It is pointed at an isolated engine with ENGINE_URL, and the arm selector is
# enabled only because PAGE_ARMS is set. Production (14900 -> 14800) is never
# touched by this script; reuse of a port that is already listening is refused
# instead of being "cleaned up".
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PAGE_PY:-/home/host/page-dmh/openwam_harness/envs/rpent/bin/python}"
PORT="${PAGE_PORT:-14901}"
ENGINE="${ENGINE_URL:-http://127.0.0.1:14821}"
ARMS="${PAGE_ARMS:-$ROOT/config/arms.json}"
RUN_ID="page-demo-$(date +%Y%m%d-%H%M%S)"
LOG="$ROOT/page_ui/logs/page_demo_$RUN_ID.log"
REC="$ROOT/records/page_ui_demo.json"
mkdir -p "$ROOT/page_ui/logs"

if ss -ltn 2>/dev/null | grep -q ":$PORT "; then
  echo "port $PORT already in use; refusing to start a second page on it" >&2
  exit 4
fi
if [[ ! -f $ARMS ]]; then
  echo "arms config not found: $ARMS" >&2
  exit 5
fi

setsid env PAGE_PORT="$PORT" ENGINE_URL="$ENGINE" PAGE_ARMS="$ARMS" \
  "$PY" "$ROOT/page_ui/server.py" >"$LOG" 2>&1 &
PID=$!
sleep 0.5
PGID=$(ps -o pgid= -p "$PID" 2>/dev/null | tr -d ' ')

"$PY" - "$REC" "$RUN_ID" "$PID" "$PGID" "$PORT" "$ENGINE" "$ARMS" "$LOG" "$ROOT" <<'PY'
import json, os, sys, time
rec, run_id, pid, pgid, port, engine, arms, log, root = sys.argv[1:10]
pid = int(pid)
def read(path, mode="r"):
    with open(path, mode) as fh:
        return fh.read()
row = {
    "run_id": run_id,
    "pid": pid,
    "pgid": int(pgid) if pgid.isdigit() else pgid,
    "started_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
    "start_epoch": round(os.stat(f"/proc/{pid}").st_mtime, 3),
    "cmdline": read(f"/proc/{pid}/cmdline", "rb").replace(b"\0", b" ").decode().strip(),
    "cwd": os.readlink(f"/proc/{pid}/cwd"),
    "ppid": int(read(f"/proc/{pid}/stat").split()[3]),
    "port": int(port),
    "engine": engine,
    "arms": arms,
    "project_root": root,
    "log": log,
    "role": "page_ui demo (isolated); production page 14900 is not this process",
}
json.dump(row, open(rec, "w"), ensure_ascii=False, indent=1)
print(json.dumps({"run_id": run_id, "pid": pid, "port": port, "log": log}, ensure_ascii=False))
PY

for _ in $(seq 1 40); do
  if curl -sf --max-time 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo "[ok] page_ui demo on :$PORT (engine $ENGINE, arms $ARMS)"
    curl -s "http://127.0.0.1:$PORT/health"; echo
    exit 0
  fi
  sleep 1
done
echo "page did not answer /health on :$PORT; see $LOG" >&2
tail -20 "$LOG" >&2
exit 6
