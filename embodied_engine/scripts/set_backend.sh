#!/usr/bin/env bash
# Select the action-model backend and restart only the inference service.
#
# Usage: set_backend.sh openwam|pi05
#
# Weights stay resident: Pi0.5 keeps running on :14510 and OpenWAM keeps
# running on :14520 either way, so switching is a routing decision, not a
# reload. The switch is explicit — an unknown name is rejected here and by
# inference/inference_service.py::_build_backend().
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY_RPENT="/home/host/page-dmh/openwam_harness/envs/rpent/bin/python"
TARGET="${1:-}"

case "$TARGET" in
  openwam|pi05) ;;
  *) echo "usage: $0 openwam|pi05" >&2; exit 2 ;;
esac

"$PY_RPENT" - "$TARGET" <<'PY'
import json, pathlib, re, sys
target = sys.argv[1]
path = pathlib.Path('/home/host/page-dmh/embodied_engine/config/engine.json')
text = path.read_text(encoding='utf-8')
pattern = re.compile(r'("action_model"\s*:\s*\{[^{}]*?"backend"\s*:\s*")([a-z0-9]+)(")', re.S)
match = pattern.search(text)
if not match:
    raise SystemExit('engine.json: action_model.backend not found')
text = text[: match.start(2)] + target + text[match.end(2):]
path.write_text(text, encoding='utf-8')
print('action_model.backend =', json.loads(text)['action_model']['backend'])
PY

if [ "$TARGET" = "openwam" ]; then
  if ! "$ROOT/scripts/openwam_server.sh" status | grep -q "listening"; then
    echo "[start] OpenWAM server is down; starting it (this takes ~2 min)"
    "$ROOT/scripts/openwam_server.sh" start
  fi
fi

pid="$(cat "$ROOT/records/inference_service.pid" 2>/dev/null || true)"
if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
  kill "$pid"
  for _ in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || break; sleep 0.2; done
fi
nohup "$PY_RPENT" "$ROOT/inference/inference_service.py" \
  > "$ROOT/records/logs/inference_service.log" 2>&1 &
echo $! > "$ROOT/records/inference_service.pid"

for _ in $(seq 1 40); do
  if curl -sf --max-time 3 http://127.0.0.1:14600/health >/dev/null 2>&1; then break; fi
  sleep 1
done
head -1 "$ROOT/records/logs/inference_service.log"
curl -s --max-time 10 http://127.0.0.1:14600/health | "$PY_RPENT" -c '
import json, sys
d = json.load(sys.stdin)
i = d.get("identity") or {}
om = i.get("openwam") or {}
print("backend={} status={} mode={}".format(d.get("backend"), d.get("status"), d.get("mode")))
print("model={} dim={} horizon={} exec={}".format(i.get("name"), i.get("action_dim"), i.get("action_horizon"), i.get("execute_horizon")))
print("openwam.integration={}".format(om.get("integration")))
print("openwam.reason={}".format(om.get("reason")))
'
