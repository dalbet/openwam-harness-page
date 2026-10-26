#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "== ports =="
ss -ltnp 2>/dev/null | grep -E ':(14510|14600|14700|14800|14900|8001) ' || true
echo "== engine health =="
curl -s --max-time 5 http://127.0.0.1:14800/health | python3 -m json.tool 2>/dev/null || echo "engine unreachable"
echo "== engine status =="
curl -s --max-time 5 http://127.0.0.1:14800/api/status | python3 -c '
import json,sys
try:
    d=json.load(sys.stdin)
except Exception as e:
    print("status unavailable", e); raise SystemExit
ident=d.get("identity") or {}
print(json.dumps({
  "phase": d.get("phase"),
  "detail": d.get("detail"),
  "task_id": d.get("task_id"),
  "episode_id": d.get("episode_id"),
  "scene_epoch": d.get("scene_epoch"),
  "agent": (ident.get("agent") or {}).get("model"),
  "action": (ident.get("action_model") or {}).get("name"),
  "inference": (ident.get("inference") or {}).get("mode"),
  "world_model": ((ident.get("action_model") or {}).get("openwam") or {}).get("integration"),
  "metrics": d.get("metrics"),
}, ensure_ascii=False, indent=2))
' 2>/dev/null
echo "== page_ui =="
curl -s --max-time 5 http://127.0.0.1:14900/health || echo "page_ui unreachable"
echo
echo "== recent episodes =="
ls -t "$ROOT/records/runs"/*.json 2>/dev/null | head -5
