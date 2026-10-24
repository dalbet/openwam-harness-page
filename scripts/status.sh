#!/usr/bin/env bash
# Short status for this project's planner and adapter. No environment dump.
set -euo pipefail
source "$(dirname "$0")/common.sh"

guard status --name planner --port "${PLANNER_PORT}"
if curl -sf --max-time 3 "${OPENAI_BASE_URL}/models" | grep -F -q "\"${PLANNER_MODEL}\""; then
  echo "planner http=ready model=${PLANNER_MODEL} weights=loaded"
else
  echo "planner http=down model=${PLANNER_MODEL}"
fi

guard status --name adapter --port "${SEQUENTIAL_PORT}"
health="$(curl -sf --max-time 3 "http://${SEQUENTIAL_HOST}:${SEQUENTIAL_PORT}/health" || true)"
if [[ -z "${health}" ]]; then
  echo "adapter http=down"
  exit 0
fi
"${PY_GUARD}" - "${health}" <<'PY'
import json, sys
body = json.loads(sys.argv[1])
env = body.get("env_server") or {}
startup = env.get("startup_args") or {}
components = []
for item in body.get("components") or []:
    if isinstance(item, dict):
        components.append(f"{item.get('name')}={'up' if item.get('alive') else 'down'}")
print(
    "adapter http=ready"
    f" mode={body.get('mode')}"
    f" planner={body.get('planner')}"
    f" env={body.get('env')}"
    f" vla={body.get('vla')}"
    f" suite={startup.get('suite')}"
    f" task={startup.get('task_id')}"
    f" sim_action_count={env.get('sim_action_count')}"
    f" components={','.join(components) or 'not_loaded'}"
)
print("model_load=on_scene_load execution=sync planning_input=scene_metadata vision=false")
PY
