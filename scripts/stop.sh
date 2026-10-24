#!/usr/bin/env bash
# Stop this project's planner and adapter after checking process identity.
# A foreign listener on the same port is left running.
set -euo pipefail
source "$(dirname "$0")/common.sh"

echo "stopping adapter"
guard stop --name adapter --port "${SEQUENTIAL_PORT}"
echo "stopping planner"
guard stop --name planner --port "${PLANNER_PORT}"
