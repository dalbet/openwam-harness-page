#!/usr/bin/env bash
# Page 启动脚本共用路径与进程身份判断。由 start/stop/status source，不要直接执行。
set -uo pipefail

ROOT=/root/Page_dmh
PUI="$ROOT/page_ui"
SH="$ROOT/Show-Harness"
VLM_PY="$SH/.venv-vllm/bin/python"
UI_PY="$SH/.venv-maniskill/bin/python"
LOG_DIR="$PUI/logs"
VLM_PIDF="$LOG_DIR/vlm_server.pid"
UI_PIDF="$LOG_DIR/server.pid"
VLM_LOG="$LOG_DIR/vlm_server.out"
UI_LOG="$LOG_DIR/server.out"
VLM_MARK="page_ui/vlm_server.py"
UI_MARK="page_ui/server.py"
MODEL_NAME="qwen3_5_2b_showharness_sim"

mkdir -p "$LOG_DIR"

page_env() {
  cd "$ROOT"
  export HF_HOME="$SH/models/huggingface"
  export MUJOCO_GL=egl
  export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
  export VK_DRIVER_FILES=/etc/vulkan/icd.d/nvidia_icd.json
  export VK_LOADER_DEBUG=error,warn
  # 官方 real2sim 场景（BlockPAP-v1）所需的 RLinf checkout
  # （AaronCaoZJ/RLinf @ fd52554，含 real_franka.real2sim_env；见 page_ui/README.md）
  export RLINF_ROOT="${RLINF_ROOT:-/root/RLinf}"
}

pid_alive() {
  local pid="${1:-}"
  [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
}

read_pid() {
  local f="$1"
  [[ -f "$f" ]] || return 1
  tr -d '[:space:]' < "$f"
}

cmdline_has() {
  local pid="$1" needle="$2"
  [[ -r "/proc/$pid/cmdline" ]] || return 1
  tr '\0' ' ' < "/proc/$pid/cmdline" | grep -F -q -- "$needle"
}

# 监听指定端口的 pid。找不到则空。
listen_pid() {
  local port="$1"
  "$VLM_PY" - "$port" <<'PY'
import os, sys
port = int(sys.argv[1])
hexport = f"{port:04X}"
inodes = set()
for name in ("/proc/net/tcp", "/proc/net/tcp6"):
    try:
        lines = open(name).read().splitlines()[1:]
    except OSError:
        continue
    for line in lines:
        parts = line.split()
        local, st, inode = parts[1], parts[3], parts[9]
        if st != "0A":
            continue
        hp = local.rsplit(":", 1)[-1].upper()
        if hp == hexport:
            inodes.add(inode)
if not inodes:
    sys.exit(0)
for pid in os.listdir("/proc"):
    if not pid.isdigit():
        continue
    fd_dir = f"/proc/{pid}/fd"
    try:
        fds = os.listdir(fd_dir)
    except OSError:
        continue
    for fd in fds:
        try:
            target = os.readlink(f"{fd_dir}/{fd}")
        except OSError:
            continue
        if target.startswith("socket:[") and target[8:-1] in inodes:
            print(pid)
            sys.exit(0)
PY
}

model_healthy() {
  curl -sf --max-time 3 "http://127.0.0.1:8000/v1/models" 2>/dev/null | grep -F -q "$MODEL_NAME"
}
