# Sanitized runtime for the original Harness baseline.
# Copy to a local file that is gitignored. Do not put real API keys here.
# The local planner uses a dummy key because vLLM does not check it.

export HARNESS_ROOT=/root/Page_dmh/openwam_harness
export PI05_CHECKPOINT_PATH="${HARNESS_ROOT}/checkpoints/RLinf-Pi05-LIBERO-130-fullshot-SFT"
export SAM3_CHECKPOINT_PATH="${HARNESS_ROOT}/checkpoints/sam3/sam3.pt"
export PLANNER_MODEL_PATH="${HARNESS_ROOT}/checkpoints/Qwen3.5-9B"
export LIBERO_TYPE=standard
export MUJOCO_GL=egl
export HF_ENDPOINT=https://hf-mirror.com
export OPENAI_API_KEY=EMPTY
export OPENAI_BASE_URL=http://127.0.0.1:8000/v1
export NO_PROXY=127.0.0.1,localhost
# Sequential planner output budget. Do not raise the server context to 65536.
export PLANNER_OUTPUT_BUDGET=2048
export PLANNER_CONTEXT_LIMIT=32768
export HF_HUB_OFFLINE=1
