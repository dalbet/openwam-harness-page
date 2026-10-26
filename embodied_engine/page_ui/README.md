# Page — Show-Harness ManiSkill 闭环 Web UI

一个简洁网页（标题为 "Page"），通过本地 vLLM 服务 + 真实 ManiSkill 仿真环境，构成
"本地模型 → 仿真环境 → 网页 UI" 的真实闭环。

## 当前采用的真实模型 / adapter / 仿真环境 / 执行模式

| 项 | 值 |
| --- | --- |
| 基础模型 | `Qwen/Qwen3.5-2B`（已下载于 `Show-Harness/models/huggingface/hub/Qwen/Qwen3.5-2B/`） |
| LoRA adapter | `qwen3_5_2b_sim`（仿真策略，训练于 RoboLab + ManiSkill 仿真数据集，v3 统一 prompt） |
| adapter 服务名 | `qwen3_5_2b_showharness_sim` |
| chat template | `models/chat_templates/qwen3_5_nothink.jinja`（与训练时 LlamaFactory `qwen3_5_nothink` 对齐） |
| 仿真环境 | ManiSkill 3。**首选官方 real2sim 场景 `BlockPAP-v1`**（RLinf `real_franka.real2sim_env`，训练分布内）；stock 任务（PickCube 等）保留作对照 |
| 控制模式 | `pd_ee_delta_pos`（仅平移，旋转锁定）— 与 adapter 训练契约一致 |
| 执行模式 | 单动作 token 模式（每步一个语义动作 token：MV_* / GRASP / RELEASE / DONE），无 planner、无 action chunk |
| 每决策仿真步 | `sim_steps_per_decision = 2`，`step_m = 0.026`（与 `configs/robot_maniskill.yaml` 一致，不改动作语义） |
| 视角 | 官方场景：Agent（`external_cam`，标定 RealSense 内参 640x480）+ Wrist（`hand_camera` 居中挂载）；stock：Agent（`base_camera`）+ Wrist（`hand_camera`）。两侧统一 `wrist_flip=both`、腕部 4:3 裁剪、256 letterbox 训练契约 |

> 说明：`qwen3_5_2b_sim` 训练于 ManiSkill 的 RLinf real2sim 场景（BlockPAP，木纹桌面，
> shard `blockpap_follow_wood`）+ RoboLab。stock ManiSkill 任务相对训练分布存在域差，
> 成功率低是预期行为；成功/失败/超时均如实报告，不伪造成功。

## 官方场景（BlockPAP-v1）

`BlockPAP-v1` 是 RLinf 的 real2sim 复刻（真实 Franka 台架：桌面/底座/橙块/绿色杯垫/
标定正面 RealSense），即 sim adapter 的训练场景。场景注册代码在 Show-Harness
`core/sim/maniskill_scenes.py`（本仓库未改动），env 模块来自 RLinf checkout（`RLINF_ROOT`）。

**来源与版本（已核验）**：

| 项 | 值 |
| --- | --- |
| 场景模块 | `real_franka.real2sim_env.pick_and_place`（`BlockPAP-v1`、`PandaHighFriction`） |
| 代码来源 | GitHub `AaronCaoZJ/RLinf`（Show-Harness 作者 Zhijun Cao 的 RLinf fork）分支 `main` |
| 固定 commit | `fd52554870dc9c0dbd4243c052f5ec32c56f1340`（2026-08-21） |
| 许可证 | Apache-2.0（继承 RLinf；`LICENSE` 已随 checkout 保存） |
| 桌面纹理 | HF 数据集 `RLinf/maniskill_assets` 路径 `carrot/more_table/textures/001..021.png`，ODC-BY |
| 本地位置 | `/root/RLinf`（稀疏检出：`real_franka/real2sim_env/*.py` + `LICENSE`；纹理在 `rlinf/envs/maniskill/assets/carrot/more_table/textures/`） |

安装（本机已执行，记录供复现）：

```bash
git clone --depth 1 --filter=blob:none --no-checkout \
  https://github.com/AaronCaoZJ/RLinf.git /root/RLinf   # 网络受限时可用 gh-proxy.com 镜像
cd /root/RLinf && git sparse-checkout init --no-cone && \
  printf '/real_franka/real2sim_env/*.py\n/real_franka/real2sim_env/*.md\n/LICENSE\n/README.md\n' \
  > .git/info/sparse-checkout && git checkout -f main
# 木纹纹理（训练分布 `blockpap_follow_wood`，table_tex "006" 必需）
mkdir -p /root/RLinf/rlinf/envs/maniskill/assets/carrot/more_table/textures
for i in $(seq -w 1 21); do
  curl -sL "https://hf-mirror.com/datasets/RLinf/maniskill_assets/resolve/main/carrot/more_table/textures/0$i.png" \
    -o /root/RLinf/rlinf/envs/maniskill/assets/carrot/more_table/textures/0$i.png
done
```

运行契约（server.py `OFFICIAL_SCENES`，与 `configs/robot_maniskill.yaml` 对齐）：
`panda_high_friction_wristcam`（Show-Harness 动态派生的腕部相机 agent）、居中腕部挂载、
`external_cam` 保留标定内参（`camera_resolution=null`）、env 自带标定起始角
（`reset_qpos=null`）、`table_tex="006"`、`traj_id=random` + `layout=wide`（训练生成器
同一采样器的评估协议）、auto_release 空抓阈值 0.005 m、每 episode seed 递增。

**BlockStack-v1 未接入**：其 `PandaExtendedGripper` 需要 `panda_v2_extended.urdf`，
该资产在 RLinf main/全部分支、上述 fork、ManiSkill 上游（mani-skill/ManiSkill）、
HF 资产库（RLinf/maniskill_assets、RLCo-maniskill-assets、haosulab/ManiSkill）均不存在，
构建时报 `definition file not found ... panda_v2_extended.urdf`。已整理索取邮件草稿
`docs/request_official_scene_assets_email.md`；收到资产后，在 `OFFICIAL_SCENES` 加一行
（robot `panda_extended_gripper_wristcam`、layout `wide`、`snap` 按 `tasks.py`）即可接入。

实测结果（`qwen3_5_2b_showharness_sim`，协议 `--traj-id random --layout wide`，80 步上限）：

| 入口 | seed 0 | seed 1 | seed 2 | 说明 |
| --- | --- | --- | --- | --- |
| 官方 runner（`scripts/run_maniskill_mvtoken.py`，table_tex 006） | ✅ 63 步 | ✅ 58 步 | ✅ 45 步 | 3/3 |
| 官方 runner（table_tex white，对照） | ❌ 空抓 | ✅ 65 步 | ❌ 未抓 | 1/3，白桌面不在训练分布 |
| Page UI（本仓库 server.py） | ✅ 58 步 | ✅ 61 步 | ✅ 45 步 | 3/3；浏览器手动再跑 seed 3 也 ✅ |

判定均为环境自身 `info["success"]`（`DONE` 不算成功）。日志：
`page_ui/logs/official_eval_wood/`（006）与 `page_ui/logs/official_eval/`（white 对照）。

## 目录结构

```
Page_dmh/
├── Show-Harness/          # 同事的项目（只读使用，不修改）
├── RLinf -> /root/RLinf   # 官方场景 checkout（见上节；稀疏、固定 commit）
├── page_ui/
│   ├── server.py          # Web UI 服务端（aiohttp + WebSocket）
│   ├── static/
│   │   ├── index.html     # 页面（标题 "Page"）
│   │   └── app.js         # 客户端：WebSocket + Canvas + jitter buffer
│   ├── configs/
│   │   └── robot_maniskill_wood.yaml  # 官方 runner 评估配置（table_tex 006）
│   ├── test_blockpap_e2e.py           # WebSocket 端到端：列场景/加载/跑 episode
│   ├── test_capture_ui.py             # 抓取 UI 广播展示帧（调试显示链路）
│   ├── docs/              # 索取官方资产的英文邮件草稿等
│   ├── logs/server.log    # 服务端日志（含每步 render/vlm/step 耗时）
│   └── README.md          # 本文件
└── scripts/page/          # start/stop/status/common（一键启停）
```

## 环境

- `.venv-vllm`：已存在，用于 vLLM 服务（不修改）。
- `.venv-maniskill`：含 ManiSkill 3 + CPU torch + harness 运行所需依赖 + aiohttp。
  接入官方场景**未新增/升级任何包**（场景模块只 import mani_skill/sapien/torch/numpy/scipy）。
  - 用 CPU torch 是为避免下载 ~2GB 的 nvidia CUDA 大包；ManiSkill `sim_backend=physx_cpu`
    下仿真在 CPU、渲染走 SAPIEN/Vulkan（GPU 驱动），CPU torch 足够。
- `RLINF_ROOT=/root/RLinf`：官方场景 checkout，`scripts/page/common.sh` 已默认导出。

## 启动 / 停止 / 重启命令（服务器端）

启动模型服务（vLLM，后台，端口 8000，仅 127.0.0.1）：

```bash
cd /root/Page_dmh/Show-Harness
source /etc/network_turbo  # 仅首次需要联网下载时；已下载则不需要
MODEL=Qwen/Qwen3.5-2B \
  LORA=qwen3_5_2b_showharness_sim=models/Show-Harness-VLMs/qwen3_5_2b_sim \
  FAMILY=qwen3_5 PORT=8000 \
  nohup .venv-vllm/bin/python -m vllm.entrypoints.openai.api_server \
    $(MODEL=Qwen/Qwen3.5-2B LORA=qwen3_5_2b_showharness_sim=models/Show-Harness-VLMs/qwen3_5_2b_sim \
      FAMILY=qwen3_5 PORT=8000 bash -c '...') \
  > page_ui/logs/vllm.log 2>&1 &
```

> 实际启动请用下面"一键脚本"中的 `start_page.sh`，它封装了正确的 vllm serve 参数。

启动 Web UI（端口 8765，仅 127.0.0.1）：

```bash
cd /root/Page_dmh/Show-Harness
nohup .venv-maniskill/bin/python /root/Page_dmh/page_ui/server.py \
  > /root/Page_dmh/page_ui/logs/server.out 2>&1 &
```

停止：

```bash
# 停 Web UI
pkill -f "page_ui/server.py"
# 停 vLLM
pkill -f "vllm.entrypoints.openai.api_server"
```

重启：先 stop 再 start。

## Windows PowerShell SSH 端口转发

```powershell
ssh -N -L 8765:127.0.0.1:8765 -p 14263 root@connect.bjb1.seetacloud.com
```

## Windows 浏览器访问地址

```
http://127.0.0.1:8765
```

## 使用流程

1. 服务器端启动 vLLM 与 Web UI（见 `start_page.sh`）。
2. Windows 端建立 SSH 隧道。
3. 浏览器打开 `http://127.0.0.1:8765`，标签页标题为 "Page"。
4. 选择场景 → 点 "加载场景"（页面显示初始观测）。
5. 在任务输入框输入/修改任务（默认填入场景指令，可改）→ 点 "开始"。
6. 观察 Agent/Wrist 双视角与一行状态。
7. 点 "重新开始" 取消旧 episode、reset 环境、用当前任务重新执行。
