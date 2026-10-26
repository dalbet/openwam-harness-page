# 边缘具身智能引擎：三层架构与运行说明

本文件描述 `embodied_engine/` 的真实组件、接口、来源与运行方式。所有判定以代码与运行记录为准；
未完成的组件在文末"当前状态"中明确标注，不以下载目录、接口类或空服务代替"已接入"。

## 0. 端口与服务

| 端口 | 服务 | 进程 |
| --- | --- | --- |
| 8001 | Planner LLM（Qwen3.5-9B，vLLM OpenAI 兼容，GPU3） | `models/../scripts/start_all.sh` 启动的 vLLM |
| 14510 | Pi0.5 动作模型 RPC 服务（GPU1，常驻） | RPent `pi05_vla_server.py` |
| 14600 | 推理引擎服务 | `inference/inference_service.py` |
| 14700 | Agent 服务（规划/恢复） | `agent/agent_service.py` |
| 14800 | 引擎服务（回合、技能、完成判定、UI 数据源） | `engine/engine_service.py` |
| 14900 | page_ui（双相机 WebSocket 页面） | `page_ui/server.py` |
| 14561 | LIBERO 环境 RPC（引擎按需拉起，回合结束回收） | `adapters/libero_env_server.py` |

旧顺序链路（adapter 8766 / RPent 44481 / SAM3 60071）未被改动，仍可独立启动，用作对照。

## 1. Agent 层

- **实际模型**：`Qwen3.5-9B`，通过 vLLM `0.24.0` 部署在 `127.0.0.1:8001`（GPU3，`--enforce-eager`）。
  同权重的共享实例 `:8000` 受其他负载影响，同一 prompt 实测 9.2 s；专用实例 1.1 s，
  因此 `config/engine.json` 指向 `:8001`。
- **职责**：把自然语言指令转成结构化计划（技能 + 对象绑定 + 目标 + relation），
  以及失败后的 `retry / replan / abort` 决策。
- **契约**：`common/plan_contract.py` 对模型输出做结构校验，不解析自然语言列表：
  - 技能来自 `skills` 目录（`pick_place`、`push_object`、`open_drawer`、`turn_on`）；
  - `object_n` / `target_n` 必须存在于本回合快照的物体或 site；
  - `open_drawer` 的目标必须是**确实拥有该层关节**的抽屉区域（例如
    `wooden_cabinet_1_top_region` 有 `wooden_cabinet_1_top_level`，
    `wine_rack_1_top_region` 没有关节，会被拒绝）；
  - 重规划时 `prior_steps` 提供已完成步骤，允许"先开抽屉、再放碗"的合法组合；
  - `preserves_goal()` 要求重规划保留初始计划的最终目标步骤，否则按有界 retry 处理，
    防止模型把"放进抽屉"漂移成"放到桌上"。
- **输入**：用户指令、场景快照（物体位姿、site 几何、关节、夹爪开度）、技能目录、
  已完成步骤与失败原因、可选决策集合、`episode`/`scene_epoch`/`observation_id`。
- **输出**：`{"interpretation", "success_condition", "steps": [...], "source": "qwen3.5-9b", "latency_s"}`；
  恢复输出 `{"status", "decision", "steps", "reason", "policy"}`。
- **Jev（TypeSafe System One）现状**：官方为云端 API（`https://api.typesafe.ai/v1/systemone`），
  需要 waitlist 账号与 `TYPESAFE_API_KEY`；本地无权重、无可用凭据。因此**没有接入 Jev**，
  也不能把 Qwen 的决策标成 Jev。Agent 层当前为"语言模型规划 + 结构化契约 + 有界恢复"。

## 2. 模型层

- **实际动作模型**：Pi0.5 LIBERO-130 SFT
  - checkpoint：`/home/host/page-dmh/openwam_harness/checkpoints/RLinf-Pi05-LIBERO-130-fullshot-SFT`
  - revision：`6222623f635769bfc73c9472e29fab9b7fd8e027`
  - 结构：`Pi0Config(pi05=True, gemma_2b + gemma_300m, action_dim=32, action_horizon=10)`，3.35B 参数
  - 推理：RPent `pi05_vla_server`（HTTP JSON RPC），5 步去噪；对外输出动作块
  - 实际执行：每块取 5 个 7 维 LIBERO 动作（`[dx,dy,dz,dax,day,daz,gripper]`，世界系 OSC_POSE）
  - 观测：`main_images`（agentview）+ `wrist_images` + 8 维状态 + 任务文本
- **OpenWAM（`OpenWAM-Alpha-Sim-LIBERO`）现状：BLOCKED，未进入执行链**
  - 权重已完整落盘：`checkpoint_step_10690.safetensors` 24,813,767,464 B，
    safetensors 头部 2089 个张量，含 `video_backbone.dit`(825)、`action_backbone`(818)、
    `text_encoder`(242)、`vae`(196)；`action_encoder.weight [1024, 80]`、`action_decoder.weight [80, 1024]`。
  - 官方配置要求 `video_backbone.model_path=/path/to/Wan2.2-TI2V-5B`，本机**没有**该主干；
    `normalization_stats.npy` 与 tokenizer 已在权重目录内。
  - 运行依赖 `diffusers>=0.38`、`transformers>=5.5`；本机 `rpent` 环境为 `transformers 4.57.6`
    且未安装 `diffusers`，新建环境需重新装 torch/diffusers，且 24.8 GB 权重在 32 GB 显卡上余量很小。
  - 因此本阶段**没有运行过 OpenWAM 的真实动作推理**，报告中的动作模型一律为 Pi0.5。

## 3. 推理引擎层

`inference/inference_service.py`：

- **模型驻留**：Pi0.5 由独立进程常驻（14510），推理服务只做观测编码与请求排队，不重复加载。
- **队列**：容量 1（`inference.queue_size=1`），新请求会取消同 token 的旧请求，避免旧观测积压。
- **请求字段**：`token`（回合隔离）、`episode_id`、`scene_epoch`、`skill`、`step_id`、
  `observation_id`、`sim_action_count`、`prompt`。
- **异步动作块预取**：`inference.mode=async_chunk_prefetch`。
  执行当前 5 动作块到第 3 个动作（index 2）时，用当时的新观测预取下一块；
  取用时校验：
  - 同 token / 同 scene_epoch；
  - `sim_action_count` 与预取时相差 ≤ `max_chunk_age_actions=5`；
  - 状态 `ready` 且未被取消。
  校验失败即丢弃并重新推理（记录在 `chunks[].prefetched/age_actions` 与 `prefetch` 统计）。
- **边界**：抓取确认、释放、技能完成、`stop/reset` 一律丢弃在途预取；不跨边界拼接旧状态动作。
- **取消/reset**：`cancel` 提升 epoch；引擎 `reset/stop` 后旧回合结果按 token 丢弃。
- **指标**：`inference_s`（模型前向）、`wait_s`（排队/等待）、`env_step_s`（仿真步进）、
  `snapshot_s`（只读快照）、`chunk_gap_s`（动作块空档）、`prefetch_inference_overlap_s_est`。

`vLLM` 只服务语言规划模型，**没有**用于 Pi0.5 或 OpenWAM；两者架构不同，不能宣称 vLLM 加速了 VLA。

## 4. 适配层与完成判定

`adapters/libero_env_server.py` 通过 RPent HTTP RPC 暴露 LIBERO：
`env.reset`、`env.step`、`env.state_snapshot`、`env.render_camera`、`env.get_env_meta`。
`state_snapshot` 返回 `sim.objects[*].pos`、`named_poses.sites`/`site_geometry`、`joints`、
`eef_pos`、`gripper_opening_m`、`native_goal`、`scene_epoch`、`observation_id`，全部带
`evidence_source`（`readonly_rpc` / `mujoco_site` / `sim_joint`）。

完成判定（`engine/completion.py`）：

| 技能 | 证据 | 停止条件 |
| --- | --- | --- |
| `open_drawer` | 所属关节 qpos | `qpos < -0.14`（与 LIBERO `WoodenCabinet.is_open` 一致） |
| `turn_on` | 炉灶 button 关节 | `qpos >= 0.5` |
| `push_object` | 目标 relation 且位移 | 位移 `> 0.03 m` 才记成功 |
| `pick_place` | relation + 高度，且释放后 3 个新样本 | relation 成立才发释放；抽屉内放置允许"对中—下探—释放"，释放后 3 帧新观测仍须满足 `in_box` |

补充机制：

- 释放是可验证事件：命令夹爪张开后测开度（≥0.05 m），再退离，最终验证必须使用
  释放之后、`sim_action_count` 更大的 3 个新样本。
- 抽屉预算按进展扩展：`open_drawer` 用最近 60 个动作的关节变化（≥0.01 rad）判断，
  仍在上限内才 +10 块；`pick_place` 用物体位移（≥0.02 m）判断。不是降门槛。
- 抽屉回归修复：放入抽屉会把抽屉面板推回 1–3 cm，最终验证若**唯一**失败项是抽屉
  （qpos 回到 -0.14 以上），重开一次后重新采 3 帧验证，最多一次。
- 已满足的 `open_drawer` 步骤在重规划中会跳过（`skipped=already_open`），避免无谓动作。
- 失败技能的第 2 次及以后尝试会先做一次**有界归位**（`skills.py::_goto_home`）：
  用同一 OSC 笛卡尔契约把 EEF 移回本回合初始位姿（每步 ≤0.06 m、最多 26 步、容差 2 cm），
  夹爪开度 <0.05 m（可能还夹着物体）时不归位。原因是有证据表明，重试若从失败位姿开始，
  机械臂会一直贴在柜面上，抽屉 qpos 整段保持 0.0。归位只改起始位姿，不改任何判定门槛，
  并记录在步骤的 `rehome` 字段与 journal 的 `rehome` 事件里。

## 5. page_ui

- 来源：本仓库 Git 历史中的旧双相机页面（`git show fc9d499:page_ui/` 恢复；该目录在
  `7864980` "去掉已经不在执行链上的旧页面" 时被删除），保留原布局、双相机
  （Agent/Wrist）、开始/停止/reset 控件；`server.py` 改为连接新引擎 14800 的 WebSocket，
  原 `get_scenes` / `load` / `start` / `restart` 文本协议与"JSON 头 + 两路 JPEG"二进制帧
  保持不变，`stop` / `reset` / 身份消息为新增。
- 展示字段全部来自后端 `/api/status` 的 `identity`：Agent 模型、动作模型、世界模型状态、
  推理后端与 async/sync 模式、阶段与失败原因。没有写死标签。
- 旧 `page_ui/README.md` 是 ManiSkill 时期的说明，仅作来源记录，不代表当前链路。

## 6. 启动与状态

```bash
cd /home/host/page-dmh/embodied_engine
./scripts/start_all.sh      # 8001 -> 14510 -> 14600 -> 14700 -> 14800 -> 14900
./scripts/status.sh         # 端口、健康、身份、最近回合
./scripts/stop_all.sh       # 只停本项目记录的 PID
python3 scripts/run_suite.py --tasks all --runs 3 --timeout 900
```

已验证的启动坑（记录在脚本里，避免重复踩）：

- vLLM 在 GPU3 编译 Triton launcher 需要 `Python.h`：脚本给 `CPATH` 指向 uv 的 cpython-3.10 头文件；
- `VLLM_USE_FLASHINFER_SAMPLER=0` + `--enforce-eager`；
- 新版 vLLM 已移除 `--disable-log-requests`；
- Pi0.5 的 RPC facade 没有 GET 健康路由，就绪检查必须 POST `/call {"method":"healthz"}`；
- 不要给常驻服务加 `--parent-watch`（stdin 为 /dev/null 会立刻触发退出）。

## 7. 当前状态

| 层 | 状态 | 说明 |
| --- | --- | --- |
| Agent | DONE（Qwen3.5-9B） | Jev 无凭据/无本地权重，未接入，未冒充 |
| 动作模型 | DONE（Pi0.5） | 常驻、真实推理、真实执行 |
| OpenWAM | BLOCKED | 权重齐全，缺 Wan2.2 主干与运行依赖，未跑过动作推理 |
| 推理引擎 | DONE | 异步块预取、token/epoch/age 校验、取消与隔离 |
| LIBERO 适配 | DONE | 快照、关节、site、RPC、生命周期 |
| page_ui | DONE（协议 + 浏览器） | 见验收摘要 |
| 十任务 | **PARTIAL（26/30）** | seed 0、每项 3 回合；失败项与断点见 `docs/acceptance_report.md` |

验收与记录：

- 最终验收 26/30（86.7%），成功回合中位 27.5 s，规划中位 6.5 s，块间空档合计 0.01 s，
  推断与动作执行重叠约 38%（块长 5 动作 ≈ 0.335 s < 单块推断 0.354 s，是剩余等待的根因）。
- 每轮验收的记录会归档到 `records/acceptance/<suite>_runs/`；`records/runs` 保留最新
  `MAX_RUNS=200` 份回合。
