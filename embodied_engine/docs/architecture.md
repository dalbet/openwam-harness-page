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
| 14710 | d1-3B 决策服务（单次前向，独立模型常驻） | `agent/d1_service.py` |
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
  - 技能来自 `skills` 目录，并按 `action_model.backend` 收缩：`pi05` 只给
    `pick_place`、`push_object`、`open_drawer`、`turn_on`；`openwam` 只给 `whole_task`
    （`agent/agent_service.py::_skills_for`，同时切换 system prompt）。
    这样规划器不会提出当前后端无法执行的步骤；
  - `whole_task` 恰好一步，`object_n`/`target_n` 必须为 `null`（整任务不做 object/site 绑定），
    `prompt` 必须是官方整句指令——检查点只在这种提示下训练过；
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

### 1.1 d1-3B 决策规划器（可选 Agent 模式）

- **模型**：`LiquidAI/d1-3B`（3.12B，基座 `LiquidAI/LFM2.5-VL-3B`，LFM Open License v1.0），
  本地权重位于 `openwam_harness/checkpoints/LiquidAI-d1-3B`，SHA256
  `50e03317847caf6df9a9aee27ed40f20554a86a21e60d1d47ba41a422b546c0c`。
  ModelScope 仓库只暴露 `Revision=master`，没有固定 commit id，因此**以本地权重哈希锁定身份**，
  不把 `master` 当成可复现版本。独立环境 `.venv-d1`（由 `.venv-openwam` 硬链接克隆），
  独立端口 `:14710`、GPU0 常驻（加载 8.34 s），不升级现有 vLLM/OpenWAM 环境。
- **接口**：d1 服务（:14710）只暴露 `POST /system_one`，内部调用
  `model.system_one(state, questions)`，一次前向返回带类型的答案（`noul` 概率、
  `choice` 概率分布、`score`），**不生成输出 token**；没有 OpenAI 兼容 chat 端点，
  因此不存在"用聊天生成冒充决策"的路径。
- **规划方式**：d1 不写自由文本计划。`agent/d1_planner.py` 从当前快照生成合法候选
  （操作、source、target、relation、抽屉层），按阶段提问；关系与 source/target 使用
  联合/受约束候选，避免把互相矛盾的独立选择拼接。模型给出绑定后，由代码组装
  `whole_task` 计划并走同一个 `common/plan_contract.py` 校验。
- **候选隔离**：候选只来自现场物体/site/关节、技能目录与用户指令；不读取 BDDL goal，
  不把官方任务文本作为查表答案。BDDL 只用于离线评测的独立期望值。
- **三种 Agent 模式**：`qwen`（默认）、`d1`（只用 d1，不回落）、
  `d1_with_qwen_fallback`（仅在 `error`/`unavailable` 时回落；语义拒绝不回落为 Qwen）。
  引擎只在 idle 阶段接受切换，执行中返回 409；回合记录写入真实的
  `agent_mode` 与 `plan_source`，不会把一个模型产出的计划记成另一个。
- **阈值来源**：置信度/边距阈值只由独立校准集（`config/offline_calibration.json`，24 例）
  选择；冻结后的正式集（`config/offline_eval.json`，48 例）只用于报告，不据其回调阈值。

## 2. 模型层

模型层有两个**可显式切换**的后端，由 `config/engine.json` 的 `action_model.backend` 选择：

| `backend` | 服务 | 动作契约 | 执行粒度 |
| --- | --- | --- | --- |
| `pi05`（基线） | `127.0.0.1:14510`（RPent HTTP RPC） | 7 维 LIBERO OSC，`action_horizon=10`，每次执行 5 个 | 子技能（`pick_place` 等） |
| `openwam`（新增） | `ws://127.0.0.1:14520`（OpenWAM 官方部署服务） | 10 维 native EEF10，模型侧 32 步块 | 整任务（`whole_task`） |

切换是显式的：`inference/inference_service.py::_build_backend()` 只接受 `pi05` / `openwam`，
未知值直接报错；两个后端**不会互相回退**。切换需要改配置并重启推理服务（权重常驻，不受影响）：

```bash
# 1) 常驻 OpenWAM 服务（已启动可跳过）
./scripts/openwam_server.sh start

# 2) 改后端并重启推理服务
./scripts/set_backend.sh openwam     # 或 ./scripts/set_backend.sh pi05
```

### 2.1 Pi0.5 LIBERO-130 SFT（基线，保留）

- checkpoint：`/home/host/page-dmh/openwam_harness/checkpoints/RLinf-Pi05-LIBERO-130-fullshot-SFT`
- revision：`6222623f635769bfc73c9472e29fab9b7fd8e027`
- 结构：`Pi0Config(pi05=True, gemma_2b + gemma_300m, action_dim=32, action_horizon=10)`，3.35B 参数
- 推理：RPent `pi05_vla_server`（HTTP JSON RPC），5 步去噪；对外输出动作块
- 实际执行：每块取 5 个 7 维 LIBERO 动作（`[dx,dy,dz,dax,day,daz,gripper]`，世界系 OSC_POSE）
- 观测：`main_images`（agentview）+ `wrist_images` + 8 维状态 + 任务文本

`backend=pi05` 时，Agent 的技能目录为五个子技能，`whole_task` 不在目录内，行为与之前一致。

### 2.2 OpenWAM-Alpha-Sim-LIBERO（已接入执行链）

**结论先行**：这个检查点是**自包含**的，部署推理**不需要**外部 `Wan2.2-TI2V-5B` 目录，
24.8 GB 单文件已包含 DiT / T5 / VAE / ActionDiT 的全部权重。证据三条，逐条可复核：

1. **配置里内嵌了组件规格**。`checkpoints/OpenWAM-Alpha-Sim-LIBERO/config.yaml` 的
   `model.video_backbone.components` 给出 `dit`、`vae`、`text_encoder` 三个 `class_path` 与构造参数；
   `openwam/deploy/model_loader.py` 打印
   `Using config-embedded component specs for video-backbone construction`，
   用空壳实例化后**直接从 safetensors 载入权重**。`model_path=/path/to/Wan2.2-TI2V-5B`
   只在 tokenizer 缺失时作兜底（`wan/loader.py` 先找 `ckpt_dir/tokenizer/...`），
   而 tokenizer 已在 `checkpoints/OpenWAM-Alpha-Sim-LIBERO/tokenizer/google/umt5-xxl/` 内。
2. **单文件体积与结构对得上**。按 safetensors 头部逐张量统计（2089 个张量，24.814 GB）：

   | 组件 | 张量数 | 体积 | 参数规模 |
   | --- | --- | --- | --- |
   | `video_backbone.dit.*` | 825 | 9.997 GB | Wan2.2-TI2V-5B DiT（5.0B，bf16） |
   | `video_backbone.text_encoder.*` | 242 | 11.363 GB | UMT5-XXL（5.68B，bf16） |
   | `action_backbone.*` | 818 | 2.042 GB | ActionDiT 1.02B + 投影 |
   | `video_backbone.vae.*` | 196 | 1.409 GB | Wan VAE（0.705B） |
   | `proprio_encoder` | 2 | 0.001 GB | 10 → 1024 |

   合计 12.4068B 参数，与文件体积一致；**没有**权重依赖外部主干目录。
3. **实测加载与推理**。`scripts/openwam_probe.py --device cpu` 以 `strict=True` 载入
   （`load_checkpoint` 对缺失/多余 key 直接抛错，成功即 key 完全匹配），得到 12.4068B 参数；
   GPU2 常驻服务对真实 LIBERO 观测返回 10 维动作，见 `docs/acceptance_report.md`。
   本机 `checkpoints/Wan2.2-TI2V-5B/` 曾是早期尝试留下的**空目录**（已删除），不影响运行。

**运行环境**（无 sudo，独立 venv）：

- `embodied_engine/.venv-openwam/`：`torch 2.7.1+cu128`（sm_120 实测算子通过）、
  `transformers 5.20.2`、`diffusers 0.41.0`、`modelscope`、`safetensors` 等；`openwam`
  以 `--no-deps -e` 安装。部署路径不引用 `deepspeed/wandb/hydra`。
- 启动：`scripts/openwam_server.sh start`（读配置、写启动记录 `records/openwam_server.json`）。
- 常驻显存实测 **24,778 MiB**（GPU2，31.4 GiB），加载约 110 s。

**服务契约**（`models/openwam_backend.py` 只做传输与校验）：

- 观测：`head_camera` = agentview，**旋转 180°** 后 LANCZOS 缩放到 320×256；
  `left_wrist_camera` = eye-in-hand，旋转 180° 后缩放到 160×128；`state` = 10 维 EEF10
  `[eef_pos(3), rot6d(quat)(6), gripper_open_scale(1)]`，`gripper_open_scale = 2*width/0.08 - 1`。
- 动作：模型返回 native EEF10 `[dxyz(3), rot6d(6), open_scale(1)]`；
  适配层转成 LIBERO 7 维 OSC：`rot6d → 四元数 → 轴角`，夹爪取反（`+1=close`），再裁剪到 `[-1,1]`。
- 转换与官方 `benchmarks/utils/action_conversion.py`、`benchmarks/libero/openwam2libero_interface.py`、
  `benchmarks/utils/client.py` **逐函数数值对拍**：`tests/test_openwam_contract.py` 用 200 组随机
  位姿/动作、50 组随机 proprio、真实图像槽位比对，全部 `allclose`。
- 一个请求返回**一个**动作；32 步块缓冲在服务端（`--inference-horizon` = `execute_horizon`），
  客户端不缓存、不重放。响应校验：形状 `(1,10)`、有限值，否则抛错。
- 失败即失败：传输/形状/服务端错误都会抛出，不会退回 Pi0.5。

**为什么是"整任务"而不是八步技能**：该检查点用 LIBERO 官方整句指令训练
（`dataloader.action_mode=eef`、`unify_action 0-9`、`num_frames=33`），
喂 `pick_place` 这类子技能提示属于训练分布外指令，会人为压低成功率。
因此 `backend=openwam` 时 Agent 的技能目录只含 `whole_task`，计划恰好一步，
`prompt` 必须是官方整句指令（`common/plan_contract.py` 对 `whole_task` 禁止 object/site 绑定）。

## 3. 推理引擎层

`inference/inference_service.py`：

- **后端选择**：`_build_backend()` 按 `action_model.backend` 构造 `Pi05Backend`（14510）
  或 `OpenWAMBackend`（WebSocket 14520）。未知后端名 = 启动失败，不做隐式回退。
- **后端身份由推理服务说了算**：引擎（`engine/engine_service.py::_action_backend()`）每次执行
  前向推理服务 `/health` 查询当前 `backend`，不用启动时缓存的配置；计划里的 `backend` 与运行
  时后端不一致时直接拒绝（`plan_backend_mismatch`）。因此切换后端只需重启推理服务，
  引擎与页面无需重启，也不会把 8 维 Pi0.5 proprio 喂给 10 维 OpenWAM 契约。
- **模型驻留**：两个后端都在独立进程里常驻；推理服务只做观测编码、请求排队与契约校验，
  不重复加载权重。OpenWAM 的 24.8 GB 权重只在上游服务启动时加载一次。
- **队列**：容量 1（`inference.queue_size=1`），新请求会取消同 token 的旧请求，避免旧观测积压。
- **请求字段**：`token`（回合隔离）、`episode_id`、`scene_epoch`、`skill`、`step_id`、
  `observation_id`、`sim_action_count`、`prompt`。
- **Pi0.5 异步动作块预取**：`inference.mode=async_chunk_prefetch`。
  执行当前 5 动作块到第 3 个动作（index 2）时，用当时的新观测预取下一块；取用时校验：
  - 同 token / 同 scene_epoch；
  - `sim_action_count` 与预取时相差 ≤ `max_chunk_age_actions=5`；
  - 状态 `ready` 且未被取消。
  校验失败即丢弃并重新推理（记录在 `chunks[].prefetched/age_actions` 与 `prefetch` 统计）。
- **OpenWAM 同步单步**：`inference.openwam_mode=sync_step`。客户端一次请求换一个动作，
  32 步块缓冲在模型服务内部管理，因此**没有**跨请求的动作缓存可被污染；
  不启用预取，边界天然干净。
- **边界**：抓取确认、释放、技能完成、`stop/reset` 一律丢弃在途请求与预取；
  不跨边界拼接旧状态动作。
- **取消/reset**：`cancel`/`reset` 提升 token，旧回合结果按 token 丢弃；
  同时向后端发送 `reset` 清空**模型侧的 receding-horizon 缓冲**
  （OpenWAM 的 `reset_ack` 是显式应答，计数记在 `backend_resets` / `backend_resets_on_cancel`）。
  这一条是"停止后旧 chunk 不得进入新回合"的关键：只丢客户端队列不够。
- **指标**：`inference_s`（模型前向）、`wait_s`（排队/等待）、`env_step_s`（仿真步进）、
  `snapshot_s`（只读快照）、`chunk_gap_s`（动作块空档）、`prefetch_inference_overlap_s_est`；
  OpenWAM 额外记录服务端 `latency_ms` 与首帧/稳态延迟。

`vLLM` 只服务语言规划模型，**没有**用于 Pi0.5 或 OpenWAM；两者架构不同，不能宣称 vLLM 加速了 VLA。

## 4. 适配层与完成判定

`adapters/libero_env_server.py` 通过 RPent HTTP RPC 暴露 LIBERO：
`env.reset`、`env.step`、`env.state_snapshot`、`env.render_camera`、`env.get_env_meta`。
`state_snapshot` 返回 `sim.objects[*].pos`、`named_poses.sites`/`site_geometry`、`joints`、
`eef_pos`、`gripper_opening_m`、`native_goal`、`scene_epoch`、`observation_id`，全部带
`evidence_source`（`readonly_rpc` / `mujoco_site` / `sim_joint`）。

`adapters/openwam_contract.py` 只服务 OpenWAM 后端：把原始 LIBERO 观测
（`agentview_image`、`robot0_eye_in_hand_image`、`robot0_eef_*`、`robot0_gripper_qpos`）
转成上面的 EEF10/图像槽位契约，并把 10 维动作转成 7 维 OSC。Pi0.5 路径完全不用它。

完成判定（`engine/completion.py`）：

| 技能 | 证据 | 停止条件 |
| --- | --- | --- |
| `open_drawer` | 所属关节 qpos | `qpos < -0.14`（与 LIBERO `WoodenCabinet.is_open` 一致） |
| `turn_on` | 炉灶 button 关节 | `qpos >= 0.5` |
| `push_object` | 目标 relation 且位移 | 位移 `> 0.03 m` 才记成功 |
| `pick_place` | relation + 高度，且释放后 3 个新样本 | relation 成立才发释放；抽屉内放置允许"对中—下探—释放"，释放后 3 帧新观测仍须满足 `in_box` |
| `whole_task` | native LIBERO goal（3 个新样本全为真） | 只由策略自己完成；释放只**测量**，不要求、不伪造 |

补充机制：

- 释放是可验证事件：命令夹爪张开后测开度（≥0.05 m），再退离，最终验证必须使用
  释放之后、`sim_action_count` 更大的 3 个新样本。
- `whole_task` 不做子技能拆解：整任务策略由自己产生抓取/搬运/释放动作，
  因此**没有** Pi0.5 风格的 release 收据。回合记录里给出的事实是分开的：
  `release.events`（测量到的张开，`measured_gripper_separation`）、
  `grasp.final_opening_m`/`min_opening_m`、`final_verification.gripper_opening_m`/
  `release_observed`。策略一直夹着物体到达 goal 时，报告会显示 `release_observed=false`，
  而不会补一条假收据。
- 抽屉预算按进展扩展：`open_drawer` 用最近 60 个动作的关节变化（≥0.01 rad）判断，
  仍在上限内才 +10 块；`pick_place` 用物体位移（≥0.02 m）判断。不是降门槛。
  这两条只作用于 Pi0.5 子技能；OpenWAM 用 `max_steps=600` 的整任务预算。
- 抽屉回归修复：放入抽屉会把抽屉面板推回 1–3 cm，最终验证若**唯一**失败项是抽屉
  （qpos 回到 -0.14 以上），重开一次后重新采 3 帧验证，最多一次。
- 已满足的 `open_drawer` 步骤在重规划中会跳过（`skipped=already_open`），避免无谓动作。
- 失败技能的第 2 次及以后尝试会先做一次**有界归位**（`skills.py::_goto_home`）：
  用同一 OSC 笛卡尔契约把 EEF 移回本回合初始位姿（每步 ≤0.06 m、最多 26 步、容差 2 cm），
  夹爪开度 <0.05 m（可能还夹着物体）时不归位。原因是有证据表明，重试若从失败位姿开始，
  机械臂会一直贴在柜面上，抽屉 qpos 整段保持 0.0。归位只改起始位姿，不改任何判定门槛，
  并记录在步骤的 `rehome` 字段与 journal 的 `rehome` 事件里。
- 停止的动作预算：`stop()` 记录 `sim_at_request` / `sim_at_ack` / `in_flight_actions`。
  已经交给仿真的那一步无法撤回，所以契约是"**停止后不再下发新动作**"，
  在途动作只会是 0 或 1，并写进 `/api/stop` 响应与 `status.last_stop`。
  `scripts/test_lifecycle.py` 断言该值 ≤1 且确认 3 s 后 `sim_action_count` 不再增长。

## 5. page_ui

- 来源：本仓库 Git 历史中的旧双相机页面（`git show fc9d499:page_ui/` 恢复；该目录在
  `7864980` "去掉已经不在执行链上的旧页面" 时被删除），保留原布局、双相机
  （Agent/Wrist）、开始/停止/reset 控件；`server.py` 改为连接新引擎 14800 的 WebSocket，
  原 `get_scenes` / `load` / `start` / `restart` 文本协议与"JSON 头 + 两路 JPEG"二进制帧
  保持不变，`stop` / `reset` / 身份消息为新增。
- 展示字段全部来自后端 `/api/status` 的 `identity`：Agent 模型与模式、动作模型、
  世界模型状态、推理后端与 async/sync 模式、阶段与失败原因。没有写死标签。
- Agent 模式切换（`qwen` / `d1` / `d1_with_qwen_fallback`）通过现有 WebSocket 文本协议新增的
  `agent_mode` 消息转发到引擎 `/api/agent_mode`；页面只显示身份消息里的真实模式，
  引擎在 planning/running/recovering/stopping 阶段拒绝切换（HTTP 409），因此不会出现
  "页面显示已切换、实际回合仍用旧模型"的情况。切换控件只在 idle 阶段可用。
- 旧 `page_ui/README.md` 是 ManiSkill 时期的说明，仅作来源记录，不代表当前链路。
- 回合隔离：引擎维护 `planner_generation`，`load` / `reset` / 执行中的 `stop` / 新回合都会
  递增；规划请求跨过任何一个生命周期转换后，其返回值代次已过期，被丢弃并记
  `planner_result_discarded`（`records/run_journal.jsonl`），不会写入新回合的 `plan`
  或触发任何动作。`/api/status` 暴露 `planner_generation` 与
  `metrics.stale_plans_discarded` 便于验证。

## 6. 启动与状态

```bash
cd /home/host/page-dmh/embodied_engine
./scripts/start_all.sh              # 8001 -> 14510 -> 14600 -> 14700 -> 14800 -> 14900
./scripts/openwam_server.sh start   # OpenWAM 常驻服务 14520（GPU2）
./scripts/set_backend.sh openwam    # 选动作后端并重启推理服务（或 pi05）
./scripts/status.sh                 # 端口、健康、身份、最近回合
./scripts/stop_all.sh               # 只停本项目记录的 PID（不含 14520）
./scripts/openwam_server.sh stop    # 停 OpenWAM 服务
python3 scripts/run_suite.py --tasks all --runs 3 --timeout 900
./scripts/d1_server.sh start         # d1-3B 决策服务 14710（GPU0）；status|stop|restart
python3 scripts/eval_offline.py      # 冻结离线集（config/offline_eval.json）
python3 scripts/run_ab.py --tasks all --seeds 0,1,2 --modes qwen,d1 \
  --out records/ab/ab_frozen.json    # 60 回合交错 A/B
python3 scripts/summarize_ab.py records/ab/ab_frozen.json --markdown
```

OpenWAM 启动记录写在 `records/openwam_server.json`（pid、commit、checkpoint、
`generate_horizon`、`execute_horizon`、`denoise_steps`、GPU），因此"实际执行 horizon"
来自真实启动参数，而不是只写在配置里。`scripts/openwam_server.sh status` 同时打印
推理服务的身份与 `integration` 字段。

已验证的启动坑（记录在脚本里，避免重复踩）：

- vLLM 在 GPU3 编译 Triton launcher 需要 `Python.h`：脚本给 `CPATH` 指向 uv 的 cpython-3.10 头文件；
- `VLLM_USE_FLASHINFER_SAMPLER=0` + `--enforce-eager`；
- 新版 vLLM 已移除 `--disable-log-requests`；
- Pi0.5 的 RPC facade 没有 GET 健康路由，就绪检查必须 POST `/call {"method":"healthz"}`；
- 不要给常驻服务加 `--parent-watch`（stdin 为 /dev/null 会立刻触发退出）；
- 本机无 `ensurepip`：venv 用 `uv venv` 建，pip 由已有环境 bootstrap；
- OpenWAM 服务是 WebSocket，就绪检查不能发裸 TCP/HTTP 探针（会让服务端抛
  `InvalidMessage`）：`openwam_server.sh` 用官方 `websockets.sync` 客户端发 `ping` 判活；
- `MAX_RUNS` 会裁剪 `records/runs`，验收脚本必须把回合记录一起归档到
  `records/acceptance/<suite>_runs/`。

## 7. 当前状态

| 层 | 状态 | 说明 |
| --- | --- | --- |
| Agent | DONE（Qwen3.5-9B 默认） | Jev 无凭据/无本地权重，未接入，未冒充；技能目录随 `action_model.backend` 收缩 |
| Agent（d1-3B 可选） | **PARTIAL** | `d1` 决策模式可用，A/B 成功率低于 Qwen，保持 Qwen 为默认；结论见验收报告 §14 |
| 动作模型（`pi05`） | DONE | 常驻、真实推理、真实执行（基线保留） |
| 动作模型（`openwam`） | **DONE，执行链** | 常驻 14520，真实 10 维动作；LIBERO-Goal 实测见验收报告 |
| 推理引擎 | DONE | 双后端选择、Pi0.5 异步块预取、OpenWAM 同步单步、token/epoch/age 校验、取消与缓冲清理 |
| LIBERO 适配 | DONE | 快照、关节、site、RPC、生命周期；OpenWAM 契约逐函数对拍 |
| page_ui | DONE（协议 + 浏览器） | 身份消息显示真实后端与 OpenWAM 状态 |
| 十任务（Pi0.5 历史验收） | **PARTIAL（26/30）** | seed 0、每项 3 回合；失败项与断点见 `docs/acceptance_report.md` |
| 十任务（OpenWAM 单轮） | 见 `docs/acceptance_report.md` | 每任务 1 回合，不重试 |
| 十任务 ×3 初始化 A/B（60 回合） | 见 `docs/acceptance_report.md` §14 | Qwen vs d1 同 OpenWAM、同初始化、交错顺序 |

验收与记录：

- Pi0.5 历史验收 26/30（86.7%），成功回合中位 27.5 s，规划中位 6.5 s，块间空档合计 0.01 s，
  推断与动作执行重叠约 38%（块长 5 动作 ≈ 0.335 s < 单块推断 0.354 s，是剩余等待的根因）。
- OpenWAM 的回合记录里，`model.backend=openwam`、`model.openwam.integration=on_execution_path`，
  步骤 `skill=whole_task`，`final_verification.method=native_libero_goal`。
- 每轮验收的记录会归档到 `records/acceptance/<suite>_runs/`；`records/runs` 保留最新
  `MAX_RUNS=200` 份回合。
