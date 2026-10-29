# 边缘具身智能引擎：三层架构与运行说明

本文件描述 `embodied_engine/` 的真实组件、接口、来源与运行方式。所有判定以代码与运行记录为准；
未完成的组件在文末"当前状态"中明确标注，不以下载目录、接口类或空服务代替"已接入"。

## 0. 端口与服务

| 端口 | 服务 | 进程 |
| --- | --- | --- |
| 8001 | Planner LLM（Qwen3.5-9B，vLLM OpenAI 兼容，GPU3） | `models/../scripts/start_all.sh` 启动的 vLLM |
| 14510 | Pi0.5 动作模型 RPC 服务（GPU1，常驻） | RPent `pi05_vla_server.py` |
| 14600 | 推理引擎服务 | `inference/inference_service.py` |
| 14700 | Agent 服务（语义路由、计划组装、有界恢复） | `agent/agent_service.py` |
| 14800 | 引擎服务（回合、技能、完成判定、UI 数据源） | `engine/engine_service.py` |
| 14900 | page_ui（双相机 WebSocket 页面） | `page_ui/server.py` |
| 14561 | LIBERO 环境 RPC（引擎按需拉起，回合结束回收） | `adapters/libero_env_server.py` |

旧顺序链路（adapter 8766 / RPent 44481 / SAM3 60071）未被改动，仍可独立启动，用作对照。

## 1. Agent 层

- **实际模型**：`Qwen3.5-9B`，通过 vLLM `0.24.0` 部署在 `127.0.0.1:8001`（GPU3，`--enforce-eager`）。
  同权重的共享实例 `:8000` 受其他负载影响，同一 prompt 实测 9.2 s；专用实例 1.1 s，
  因此 `config/engine.json` 指向 `:8001`。
- **职责**：把用户的原始自然语言指令路由到十任务目录中的**一个**能力键（新任务语义路由，
  见 §1.1），再由代码取官方标准指令组装 `whole_task` 计划；另外负责失败后的
  `retry / replan / abort` 决策。
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
- **输入**：用户原始指令、场景快照（物体位姿、site 几何、关节、夹爪开度）、技能目录、
  已完成步骤与失败原因、可选决策集合、`episode`/`scene_epoch`/`observation_id`。
  语义路由的 prompt 只含十任务能力目录与用户原句，**不含** `task_id`、当前官方指令、
  场景快照或 BDDL goal。
- **输出**：`{"choice": "<catalog key>"}`、`{"choice":"clarify"}` 或 `{"choice":"unsupported"}`；
  代码校验后组装 `{"interpretation", "success_condition", "steps": [whole_task], "source":
  "qwen3.5-9b-semantic-router", "latency_s"}`；恢复输出
  `{"status", "decision", "steps", "reason", "policy"}`。
- **Jev（TypeSafe System One）现状**：官方为云端 API（`https://api.typesafe.ai/v1/systemone`），
  需要 waitlist 账号与 `TYPESAFE_API_KEY`；本地无权重、无可用凭据。因此**没有接入 Jev**，
  也不能把 Qwen 的决策标成 Jev。Agent 层当前为"语言模型语义路由 + 代码组装 + 结构化契约 + 有界恢复"。

### 1.1 Qwen 语义路由（当前唯一 Agent 模式）

链路固定为：**用户原始指令 → Qwen 选择目录键 → 代码校验 → 代码取官方标准指令 → whole_task → OpenWAM**。

- **唯一任务表**：`config/tasks.json`（schema `embodied_engine.task_catalog.v2`，版本 `2026-10-27.1`）
  是项目中唯一的十任务目录，UI、Agent、引擎、测试都通过 `common/task_catalog.py` 读取。
  每项记录：`task_id`、`key`、官方 `official_instruction`（本 fork 的 `Task.language` 由文件名生成）、
  BDDL `:language`、BDDL 与 init 文件 SHA256、原生 goal 原子、完整语义绑定
  （operation / source / target / relation / subgoals）以及给模型看的 `capability` 描述。
- **候选隔离**：发给模型的是十项能力的语义描述（`capability_rows()`），不含 `task_id` 与官方句子；
  路由 prompt 里也没有当前加载任务、BDDL goal 或场景快照。被选中的键由代码回查目录，
  因此模型不能"看着答案选答案"。
- **紧凑输出**：`max_tokens=24`、`temperature=0`，只返回一个 JSON 对象。
  上限与修复次数由 `config/engine.json` 的 `agent.semantic_router` 读取
  （`router_options()`，写入 health 与响应遥测），不在三处各写常量。
  解析失败（非 JSON、缺字段、键不在目录）才允许一次修复，修复仍失败则 `planning_failed`。
- **目录一致性由测试守住**：`tests/test_task_catalog.py` 从安装的 LIBERO 包重新推导
  任务顺序、文件名指令、BDDL/init SHA256 与 goal 原子，和 `tasks.json` 逐项比对；
  改错一个字、或升级 LIBERO 包，都会直接测试失败。
- **非思考模式**：使用该 vLLM 实例正式支持的 `chat_template_kwargs={"enable_thinking": false}`
  （见 `agent/planner.py::chat`），不是只在提示词里写"不要思考"。
- **校验与场景检查**（全部在代码里，不在模型里）：
  1. `parse_choice()` 只接受目录键、`clarify`、`unsupported`；
  2. `task_catalog.scene_support()` 用实时快照检查所选任务的 source/target 实体是否存在
     （`sim.objects` 与 `named_poses` 的 site/body，fixture 用 site 前缀匹配）；
  3. 选中任务与已加载场景不一致时返回 `task_mismatch`（HTTP 409），提示切换任务，
     **不会偷偷加载另一个场景**；
  4. 一致时取目录中的官方指令组装唯一 `whole_task` 步骤，再走 `plan_contract`。
- **可追溯字段**：状态与回合记录里的 `understanding` 保存用户原文
  （`raw_instruction`）、模型语义选择（`semantic_choice` / `decision_status` / `reason`）、
  目录键与 `task_id`、官方标准指令、模型与模式、目录版本与 SHA256、路由耗时、
  场景支持结果、`needs_scene_switch`。回合记录同时写入
  `plan.steps[0].prompt` 与策略侧 `verification.prompt`，因此"用户原文 → 语义 → 标准指令 →
  OpenWAM 实际输入"是同一条可核对链。
- **不做的事**：不把 `clarify` 当低置信度回退、不要求模型自报概率、不用关键词/同义词表代替模型、
  不因为语义相近就替换对象或目标（例如不会把"把碗放柜顶"换成"放盘子"）。
- **历史**：`d1-3B` 决策模式已按用户授权下线并从活动代码删除（服务、规划器、启动脚本、
  UI 选项、health 探测），只保留 Git 历史与 `records/ab/` 里的精简对照证据。

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
  `sim_at_ack` 由 `_settled_sim_count()` 通过 `env.client` 连续取快照、直到两次读数一致
  （上限 2 s、取最大值）后写入，不能只取一次：rollout 线程 join 返回时，最后一步可能仍在
  env 进程里执行，单次快照会少算一步，让"停止后无新动作"这条证据失真。
  `scripts/test_lifecycle.py` 断言 `in_flight_actions ≤1`，且 3 s 后 `sim_action_count`
  与 ack 值相等（连续两次运行通过）。
- 停止回合的记录：`except Stopped` 分支在落盘前重新取一次快照，因此
  `sim_action_count` / `episode_sim_actions` 反映**停止前真实执行过的动作数**
  （此前用的是 prepare 阶段快照，恒为 0）。取不到时写 `stop_snapshot_failed` 日志，
  不静默吞异常。

## 5. page_ui

- 来源：本仓库 Git 历史中的旧双相机页面（`git show fc9d499:page_ui/` 恢复；该目录在
  `7864980` "去掉已经不在执行链上的旧页面" 时被删除），保留原布局、双相机
  （Agent/Wrist）、开始/停止/reset 控件；`server.py` 改为连接新引擎 14800 的 WebSocket，
  原 `get_scenes` / `load` / `start` / `restart` 文本协议与"JSON 头 + 两路 JPEG"二进制帧
  保持不变，`stop` / `reset` / 身份消息为新增。
- 展示字段全部来自后端 `/api/status` 的 `identity`：Agent 模型与模式、语义路由的
  非思考模式/最大 token/目录版本、动作模型、世界模型状态、推理后端与 async/sync 模式、
  阶段与失败原因。没有写死标签。
- 语义链路展示：用户原始输入（`raw_instruction`）、模型语义选择与目录键
  （`semantic_choice` → `task_id key`）、理解后的官方标准任务（`official_instruction`）、
  需要切换任务时的目标任务（`needs_scene_switch`），以及阶段与失败原因。
  没有 `--` 之外的占位值：未识别时对应字段为空并在页面上标红。
- 本层已没有 Agent 模型切换控件（只有 Qwen 一个语言模型），旧的
  `qwen` / `d1` / `d1_with_qwen_fallback` 消息通道随 d1 下线一起删除。
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
python3 scripts/eval_offline.py --set config/semantic_eval_20261027c.json \
    --out records/offline/semantic_eval_20261027c.json --legacy --legacy-limit 20
python3 scripts/run_language_rounds.py --rounds config/language_rounds_20261027.json
python3 scripts/test_lifecycle.py   # stop/reset/planner_generation 回归（不需要重新加载模型）
python3 scripts/progress_probe.py   # 离线：把已记录回合折成"手在动、物体不动"的窗口，不 step
python3 scripts/render_determinism_probe.py --env http://127.0.0.1:14561   # 同 seed 重渲染，需先 load 一次
python3 scripts/composite_matrix.py       # 从已安装 BDDL 重新解析十任务资产，输出组合矩阵
python3 scripts/inference_determinism_probe.py   # 同一观测两次请求，检查动作是否逐位相同
python3 -m unittest discover -s tests -p "test_*.py" -v   # 语义路由、任务目录、计划契约、完成判定、OpenWAM 契约
```

`scripts/d1_server.sh`、`agent/d1_service.py`、`agent/d1_planner.py`、`scripts/run_ab.py`
已按下线要求删除，`records/ab/` 中的精简 A/B 证据与 Git 历史仍保留。

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
| Agent | DONE（Qwen3.5-9B，语义路由） | Jev 无凭据/无本地权重，未接入，未冒充；十任务语义路由 + 代码组装 + 有界恢复 |
| Agent（d1-3B） | **已下线并删除** | 用户授权后停止服务、删除权重与专用 venv、移除代码/UI/health 依赖；证据见验收报告 §15 |
| 动作模型（`pi05`） | DONE | 常驻、真实推理、真实执行（基线保留） |
| 动作模型（`openwam`） | **DONE，执行链** | 常驻 14520，真实 10 维动作；LIBERO-Goal 实测见验收报告 |
| 推理引擎 | DONE | 双后端选择、Pi0.5 异步块预取、OpenWAM 同步单步、token/epoch/age 校验、取消与缓冲清理 |
| LIBERO 适配 | DONE | 快照、关节、site、RPC、生命周期；OpenWAM 契约逐函数对拍 |
| 任务目录 | DONE | `config/tasks.json` 为唯一任务表，含官方指令、BDDL/init 指纹与语义绑定 |
| page_ui | DONE（协议 + 浏览器） | 身份消息显示 Qwen＋OpenWAM；显示原文/语义/标准任务/阶段/失败原因 |
| 十任务（Pi0.5 历史验收） | **PARTIAL（26/30）** | seed 0、每项 3 回合；失败项与断点见 `docs/acceptance_report.md` |
| 十任务（OpenWAM 单轮） | 见 `docs/acceptance_report.md` | 每任务 1 回合，不重试 |
| 十任务 ×3 初始化 A/B（60 回合，历史） | 见 `docs/acceptance_report.md` §14 | d1 已下线，仅作历史对照 |
| 同义指令映射（离线） | DONE | 回归集 101 例 + 新冻结集 100 例，映射 100%，中位 0.68 s，见验收报告 §17 |
| 20 回合语言闭环（真实执行） | DONE（映射 20/20，原生 19/20） | 10 任务 × 1 英文改写 + 1 中文表达，指令运行前冻结；释放证据单列，见 §18 |
| 放置收尾（释放 + 退离 + 验证） | DONE（最终 12 回合：11 原生，9 策略 + 2 执行层） | 见验收报告 §19；`policy_owned` 逐回合标注，不与"自主成功率"混算 |
| 同场景两步组合 | **设计入口，未执行** | 可行性矩阵见验收报告 §21；不实现调度、不改路由输出 |

验收与记录：

- Pi0.5 历史验收 26/30（86.7%），成功回合中位 27.5 s，规划中位 6.5 s，块间空档合计 0.01 s，
  推断与动作执行重叠约 38%（块长 5 动作 ≈ 0.335 s < 单块推断 0.354 s，是剩余等待的根因）。
- OpenWAM 的回合记录里，`model.backend=openwam`、`model.openwam.integration=on_execution_path`，
  步骤 `skill=whole_task`，`final_verification.method=native_libero_goal`。
- 每轮验收的记录会归档到 `records/acceptance/<suite>_runs/`；`records/runs` 保留最新
  `MAX_RUNS=200` 份回合。

## 8. 放置收尾：谁结束回合，谁负责松手

LIBERO 的 `terminated=True`、服务端的 `can_continue`、策略 chunk 的剩余动作，是三个
互相独立的事实。把它们混为一谈，就会出现"原生目标成立但手指还夹着物体"被记成完成，
或者反过来把"策略本来会松手"的回合强行截断。

实际链路（`engine/engine_service.py` + `engine/openwam_rollout.py`）：

| 事实 | 来源 | 本栈的处理 |
| --- | --- | --- |
| 原生目标成立 | LIBERO `terminated` | 记录**事件** `native_goal`、当时的动作计数与夹持证据；不改写原始 `terminated` |
| 能否继续 step | 适配层 `mode="free"` 返回的 `can_continue`，配合 `EpisodeFlags.absorb` 清 `terminated` | 允许继续时进入收尾；`truncated`/真终止/`stop` 一律优先，不进收尾 |
| 策略是否还要松手 | 当前 chunk 内 `native_goal` 之后的动作 | Tier 1：只消费**本 chunk 剩余动作**，不重规划 |

收尾分两层，且来源必须分开记账：

- **Tier 1 策略收尾（`evidence_source="policy_chunk"`，`policy_owned=true`）**
  `openwam_rollout.py` 不再在首个 `native_goal=True` 处 `break`，而是把当前 32 步
  receding-horizon chunk 里剩余的动作走完。若其中出现夹爪命令张开、实测开度 ≥0.05 m
  连续 2 帧、且物体已在手指跨度之外，判为释放成功（`release_verified`），
  随后用**新的**仿真样本验证目标关系（`placement_verified`）。这些动作是 OpenWAM 自己
  产生的，可以计入策略完成。
- **Tier 2 执行层收尾（`evidence_source="execution_finalizer"`，`policy_owned=false`）**
  chunk 耗尽仍未释放时，`engine/finalize.py::PlacementFinalizer` 先**清空策略缓冲**
  （旧动作基于过旧观测，不能和收尾动作混发），再在**原姿态原地**张开夹爪，
  沿 `+z` 做有界退离探针，用 3 个**新仿真样本**（`sim_action_count` 递增 / `observation_id`
  变化）验证"物体已脱离手指且目标关系保持"。这些动作不是 OpenWAM 生成的，
  报告里单列为执行层辅助。

共同约束（写在 `engine/completion.py` 与 `finalize.py` 里，非调用点约定）：

1. 只在**可靠的放置候选状态**允许张开夹爪。候选判定用已有的关系谓词与几何
   （`engine/grasp_geometry.py`：手指间距、物体到指垫线的间隙、手指跨度包含关系）。
   `native_goal=true` 本身**不是**释放许可：task 3 第 7 回合就是因为 `in_box` 不成立
   被拒（`placement_reason="not_at_release_candidate"`）。
2. 开度只是状态量。开度 <0.05 m 不能单独证明"仍夹着物体"（物体可能已被放下而手指恰好闭合），
   开度 ≥0.05 m 也不能单独证明"已稳定放下"；两者都要配合物体位置、手指跨度与关系谓词。
3. 稳定性样本必须跨真实仿真状态。连续查询同一快照不算样本，`finalize.py` 因此按
   `sim_action_count` 去重，而不是数调用次数。
4. 退离单独记账（`retreat_status`）。退离失败不抹掉已经证实的目标完成，也不把
   `placement_verified` 反写成 false。
5. `stop` / `reset` / 超时 / 真终止优先于一切收尾：收尾循环每一步先查回合取消标识，
   取消后不再下发动作，且迟到的收尾结果不写入新回合。

回合记录因此新增这些字段（同一份记录里并列，不互相推导）：

`native_goal`、`release_executed`、`release_verified`、`placement_verified`、
`release_state`（`observed` / `unknown` / `not_applicable`）、`evidence_source`
（`policy_chunk` / `execution_finalizer` / `none`）、`policy_owned`、
`retreat_status`、`execution_status`、`policy_actions_after_goal`、`policy_stop_reason`、
`finalizer_actions`、`finalizer_s`、`placement_reason`。

`completion.verification()` 的出口是四态：完整放置成立才 `success`；原生成立但证据不足是
`unknown`（附 `placement_reason`），不是 `success`；关系丢失是 `failure`；
不需要释放的技能（开抽屉/推/开炉灶）走各自的关节或位移条件。

## 9. 同场景两步组合（0→8 已实现）

十个 LIBERO-Goal BDDL 声明的是**同一套资产**（`main_table`、`wooden_cabinet_1`、
`flat_stove_1`、`wine_rack_1` + 四个物体）与语义相同的 `:init` 区域，区别只在
`:goal` 与各自采样的 `pruned_init`。可行性矩阵与两个候选（`0→8`、`1→7`）见
`docs/acceptance_report.md` §21；**本轮只实现并真实执行 `0→8`**，`1→7` 只在报告里
记录"旋钮可达性未测量"。

### 9.1 契约（`common/composite_plan.py`，纯数据，无仿真）

- `options()`：`enabled`、`supported_pairs`（当前只有 `["open_middle_drawer",
  "put_bowl_on_plate"]`）、`max_steps=2`、每步预算（`1:240`、`2:600`）、
  `goal_confirm_samples=2`。
- `completion_for(task, is_anchor)`：**锚定步**用加载场景的原生目标
  （`native_liberio_goal`），早于锚定的步用引擎谓词（抽屉 `drawer_open`、
  开关 `stove_button`、放置/推物 `relation`）。第一步的原生标志照常记录，但
  **不作为它的目标**。
- `support(keys, loaded_task, catalog)`：识别与执行分离。返回 `ok/executable`
  或明确的 `reason_code`：`not_a_composite`、`step_limit_exceeded`、
  `duplicate_step`、`key_not_in_catalog`、`unsupported_combo`、
  `requires_scene_switch`、`step_goal_unsupported`。复合以**最后一步**的 BDDL
  为锚，因此只有 `loaded_task` 是锚定任务时才可执行。
- `build_plan()`：生成 `step_id="c1"/"c2"`、`step_index`、`step_task_id/key`、
  `prompt`（目录官方指令）、`goal_kind` 与 `completion`；`validate_plan()`
  是执行前的结构自检。
- 路由与执行共用这一份定义，"哪些有序对能执行"没有第二处副本。

### 9.2 执行（`engine/engine_service.py` + `engine/openwam_rollout.py`）

1. `/api/load` 用**锚定任务**（task 8）的 BDDL 与固定 init 索引；episode 内不
   reset、不重载、不改 BDDL、不传送物体。
2. 计划来自 Qwen 的有序 key 列表；`task_scope="combo"` 时 `goal_step` 仍是最后
   一步，但每一步都带自己的 `step_id`，请求行、动作、完成状态按 `step_id` 隔离。
3. 第一步向 OpenWAM 发**第一步的官方指令**，由 `goal_state` 的引擎谓词判定完成
   （`drawer_open`：`qpos < -0.14`，连续 2 个新样本）。`WholeTaskRunner` 只在
   `goal is None`（非组合步）时信任加载场景的 `native_goal`；组合步即使
   `native_goal` 为真也不会提前结束第一步。
4. 每步有界预算（240/600 动作）；预算耗尽即失败并写明阶段
   （`composite_step_1_failed` / `composite_step_2_failed` / `composite_handoff_failed`），
   不会靠整场 reset 续命。

### 9.3 步间处理（`engine/composite.py::Handoff`）

顺序固定，任何一项不满足就 `not_ready` 并拒绝启动第二步：

1. **校验第一步目标**：`state_check(pre)` 必须成立，否则 `step_1_unverified`。
2. **按动作类型松手**：`release_required(step_task)` 为假的抽屉/开关步不套放置
   finalizer（记录里写 `release.required=false` 与理由），只有放置/推物步才走
   §8 的释放路径。
3. **有界退离到 episode 初始 EEF 位姿**：位置按 0.04 m/步、姿态按轴角闭环
   （OSC 单位旋转指令约 0.05 rad，代码按实测四元数逐步收敛），上限 60 动作，
   收敛阈值 0.01 m / 0.05 rad（≈2.9°）。退离**只动机械臂**，不关门、不挪物。
   记录 `start_angle_deg`/`end_angle_deg`/`rotation_applied`/`moved_m`/`reason`。
   为什么必须恢复姿态：OpenWAM 的 proprio 是 EEF10（位置 3 + 旋转 6D + 夹爪 1），
   姿态不恢复时第二步从"手腕朝向抽屉"的分布起步，手腕相机看到的是抽屉内壁。
4. **检查退离没有破坏第一步目标**（`retreat_preserved_goal`），`scene_epoch` 与
   `scene_generation` 不变。
5. **清空策略缓冲**：`inference.reset("handoff:<episode>:<step_id>")` 返回新 token，
   清队列、清结果、清模型侧 receding-horizon 缓冲；第二步请求带新 token，
   不会有第一步在途 chunk 被执行。
6. **取新观测并确认新鲜**：`raw_obs()` + `state_snapshot()`，`observation_id`
   必须不同于交接前；第二步以该观测开始。

### 9.4 终局与记录

- 第二步发**第二步的官方指令**；放置沿用 §8 的释放与稳定性验证，保留
  `policy_chunk` 与 `execution_finalizer` 的来源区分。
- 回合成功（`completed`）后，用**新的**仿真样本对两步同时复验：
  `step_1_final_ok ∧ step_2_final_native`，再叠加释放证据得到 `combo_verified`。
- 记录字段（`composite` 段）分开记账，不互相推导：
  `scene_epoch_start/end/continuous`、`sim_action_count_start/end`、
  `env_reset_between_steps=0`、`env_reload_between_steps=0`、`bddl_mutation=false`、
  `object_teleport=false`、每步 `ok`（该步结束时）/`ok_final`（终局复验）/
  `native_current`/`native_ever`/`sim_actions`/`release_state`/`release_source`、
  `handoff`（§9.3 全部字段），以及 `native_goal_current`/`native_goal_ever`、
  `step_1_ok`/`step_1_final_ok`/`step_2_native`/`step_2_final_native`/
  `final_verification_read`/`combo_verified`。
- 只读证据扩展：环境快照 `sim.eef_quat`（`adapters/env_runtime/episode_service.py`），
  用于退离闭环与记录，不参与任何动作生成。

### 9.5 路由（`agent/semantic_router.py`）

- Qwen 只输出**有序目录 key 列表**或 `clarify`/`unsupported`；官方动作指令一律由
  代码查 `config/tasks.json`，提示词里没有官方指令、task_id 或 BDDL 目标。
- 组合语义先识别再判可执行：模型可以识别任意"两步能力"，只有 `support()` 放行的
  有序对才组装计划；其余返回 `unsupported` + `unsupported_combo`（附支持对），
  不降级成单任务、不交换顺序。
- 顺序（词序/显式先后）、实体、场景、目标冲突、超过两步、缺槽（抽屉层级、代词）
  都有对应出口：`clarify` 只用于缺槽，`unsupported` 用于说清了但目录没有。
- 单任务路径与既有输出（`task_scope="single"`、`semantic_choice`、官方指令一致）
  保持不变；组合额外返回 `composite_keys`、`composite_executable`、
  `composite_reason_code`、`step_official_instructions`、`step_goals`。

### 9.6 已知偏移（写进结果，不隐藏）

第二步从第一步的终态起步：抽屉是开的、机械臂带着第一步的动作历史。即使位姿已回到
初始 EEF（位置 ≤1 cm、姿态 ≤3°），**主相机的画面与官方 init 不同**，这是组合的
固有分布偏移；验收报告把"第二步在偏移下的表现"与"第二步单步对照"分开写，
不把偏移导致的失败归因为模型采样或渲染。


### 9.7 实测结果（2026-10-27/28，6 回合冻结 + 补测）
机制与能力的结论必须分开读：

| 层面 | 结果 | 证据 |
| --- | --- | --- |
| 组合路由（Qwen） | 开发集 17/17、冻结验收集 16/16；median ≈0.81 s、p95 ≈0.95–1.04 s | `records/offline/composite_routes_*.json` |
| 第一步（`drawer_open`） | 6/6，114–123 动作，未用满 240 预算 | `records/acceptance/composite_rounds_20261027.json` |
| 交接 | 6/6 `ready`，退离 31–45 动作，姿态 101.4–105.5° → 1.1–2.8° | 同上 `composite.handoff` |
| 第二步原生目标 | **1/6** | 同上 `composite.step_2_native` |
| 双目标同真（`combo_verified`） | **0/6** | 同上 |
| 单步对照（task 8，同 3 索引） | 3/3，88–98 动作 | `composite_control_single_20261027.json` |
| stop（第一步/步间/第二步/finalizer 中途） | 4/4 停住、ack 后不再动作 | `composite_stop_tests_20261027.json`、`..._finalizer_20261027.json` |

失败结构（写进记录，不隐藏）：

1. 5 回合第二步 600 动作内一次都没出现 `On(bowl, plate)`；
2. 1 回合第二步完成了自己（479 动作、`policy_chunk` 释放、`placement_verified=true`），
   但终局复验发现第一步目标已回退：抽屉 `qpos` 从第二步开始时的 −0.1547 变成 −0.0999
   （阈值 −0.14），引擎报 `composite_final_verification_failed` / `step_goal_regressed`；
3. 修复后补测 1 回合（同指令、同 seed 0）在失败瞬间测到抽屉 `qpos=−0.1167`，
   即第二轮独立复现"第二步把第一步的开抽屉状态推回 3.4–5.5 cm"。

因此 `0 → 8` 在本轮**不是**一条可用能力：执行机制（步身份、缓冲隔离、交接、双目标分开记账）
成立，**物理上两个目标会互相破坏**。因果方向（抽屉挡臂 vs 臂撞抽屉）未分离，见
`docs/acceptance_report.md` §23.5–§23.6 与 §29.2。`1 → 7` 因此没有进入放行清单，
只做了只读几何测量（`scripts/knob_reach_probe.py`），动态可达性标注 `not_tested`。

### 9.8 组合失败证据的字段（本轮新增，只读）
- `composite_steps[i].failure_evidence.prior_steps_at_failure`：某一步失败时，用**同一份**
  失败快照复验此前各步的目标（`engine_predicate` 结果 + `qpos`）。没有它，
  "第二步没到"和"第二步把第一步拆了"在记录里长得一样。
- 被 stop 打断的回合保留 `placement` 块（`attempted`/`extra_actions`/逐动作行）：
  finalizer 内部把 `Stopped` 写进块后由外层重新抛出，避免"释放半途被打断却记成 completed"。

### 9.9 步间交接的两种模式（2026-10-28 新增）

交接模式是**引擎运行时配置**（`composite.handoff.mode`），不是回合内可变项：

| 模式 | 语义 | `physical_continuity` |
| --- | --- | --- |
| `action_retreat`（默认，原有） | 只发动作，把 EEF 闭环退离到 episode 初始位姿，物体与抽屉一个字节都不写 | true（动作连续） |
| `robot_state_restore`（新增） | 保留场景，把机器人 9 个关节**直接写回**该回合初态；一次性状态写入，不是动作 | false（teleport） |

- 读取入口：`GET /health` → `identity.engine.handoff`（`mode`/`default_mode`/`modes`/`options`）；
  `POST /api/handoff {"mode": ...}` 只允许在**没有回合在跑**时切换（409 拒绝），
  每个回合在 `_run_episode` 开头快照当前模式，记录里写的是真正生效的模式。
- 回合记录：`composite.handoff_mode`、`composite.robot_state_restore`、
  `composite.robot_teleport`、`composite.physical_continuity=false`、
  `composite.robot_state_capture`、`composite.action_accounting.*`。

**初态捕获**（`episode init`，`/api/reset` 稳定动作之后、第一个任务动作之前）：
`env.robot_state_capture` 在 worker 进程里读 `robot.robot_joints` + `robot.gripper_joints`
的模型地址（`jnt_qposadr`/`jnt_dofadr`/`jnt_type`），不写死切片；返回 qpos/qvel/EEF/
夹爪/控制器段/非机器人 qpos 摘要/SHA256。捕获编号绑定 `service_instance_id`，
env 进程重启后旧编号一律 `capture_unknown`，不会恢复一个没人读过的状态。

**恢复顺序**（`composite.Handoff` → `env.robot_state_restore`）：

1. 第一步目标必须已经成立（否则 `step_1_unverified`，不写任何状态）。
2. 放置类步骤必须先有已测释放（`release_not_verified` 时不写）。
3. 停止动作 → 等在途推理（`inflight_wait_s`）→ `inference.reset` 清缓冲。
4. **测量式脱离**：`env.robot_state_inspect` 读手指与作用对象的真实接触；
   有接触时用 ≤8 个开爪小步退离（`detach_*`），每一步重新测接触，
   清不掉就 `gripper_contact_after_detach` 拒绝。抽屉步不能凭"任务类型"假定无接触。
5. **候选构型碰撞检查**：`mujoco.MjData(model._model)` 独立副本上写入候选 qpos、
   清 qvel、`mj_forward`，读接触穿透；超 `restore_penetration_tol_m` 就
   `candidate_penetration` 拒绝。主环境不试写、不回滚。原始对象拿不到时返回
   `raw_mujoco_unavailable`，本仓库因此**拒绝**恢复而不是宣称安全。
6. 只写机器人 qpos 索引、清零对应 qvel，`sim.forward()`；随后逐部件
   `update(force=True)` → `reset_goal()` → `composite_controller.reset()`，
   否则 OSC 会用陈旧 `ref_pos` 把机械臂拉回旧目标（实测会导致 restore 后跳动）。
7. 复读校验：`robot_qpos_error ≤ 1e-6`、`robot_qvel_abs_max ≤ 1e-9`、
   非机器人 qpos 摘要**逐字节相等**、`sim.time` 不变。校验失败即 `not_ready`，
   且在写后**不追加任何动作**。
8. 已计数稳定动作（`settle_actions`，零位移 + 开爪的保持指令），每步记录 EEF/抽屉
   qpos；稳定阶段用毫米级容差（`settle_scene_tol_m`/`settle_object_tol_m`）判定
   "场景没被推动"，写入本身的精确性由第 7 步的摘要相等保证。
9. 复验第一步目标 + 取新观测（`observation_id` 必须变），才允许第二步启动。

**记账**：`action_accounting` 分开记 `policy_actions` / `handoff_detach_actions` /
`handoff_settle_actions` / `handoff_retreat_actions` / `finalizer_actions` /
`sim_actions_total`；`handoff.counts` 是同一回合的逐项来源。
`robot_state_restore=true` 与 `robot_teleport=true` 同时写死，避免把"场景连续"
误读成"物理连续"。

**观测/缓存同步**：恢复后 `_post_process()` + `_update_observables(force=True)`
刷新 worker 的观测缓存；模型侧缓冲由第 3 步的 `inference.reset` 清空。
不做 `env.reset`/`load`/`set_init_state`，不回退仿真时间，不 `sim.step` 绕过计数。

**能力缺口（如实标注）**：
- 采样随机性：动作噪声由 `torch.Generator(device).manual_seed(seed)` 生成，`seed` 缺省 42
  （`openwam/deploy/engine.py` 的 `conditions.get("seed", 42)`、
  `openwam/model/architectures/base.py` 的 `generate(seed=42)`），推理路径上**没有**别的随机源。
  所以相同 `(obs, prompt)` 必然给出相同动作，重复回合的差异只来自观测不同，不是采样抖动。
  代价是常驻服务**没有**按回合/按请求改种子的入口：重复回合仍然不能当独立样本，
  也没法用"换种子"来估计策略方差。详见 `acceptance_report.md` §32.7。
- 候选检查只能报穿透，不能报"未来是否会被臂扫到"；它不是运动规划。
- `physical_continuity=false`：本模式不做物理连续的组合，只回答"机械臂状态是否是
  第二步失败的主因"。

### 9.10 第二步对前序目标关节的追踪（2026-10-28 新增，只读诊断）

要回答"第二步是在第几个动作上把第一步打开的抽屉推回去"，光有回合末的
`qpos` 不够：那只能说明回退了，不能说明什么时候开始退。所以第二步执行期间追踪
**前序步骤目标关节**：

- 关节名不写死。`_watched_joints()` 从前序步骤的 `completion` 里取 `drawer_open`
  目标（例如 `wooden_cabinet_1_middle_region`），用 `completion.drawer_level()` +
  `completion.drawer_joint()` 在当前快照的 `joints` 里解析出真实关节名；
  owner 前缀对不上时按 `_<level>_level` 后缀回退，仍解析不到就**不追踪**（宁缺不猜）。
- 追踪点接在**实际执行第二步的那个 runner** 上：组合的两步都走 `WholeTaskRunner`
  （OpenWAM whole-task），所以追踪必须在 `engine/openwam_rollout.py` 里，
  只有 `engine/skills.py` 的追踪在组合回合里永远不会触发（本轮已修）。
- 只读：`_watch()` 复用该动作已经取回的 `snapshot["joints"]`，不发额外 RPC、
  不改动作流；基线取第二步**第一个动作时**的值，也就是交接交出来的状态，
  这样"回退"度量的是第二步自己推动了多少。
- 首个超过 `1e-4 m` 的变化记一条 `watched_joint_moved` 日志，并读一次
  `env.robot_state_inspect` 的接触，记录 `sim_action_count`（全回合动作号）、
  `skill_sim_actions`（本步内动作号）、关节前后值、EEF、夹爪开度、
  `robot_vs_nonrobot`/`gripper_vs_nonrobot` 的 geom 对与穿透深度。
- 回合记录：`steps[*].watch_joints`、`steps[*].watch_baseline`、
  `steps[*].watch_events`；`run_rounds.py` 把它们提到 `watch_joints`/`watch_events` 列。
- 交接内部的动作行（`handoff.detach.steps[*]`、`handoff.restore.settle[*]`）里的
  目标关节名同样由 `Handoff._goal_joint()` 现场解析，不写死层级或 owner，
  字段是 `goal_joint`/`goal_joint_qpos`（旧字段 `drawer_qpos` 保留，值改为解析结果）。

**符号约定（2026-10-29 补充，两处口径不同）**：

- `watch_events[*].delta` 是**有符号**的（`now - baseline`）：抽屉更接近 0 为正（关回），
  更负为负（继续拉开）。`summarize_drawer_regression.py` 的
  `first_close_5mm`/`first_open_5mm` 也按这个符号分方向，`first_move_5mm` 方向无关。
- `detach_check.joints[*].delta` 是**绝对值**（`abs(after - before)`），方向另存
  `delta_signed`（2026-10-29 新增，追加字段，旧的 `delta` 语义不变）。
  跨表比较时先统一符号，否则会把"继续拉开"读成"关回"。

**非机器人关节速度（2026-10-29 补充）**：`_nonrobot_state` 现在同时返回 `joints`（qpos）与
`qvel`（按关节名），后者**不参与** `digest`，所以"恢复后非机器人 qpos 逐字节相等"
这条验收口径不变。用途是区分"抽屉是带着速度过去的"和"抽屉是被推的"。
