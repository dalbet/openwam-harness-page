# 阶段 A/B：顺序 Harness 最小可运行版本

顺序页在 `8766`。`HARNESS_LIVE=1` 才创建 LIBERO。干跑会标明自己，不会用占位帧进入确认，也不会在环境或 VLA 缺失时偷偷改成干跑。模型计划两次失败返回 `PLANNING_FAILED`，不会偷偷换成人工计划。人工计划只有用户显式选择，`plan_source=manual`。

## 现场位置

- Git 根目录是 `/root/Page_dmh`，远程是 `git@github.com:dalbet/page.git`。
- 嵌套上游 `openwam_harness/RPent` 不进入本仓库。现场改动保持为可重放补丁 `openwam_harness/baseline/runtime_patches.py`（robosuite `default_base is None` 时使用 `NullMount`）。
- 原 Page UI 在 `/root/Page_dmh/page_ui`，协议不变：`load` / `start` / `restart`，二进制双相机帧。它仍是 ManiSkill 路径。
- 顺序模式走薄适配层 `openwam_harness/sequential/adapter.py`，端口 `8766`。视频帧格式与 Page UI 相同。

## 保留的原模式

原 Harness 仍是 `rpent --robot libero --planner api`。Planner 仍是已下载的 Qwen3.5-9B，VLA 仍是 Pi0.5，分割仍是 SAM3。不接 FlashRT，不换 VLA，不训练。

`openwam_harness/baseline/start_planner.sh` 保持正在运行的服务参数：`max-model-len 32768`，`gpu-memory-utilization 0.48`。不要把它改成 65536。

## 已保存的失败

分类在 `openwam_harness/sequential/records/failure_catalog.json`。原始日志在 `openwam_harness/runs/baseline_original/`（不上传）。

| episode | 类别 | Pi0.5 `pi0_pick` |
|---|---|---|
| `libero_10_t0_s0_r0` | planner 超时 1200s | 未调用。调用了 SAM3 `segment` |
| `libero_10_t0_s0_r1` | context 错误，输入至少 24577 + 请求输出 8192 > 32768 | 未调用 |
| `libero_10_t0_s0_r2` | 同上 | 未调用 |
| `libero_10_t1_s0_r0` | 同上 | 未调用 |
| `libero_10_t1_s0_r1` | 本轮 SIGINT，transcript 未写完 | 未调用 |

没有操作失败记录，因为这几回都没有进入抓取。未启动的 `t1` 重复回合已取消。

## 预算和短记忆

配置：`openwam_harness/sequential/budget_config.json`。

- Planner 输出额度 2048。
- 上下文上限 32768，与当前 vLLM 一致。
- 每次请求前把任务、技能约束、当前失败证据、工具 schema 和短历史分开计数。
- 没有 tokenizer 时按每 2 个字符至少 1 个 token 高估。超过上限在本地抛 `BudgetError`，不发送。
- 不裁任务、技能约束和当前失败证据。不把整份 memory corpus 或 guide 放进请求。
- 初始规划一次。格式不合法时最多修复一次。仍不合法返回 `PLANNING_FAILED`，不执行，也不换成人工计划。人工计划只有用户显式选择，`plan_source=manual`。
- 未完成的流式输出不会进入执行器，也不会靠拼接补全。客户端关闭流会只取消该 `X-Request-Id`，不停止 vLLM 进程。
- 同一工具失败后只重试一次，再失败则 `no_progress` 退出。

## 顺序执行

人工计划：`openwam_harness/sequential/plans/two_object.json`。

- 任务 0：alphabet soup 和 tomato sauce 放入 basket。
- 任务 1：cream cheese box 和 butter 放入 basket。
- 允许的动作只有 `view_env_state`、`segment`、`pi0_pick`、`move_to`、`set_gripper`、`pause_for_confirm`。
- `pi0_doubled`、任意代码和子任务间 `reset` 会被 schema 拒绝。
- `pi0_pick` 的返回只记为 `grasp_signal`。子任务在确认前是 `unknown`。整体成功只看 LIBERO `env.terminated`（toolkit `solved()`）。
- 每个物体放下后暂停，等用户确认成功、失败或未知。确认前不进入下一个物体。
- 动作之后只取新观测，不因此再请求 LLM。
- `episode_id`、`scene_epoch`、`plan_version` 变化或取消后，旧确认被丢弃。

`HARNESS_LIVE=1` 时适配层调用现有 `LiberoToolkit`，复用 Pi0.5 和 SAM3。默认 `HARNESS_LIVE=0`，只跑契约和 UI 协议，不占第二套模型。

## 启动

Planner 已经在 `127.0.0.1:8000`。不要再开一个 vLLM。

```bash
# 原模式 Planner（仅当 8000 没在听时）
bash /root/Page_dmh/openwam_harness/baseline/start_planner.sh

# 干跑只检查协议，页面会显示干跑并拒绝执行。
HARNESS_LIVE=0 bash /root/Page_dmh/openwam_harness/sequential/start_adapter.sh

# 真实闭环。脚本若发现本项目的旧 adapter 占着 8766，会停掉那个 pid 再启动，不会复用旧进程。
# 不杀其他端口上的进程。显存不够就先停 8000 上的 vLLM，不要再开第二个。
HARNESS_LIVE=1 bash /root/Page_dmh/openwam_harness/sequential/start_adapter.sh
```

原 Page UI 仍用 `page_ui` 的 8765。顺序页是 8766。

## 回归

协议回归不连接 LIBERO，也不占用 8766。环境、Planner 和分割都是替身。没有浏览器，所以不能把结果说成按钮点击通过。

```bash
cd /root/Page_dmh/openwam_harness/sequential
/root/Page_dmh/openwam_harness/envs/rpent/bin/python -m unittest \
  test_protocol_regression.py test_retreat_geometry.py \
  test_lifecycle.py test_run_life.py test_start_contract.py test_load_contract.py \
  test_free_task.py test_contract.py test_stagec.py test_binding_chain.py
```

`test_protocol_regression.py` 走适配器的 `/ws`：加载、刷新、开始门闩、Planner 离线后恢复、规划失败后重新加载、双击开始、低分分割后停止和确认超时、执行异常后生产者退出再加载、旧 `ended`、重连。报告写到 `openwam_harness/sequential/records/protocol_regression_report.json`。`test_retreat_geometry.py` 用回合 `2b2a7e4b1d0d` 的坐标，不读被忽略的 `runs/` 目录。

加载结束时会再发一次快照。这次 `loading` 已经是 false，所以不刷新也能看到真实的 `can_start`。刷新仍然会收到当前快照和最后一帧。生产者还没退出时，重复停止的回执是 `already_stopping`。确认超时写的根因是「确认超时，已结束等待」，不会被进入等待时的低分原因盖住。

## Windows 转发

容器里的 `SSH_CONNECTION` 是 `127.0.0.1`，公网 SSH 端口不在环境变量中。当前实例区域是 `bj-B2`。仓库里原 Page UI 记录的例子来自另一台 `bjb1`：

```powershell
ssh -N -L 8766:127.0.0.1:8766 -p 14263 root@connect.bjb1.seetacloud.com
```

请用 AutoDL 控制台里这一台 `bj-B2` 机器的 SSH 命令，只把本地转发改成上面这一行的 `-L`。原 Page UI 把 `8766` 换成 `8765`。浏览器打开 `http://127.0.0.1:8766`。

## 套件

页面按「套件 → 任务 → seed」从 LIBERO 注册表读取，只列四个标准套件：

| 页面 | suite | 任务数 | 初始状态 |
|---|---|---|---|
| Spatial | `libero_spatial` | 10 | 每任务 `get_task_init_states` |
| Object | `libero_object` | 10 | 同上 |
| Goal | `libero_goal` | 10 | 同上 |
| Long | `libero_10` | 10 | 同上，t0/t1 是长任务候选 |

`libero_90` 和 `*_task` / `*_swap` / `*_lan` 不会出现在这四个名字下面。任务原文来自 `task.language`。中文摘要是对照表，原文为准。

同套件、同任务、同 seed 的再次加载走环境 `reset`，VLA 和 SAM3 保持进程。切换任务会停掉 env 进程并新建，同时增加 `scene_epoch`。加载进行中的第二次点击会被拒绝。

## 图像

- Pi0.5 的输入仍是环境观测里的 256 相机，这条预处理没有改。
- `agentview_high.png` / `wrist_high.png` 是 1024 方图，给分割和反投影用。
- 页面传输前把展示图最长边缩到 512，JPEG quality 70，宽高比保持原帧。
- Planner 请求标记为 `planning_input=scene_metadata`，`vision=false`。只带图像 id，不把这次规划叫做视觉规划。若要附图像，最长边不超过 384。

## 双卡示例

这台机器当前只有一张 RTX 5090。下面的命令是配置示例，本轮没有执行，也没有改正在跑的 vLLM。不要默认 `--tensor-parallel-size 2`。

```bash
# GPU0 只跑 Planner。先不要在这张卡上加载 Pi0.5 / SAM3。
CUDA_VISIBLE_DEVICES=0 /root/Page_dmh/Show-Harness/.venv-vllm/bin/vllm serve \
  /root/Page_dmh/openwam_harness/checkpoints/Qwen3.5-9B \
  --served-model-name Qwen3.5-9B --host 127.0.0.1 --port 8000 \
  --dtype bfloat16 --max-model-len 32768 --max-num-seqs 1 \
  --gpu-memory-utilization 0.90 --cpu-offload-gb 0 --enforce-eager \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --limit-mm-per-prompt '{"image":4}' --trust-remote-code

# GPU1 跑 Pi0.5 和 SAM3。顺序页启动前设置，不要和上面抢 GPU0。
SEQUENTIAL_CUDA_DEVICE=1 HARNESS_LIVE=1 \
  bash /root/Page_dmh/openwam_harness/sequential/start_adapter.sh
```

32768 继续保留。65536 大约要再留一倍 KV cache。9B bf16 权重大约 18GB；单卡 32GB 在 offload、并且还要同卡放 VLA 时放不下 65536。放不下就退回 32768。不要把 `--enforce-eager` 关掉后直接当成主链路；图模式要单独验证兼容性和回退。

当前服务仍是 bfloat16、无量化、`--max-model-len 32768`、`--gpu-memory-utilization 0.48`、`--cpu-offload-gb 6`、`--enforce-eager`、vLLM 0.24.0。日志写明 `Total CPU offloaded parameters: 6.05` GiB，权重加载占 11.63 GiB，可用 KV cache 只有 1.33 GiB。进程占用约 14470 MiB。这是长解码慢的主要原因。本轮没有改启动参数，也没有做 offload=0 的持续解码对比，所以没有新的加速数字。pid 以 `ss` 或页头为准，不要沿用旧数字。

## 页面状态

`/health` 只返回适配器状态和最多 1.5 秒的 Planner 探测缓存。它不启动模型、不 reset、不做推理。超时记为 `planner=timeout`，失败记为 `down`。套件从 `/suites` 读取；注册表还在加载时页面显示「套件加载中」，并有「重试检查」。主界面只有套件、加载场景、中文任务、边缘/云端模型、开始、停止、双相机和一行状态。任务和 seed 没有下拉框。seed 固定为 0，并写在加载日志里。加载后单独显示当前中文目标，不改写用户已经输入的文字。确认按钮只在等待确认时出现。Planner、性能、绑定和重置在折叠详情里。没有复用计划和人工计划入口。Planner 不可用时「开始」禁用，也不会改成干跑或已保存计划。

## 手动测试

1. 打开 `http://127.0.0.1:8766` 并刷新。页头应在几秒内变成「适配器在线」，并带上当前 pid、planner、env、vla。若停在「检查超时」，点「重试检查」。
2. 选套件后点「加载场景」。页面上没有任务和 seed。日志里有 `seed fixed=0`。加载后状态行上方显示当前中文目标，任务输入框保持用户原文。
3. 选边缘或云端模型，用中文写下任务，点「开始」。每次开始都重新请求 Planner。没有人工计划，也没有复用已保存计划。
4. 若中文对应多条任务，页面只问一个问题，不默认第一条。对不上当前套件时提示不支持，机器人不动。
5. 两个黑碗不能靠名称区分。详情里的绑定会写明证据来源。未唯一绑定时要指定编号，不会默认第一只。
6. 确认按钮平时不出现。只有执行停在等待确认时才出现。重置在「性能、绑定和重置」里。

## 日志

- 失败分类：`openwam_harness/sequential/records/failure_catalog.json`
- 原回合目录：`openwam_harness/runs/baseline_original/libero_10_t*_s0_r*/`
- 原回合包装日志：`openwam_harness/tmp/baseline_run.log`
- Planner 服务日志：`openwam_harness/tmp/planner/vllm.log`
- 短请求记录：`openwam_harness/sequential/records/planner_check.json`
- 真实顺序回合会写到 `openwam_harness/runs/sequential/<episode_id>/`，该目录不上传。

## 回退

```bash
git checkout feat/official-show-harness-sim
```

该分支停在 `fc9d499`。原模式命令仍是：

```bash
cd /root/Page_dmh/openwam_harness/RPent
/root/Page_dmh/openwam_harness/envs/rpent/bin/rpent \
  --robot libero --suite libero_10 --task 0 --seed 0 \
  --libero-type standard --planner api \
  --model openai:Qwen3.5-9B --base-url http://127.0.0.1:8000/v1
```

## 能力矩阵与对象绑定

矩阵在 `openwam_harness/sequential/records/capability_matrix.json`。四套件 40 条都来自本地注册表语言，并带上 BDDL `:objects` / `:fixtures`。`local_acceptance` 全部是 `not_run`。可展开计划不算已成功。

开放范围：Object 搬运、Spatial 抓放、Goal 里不含开合/推动/旋钮的搬运、Long 里两条食品放入篮子。接触任务在动作前返回拒绝，不会调用 `pi0_pick`。

Planner 收到带编号的场景图。同类型多实例都保留。SAM3 的 `segment` 经 `Sam3Facade._select_top` 只返回最高分；点提示 `segment(point=[row, col])` 存在，但同样只返回一张最高分掩码，没有多候选 RPC。最高分掩码在没有点提示或唯一实例证据时不绑定到实例，也不默认第一只。

中文输入匹配到英文规范指令。多条都像时只问一个问题。自由输入若对应另一个任务，会显式加载该任务并改用它的 `env.terminated`；对不上时提示当前场景不支持，不沿用原来的成功标准。

## VLA 异步

`robots/libero/tools.py` 的 `_vlm_chunk` 在环境线程上先 `model.predict`（内部是 `vla.predict` RPC），返回后才 `env.chunk_step`。这两步不重叠。画面线程只编码和发送 512 JPEG，不能算 VLA 异步。用户看到的两段停顿，一段是这次推理加 RPC，一段是同线程上的 512 渲染或工具切换。

执行保持同步。原先 `VLA_ASYNC=async` 会在 `chunk_step` 期间再发一次 `model.predict`，然后用 `apply=False` 丢掉，下一步仍同步预测。这次额外推理不再发送。页面详情写明「同步」。能消费并对齐动作结果之后，才重新打开异步实验。

reset、取消、重规划、技能切换和原生成功会清空队列，并丢掉旧 `episode_id` / `scene_epoch` / `plan_version` 的结果。环境线程是唯一的 `chunk_step` 调用者。checkpoint、归一化、采样和 `action_chunk` 没有改。

本轮没有做同步/异步各一次的实机诊断：`127.0.0.1:8000` 没有 Planner，8766 上仍是旧的 live 进程。CPU 队列回放在 `test_stage_d.py`。

## 本轮验证

记录在 `openwam_harness/sequential/records/planner_speed.json`。没有第三次 Planner 请求，也没有批量评测。

Planner：

- 纯文本：TTFT 0.267s，总时长 0.409s，prompt 17 / completion 2，`finish_reason=stop`，content `pong`，没有 reasoning。请求里带了 `enable_thinking: false`。
- 单图 256×256：TTFT 0.574s，总时长 0.718s，prompt 83 / completion 2，content `seen`。
- 一次真实语义计划：初始 `seq-d7a38087ddd3` prompt 1482 / completion 391；唯一一次修复 `seq-e653ae06b66d` prompt 2680 / completion 596。两次都是 `finish_reason=stop`，没有 reasoning，没有截断。总耗时 157.535s。模型把 `objects` 展平成顶层字符串，schema 失败，`executed=false`。共享卡上长输出大约 6–7 token/s。420 秒不是修复。
- 非流式请求在客户端超时后，服务端仍会把这条请求生成完。流式断开会按 `X-Request-Id` 只取消这一条；随后的短请求能返回。不停止 vLLM。

场景：

- 注册表四个标准套件各 10 个任务、50 个初始状态。Long t0/t1 仍是长任务候选。
- `libero_10` task 0 seed 0 冷加载：service_start 34.6s，env_create 12.233s，reset 2.394s，first_frame 2.808s，`scene_epoch=1`。画面 512，来源 1024。
- 同一任务热 reset：service_start 0，env_create 0，reset 2.338s，first_frame 2.766s，`scene_epoch=2`。VLA 没有重启。
- 加载进行中的第二次请求被拒绝，文案是「已有加载在进行，不会再创建环境。」

抓取：回合 `6838b71aaab1` 的服务端 `states.json`。第一次 `pi0_pick` 是 record_index 2，`chunks_used=12`，工具 `success=false`，`peak_lift_m=0.011`。页面上的 `env_step` 1→4 是记录序号，不是 12 个 chunk 只走了 3 个仿真步。LIBERO Pi0.5 配置是 `action_chunk=5`、`action_dim=7`。当时没有逐 chunk 的动作形状，所以不用别的快照回填动作数。之后的记录是恢复和未确认就继续的 move/place。抬升阈值仍是 0.05 m。这不算抓取通过。计数说明在 `sequential/records/grasp_units.json`。

扁平计划：已保存的旧输出在 CPU 上因 `objects` 等未知字段被拒绝。修好后只发了一次真实初始请求 `seq-a302f9505e25`，prompt 1150 / completion 168，`finish_reason=stop`，没有 reasoning，没有修复，没有执行。展开后的技能是观察、分割、`pi0_pick`、移动、夹爪和暂停。记录在 `sequential/records/planner_flat_check.json`。输入是场景元数据和图像 id，不是视觉规划。

CPU 测试通过只说明契约，不表示整条系统完成。没有批量评测，上下文仍是 32768。offload=0 的持续解码还没有做。

## 状态与快照

按钮只看服务端 `command_gates()`。前端再加一条连接状态：没连上时原因是「未连接」，还没有快照时是「正在同步」。连上之后，按钮旁只显示 `block_reason`。`load_ack` 被拒绝时不改写这次运行的 `root_error`。

`RunContext` 是运行相位的唯一业务状态。`loading` 只表示场景加载。成功加载会把上一轮 `public()` 放进 `history`，然后换一个新的 `RunContext`。历史不参与 `can_start`。

同一进程里 `state_revision` 每次发出控制消息加一。`server_instance_id` 在进程启动时生成。客户端丢掉更小的 revision、更旧的 `scene_epoch`，以及回合或 run 对不上的 `ended`。实例 id 变了就清空本地相位并重新要快照。

快照字段：`server_instance_id`、`state_revision`、`scene_epoch`、`run_id`、`phase`、`scene_loaded`、`producer_alive`、`planner.status`、`can_start`、`can_load`、`can_stop`、`block_reason`。

生产者在自己的 `finally` 里不能把 `is_alive()` 当成已经退出。`complete_from_producer` 等线程结束后才 `release`，然后发新快照。这之前 `block_reason` 是「正在停止」，加载和开始都不放行。不把 `thread` 提前设成 `None`。

停止、确认超时、异常和正常结束都走 `_close_run`。停止和确认各自有 `stop_ack` / `confirm_ack`。重复停止得到 `already_stopping`。空任务字符串表示用当前场景目标，不是禁用原因。

刷新时连接会收到当前快照和 `last_payload`。收到帧后 `scheduleDraw()` 会排进下一帧绘制，不等下一次机器人动作。

2026-10-19 审计时，8766 上的 live 进程是 pid 3195、commit `9c159a8`。当时快照是 `block_reason=""`、场景已加载、`producer_alive=false`、Planner `ok`。没有浏览器，不能把这次推算当成按钮点击通过。原始快照在 `openwam_harness/sequential/records/audit_pre_lifecycle.json`。代码上会禁用开始的条件是：正在加载、正在停止、等待确认、正在执行、场景未加载、回合已结束、模型未就绪。其中「正在停止」会在收尾快照里一直留着，直到生产者线程真正退出后又发一次快照。加载解析失败不再先把当前回合标成 `cancelled`。

## 坐标

三个世界坐标分开命名。`support_origin` 是目标表面物体原点。`object_origin` 是被放物体原点。`eef_origin` 是末端原点。`hold_offset = object_origin - eef_origin`，只在仍抓着物体的 `travel` / `descend` 里使用。释放之后的 `retreat` 不再减这个偏移。

落点在支持半径内做占用检查：先试中心，再试 0.04 m 和 `min(radius, 0.055)` 的八个方向，与已有物体保持 0.05 m。没有通过检查的点就返回 `support_occupied`，不会让两个物体共用一个未验证落点。物体目标的 z 是表面 z 加 `z_offset`（接近 0.12 m，下降 0.03 m）。末端目标是 `object_origin - hold_offset`。超出 `thresholds.json` 的 `workspace_m` 直接失败，不把目标夹进上限。

退离从释放时的真实 `eef_origin` 出发。候选是向上 0.04 / 0.08 / 0.12 m，以及远离刚释放物体的水平 0.04 / 0.08 m 和斜向。候选必须留在工作空间里、比释放点更远离该物体，并且不靠近其他物体 0.05 m。没有这样的路径时 code 是 `retreat_no_feasible_path`。

回合 `2b2a7e4b1d0d` 的 `retreat_step2` 旧公式是 `support + [0.04, 0] + 0.12 - hold_offset`。表面 `[0.0466, -0.0276, 0.9097]`，持物偏移 z 为 `-0.12636852116610497`，得到末端 z `1.156068521166105`，被工作空间 z 上限 1.1 拒绝。这组数在 `placement.RECORDED_RETREAT`。按释放末端 `[0.0835, -0.0633, 1.0495]` 向上 0.04 m 得到 z 约 1.0895，留在工作空间内，且不再减持物偏移。

放置、退离和最终复查分开记在 `place_records`、`retreat_records` 和 `final_check`。退离失败后如果 `sim_state()` 仍能只读，就做最终复查，`extra_step` 为 false。

## 成功判据

五个字段分开：

| 字段 | 含义 |
|---|---|
| `execution_status` | 计划执行是否走完。退离失败时是 `retreat_error` |
| `user_goal_status` | `met` / `unmet` / `unknown`。只读仿真状态，不看退离是否成功 |
| `native_task_success` | 原生 LIBERO 目标是否已经出现 |
| `retreat_error` | 退离自己的原因。空表示没有退离错误 |
| `evidence_source` | 用户目标证据从哪来。读不到状态时是 `unavailable` |

退离错误不把用户目标改成失败或成功。自由模式里用户目标 `met` 时相位是完成；`unmet` 时是未完成。原生目标已达成而用户组合还没完成，原因是「原生目标已达成，用户组合目标未完成」。

终止分层：Gym / RLinf 在 `ignore_terminations=false` 时原样返回 `terminated`，`truncated` 是时间上限。`LiberoEnvFacade` 原样转发。客户端 `LiberoEnvClient` 用 `|=` 闩住这两个标志，之后 `step` 会拒绝。工具层看到客户端 `terminated` 或 `truncated` 就停。自由模式不再把 `env.terminated = False`。`episode_service` 在 env server 进程里包装 facade：基准模式原样返回；自由模式把原生成功记成 `native_goal`，返回给客户端的 `terminated=false`，不 reset。`truncated` 仍然结束且不能继续。只读 RPC 是 `env.native_goal`。模式文件是 `sequential/records/episode_mode.json`，适配器启动时写成 `benchmark`，每次 `set_execution_mode` 再写。

本轮没有真实跑到「原生目标先达成、用户组合目标仍未完成」。这一条未验收。

`finalize_run` 是成功、失败、停止、确认超时、预算到时和异常的同一出口。它只读复查，不调用 `env.step`。读失败时五个字段仍然写出：`user_goal_status=unknown`，`evidence_source=unavailable`，并带 `unknown_reason`。`completed_subgoals` 只表示步骤曾经 `PLACED`，`subgoal_results` 才是最终条件。`first_error` 和 `verification_error` 分开保留。finalize 自己出错也不挡住运行所有权释放。

## 自由模式字段流

原句直接进入 Planner，不经过名称正则或任务目录门槛。Planner 只提交 `object_n` / `target_n`。`resolve_plan_bindings` 用同一次 `scene_graph` 派生实例，不另建映射。

每条语义步骤记下 `candidate_id`、`instance_ref`、`canonical_name`、`grounding_query`、`vla_prompt`、`evidence_source`。候选表和计划同时记下 `scene_epoch`、`observation_id`、`candidate_set_version`。位姿带 `sampled_at`，状态是 `not_sampled` 直到当次查询，不当成永久坐标。

绑定缺编号、编号和实例不一致、类型名冒充实例、候选版本过期，或设施没有唯一 BDDL 放置区域时，返回契约错误，动作不会开始。

页面「绑定对象」读 `plan_bindings`。自由模式进 Planner 前的 `binding.object_ref` 仍是空的，不能用来显示未绑定。

## 状态查询与错误

`query_bound_state` 按已绑定的 `instance_ref` 读当前观测。可动物体的查询键就是实例名，对应观测里的 `{instance}_pos`。设施目标的查询键是场景里该设施唯一的操作 site，键为 `owner_instance + "_" + region_name`。`put_the_bowl_on_the_stove` 的炉灶 site 是 `flat_stove_1_cook_region`。同一份 BDDL 里酒架的操作 site 是 `wine_rack_1_top_region`，它不在这个任务的 `:goal` 里，但自由模式仍然注册。不截后缀，也不用设施 body 原点。裸 `top_region` 不能当全局键：柜子和酒架各有一个。

只读 RPC 是 `env.named_poses`。位姿成功时证据是 `sim_state` 或 `mujoco_site`。场景清单里有名字不算已经有位姿。

错误码分开：

| 码 | 含义 | 去向 |
|---|---|---|
| `instance_not_found` | 实例不在当前场景，或拿到的是类型名 | 失败，不请求抓取确认 |
| `pose_not_provided` | 实例在，但没有位姿或放置区域坐标 | 失败。状态可读时最多再读一次 |
| `stale_observation` / `STALE_OBSERVATION` | `scene_epoch` 与计划不一致，属于另一个场景 | 失败，不把另一场景的快照读进来冒充 |
| `stale_snapshot` | 同一场景、同一 `tool_record` 时钟，记录序号落后 | 失败前最多再读一次当前动态状态 |
| `state_service_error` | 状态服务报错或缺字段 | 失败。最多再读一次 |
| `visual_instance_ambiguous` | 定位无法区分多个同类实例 | `awaiting_clarification`，列出候选 |

抓取证据仍走原来的持有验证和有限采样。点提示沿用现有的一次重试，不降低 SAM3 阈值，也不默认第一只。

第一项动作之后，对象、目标和空间关系仍按原有 `invalidate` 规则失效，下一步重新 `observe`。

搬运到释放沿用同一步骤链，阶段名为 `localized`、`grasp_verified`、`transporting`、`preplace_ready`、`descending`、`releasing`、`placement_verified`、`retreat`。放置前只检查仍持有、目标区域、落点和下降姿态；通过后才下降并松爪。炉灶区域证据是 `mujoco_site`，碗的位姿仍是 `sim_state`，整项结果用 `per_item`，不把 site 写成全部证据。`move_to` 的调用返回不算末端已经到达。内部缺字段返回 `pose_not_provided` 或 `state_service_error`，不进入人工确认。

`flat_stove_1_cook_region` 在 `flat_stove.xml` 里是半尺寸 `0.075 0.075 0.0025` 的 box。site 原点是区域参考。对象原点的高度是区域高度加一次 `z_offset`，持物偏移只从对象原点换算到末端一次。

## 最小回归

```bash
cd /root/Page_dmh/openwam_harness/sequential
/root/Page_dmh/openwam_harness/envs/rpent/bin/python -m unittest test_regions.py test_binding_chain.py test_transfer_release.py test_lifecycle.py
```

`test_binding_chain.py` 回放 `fixtures/planner_seq_68c884edeb80.json` 里的 `seq-68c884edeb80`，不请求模型，不执行动作。`test_regions.py` 回放 `seq-8a435cf576eb` 的酒架计划参数，并检查 Reset 与观测时钟。

## 场景区域

安装包里的两份 Goal BDDL 都声明了酒架，不是仓库里的副本：

- `liberopro/bddl_files/libero_goal/put_the_bowl_on_the_stove.bddl`
- `liberopro/bddl_files/libero_goal/put_the_wine_bottle_on_the_rack.bddl`

碗任务的 `:goal` 是 `(On akita_black_bowl_1 flat_stove_1_cook_region)`。酒瓶任务的 `:goal` 是 `(On wine_bottle_1 wine_rack_1_top_region)`。两份的 `:regions` 都有 `top_region` → `wooden_cabinet_1` 和 `top_region` → `wine_rack_1`。运行时 site 名是 `wooden_cabinet_1_top_region` 与 `wine_rack_1_top_region`。

`main_table_wine_rack_region` 来自 `(wine_rack_region (:target main_table) (:ranges ...))`，只在 `:init` 里把酒架放到桌面。它不是酒瓶落点。`bowl_drainer_1` 的 `left_region` / `right_region` 的 owner 不在这份 `:fixtures` 里，记为 `owner_not_in_scene`，不进入放置表。

先前 `target_region_missing` 的第一处丢失在 `suites._place_regions`：它只保留出现在 `:goal` 或 `:obj_of_interest` 里的 site。碗任务因此没有 `wine_rack_1_top_region`。`seq-8a435cf576eb` 和随后的格式修复 `seq-b2ee7edc020a` 都选了合法编号 8（`wine_rack_1`），错误出在注册表，不是 Planner 选错对象。现在放置区域来自当前问题里全部设施 site。`target_region_missing` 和 `target_region_ambiguous` 不再消耗格式修复。柜子有多个区域时仍是 `target_region_ambiguous`。

酒架模型 `libero/libero/assets/turbosquid_objects/wine_rack/wine_rack.xml` 的 site：

```xml
<site type="box" pos="0 -0.08003 0.21745" quat="0.8549119 -0.5187733 0 0" size="0.10087 0.02200 0.00251" name="top_region"/>
```

MuJoCo 四元数顺序是 wxyz。这个局部四元数不是单位四元数，表示绕 site 的 X 轴旋转。`size` 是半尺寸。`env.named_poses` 仍用 `sites[name] = [x, y, z]`。朝向在 `site_geometry[name]`：`quat_wxyz`、row-major `mat`（`site_xmat`）、`size`。世界位姿每次从当前 MuJoCo 状态读取。

官方 On 是 `SiteObjectState.check_ontop` → `SiteObject.under`：`delta = site_xmat @ (object_pos - site_pos)`，要求 `size[2] - 0.005 < delta_z < size[2] + 0.10` 且 `abs(delta_xy) < size[:2]`。有 site 几何时放置复查用这个判据。没有 mat/size 时不把中心距离说成 On。

炉灶 `cook_region` 的四元数是单位四元数，现有 `move_to` 平移可以生成末端目标。酒架 site 带旋转。`move_to` 只伺服位置并保持朝向。`RPent` 的 `move_pose` 只同时伺服 xyz、pitch 和 yaw，顺序执行没有把它接上完成检查和取消。因此旋转区域在动作前返回 `orientation_not_commandable`，不调用未接通的姿态命令。几何参数由后端从当前状态生成，Planner 不输出落点或旋转角。

刚性区域的世界位姿按 `scene_epoch` 和设施 body 位置缓存。设施被挪动、场景版本变化，或区域名含 drawer/door/knob/handle/joint 时，不沿用缓存。动态观测可以刷新位姿，不能删掉注册表里的区域定义。

观测时钟分开：`scene_epoch` 是场景版本，`episode_id` 是回合，`tool_record` 是工具记录序号，`env_action_count` 是环境动作计数。只有同一 `tool_record` 时钟才比较序号。Planner 等待期间墙钟变了，但环境没变，不因此把对象或静态区域判失效。动作前仍读当前动态状态。可刷新的落后快照最多再读一次。跨 `scene_epoch` 直接拒绝。

主栏「重置场景」发送 `reset`。执行中先停掉动作所有者和在途动作，退出后才 `warm_reset`。它重置同一 suite/task/seed，不重启 VLA、SAM3 或环境进程。`scene_epoch` 和 `episode_id` 更新，`plan_version` 增加，旧计划、定位、确认和持有基准作废。历史和输入框保留。新状态和首帧之后才回到 idle。失败时 `reset_failed`，开始保持禁用。重复点击在重置进行中返回 `already_resetting`。

页面手动检查：打开 `http://127.0.0.1:8766/`，套件选「目标任务」，加载默认 task 1、seed 0。空闲时点「重置场景」。执行中点它会先停止再重置。输入仍留在任务框。

2026-10-19 只读和单任务结果，live commit `2be2f87`，适配器 pid `28636`，页面 `view12`：

- Goal task 1、seed 0 只读没有动作。`env_step` 前后都是 0。碗 `[-0.0944, -0.0018, 0.8984]`，酒瓶 `[-0.1887, -0.0495, 0.8989]`，都来自 `sim_state`。
- `flat_stove_1_cook_region` 世界位置约 `[-0.2650, 0.2091, 0.905]`，四元数 `[1, 0, 0, 0]`，半尺寸 `[0.075, 0.075, 0.0025]`，放置模式 `translation`。
- `wine_rack_1_top_region` 世界位置约 `[-0.2605, -0.1846, 1.1374]`，世界四元数 `[0, 0, -0.518773, 0.854912]`（wxyz）。这和模型局部四元数 `0.8549119 -0.5187733 0 0` 不是同一组数，因为酒架初始化还有 yaw。半尺寸 `[0.10087, 0.022, 0.00251]`。放置模式 `orientation_not_commandable`。
- `main_table_wine_rack_region` 返回 `region_not_placeable`。柜子 `wooden_cabinet_1_top_region` 仍单独存在，没有覆盖酒架。
- 主栏「重置场景」把回合从 epoch 1 换成 epoch 2、episode `4f8e992b89eb`。开始保持可用，停止保持禁用。
- 一次自由模式「把碗放到炉灶上」：Planner `seq-5d9f1c256515`，`repairs=0`，`plan_from=initial`。约 66 秒结束，`execution_status=completed`，`native_task_success=true`，`user_goal_status=met`。当时用户目标记录仍是中心距离 `xy_m=0.0008`。事后只读同一场景，`SiteObject.under` 为真，`delta_site` 约 `[0.0003, 0.0007, 0.0239]`。之后的复查代码改为有 site 几何时用这个判据。
- 酒瓶没有执行。控制器不能按酒架的世界朝向放置。缺的是完整朝向的末端命令，以及它的完成检查和取消接线。
- 碗先到炉灶之后不 reset、继续酒瓶，并复查两个目标：这条组合没有跑，不能当成通过。

记录：`records/readonly_regions_goal_t1.json`、`records/live_bowl_stove_once.json`。这些运行记录不进入提交。
