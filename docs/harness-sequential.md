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
