# LIBERO 顺序系统

当前页面是 `openwam_harness/sequential` 的顺序页，地址 `http://127.0.0.1:8766`。它执行 LIBERO 标准单任务。Planner 使用场景元数据，请求标记为 `planning_input=scene_metadata`、`vision=false`，不接收图像像素，不是视觉自主规划。

`HARNESS_LIVE=1` 才会在加载场景时创建 LIBERO、Pi0.5 和 SAM3。权重或环境缺失时启动入口直接失败，不会改成 DryBackend。

## 版本

| 位置 | 提交 | 说明 |
| --- | --- | --- |
| 本仓库 `main` | `a9c559e` | PHASE2A 运行时修复；比 `origin/main` 的 `df7e083` 超前 3 个提交（文档提交随后 +1） |
| `openwam_harness/RPent` | `a8c5311` | detached，editable 安装，`import rpent` 指向这份源码 |
| `openwam_harness/OpenWAM` | `4d7f642` | detached，当前执行链不调用它，源码保留 |

远程：本仓库 `https://github.com/dalbet/openwam-harness-page.git`，RPent `https://github.com/RLinf/RPent.git`，OpenWAM `https://github.com/OpenWAM-Official/OpenWAM.git`。

## 目录

| 路径 | 职责 |
| --- | --- |
| `scripts/` | 启动 Planner、启动真实 adapter、查看状态、停止本项目服务 |
| `config/libero.env.example` | 非敏感配置模板。本机覆盖写到 git 忽略的 `config/libero.env` |
| `docs/harness-sequential.md` | 当前架构和操作说明 |
| `runtime/` | PID 和服务日志，可再生成，不提交 |
| `openwam_harness/sequential/` | 顺序页、适配器、计划校验和本地执行 |
| `openwam_harness/baseline/start_planner.sh` | vLLM 启动参数的唯一实现 |
| `openwam_harness/RPent/` | Pi0.5、SAM3、LIBERO 环境 RPC |
| `openwam_harness/OpenWAM/` | 后续源码，不在本阶段执行链 |
| `openwam_harness/checkpoints/` | Qwen3.5-9B、Pi0.5、SAM3 |
| `openwam_harness/envs/vllm`、`envs/rpent` | 两个虚拟环境 |
| `openwam_harness/runs/sequential/` | 回合输出。正常模式只保留最近 3 个回合目录，每个目录保留最近 2 步观测文件 |

## 模型

| 模型 | 路径 | 来源 |
| --- | --- | --- |
| Qwen3.5-9B | `openwam_harness/checkpoints/Qwen3.5-9B` | Hugging Face revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a` |
| Pi0.5 | `openwam_harness/checkpoints/RLinf-Pi05-LIBERO-130-fullshot-SFT` | `RLinf/RLinf-Pi05-LIBERO-130-fullshot-SFT` revision `6222623f635769bfc73c9472e29fab9b7fd8e027`。下载时跳过 `optimizer.pt` |
| SAM3 | `openwam_harness/checkpoints/sam3/sam3.pt` | `facebook/sam3` 的 `sam3.pt` |

恢复时在对应目录重新执行上述 Hugging Face 下载。不要把权重提交到 Git。

Planner 在 GPU 0，LIBERO、Pi0.5 和 SAM3 在 `SEQUENTIAL_CUDA_DEVICE`（默认 1）。解释器分别是 `openwam_harness/envs/vllm/bin/python` 和 `openwam_harness/envs/rpent/bin/python`。

## 启动、状态、停止

可以从任意工作目录调用。重复启动不会再起一份服务。端口被其他程序占用时退出并说明，不会结束那个进程。

```bash
bash /home/host/page-dmh/scripts/start_planner.sh
bash /home/host/page-dmh/scripts/start_adapter.sh
bash /home/host/page-dmh/scripts/status.sh
bash /home/host/page-dmh/scripts/stop.sh
```

三种状态不要混在一起：

- 进程已启动：PID 已记录，命令行身份匹配。
- 服务 ready：Planner 的 `/v1/models` 返回 `Qwen3.5-9B`，或 adapter 的 `/health` 返回 `adapter=online`。
- 模型加载完成：Planner ready 时权重已经在显存里。Pi0.5、SAM3 和 LIBERO 要等页面加载场景，adapter 启动本身不加载它们。

停止时先按 PID 文件里的 `starttime` 核对身份，再发送 SIGTERM，超时后只对仍匹配的进程发送 SIGKILL。adapter 会连同它拉起的 env、Pi0.5 和 SAM3 子进程一起退出。

## 页面

布局、按钮和双相机协议保持原样。页面提供套件、加载场景、任务输入、边缘/云端模型、开始、停止、重置和双相机。确认控件只在执行停在等待确认时由现有脚本处理。帧仍是二进制双相机消息，页面只画最新帧。

代表任务以注册表为准：`libero_goal` task 1，语言是 `put the bowl on the stove`，BDDL 是 `put_the_bowl_on_the_stove.bddl`，目标原子是 `On(akita_black_bowl_1, flat_stove_1_cook_region)`。

成功要分开看：工具是否执行、抓取信号、技能是否走完、LIBERO `terminated` 且未 `truncated`、用户目标几何、释放、退离、人工确认。`grasp_signal` 或单独的 `terminated` 误读不是成功。

## 完成条件

原生目标、执行完成和用户目标不是同一件事：

- LIBERO 的 `On` 是几何/接触判定（`ObjectState.check_ontop`、`SiteObject.under`），不要求松手。抓着碗进入炉灶区域也可能 `terminated=true`。
- 自由模式下原生目标只发布为 `NATIVE_GOAL_FINISH` 事件（`native_goal=true`、`terminated=false`、`episode_continuable=true`），保留首次原生成功。benchmark 模式仍按 `terminated` 结束，`truncated` 始终终止。
- 原生成立后只允许当前 transfer 剩余的 `descending`、`releasing`、`retreat` 阶段继续。预算取同一子任务剩余阶段的既有预算（本任务为 place 30s + retreat 30s，共 60s），并记录 `native_finish_scope`、`native_finish_budget_s`、`native_finish_start_sim_action`。不运行其他步骤或其他事务；stop、reset、取消、真实终止立即生效。
- 用户目标要求释放后证据：`PLACED` 收据带 `released_required`，夹爪开度不低于 `gripper_open_min_m=0.05`，释放后观测有 `sim_action_count`，最终观测的 `sim_action_count` 不小于释放后观测，观测 ID 不同，最终夹爪仍打开，并且最终关系仍成立。抓着碗的 `On` 只是 `held_at_end` 几何，不算放置完成。
- 关系成立但缺少释放证据记为 `unknown` 并列出缺失项；关系不成立记为 `unmet`。用户目标 `met` 但退离失败时保留 `user_goal_status=met` 和 `retreat_error`，同时 `finalization=user_goal_met_retreat_error`，执行不记为完成。
- 时间、动作数和观测 ID 取 env server 的 `sim_action_count` 与观测编号，不用日志行号、帧号或工具步数代替；只读查询不推进仿真。

## 已知限制

- `On` 的原生判定不含释放，原生成功只证明关系可达，不能单独证明放置完成。
- 自由模式的“原生后有限收尾”是本顺序系统自己的契约，不是 LIBERO 或 RPent 的通用行为；benchmark 仍保持原生终止语义。
- 收尾窗口沿用现有 `place`/`retreat` 预算（任务 1 为 60s），只覆盖当前 transfer；任务 1 之外的目标尚未做三回合回归。
- 成功结束消息的内联 `ended` 仍不带 `release`、`retreat_complete`、`native_finish_window` 字段；释放证据在 `subgoal_results[*].release_evidence`，收尾窗口在事件流和执行结果里。失败/取消路径已带这些字段。
- adapter `/health` 的 `commit` 是进程启动时的 HEAD；运行中提交代码后 `disk_head` 会变化，`running_commit_matches_disk_head=false` 不能单独当作运行代码不同的证据。

## 日志

只使用 `openwam_harness/sequential/log_budget.py` 这一套上限：

- 服务标准输出：2 MiB，另留 1 个备份，写在 `runtime/logs/`。
- 回合流水 `records/run_journal.jsonl`：1 MiB，另留 1 个备份。
- 回合摘要：最近 8 份，写在 `records/runs/`。
- 回合目录：最近 3 个。清理只在 `adapter.py` 作为服务进程启动时打开，单元测试导入模块不会删共享的 `runs/` 或 `records/runs/`。
- 观测图像、深度和 world 文件：每个回合保留最近 2 步。`states.json` 保留。
- `SEQUENTIAL_DEBUG=1` 时才持续保存展示 JPEG，并把观测步上限放到 20。

正常模式记录启动和退出、模型版本、`run_id`、`scene_epoch`、规划、技能、失败原因和终态。不持续保存每帧图像、完整观测或每个动作的大数组。

## 回归

```bash
cd /home/host/page-dmh
/home/host/page-dmh/openwam_harness/envs/rpent/bin/python -m unittest discover \
  -s openwam_harness/sequential -p 'test_*.py'
```

这些测试不占用 8766，也不加载 Pi0.5。PHASE1 的原生可达记录在 `docs/phase1-acceptance.md`；PHASE2A 的收尾修复与复验在 `docs/phase2a-acceptance.md`。
