# LIBERO 顺序系统

当前页面是 `openwam_harness/sequential` 的顺序页，地址 `http://127.0.0.1:8766`。它执行 LIBERO 标准单任务。Planner 使用场景元数据，请求标记为 `planning_input=scene_metadata`、`vision=false`，不接收图像像素，不是视觉自主规划。

`HARNESS_LIVE=1` 才会在加载场景时创建 LIBERO、Pi0.5 和 SAM3。权重或环境缺失时启动入口直接失败，不会改成 DryBackend。

## 版本

| 位置 | 提交 | 说明 |
| --- | --- | --- |
| 本仓库 `main` | `885e64a` | 比 `origin/main` 的 `df7e083` 超前 1 个提交 |
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

成功要分开看：工具是否执行、抓取信号、技能是否走完、LIBERO `terminated` 且未 `truncated`、用户目标几何、退离、人工确认。`grasp_signal` 或单独的 `terminated` 误读不是成功。

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
cd /home/host/page-dmh/openwam_harness/sequential
/home/host/page-dmh/openwam_harness/envs/rpent/bin/python -m unittest \
  test_log_budget test_lifecycle test_run_life test_start_contract test_load_contract \
  test_binding_chain test_state_contract test_protocol_regression
```

这些测试不占用 8766，也不加载 Pi0.5。本阶段验收记录在 `docs/phase1-acceptance.md`。
