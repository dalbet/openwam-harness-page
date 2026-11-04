# 原 page-ui 集成审计（2026-11-03）

本轮把已验收的三种方案接进**主工程原有的 `page_ui`**，用独立配置与独立端口运行，形成可操作的演示版本。
不新增模型、不做组合任务、不改生产服务；生产端口 `14510 / 14520 / 8001 / 14600 / 14700 / 14800 / 14900`
的进程 PID 与启动时间在本轮前后一致（`raw/acceptance/production_ports_20261103_1210.json`）。

## 1. 结论

| 项目 | 结论 | 证据 |
| --- | --- | --- |
| 原 page-ui 集成 | 通过 | `page_ui/server.py` 直接对外服务；`/health` 返回 `page_ui`；浏览器与协议两条路径都跑通真实回合 |
| 连续实时画面 | 通过 | 每控制步 `_set_frames`，回合内帧龄中位数 35–69 ms；8 任务 × 2 方案 16 回合逐回合记录 |
| 三层自然语言链路 | 通过 | 中文原文 → Agent 语义路由（:14571）→ 引擎（:14821）→ 推理路由（:14621）→ 生成后端（:14551/:14552）→ Env（:14591）；页面显示"理解/标准任务" |
| 单任务支持 | 通过 | 10 个 LIBERO-Goal 任务全部可加载；8 个任务 × S1/S2 共 16 回合 16/16 成功，另有 task1/task5 复验 4/4 |
| 干净启停 | 通过 | `page_ui/demo_start.sh` / `demo_stop.sh` 与 `tools/demo_stack.sh`，均带台账与归属校验 |
| 回滚 | 按设计"不动生产" | 演示栈只用自己的端口与配置目录；生产进程未被重启，无需回滚动作 |

## 2. 接线：页面到仿真

```
浏览器 14901  --WS-->  page_ui/server.py
                        |-- HTTP 14571  agent_service（语义路由 + whole_task 计划）
                        |-- HTTP 14821  engine_service（场景/回合/验证/记录）
                        |-- HTTP 14621  inference_service（动作分块 + 指标）
                        |-- HTTP 14591  libero_env_server（仿真步进）
                        `-- 二进制帧：header{fid,phase,term,fts,age_s,sim_action_count} + agentview JPEG + wrist JPEG
```

- 页面**不调用任何 GPU 脚本**。方案切换只发 `{"type":"set_arm","arm":"S2"}`，由后端执行
  `tools/arm_switch.sh`，再用引擎**实时 identity**（`action_model.backend` / `server_url` / `precision`）
  反查，匹配才把 `verified` 置真；不匹配时页面明确显示"未生效（实际 …）"。
- 帧的刷新点在引擎的 `_set_frames()`：每个控制步写 `model_obs`/`raw_obs` 两路画面，带单调 `frame_seq`
  与 `frame_ts`。页面对同一个 `(eid, seq)` 只取一次；回合结束时额外重发一帧同 fid、`term=true` 的帧
  用于终止横幅，因此"严格递增"检查在 2 个回合里为假，"非降 + 仅重复 1 帧"才是真实语义。
- 回合开始/重置/换场景都会清空帧缓存，页面在首帧到达前显示"等待首帧…"，不会把上一回合画面当作本轮。

## 3. 迁回主工程的最小改动

演示引擎与推理路由现在**直接运行主工程源码**（`/home/host/page-dmh/embodied_engine`），
而不是实验目录里的副本；两者用 `EMBODIED_CONFIG_DIR` / `EMBODIED_RECORDS_DIR` 指向隔离配置与记录目录，
不设置这两个变量时行为与本轮之前完全一致（生产启动路径不变）。

| 文件 | 改动 | 为什么必须改 |
| --- | --- | --- |
| `adapters/pi05_contract.py`（新增） | LIBERO 7 维动作 ↔ π0.5 观测/动作契约 | 主工程原先只有 OpenWAM 的 10 维 EEF 契约 |
| `engine/pi05_rollout.py`（新增） | whole_task 回合循环（观测→生成→执行→验证→释放），复用既有验证/释放/记录骨架 | 主工程没有 π0.5 的整任务执行器 |
| `models/pi05_chunk_backend.py`（新增） | 生成服务 `/infer` 的**统一客户端**（openpi 与 FlashRT 同一请求形状、同一噪声种子） | 保证 S1/S2 只差生成后端，不靠更便宜的客户端"赢" |
| `engine/engine_service.py` | 按 `action_model.backend` 选择执行器；π0.5 也要求计划就是 1 个 whole_task | 否则 π0.5 会走短技能分支或接受不匹配的计划 |
| `engine/openwam_rollout.py` | 抽出 `_raw_gripper()` / `_to_action7()` 两个接缝 | 夹爪在 10 维里是 slot 9、7 维里是 slot 6；写死 slot 9 会把 OpenWAM 假设漏进 π0.5 |
| `inference/inference_service.py` | `ACTION_BACKEND` / `PI05_GEN_URL` 环境优先级；`pi05` 走 chunk 客户端；identity 带 `serving_arm` / `precision` | 切换只重启路由（秒级），权重常驻；identity 是页面校验的唯一依据 |
| `agent/agent_service.py` | `whole_task` 能力由 `action_model.<backend>.whole_task` 声明，不再写死 openwam | 否则 π0.5 的技能目录里没有 whole_task，计划形状退回短技能 |
| `common/config.py`、`common/task_catalog.py` | 新增两个**可选**环境变量覆盖配置/记录目录 | 让主工程源码能以隔离配置跑演示，且不污染生产 `records/` |
| `page_ui/server.py` | 方案表/切换、帧元数据、指标、`ended.phase`、启动时同步实际方案、并发 load 拒绝 | 原页面没有方案选择、没有实测后端校验、没有并发保护 |
| `page_ui/static/*` | 方案选择、实时指标（规划/首动作等待/生成/推理累计/任务耗时/帧序号）、停止显示"已停止"、陈旧帧提示 | 区分"完成/已停止/失败"，避免把旧帧当实时画面 |

风险与边界：

- 生产 `action_model.backend` 仍是 `openwam`，上述 π0.5 路径不会被生产进程走到；`models/pi05_backend.py`
  （对接 :14510 的旧协议）保留未删，路由的 `pi05` 后端改为 chunk 协议。
- 未改 `tools/demo_console.py`（上一轮实验台）。它仍在 :14831 运行，但**不是**本轮交付物；
  无台账记录导致归属校验拒绝自动停止，故保留并在验收里注明。

## 4. 方案矩阵

| 方案 | 后端 | 生成服务 | 精度 | 说明 |
| --- | --- | --- | --- | --- |
| S0 | openwam | `:14541` OpenWAM 候选 C | bf16 | 与两个 π0.5 服务不同卡常驻，切换时先释放 |
| S1 | pi05 | `:14552` openpi eager | fp32 权重 + bf16 选定参数 | 原始后端 |
| S2 | pi05 | `:14551` FlashRT | bf16 | 同一检查点，CUDA Graph + PromptRuntime |

页面"实际后端"以引擎 identity 为准；16 回合与 4 回合复验中，每条记录的 `action_server_url`
都与所选方案一致（S1→:14552，S2→:14551）。

## 5. 测试证据

### 5.1 8 任务 × 2 方案（16 回合，`raw/records/pagesmoke-04.jsonl`）

任务 0/2/3/4/6/7/8/9，各 1 个固定初始化，S1 与 S2 各一次：**16/16 成功**。

| 方案 | 回合 | 任务耗时中位数 | 生成 median | 生成 p95 中位数 | 规划中位数 | 首动作等待中位数 | 帧龄中位数 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| S1 | 8 | 16.03 s | 60.8 ms | 61.6 ms | 0.75 s | 0.162 s | 69 ms |
| S2 | 8 | 15.71 s | 34.2 ms | 40.4 ms | 0.754 s | 0.141 s | 45 ms |

帧：每回合 34–111 帧，去重后 33–111 帧；task 7/8 的 S2 各多 1 帧，是终止帧重发（同 fid）。

### 5.2 迁移后复验（4 回合，`raw/records/pagesmoke-07.jsonl`）

引擎与路由改为主工程源码后，task 1 / task 5 × S1 / S2：**4/4 成功**，
`action_server_url` 分别为 :14552 / :14551，生成 median 60.7 ms / 33.1 ms。

### 5.3 页面规则（`raw/records/page_checks.json`）

| 检查 | 结果 |
| --- | --- |
| 运行中再次 `start` | 拒绝：`启动失败 — an episode is already running` |
| 运行中加载另一场景 | 拒绝：`拒绝加载 — 正在running…` |
| 两个客户端同时加载 | 拒绝：`拒绝加载 — 正在loading…`（单客户端消息串行，本就不会并发） |
| 10 任务之外的自然语言 | 拒绝：`the instruction is outside the ten supported capabilities` |
| 运行中停止 | `phase=stopped`，清空 chunk，`in_flight_actions=0`；随后 reset 回 `ready` |
| 停止后再执行 | 成功（109 个动作，14.75 s，45 帧 / 45 张不同画面） |
| 加载期间切换方案 | 串行化：等加载结束再切换，完成后 identity 校验通过（S2→S1 真实生效） |

### 5.4 浏览器真实操作

浏览器内点过：方案选 S2 → 切换成功（`已校验`、`:14551`、`bf16`）；加载 task 1 → 中文"把碗放到炉子上"
→ 成功（15.02 s，114 动作，生成 median 32 ms）；加载 task 5 → "把盘子推到炉子前面" → 成功（16.01 s，
118 动作，生成 median 34 ms，p95 77 ms）；运行中点"停止"→ `phase=stopped`。
**限制**：最后一次重启页面后，浏览器（远端）已无法访问本机端口（连生产 :14900 也失败），
因此"已停止"文案与"等待新帧"提示改用页面 WebSocket 协议验证，未再用点击复核。

## 6. 时延表（既有 run record，未重跑）

见 `raw/acceptance/three_arm_timing.md`（S0 8 回合 / S1 28 回合 / S2 30 回合）：

| 方案 | server median | p95 | 首动作等待 | 规划 | 推理累计 | 任务耗时 |
| --- | --- | --- | --- | --- | --- | --- |
| S0 | 222.82 ms | 224.51 ms | 0.40 s | 0.01 s | 4.67 s | 36.38 s |
| S1 | 61.31 ms | 61.70 ms | 0.17 s | 0.70 s | 1.78 s | 16.86 s |
| S2 | 34.16 ms | 86.69 ms | 0.13 s | 0.69 s | 1.17 s | 16.18 s |

页面内实测与上表一致；S2 的 p95 在个别回合抬到 77–86 ms，属生成后端尾部抖动，不据此下加速比结论。

## 7. 运行手册

启动（干净状态，按顺序）：

```bash
cd /home/host/flashrt-exp/pi05_round
bash tools/demo_stack.sh start env             # :14591 libero env server
bash tools/demo_stack.sh start inference S2    # :14621 router -> :14551
bash tools/demo_stack.sh start engine          # :14821 project engine (config_demo)
cd /home/host/page-dmh/embodied_engine
PAGE_PORT=14901 PAGE_ARMS=/home/host/page-dmh/embodied_engine/config/arms.json \
  bash page_ui/demo_start.sh                   # :14901 the real page_ui
```

方案切换：页面下拉框选 S1/S2/S0 → "切换方案"；等价命令 `bash tools/arm_switch.sh S2`
（台账校验；失败会打印 `RESULT {"status":"error",...}`）。

停止：

```bash
cd /home/host/page-dmh/embodied_engine && bash page_ui/demo_stop.sh
cd /home/host/flashrt-exp/pi05_round && bash tools/demo_stack.sh stop engine
```

回滚：演示栈不写生产配置、不占生产端口；按上面的 stop 顺序停掉本栈服务即可，
生产 `14510/14520/8001/14600/14700/14800/14900` 自始至终未被本轮启停脚本调用。

## 8. 已知限制

1. `load` 与 `run` 在页面内串行；并发请求会被明确拒绝，同一客户端的重复点击只会排队。
2. 停止后若首动作未测得（回合在首动作前结束），页面显示"—"而不是 0 ms。
3. 浏览器侧可达性依赖 IDE 端口转发；页面重启后该转发可能失效，需要在本地重新打开端口。
4. :14831 是上一轮实验台，仍在运行且无台账，归属校验拒绝自动停止；不属于本轮交付。
5. `docs/audits` 此前文件保持原样；本轮只新增本篇与 `raw/acceptance/page_ui_integration_acceptance.json`。
