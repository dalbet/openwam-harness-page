# 十任务验收摘要（LIBERO-Goal，seed 0，每项 3 回合）

> 阶段说明：§1–§10 是 **Pi0.5 基线**的 30 回合验收；§11 是本阶段新增的 **OpenWAM 模型层接入**验收
> （独立代码状态、独立数据）。两节的数字不可混用。
> §12–§14 是速度测量、d1 对照与 60 回合 A/B 的历史记录；§15 起记录 2026-10-27 的
> **d1 下线、十任务目录统一、Qwen 语义路由、20 回合语言闭环**，其中 d1 已不再运行。
> §23–§30 记录 **同场景两步组合 `0 → 8`**：路由与执行机制通过，组合的物理目标未同时成立
> （`combo_verified` 0/6），失败位置与证据见 §23.5–§23.6。

本文件只记录已经跑出来的数字和失败原因。所有结果来自 `records/acceptance/final_3x_rehome.json`
及其引用的 30 份回合记录（归档在 `records/acceptance/final_3x_rehome_runs/`），可用

```bash
python3 scripts/summarize_acceptance.py records/acceptance/final_3x_rehome.json
```

复现表格。原生目标（LIBERO `goal`）和用户目标（本引擎的完成证据）分开记录，
不以原生 `On` 代替用户目标，也不在失败回合上做人工确认。

本次验收对应的代码状态：`embodied_engine/` 的首次提交（本文件随该提交写入，见
`git log --oneline -- embodied_engine`）；两次 30 回合分别对应提交前的"顺序重试"版本与
提交后的"重试前归位"版本，差别只有 §5.1 描述的归位机制。

## 1. 结论

| 项目 | 结果 |
| --- | --- |
| 最终验收 | **26 / 30 回合成功（86.7%）**，同一代码版本、seed 0、每任务 3 回合 |
| 成功回合耗时 | 中位 **27.5 s**，均值 32.4 s，最短 23.9 s |
| 全部回合耗时 | 中位 28.2 s，均值 38.3 s，最长 111.7 s（task 3 失败回合） |
| 一次通过（无恢复） | 20 / 30 回合；10 个回合用到恢复，共 14 次恢复调用 |
| 规划延迟 | 中位 **6.47 s**（每回合 1 次规划），最大 9.58 s |
| 动作块空档 | 30 回合合计 **0.01 s**（预取覆盖了块间等待） |
| 推断隐藏比例 | 推断总时长 377.4 s，其中约 **38%** 与仿真动作执行重叠，其余以排队/等待形式暴露 |
| 未通过 | task 3 ×1（放置进抽屉）、task 5 ×2（推盘子）、task 6 ×1（原生目标已达成但用户证据不足） |

`vLLM` 只服务语言规划模型；Pi0.5 由 RPent 常驻服务推理。两者都不是同一层的加速关系，
报告中的"推断"一律指 Pi0.5 动作块推理，不指 vLLM。

## 2. 运行配置（本次验收固定）

| 项 | 值 |
| --- | --- |
| 工程 | `embodied_engine/`（旧顺序链路未改动，端口不重叠） |
| Agent 模型 | Qwen3.5-9B，vLLM 0.24.0，`127.0.0.1:8001`，`temperature=0`，GPU3 |
| 动作模型 | Pi0.5 LIBERO-130 SFT，revision `6222623f635769bfc73c9472e29fab9b7fd8e027`，RPent `pi05_vla_server`，`14510` |
| 推理模式 | `async_chunk_prefetch`，队列 1，`max_chunk_age_actions=5`，动作块 5 |
| 环境 | LIBERO-Goal，seed 0，`env_server` `14561`，每回合同一次 `reset` |
| 引擎/Agent/UI | `14800` / `14700` / `14900` |
| 世界模型 | OpenWAM-Alpha-Sim-LIBERO：**BLOCKED，不在执行链**（见 `architecture.md` §2） |

> 验收跑完之后只改过一处非行为代码：`engine_service.py` 里 OpenWAM 的 `reason` 字符串
> （原来写的是"没有 LIBERO 动作权重"，与实测不符），改为"缺 Wan2.2-TI2V-5B 主干与依赖、
> 从未跑过动作推断"。它只影响 `/health` 与 UI 的说明文字，不涉及动作路径、调度或判定，
> 因此本节表格对应的仍是实际跑出这些数字的那份代码。

## 3. 逐任务结果（最终验收，每项 3 回合）

| task | 指令 | 成功 | 耗时（s） | 仿真动作 | 恢复次数 | 失败码 |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | open the middle drawer of the cabinet | 3/3 | 26.4 / 28.5 / 27.9 | 151 / 168 / 162 | 0 / 0 / 0 | — |
| 1 | put the bowl on the stove | 3/3 | 27.0 / 27.1 / 26.4 | 145 / 146 / 136 | 0 / 0 / 0 | — |
| 2 | put the wine bottle on top of the cabinet | 3/3 | 23.9 / 26.8 / 24.7 | 123 / 137 / 121 | 0 / 0 / 0 | — |
| 3 | open the top drawer and put the bowl inside | **2/3** | 111.7 / 84.6 / 77.4 | 687 / 554 / 504 | 2 / 1 / 1 | `chunk_budget_exhausted` |
| 4 | put the bowl on top of the cabinet | 3/3 | 29.5 / 26.5 / 24.9 | 168 / 142 / 131 | 0 / 0 / 0 | — |
| 5 | push the plate to the front of the stove | **1/3** | 60.4 / 34.4 / 60.5 | 320 / 157 / 352 | 2 / 1 / 2 | `chunk_budget_exhausted` ×2 |
| 6 | put the cream cheese in the bowl | **2/3** | 75.1 / 32.1 / 26.9 | 365 / 184 / 145 | 2 / 0 / 0 | `insufficient_stable_samples` |
| 7 | turn on the stove | 3/3 | 29.9 / 31.9 / 27.8 | 104 / 110 / 103 | 1 / 1 / 1 | — |
| 8 | put the bowl on the plate | 3/3 | 25.4 / 25.8 / 24.6 | 136 / 137 / 130 | 0 / 0 / 0 | — |
| 9 | put the wine bottle on the rack | 3/3 | 35.1 / 32.9 / 33.6 | 218 / 197 / 206 | 0 / 0 / 0 | — |

task 3 是"开上层抽屉再放碗"的标准组合任务，全部 3 回合都先成功开抽屉（qpos < -0.14），
失败只发生在"把碗放进抽屉"这一步。

## 4. 失败归类与根因

### 4.1 task 6 run 1：原生目标达成，用户证据不足（`insufficient_stable_samples`）

这是唯一一次"LIBERO 原生 goal = true，用户目标 = false"的回合，也是最值得修的判定缺陷。

- 释放前：`xy_m = 0.0248`（阈值 0.025），`object_bottom_z = 0.9236`（碗沿 0.939）→ 关系成立，允许释放；
- 释放命令在 `sim_action_count=346` 生效，夹爪开度 0.0796 → 释放事件本身通过；
- 释放后 3 个新样本：`xy_m = 0.0338 > 0.025`，三次都不满足，`stable=false`。

即：奶油奶酪块落进碗里后偏向碗沿 3.4 cm，仍被 LIBERO 判为在碗内，但本引擎的"质心距碗心
≤ 2.5 cm"条件不通过。**本次没有为了让这条通过而放宽阈值**；正确做法是让判定改用碗的实际
内径几何（贴图资产半径），而不是写死 2.5 cm。列为下一阶段第一优先修复项。

### 4.2 task 5（推盘子）2 次失败

`push_object` 预算 20 块（100 个动作），意图是"把盘子推到炉灶前沿 `main_table_stove_front_region`"。
两次失败都是 3 次尝试（100 + 100 + 120 动作）后仍未达到 0.03 m 位移，恢复决策是 replan；
第 3 次尝试前的归位动作实际执行了 20 / 26 步（见 §5）。第一次尝试就没有产生足够位移，
说明 Pi0.5 对该目标的推动方向/接触点本身不稳定，不是判定问题（同一代码在上一轮 30 回合里
task 5 是 3/3，见 §5 的方差说明）。

### 4.3 task 3 失败回合（run 1）

- `open_drawer` 12 块成功（75 个动作，qpos -0.157）；
- `pick_place` 连续 3 次各 40 块失败，共 200 + 200 + 200 个动作；
- 结束时碗位于 `y ≈ +0.03`，而抽屉开口 `y ∈ [-0.227, -0.076]`，即碗被举到柜体上方/后侧，
  始终没有进入抽屉口；最后两次尝试因为夹爪仍夹着碗（`start_opening_m = 0.002`）而跳过归位。

这是策略层面的空间失败（把碗举过柜顶），不是预算不够；同任务另外两回合的放置分别用
21 块和更少块数一次完成。

### 4.4 task 7（开炉灶）每次都多花一次恢复

3/3 成功，但 3 个回合都是同一个模式：第 1 次 `turn_on` 用满 16 块（80 个动作）后
`chunk_budget_exhausted`，恢复决策 replan，第 2 次从按钮位置继续才把 `qpos` 推到 ≥ 0.5。
这不是失败，但它固定浪费一次 Agent 决策（约 5–6 s）和一段重复动作；由于按钮关节在推进，
本可以按"关节有进展就扩展预算"处理（`_extend_budget` 目前只覆盖 `open_drawer` 和
`pick_place`）。列入下一阶段的第一项低成本优化。

### 4.5 上一轮 30 回合（改动前）的失败

`records/acceptance/final_3x_1155.json`：task 0 一次 `recovery_abort`（两次尝试关节 qpos 始终 0.0，
第二次尝试开始时机械臂已经贴在柜面上）、task 3 两次失败（1/3）、task 6 一次
`chunk_budget_exhausted`。两次 30 回合的失败集合不同，见 §5。

## 5. 局部优化与同条件回归

### 5.1 优化：重试前回到回合一初始 EEF 位姿（`skills.py::_goto_home`）

证据来自改动前的 task 0 失败回合（`ep-20261026-115605-94677b.json`，该文件后来被
`records/runs` 的 60 份上限清理，以下数值是诊断时读取的原始值，汇总仍保留在
`records/acceptance/final_3x_1155.json` 的 task 0 行：72.8 s / 450 动作 / 2 次恢复 / `recovery_abort`）：
第 2 次尝试的第一个动作就已经在 `eef=[0.046, -0.106, 1.017]`（贴着抽屉面板），整段 50 块
`qpos` 没有离开 0.0；而所有成功回合都是从 `eef≈[-0.215, 0.01, 1.17]`（回合一初始位姿）出发，
在第 131–177 个动作才把抽屉拉开。也就是说**重试继承了失败位姿**，而不是重新接近。

实现（`engine/skills.py` + `engine/engine_service.py`）：

- `engine_service` 对每个 `(skill, object_n, target_n, relation)` 计次，第 2 次及以后把 `attempt` 传进步骤；
- `SkillRunner.run()` 在 `attempt > 1` 时先执行 `_goto_home()`：同一 OSC 笛卡尔契约，
  每步 ≤ 0.06 m，最多 26 步，容差 2 cm，夹爪保持张开；
- 夹爪开度 < 0.05 m（可能还夹着物体）时**不归位**，避免把物体拖走；归位记录写入
  `rehome`（`applied/actions/start_eef/end_eef/reason`）并落盘；
- 判定门槛、样本数、预算都没有改动，归位只改变重试的起始位姿。

### 5.2 同条件回归

同一 seed、同一模型、同一任务表，改动前后各 30 回合：

| task | 改动前（`final_3x_1155`） | 改动后（`final_3x_rehome`） |
| --- | --- | --- |
| 0 | 2/3（34.2 s 中位） | **3/3**（27.9 s） |
| 1 | 3/3（26.1 s） | 3/3（27.0 s） |
| 2 | 3/3（21.8 s） | 3/3（24.7 s） |
| 3 | 1/3（105.0 s） | **2/3**（84.6 s） |
| 4 | 3/3（36.8 s） | 3/3（26.5 s） |
| 5 | 3/3（34.5 s） | **1/3**（60.4 s） |
| 6 | 2/3（27.4 s） | 2/3（32.1 s） |
| 7 | 3/3（35.2 s） | 3/3（29.9 s） |
| 8 | 3/3（25.0 s） | 3/3（25.4 s） |
| 9 | 3/3（32.6 s） | 3/3（33.6 s） |
| **合计** | **26/30** | **26/30** |

必须如实说明：

- 总成功数**没有变化**（26/30 → 26/30）。task 0 从 2/3 到 3/3、task 3 从 1/3 到 2/3，
  但 task 5 从 3/3 掉到 1/3；
- 失败回合的位姿证据只在 task 3 run 1 和 task 5 的两次失败里出现归位记录：task 3 那次因为
  夹爪夹着碗被跳过，task 5 的两次实际执行了 20 / 26 步归位后仍然失败；
- 因此**不能声称"归位提高了成功率"**。能确认的是：①机制按设计生效且不破坏正常路径
  （20 个一次通过回合全部没有归位动作）；②两次 30 回合的差异主要来自 Pi0.5 采样方差——
  task 0 / 3 / 5 在两轮之间分别是 2↔3、1↔2、3↔1，单任务 3 回合不足以区分改进与噪声；
- 要把单任务成功率的分辨率做到 ±10%，需要每任务 ≥ 20 回合，或固定住策略采样噪声（当前
  Pi0.5 每块 flow-matching 采样不固定随机种子）。

## 6. 耗时分解（30 回合合计 1149.7 s 墙钟）

| 阶段 | 合计（s） | 占墙钟 | 说明 |
| --- | --- | --- | --- |
| 规划（Qwen3.5-9B，1 次/回合） | 205.5 | 17.9% | 中位 6.47 s，最大 9.58 s |
| 技能执行（含推断、仿真步进、快照） | 731.8 | 63.6% | 与下面的推断/仿真项重叠，不重复相加 |
| ↳ 推断（Pi0.5） | 377.4 | 32.8% | 每块 0.354 s 均值 |
| ↳ 其中与动作执行重叠 | ~145 | ~38% of 推断 | 由块内预取产生，块间空档合计 0.01 s |
| ↳ 仿真步进 | 425.8 | 37.0% | 每动作 0.067 s（≈14.9 Hz） |
| ↳ 只读快照 | 55.6 | 4.8% | 每动作一次 `state_snapshot` |
| ↳ 块内决策等待（暴露的推断） | 232.6 | 20.2% | 队列等待 + 预取残余等待 |
| 恢复（Agent 决策） | 90.5 | 7.9% | 14 次调用，平均 6.5 s/次 |
| 释放动作 | 50.7 | 4.4% | 张开夹爪 + 上抬退离 |
| 完成判定 | 50.0 | 4.4% | 3 帧稳定样本 |
| 其他（加载/reset/结束） | ~15 | ~1% | 环境准备与收尾 |

关键路径（有重叠，不能把上面各项直接相加成总时长）：

```
墙钟 ≈ 环境准备 + 规划(Qwen) + Σ技能执行 + 恢复(Agent) + 释放 + 完成判定
技能执行 = max(推断, 仿真步进) + 暴露的排队等待 + 快照
```

为什么推断只能隐藏 38%：

- 一个动作块只有 5 个动作 × 0.067 s = **0.335 s** 的执行时间；
- 一块推断 **0.354 s**，比整块执行时间还长；
- 预取在第 3 个动作（index 2）时发出，只剩 2 个动作 ≈ 0.13 s 可以覆盖，剩下的只能等；
- 实测 655/696 块走了预取路径、19 次在抓取/释放/技能结束边界被丢弃、0 次因过期重算，
  说明预取本身工作正常，**瓶颈是"块长 5 < 一次推断能覆盖的动作数"**，而不是队列或 HTTP。

若要继续压缩：把预取提前到块内第 2 个动作（index 1，消耗时块龄 4 ≤ 上限 5），理论可把覆盖
提高到 ~0.27 s/块；或者改用更长的动作块。两者都会改变动作契约，留待下一阶段做同条件对比，
本阶段未改动。

## 7. 生命周期与 UI 验证

### 7.1 生命周期（`scripts/test_lifecycle.py`，实测输出）

```
[1] load task 1
[2] start
    phase while running: running episode=ep-20261026-125531-07b95d
[3] stop
[ok] stopped: stopped
    stopped result: execution=stopped failure=user_stop
[4] reset
    clean phase=ready scene_epoch=3
[5] run again after reset
[ok] second episode terminal: ...
lifecycle: PASS
```

- stop 之后旧回合结果 `execution_status=stopped`、`failure_code=user_stop`，没有保持 running；
- reset 后 `scene_epoch` 从旧回合推进到 3，phase=ready；
- reset 后的新回合（`ep-20261026-125542-f4cf86`）独立跑完并成功：63.6 s、380 动作、
  `success=true`；它比常规 task 1 慢，是因为 reset 时机械臂停在半空（上一条被 stop），
  抓取需要多试一次——这是环境状态，不是旧回合结果泄漏（token/epoch 全部换新）。

### 7.2 浏览器（真实页面 `http://192.168.5.11:14900/`）

页面版本：由 `git show fc9d499:page_ui/` 恢复（该目录在 `7864980` 被删除），
`server.py` 换成到引擎 14800 的桥接，`static/index.html`、`static/app.js` 保持原布局与
双相机控件。用浏览器自动化实际点击，读取页面 DOM 文本与画布尺寸：

| 操作 | 结果 |
| --- | --- |
| 打开页面 | 10 个任务下拉、加载场景/开始/重新开始/停止/重置 五个控件齐全，双相机画布各 256×256 |
| 选 task 0 + 加载场景 | 任务框回填 `open the middle drawer of the cabinet`，开始/重置可用 |
| 开始 | `Phase: running`，`running — step 1/1 open_drawer -> wooden_cabinet_1_middle_region`，两路画布 256×256 实时刷新 |
| 停止 | `Phase: stopped`，`stopped — episode stopped`；引擎侧 `execution_status=stopped`、`failure_code=user_stop`、8.78 s |
| 重置 | 回到 `Phase: ready`，`scene_epoch=3`，开始按钮恢复可用 |

页面上的身份信息全部来自后端 `/api/status` 的 `identity`，不是写死字符串：

```
Agent: Qwen3.5-9B
Action: Pi0.5 LIBERO-130 SFT (pi05)
World: OpenWAM-Alpha-Sim-LIBERO [not_on_execution_path]
Inference: async_chunk_prefetch / pi05
```

`World` 明确标 `not_on_execution_path`（OpenWAM 未接入执行链时不冒充已验证）。

### 7.3 证据保留

- 引擎原来只保留最新 60 份回合记录（`common/records.py::MAX_RUNS`），跑完一轮 30 回合的验收
  就会把上一轮的原始记录挤掉。已改为 200，并在 `scripts/run_suite.py` 结束时把本轮引用的
  记录归档到 `records/acceptance/<suite>_runs/`，行的 `archived_record` 指向归档副本，
  `summarize_acceptance.py` 优先读归档（本文件引用的 30 份记录即
  `records/acceptance/final_3x_rehome_runs/`）。
- 改动前那一轮（`final_3x_1155`）的原始记录有一部分在修复前已被清理，本文件只引用它的
  汇总数值；表格里 `inf_s/env_s` 为 0 的行就是记录已被清理的行，不代表 0 推断。

## 8. 与旧顺序链路的对照

旧链路记录（`openwam_harness/sequential/records/runs/*.json`）没有落盘总时长，但保留了事件
时间线，可以精确比较同一模型、同一任务的规划延迟：

| 旧记录 | 指令 | 规划延迟 | 旧链路结果 | 新链路同一任务 |
| --- | --- | --- | --- | --- |
| `8395a0b22492-5` | 把碗放在炉灶上 | **51.7 s** | 用户目标完成（含确认流程） | task 1：3/3，中位 27.0 s，规划 6.9 s |
| `27a055aef009-7` | open the middle drawer | **48.2 s** | `no_joint_progress`（1 块后停） | task 0：3/3，中位 27.9 s，规划 5.6 s |
| `c921e9392f25-9` | open the top drawer and put the bowl inside | **128.4 s（失败）** | `PLANNING_FAILED`（重复自然语言列表） | task 3：2/3，规划一次通过、结构化校验 |
| `5a15a64f7072-6` / `b6e9302e9783-5` | put the bowl on the stove | — | `confirm_timeout`（90 s 确认超时） | 无人工确认环节，完全由完成证据判定 |

可确认的差异：

1. **规划延迟 48–52 s → 5.6–9.5 s**（中位 6.5 s；短指令 5.3 s，组合任务 9.5 s），旧链路 task 3 直接
   `PLANNING_FAILED`，新链路改用结构化 JSON 契约（技能 + 对象绑定 + 关系），task 3 的
   "开抽屉 + 放碗"两步组合一次校验通过；
2. **块间等待**：旧链路是顺序执行（拿一块、跑完、再请求），新链路块间空档合计 0.01 s；
3. **确认等待**：旧链路靠 90 s 人工/超时确认（`confirm_timeout`），新链路用释放事件 +
   3 帧新样本验证，task 6 run 1 正是这条差异暴露的边界（严格但可解释）；
4. **旧链路 task 0 的 `no_joint_progress`** 在新链路没有复现：最终验收的 6 次抽屉开启
   （task 0 ×3、task 3 ×3）全部一次成功，没有一次靠重试；跨两轮 30 回合的 12 次抽屉开启
   中 10 次成功，两次失败都是策略没接触到把手（qpos 恒为 0.0），不是"1 块后停"的判定问题；
5. 旧链路的 `stable_samples`、`confirm_timeout` 问题在新链路改成了显式失败码
   （`insufficient_stable_samples`），成功/失败路径字段一致。

不能声称的部分：旧链路没有同条件 30 回合数据，因此"总耗时下降多少"只能给单项（规划），
不能给整回合的对比倍数。

## 9. 尚未解决的断点

| 优先级 | 断点 | 现状 | 下一步 |
| --- | --- | --- | --- |
| P0 | task 6 用户判定 vs 原生目标（奶油奶酪 3.4 cm 偏心） | `insufficient_stable_samples`，判定用写死的 `xy_limit_m=0.025` | 用碗资产的实际内径计算允许偏移，同时保留"释放后 3 帧"要求 |
| P0 | task 5（推盘子）方差 | 同一代码两次回归 3/3 与 1/3 | 每任务扩到 ≥20 回合量化；检查推动接触点是否受初始位姿影响 |
| P1 | task 3 放置（举过柜顶） | 2/3，失败回合 3 次放置都未进抽屉 | 记录失败回合的碗/抽屉相对位姿，考虑放置前的水平重定位触发条件（不得降门槛） |
| P1 | task 7 每次都要 1 次恢复 | 6/6 回合 3/3，但每次首个 `turn_on` 都在 16 块（80 动作）处 `chunk_budget_exhausted`，第 2 次从按钮处继续才成功 | 让 `turn_on` 也走"按钮关节有进展就 +10 块"的扩展（与 `open_drawer` 同一机制），预计省掉 3 次恢复 / 约 10 s/回合 |
| P2 | 推断只重叠 38% | 块长 5 动作 ≈ 0.335 s < 单块推断 0.354 s | 试 index 1 预取或更长动作块，同条件对比 |
| — | OpenWAM | 已解除：自包含检查点、独立环境、真实推理与十任务验收 | 见本文件 §11 与 `architecture.md` §2.2；剩余限制见 §11.8 |
| P3 | 旧链路同条件对照 | 只有单项（规划）可比较 | 若需要整回合对照，旧链路需补计时埋点 |

## 10. 产物

| 文件 | 内容 |
| --- | --- |
| `records/acceptance/final_3x_rehome.json` | 最终验收 30 回合汇总（本文件的表格来源） |
| `records/acceptance/final_3x_rehome_runs/` | 上述 30 回合的完整回合记录归档（防止被运行时清理） |
| `records/acceptance/final_3x_1155.json` | 改动前同条件 30 回合（对照；部分原始记录已被清理） |
| `records/runs/ep-*.json` | 最近 200 回合的完整记录：计划、技能块、动作轨迹、观测、时间分解、失败证据 |
| `records/logs/final_3x_rehome.log` | 验收过程日志 |
| `records/run_journal.jsonl` | 运行期事件（预取丢弃、预算扩展、归位、释放等） |
| `docs/architecture.md` | 三层组件、接口、来源、启动/状态/停止命令 |

## 11. OpenWAM 模型层接入验收（2026-10-26）

本节是**独立阶段**的验收：把 `OpenWAM-Alpha-Sim-LIBERO` 真正接到 `embodied_engine` 模型层，
与前面的 Pi0.5 30 回合验收不是同一份代码状态。上面的数字仍只属于 Pi0.5。

### 11.1 结论

| 项目 | 结果 |
| --- | --- |
| 资产缺失判断 | **不需要下载** `Wan2.2-TI2V-5B`：检查点自包含（12.4068B 参数 / 24.814 GB 单文件），配置内嵌组件规格 |
| 独立环境 | `.venv-openwam`（无 sudo）：`torch 2.7.1+cu128`、`transformers 5.20.2`、`diffusers 0.41.0`，sm_120 实测可用 |
| 真实加载 | GPU2 常驻，加载约 110 s，显存 **24,778 MiB**；`strict=True` 载入无缺失/多余 key |
| 接入方式 | `action_model.backend=openwam`，WebSocket 常驻服务 `14520`，整任务 `whole_task` 一步计划 |
| 动作契约 | 10 维 native EEF10 → 7 维 LIBERO OSC，与官方实现**逐函数数值对拍**通过 |
| Task 1 连续 3 回合 | **3 / 3 成功**（29.8 / 31.0 / 31.8 s，84 / 86 / 87 动作） |
| LIBERO-Goal 十任务单轮 | **10 / 10 成功**，0 次恢复；耗时中位 33.3 s，动作中位 91.5 |
| stop/reset 隔离 | `scripts/test_lifecycle.py` PASS：停止后 0 个在途动作、模型侧缓冲清零、新场景 epoch |
| 与 Pi0.5 共存 | 同一进程内双向切换：`pi05` task 1 成功（26.46 s / 144 动作），再切回 `openwam` 成功（28.90 s / 83 动作）；切换后引擎身份与动作维度随动更新，无缓存串味 |

### 11.2 资产核实（为什么没有下载主干）

| 检查 | 事实 |
| --- | --- |
| `config.yaml` 的 `model.video_backbone.components` | 存在 `dit` / `vae` / `text_encoder` 三个组件的 `class_path` 与构造参数 |
| 部署加载路径 | 日志 `Using config-embedded component specs for video-backbone construction`，空壳实例化 + 从 safetensors 载入 |
| `model_path` 的作用 | 仅 tokenizer 兜底；tokenizer 已在本目录 `tokenizer/google/umt5-xxl/` |
| 权重分布（safetensors 头部） | DiT 9.997 GB + UMT5 11.363 GB + ActionDiT 2.042 GB + VAE 1.409 GB + proprio 0.001 GB = 24.814 GB |
| 参数总量 | 12.4068B（DiT 5.0B + T5 5.68B + Action 1.02B + VAE 0.705B） |
| 加载校验 | `load_checkpoint(strict=True)` 成功即 `missing=[] / unexpected=[]` |

结论：外部 Wan 目录缺失**不是**阻断项；本阶段唯一真正缺的是运行依赖与隔离环境，已补齐。

### 11.3 契约对拍（`tests/test_openwam_contract.py`）

| 项 | 样本 | 结果 |
| --- | --- | --- |
| `quat → rot6d` / `rot6d → quat` / `rot6d → axis-angle` | 200 组随机旋转 | 与 `benchmarks/utils/action_conversion.py` `allclose` |
| 原始 proprio → EEF10 | 50 组随机 pos/quat/gripper | 与 `libero_obs_to_eef10` 一致 |
| native EEF10 → 7 维 OSC | 200 组随机动作 | 与 `openwam2libero_interface.native_eef10_to_libero7d` 一致，且落在 `[-1,1]` |
| 图像槽位 | 真实随机 256×256 双相机 | 旋转 180° + LANCZOS：head 320×256、wrist 160×128，与 `client.resize_for_lshape_slot` 逐像素相等 |

### 11.4 十任务单轮结果（seed 0，每任务 1 回合，不重试）

| task | 指令 | 结果 | 耗时（s） | 仿真动作 | 首次推断（s） | 稳态推断中位（ms） | goal 处夹爪开度（m） |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | open the middle drawer of the cabinet | ✅ | 43.9 | 129 | 0.524 | 29 | 0.077（张开） |
| 1 | put the bowl on the stove | ✅ | 32.7 | 86 | 0.490 | 31 | 0.006 |
| 2 | put the wine bottle on top of the cabinet | ✅ | 31.4 | 82 | 0.525 | 30 | 0.031 |
| 3 | open the top drawer and put the bowl inside | ✅ | 54.4 | 170 | 0.521 | 29 | 0.008 |
| 4 | put the bowl on top of the cabinet | ✅ | 33.0 | 88 | 0.532 | 31 | 0.012 |
| 5 | push the plate to the front of the stove | ✅ | 47.1 | 146 | 0.528 | 30 | 0.078（张开） |
| 6 | put the cream cheese in the bowl | ✅ | 33.6 | 95 | 0.468 | 29 | 0.053～0.067（张开） |
| 7 | turn on the stove | ✅ | 28.3 | 77 | 0.522 | 29 | 0.028 |
| 8 | put the bowl on the plate | ✅ | 29.2 | 81 | 0.464 | 31 | 0.015 |
| 9 | put the wine bottle on the rack | ✅ | 42.6 | 126 | 0.525 | 30 | 0.041 |

汇总（`python3 scripts/summarize_openwam.py records/acceptance/openwam_libero_goal_all10.json`）：

- 成功 **10/10**，原生目标与用户目标一致（`final_verification.method=native_libero_goal`，3 个新样本全真）；
- 规划延迟中位 **5.72 s**（每回合 1 次）；
- 首次推断中位 0.52 s（服务端首块生成），稳态 **29–31 ms**（服务端缓冲区出队）；
- 单回合推断累计 3.7–7.8 s，仿真步进累计 5.4–12.0 s，回合总时长 28.3–54.4 s；
- 峰值显存 **24,778 MiB**（GPU2 独占，未影响 GPU0/1/3 上已有进程）；
- 恢复次数 **0**。

**必须说明的口径**：官方 LIBERO goal 不要求"松手"。这 10 回合里有 3 回合在 goal 处测得
夹爪张开（抽屉/推盘子/把奶酪放进碗），其余 7 回合到达 goal 时夹爪仍是闭合的
（目标原子 `On/In` 成立，物体已接触目标面，但策略没有张开手指）。
因此本阶段的 `user_goal_success` 以官方 goal 为准；若额外要求"释放证据"，
只能算 **3/10**。报告不把闭合夹爪写成一次释放，也不补 Pi0.5 式的 release 收据。

> **口径补丁（2026-10-27）**：上表 7 个"夹爪闭合到达 goal"的回合，按当前规则应记为
> **释放证据 unknown（未验证）**，而不是"已放下并释放"；`native_task_success=True` 保留，
> 历史结果不改写。引擎现已把这条口径写进每条记录：`release_evidence.state` 取
> `observed / unknown / not_applicable`，`applicable` 只对 `pick_place` 与 `composite`
> 为真，开抽屉、开炉灶、推盘子不会被要求释放。详见 §19。

### 11.5 生命周期与隔离

`scripts/test_lifecycle.py`（stop 在第 11 个动作后发起）：

| 检查 | 观测值 |
| --- | --- |
| `sim_at_request` / `sim_at_ack` | 11 / 11，`in_flight_actions=0` |
| 停止后 3 s 的动作数 | 冻结在 11（没有新动作下发） |
| 模型侧缓冲 | `backend_resets +1`（OpenWAM 返回 `reset_ack`） |
| 停止回合的结果 | `execution_status=stopped`、`failure_code=user_stop`，不报成功 |
| reset | `scene_epoch 8 → 9`，`last_result`/`episode_id` 清空 |
| reset 后重跑 | task 1 成功（新场景，无旧动作） |

### 11.6 一次真实故障与修复（保留记录）

切换后端后的第一次回归（`openwam_switchback_task1.json` 的第一版）失败了：
`inference_failed，ValueError: OpenWAM proprio must be 10-D EEF10, got 8`，0 动作。

- **根因**：引擎在启动时缓存了 `config/engine.json`。`set_backend.sh` 只重启推理服务，
  引擎仍以为后端是 `pi05`，于是给 Agent 下发 Pi0.5 的技能目录，规划出 `pick_place`，
  再把它交给 OpenWAM 后端 —— 8 维 proprio 进了 10 维契约。
- **修复**（两处，都是真问题，不是绕过）：
  1. `engine_service._action_backend()` 改为向推理服务查询**实际在服务的后端**，配置只作兜底；
  2. 计划落库后加"后端—技能"闸门：`openwam` 要求恰好一个 `whole_task`，
     其它后端禁止 `whole_task`，不匹配直接 `plan_backend_mismatch` 失败，不进入执行。
- **复测**：同一命令重跑通过（28.9 s / 83 动作），并已在 §11.1 记为可用。

### 11.7 产物

| 文件 | 内容 |
| --- | --- |
| `records/acceptance/openwam_libero_goal_all10.json` | 十任务单轮汇总 |
| `records/acceptance/openwam_libero_goal_all10_runs/` | 上述 10 回合完整记录归档 |
| `records/acceptance/openwam_task1_repeat.json`（+ `_runs/`） | Task 1 连续 3 回合 |
| `records/acceptance/pi05_regression_task1.json`（+ `_runs/`） | 切回 Pi0.5 的回归回合 |
| `records/acceptance/openwam_switchback_task1.json`（+ `_runs/`） | 切回 OpenWAM 的回归回合 |
| `records/openwam_server.json` | 常驻服务启动记录（commit、checkpoint、horizon、GPU） |
| `records/logs/openwam_server.log`、`openwam_suite_all10.log` | 服务与验收日志 |
| `scripts/openwam_server.sh`、`scripts/set_backend.sh` | 常驻服务管理与后端切换 |
| `scripts/openwam_probe.py`、`scripts/summarize_openwam.py` | 加载探针与结果汇总 |
| `tests/test_openwam_contract.py`、`tests/test_openwam_backend_smoke.py`、`tests/test_whole_task_plan.py` | 契约对拍、模型层冒烟、整任务计划契约 |

### 11.8 已知限制

| 优先级 | 限制 | 现状 |
| --- | --- | --- |
| P1 | 整任务策略不做子技能拆解 | Agent 只能发 `whole_task`；OpenWAM 无法被要求"只做 open_drawer"之类的子任务 |
| P1 | 7/10 回合到达 goal 时夹爪未张开 | 官方 goal 成立，但没有"释放"证据；如要求释放需另行训练/约束 |
| P2 | 十任务只跑了 1 回合/任务 | 方差未知（Pi0.5 在同任务上曾出现 3/3 → 1/3 的波动），需要 ≥20 回合/任务 |
| P2 | `execute_horizon` 与 `generate_horizon` 都是 32 | 未做"缩短执行 horizon 提升反应性"的对照实验 |
| P3 | 编译未启用 | `--compile-enabled false`；开启后的延迟/稳定性未测 |

## 12. 当前 Qwen + OpenWAM 速度测量（2026-10-27）

固定条件：HEAD `21e262e`，配置 SHA256 `34c98a8b68eed4f2391b17e1b492ca3df28a57b999f99f898751c717c0f5cbe9`；
Qwen3.5-9B 在 :8001；OpenWAM-Alpha-Sim-LIBERO（`checkpoint_step_10690.safetensors`，
OpenWAM repo `4d7f642`，HF revision `110b07e`）在 :14520/GPU2；
`generate_horizon=execute_horizon=32`、`denoise_steps=10`、sync；LIBERO-Goal task 1、seed 0；
控制频率 20 Hz。机器可读结果：`records/acceptance/speed_qwen_openwam.json`、
`records/acceptance/speed_openwam_cold.json`。

### 12.1 Planner（同一请求重复 12 次）

| 指标 | 冷启动首个请求 | 热请求中位数 | p95 | 最大 |
| --- | --- | --- | --- | --- |
| 端到端 wall | 5.843 s | 5.535 s | 5.904 s | 5.904 s |
| vLLM server e2e | 5.793 s | 5.793 s | 5.841 s | 5.852 s |
| TTFT | 0.197 s | 0.183 s | 0.194 s | 0.197 s |
| decode | 5.596 s | 5.596 s | 5.658 s | 5.668 s |
| queue | 0.004 s | 0.000 s | 0.000 s | 0.004 s |

输入 1311 tokens、输出 123 tokens；12 次均为 1 次请求、0 次修复，解析/契约校验合计 <0.1 ms。
**解码占 Planner 端到端约 96%**；排队、TTFT、校验都可以忽略。

### 12.2 OpenWAM 请求（task 1 完整回合，84 次动作请求）

| 请求类型 | 次数 | server 延迟 | client 推理 | engine 总耗时 |
| --- | --- | --- | --- | --- |
| `chunk_generation`（热） | 3 | 469.8 ms 中位 | 484.1 ms | 565.6 ms |
| `cache_hit` | 81 | 15.3 ms 中位 | 30.5 ms | 113.6 ms |

- 冷进程首个 chunk：server 2072 ms、engine 2184 ms（CUDA kernel 初次启动），之后同进程热生成约 470 ms。
- 生成 96 个动作、实际消费 84 个；3 个 chunk id `[0,1,2]`，每个请求都记录
  `chunk_generation`/`cache_hit`、`chunk_id`、生成数/消费数。
- **缓存取动作的 client 30.5 ms 不是模型推理**；完整 chunk 推理按 470–485 ms 计
  （摊到 32 步约 15 ms/步），不能把 30 ms 或 server 15 ms 当作整块推理。

### 12.3 动作执行（task 1 成功回合，87 个 sim 动作）

| 指标 | 观测 |
| --- | --- |
| 回合总时长 | 31.66 s（Planner 5.82 s + 执行 22.40 s + 校验 0.26 s） |
| OpenWAM 控制动作 | 84 步；仿真推进 4.2 s |
| wall 动作吞吐 | 3.75 动作/s（配置 20 Hz） |
| 仿真实时因子 | 0.187（仿真 1 s 需 wall 5.35 s） |
| 相邻动作间隔 | 中位 251 ms，p95 264 ms，最大 718 ms |
| 执行阶段拆分 | inference 11.94 s、env step 5.80 s、observe 3.76 s、snapshot 0.81 s、convert 0.05 s |

**瓶颈证据**：执行阶段 53% 是 inference 等待，但其中模型生成只有 3×0.57≈1.7 s；
81 次 cache 取动作 81×0.114≈9.2 s。也就是说，主要成本在"每步 RPC/快照/观测/取缓存"的
流水线开销，而不是 OpenWAM 前向本身。env step 5.8 s、observe 3.76 s 也大于模型生成。
因此本轮没有依据用"更换推理引擎"解释或解决速度问题；先缩短每个动作的采样/传输/校验
链路才有明确收益。

## 13. d1-3B 离线规划对照（阈值冻结，2026-10-27）

- **阈值样本**：`config/offline_calibration.json`（24 例：18 正、6 负），只用于选择阈值；
- **正式样本**：`config/offline_eval.json`（48 例：40 正 = 10 官方 + 30 等价改写，8 负），
  冻结后只报告，不据结果回调；
- 阈值：supported/operation/specified/relation/source/target ≥0.5，
  operation 动词兜底 ≥0.3，top-1 对 runner-up 的 margin ≥0.2（`margin_min`）；
- 机器可读：`records/offline/calibration_d1_v4.json`、`records/offline/eval.json`。

### 13.1 校准集（18 正 / 6 负）

| 指标 | d1 |
| --- | --- |
| 正例接受 | 18/18 |
| accepted 后完整绑定 | 1.0 |
| 契约通过 | 1.0 |
| 负例误接受 | 0 |
| 拒绝（任意类别 / 类别完全一致） | 6/6 / 3/6 |
| 延迟（24 例，中位 / p95 / max） | 0.133 / 0.144 / 0.145 s |

### 13.2 正式集（48 例）

| 指标 | Qwen 完整 Planner | d1 决策 Planner | Qwen（同一候选决策） |
| --- | --- | --- | --- |
| 操作准确率 | 不适用（整任务无绑定） | 0.900（accepted 1.0） | 1.000 |
| source 准确率 | 不适用 | 0.875（accepted 1.0） | 1.000 |
| target 准确率 | 不适用 | 0.900（accepted 1.0） | 1.000 |
| relation 准确率 | 不适用 | 0.900（accepted 1.0） | 1.000 |
| 抽屉 open_target | 不适用 | 0.500（accepted 1.0） | 1.000 |
| 完整计划语义 | 0.800（修正 push 检查器后 0.900） | 0.900（accepted 1.0） | 1.000 |
| 契约通过率 | 0.979（47/48） | 1.000（accepted 36/36） | 1.000 |
| 负例拒绝（任意 / 类别一致） | 1/8 / 1/8 | 7/8 / 4/8 | 7/8 / 5/8 |
| 负例误接受 | 7 | 1（`put the bowl on the table`） | 1（同一例） |
| 延迟中位（n=48） | 6.237 s | **0.133 s** | 7.925 s |
| 延迟 p95 / max | 8.866 / 9.889 s | 0.145 / 0.147 s | 8.943 / 10.727 s |
| 首例 / 热请求中位 | 6.486 / 6.237 s | 0.136 / 0.133 s | 7.904 / 7.928 s |

- **d1 的 4 个漏接受**：t03_official 与 t03_p2（`source_uncertain`，source 仍是
  `akita_black_bowl_1`，但置信度/边距低于冻结阈值）、t08_official（`source_uncertain`，
  source 置信度 0.492）、t06_p3（动词 `drop` 被判为 `other`）。所有 36 个 accepted
  绑定都正确，说明失败模式是"过度弃权"，不是"绑定错误"。
- **Qwen 完整 Planner 的真实语义失败**：task 3 的四种说法都被改写成
  `open the top drawer of the cabinet`，**丢失了"把碗放进去"子目标**（4/4），
  这是 0.8→0.9 修正后仍然存在的唯一语义失败；task 5 的 4 个假阴性来自
  "push … to the front" 不含词面 `on` 的检查器缺陷，已单独重算并保留原始值
  （`records/offline/eval_semantic_recheck.json`）。
- **Qwen 完整 Planner 的负例行为**：8 个负例中只有 1 个因两次格式失败而拒绝，
  其余 7 个照常生成计划；它没有稳定的歧义/缺参拒绝能力。
- **两次比较的结论**：
  1. **真实替换收益**：d1 把 Planner 延迟从 6.24 s 降到 0.133 s（约 47×），
     但正式集上弃权 2/10 官方任务，不能作为默认 Agent 替换 Qwen；
  2. **同一候选决策**：Qwen 40/40 全部绑定正确、拒绝 7/8；d1 accepted 36/40 且
     同样全部正确、拒绝 7/8。模型差异主要在"是否弃权"和延迟（59.6×），
     而不是输出格式导致的能力差；d1 的低置信门是双刃剑。
- **处置**：保留 `agent.mode=qwen` 为默认；d1 随后以 `d1` 模式参与真实 A/B，
  不通过放宽阈值或硬编码任务答案提升指标。目标绑定结论为 **PARTIAL**。
  60 回合 A/B 判定 d1 不划算后，d1 已在 §15 整体下线，当前代码不再含 d1 模式。
## 14. Qwen vs d1-3B 真实执行对照（60 回合，2026-10-27）

### 14.1 冻结条件与协议

| 项 | 值 |
| --- | --- |
| 源码 | 运行前冻结：`agent/d1_planner.py` `3e29251e`、`agent/agent_service.py` `ddc1f1a8`、`engine/engine_service.py` `a822b244`、`engine/openwam_rollout.py` `cdd35cf1`、`common/plan_contract.py` `f9586cac`、`config/engine.json` `ddf00e52`、`config/offline_eval.json` `51adf51d` |
| Agent A | Qwen3.5-9B，vLLM :8001（GPU3，`--gpu-memory-utilization 0.72 --max-model-len 16384`），`temperature=0`、`max_tokens=900`、`planner_vision=false` |
| Agent B | `LiquidAI/d1-3B`，本地权重 SHA256 `50e03317847caf6…6c0c`，bf16，:14710（GPU0），`system_one` 单次前向、0 输出 token，无 Qwen 兜底（严格 `d1` 模式） |
| 动作模型 | OpenWAM-Alpha-Sim-LIBERO，`checkpoint_step_10690.safetensors`，HF revision `110b07e`，repo `4d7f642`，:14520（GPU2），`generate_horizon=execute_horizon=32`、`denoise_steps=10`、sync 单步 |
| 环境 | LIBERO-Goal 10 任务（task 0–9），控制频率 20 Hz，动作预算 600 sim step，回合超时 900 s |
| 初始化 | 每任务 seed 0/1/2 三组固定初始化，两方案共用同一清单 |
| 顺序 | 每 (task, seed) 交替 A/B 顺序：`qwen,d1` 与 `d1,qwen` 各 15 组，减少时间/资源漂移 |
| 同起点校验 | 同一 (task, seed) 两回合的初始化场景指纹 **30/30 完全一致** |
| 记录 | `records/ab/ab_frozen.json`（完整）、`records/ab/ab_frozen_slim.json`、`records/ab/ab_frozen_episodes.csv`、`records/ab/ab_frozen_tables.md`、失败证据 `records/ab/failure_evidence/` |
| 未做的事 | 不重跑、不挑回合、不放宽验收、不为 d1 追加任务专属答案；失败/拒绝/超时全部保留 |

补充说明：A/B 跑完后才加入 `planner_generation`（旧规划结果丢弃）改动，
当前 `engine/engine_service.py` 为 `45143682`。该改动只影响"规划返回值是否允许落库"，
不改变规划内容、动作块与执行路径；改后已单独验证（§14.6）。

### 14.2 总表（30 回合/方案）

| 指标 | Qwen | d1-3B |
| --- | --- | --- |
| 尝试回合 | 30 | 30 |
| 计划语义正确 | **30/30 (100%)** | 24/30 (80.0%) |
| 原生目标成立 | **29/30 (96.7%)** | 23/30 (76.7%) |
| 用户目标满足 | **29/30 (96.7%)** | 23/30 (76.7%) |
| 完整执行完成 | **29/30 (96.7%)** | 23/30 (76.7%) |
| 回合终止码 | `native_goal_not_reached` ×1 | `plan_invalid`（拒绝）×6、`native_goal_not_reached` ×1 |
| 整回合中位（全部尝试） | 33.67 s | 26.47 s |
| 整回合中位（仅成功） | 33.24 s | 28.31 s |
| 整回合 p95（全部尝试） | 95.35 s | 90.15 s |
| 仿真动作中位 | 96.0 | 89.5 |

快速失败不计为加速：d1 的 6 个 `plan_invalid` 回合各约 3.2 s 结束（0 动作），
但同一任务上 Qwen 是完整成功；d1 成功回合（28.31 s）比 Qwen 成功回合（33.24 s）
快 4.9 s，差异几乎全部来自 Planner。

### 14.3 逐任务（成功率 / 成功耗时）

| task | 指令 | Qwen 用户目标 | d1 用户目标 | Qwen 成功中位 | d1 成功中位 |
| --- | --- | --- | --- | --- | --- |
| 0 | open the middle drawer of the cabinet | 3/3 | 3/3 | 41.47 s | 37.82 s |
| 1 | put the bowl on the stove | 3/3 | 3/3 | 30.23 s | 25.77 s |
| 2 | put the wine bottle on top of the cabinet | 2/3 | 2/3 | 31.76 s | 26.47 s |
| 3 | open the top drawer and put the bowl inside | **3/3** | **0/3（拒绝）** | 54.80 s | — |
| 4 | put the bowl on top of the cabinet | 3/3 | 3/3 | 31.67 s | 26.04 s |
| 5 | push the plate to the front of the stove | 3/3 | 3/3 | 46.30 s | 41.45 s |
| 6 | put the cream cheese in the bowl | 3/3 | 3/3 | 34.10 s | 28.31 s |
| 7 | turn on the stove | 3/3 | 3/3 | 27.58 s | 23.49 s |
| 8 | put the bowl on the plate | **3/3** | **0/3（拒绝）** | 29.69 s | — |
| 9 | put the wine bottle in the wine rack | 3/3 | 3/3 | 45.11 s | 41.97 s |

同一任务三组初始化的成功率完全一致，说明差异来自 Agent 决策，而非初始化难度抽签。

> **表内指令的更正（2026-10-27）**：本表早先手写的 task 4 与 task 6 指令有误，写成了
> "open the bottom drawer and put the bowl inside" 与 "put the cream cheese in the basket"。
> 真实执行记录 `records/ab/ab_frozen.json` 中这两项是 `put the bowl on top of the cabinet`
> 与 `put the cream cheese in the bowl`，与 `config/tasks.json` 的官方指令一致；
> 只是本表格文字写错，回合数据本身没有错。task 9 同理写作 `put the wine bottle on the rack`。
> 现在十任务只由 `config/tasks.json` 维护，见 §16。

### 14.4 失败证据（全部保留）

- **d1 task 3（3/3 全拒绝）**：`source_uncertain`，`source='akita_black_bowl_1' margin=0.178`
  （冻结阈值 margin ≥0.2）。计划被拒绝，0 动作，3.13 s 结束。
  证据：`records/ab/failure_evidence/task3_seed{0,1,2}_d1.json`。
- **d1 task 8（3/3 全拒绝）**：`source_uncertain`，`source='akita_black_bowl_1' confidence=0.492`
  （阈值 ≥0.5）。差 0.008 触发弃权，0 动作，3.21 s 结束。
  证据：`records/ab/failure_evidence/task8_seed{0,1,2}_d1.json`。
- **d1 task 2 seed 0**：语义正确（wine_bottle → cabinet top side，source 0.915 / target 0.952），
  但执行 1200 sim step（两个 600 步预算，恢复一次）仍未达成原生目标。
  与 Qwen 在 task 2 seed 1 的失败同类：这是 OpenWAM 长程放置能力问题，不是决策问题。
- **Qwen task 2 seed 1**：600 步预算用尽，`native_goal_not_reached`（无错误计划，纯执行失败）。
- 两次独立 60 回合运行（`records/ab/ab_full.json` 与 `ab_frozen.json`，同一冻结代码）：
  Qwen 两次都是 29/30；d1 两次为 24/30 与 23/30，差异只在 task 2 的执行成败。
  失败模式稳定复现（task 3/8 恒定拒绝）。`records/ab/failure_evidence/` 对应冻结版
  （`ab_frozen.json`）；第一轮的完整逐请求记录仍在 `ab_full.json`，
  其中的 Qwen task 9 seed 1 超时证据保留为 `failure_evidence/run1_task9_seed1_qwen.json`。

### 14.5 速度分解（同一 OpenWAM、同一场景、交错顺序）

| 阶段 | Qwen（30 回合） | d1（24 个有执行回合） |
| --- | --- | --- |
| Planner 中位 / p95 | 5.451 s / 6.123 s | **0.158 s / 0.160 s** |
| 规划完成 → 首个动作 | 0.645 s | 0.639 s |
| OpenWAM 完整 chunk 生成（server 延迟中位） | 465.1 ms（区间 419.9–476.0 ms） | 465.9 ms（区间 460.9–488.0 ms） |
| 缓存取动作（client 推理中位） | 29.8 ms | 29.9 ms |
| 每回合 chunk 生成 / 缓存命中 | 3.0 / 90.0 | 3.5 / 91.5 |
| 执行阶段墙钟中位 | 24.68 s | 25.38 s |
| 动作墙钟控制 Hz | 3.760 | 3.765 |
| 仿真实时因子（sim 时间 / wall 时间） | 0.188 | 0.188 |
| 执行阶段分段（中位）：observe / inference / env_step / snapshot / convert | 4.11 / 13.00 / 6.68 / 0.88 / 0.06 s | 4.18 / 13.41 / 6.79 / 0.89 / 0.07 s |
| 校验中位 | 0.252 s | 0.246 s |

同一观测、同一指令、同一模型服务下的 Planner 定点复测
（`records/acceptance/speed_planner_qwen.json`、`speed_planner_d1.json`，各 12 次
同一 task 1 请求，交替切换 Agent 模式，切换后恢复 `qwen`）：

| 指标 | Qwen | d1 |
| --- | --- | --- |
| 端到端中位（n=12） | 5.916 s | **0.141 s** |
| 首个请求 / 热请求中位 | 5.956 s / 5.904 s | 0.154 s / 0.140 s |
| 输入 / 输出 token | 1312 / 126 | 无可比 token 数（单次前向，0 输出 token） |
| TTFT / decode | 0.183 s / 5.668 s | 不适用（无生成阶段） |

结论：Planner 端到端约 **42×**（定点复测）到 **35×**（60 回合中位）加速，
与离线 48 例的 47× 同量级；`plan_to_first_action` 不变（0.64 s，来自 OpenWAM
首个完整 chunk 生成），因此**回合级收益上限就是 Planner 的 5.3 s**。

### 14.6 生命周期与切换验证（本轮新增）

- 执行中切换 Agent：`POST /api/agent_mode` 返回 **409**，`{"status":"busy",
  "error":"agent mode can only change while idle; phase=running"}`，模式保持 `qwen`。
- 规划中途 stop：`planner_generation` 5 → 6，规划 6.204 s 后返回的 Qwen 计划被丢弃，
  `records/run_journal.jsonl` 记 `planner_result_discarded`（`planner_generation=5`,
  `current_generation=6`），`/api/status` 的 `plan` 保持 `null`，
  `metrics.stale_plans_discarded=2`，回合终态 `stopped` / `user_stop`，无任何动作发出。
- 新回合隔离：停止后切换到 `d1`，task 1 新回合以 `plan_source=d1-3b`、
  `agent_mode=d1` 规划并成功（原生目标成立），未复用上一回合任何计划。
- OpenWAM 缓冲：`stop` 会 `cancel` 推理请求，`:14600/metrics` 的
  `backend_resets_on_cancel` 递增（本轮 5），`pending_prefetch=0`。
- `scripts/test_lifecycle.py` 全量回归 **PASS**（切换、stop 预算、缓冲清理、
  reset 场景 epoch 推进、reset 后二次回合成功）。
- 证据：`records/acceptance/lifecycle_planner_generation.json`、
  `records/acceptance/lifecycle_agent_switch.log`。

### 14.7 结论与处置

1. **d1 是否值得替换 Qwen：不值得，本轮结论 PARTIAL。**
   d1 把规划延迟从 5.45 s 降到 0.158 s（约 35×），计划和契约 100% 合法、
   accepted 后绑定 100% 正确、零幻觉执行；但用户目标成功率 76.7% vs 96.7%，
   在 task 3（开抽屉并放碗）和 task 8（碗放到盘子上）稳定弃权，
   真实执行中直接损失 6/30 回合。task 3 恰好是"多子目标不能拆分丢失"的任务，
   d1 的失败不是拆错，而是"不敢选"，仍不能作为默认 Agent。
2. **混合兜底的真实收益与成本。** 已实现的 `d1_with_qwen_fallback` 只在
   `error`/`unavailable` 时回落，语义拒绝不回落——本轮 d1 的 6 次失败全部是
   `needs_clarification`（语义拒绝），因此**该兜底对本轮失败零收益**。
   若把 `rejection` 也纳入兜底：这 6 个回合会各多付一次 Qwen 规划（约 5.5 s），
   成功率最多回到 Qwen 水平，而每回合都要先付 d1 的 0.16 s 与一次 409 风险；
   收益等于"用 d1 做前置过滤器"，但 d1 的过滤在本轮是假阴性（碗明明在场景里）。
   因此不建议在当前阈值下启用混合模式。
3. **当前主要瓶颈不是 Agent，也不是 OpenWAM 前向。**
   成功回合里 Planner 占 Qwen 5.45 s、d1 0.16 s；执行阶段 24–25 s 中
   inference 等待 13 s、env_step 6.7 s、observe 4.1 s，而真正的 chunk 生成只有
   3–3.5 次 × 0.47 s ≈ 1.6 s；缓存取动作 ~90 次 × 0.03 s（client）≈ 2.7 s，
   单步 RPC + 观测 + 快照的流水线才是主要成本（墙钟 3.76 Hz vs 配置 20 Hz，
   实时因子 0.188）。
4. **是否值得下一轮更换推理引擎：不以此为依据。**
   换推理引擎最多改善那 1.6 s 的生成时间；即使生成变 0，回合也只快 ~5%，
   而每步 0.26 s 的流水线开销不变。下一轮应先做动作循环的批量化/异步化
   （观测与快照合并、每步 RPC 合并、缓存取动作本地化），再谈引擎替换。
   本轮按要求**未改** OpenWAM 精度、采样步数、动作块或异步策略。
5. **默认模式保持 `qwen`**；本轮结论只是「d1 不划算」，因此 d1 随后按用户要求在
   §15 整体下线（服务、权重、venv、代码、UI 选项全部删除），而不是放宽阈值
   （如 source 0.492→0.49）或硬编码 task 3/8 答案来"修好"指标。
   若将来还要做前置过滤，应在独立校准集上重选阈值，重建一条独立服务。

产物清单（本轮）：`records/ab/ab_frozen*.json|csv|md`、`records/ab/failure_evidence/*.json`、
`records/acceptance/speed_{qwen_openwam,openwam_cold,planner_qwen,planner_d1}.json`、
`records/acceptance/lifecycle_planner_generation.json`、`records/offline/{eval,calibration_d1_v4,eval_semantic_recheck}.json`。

---

## 15. d1-3B 下线与资源回收（2026-10-27）

d1 作为 §12–§14 的第二条决策链路已完成对照任务：在 60 回合交错 A/B 中它用 0.158 s
的决策把成功率从 29/30 降到 23–24/30，收益为负（§14.2）。本阶段按用户要求**彻底移除**，
生产栈回到 Qwen3.5-9B + OpenWAM 两条模型。

### 15.1 停止服务

| 项 | 停止前 | 停止后 |
| --- | --- | --- |
| 进程 | PID 1410342，`2026-10-27 03:07:15 +08:00` 启动 | 进程不存在 |
| 命令 | `.venv-d1/bin/python agent/d1_service.py` | 脚本与入口文件已删除 |
| 监听 | `127.0.0.1:14710` | 端口无监听 |
| 显存 | GPU0 6800 MiB | GPU0 4 MiB（仅驱动上下文） |

停止方式为 `bash scripts/d1_server.sh stop`（脚本等待进程退出后核对端口与显存），
不是 `kill -9`。停机发生在删除之前，确保没有进程持有被删文件的句柄。

### 15.2 删除范围

| 路径 | 内容 | 释放 |
| --- | --- | --- |
| `openwam_harness/checkpoints/LiquidAI-d1-3B` | 16 个权重文件 | 6,265,092,138 B（5974.9 MiB） |
| `embodied_engine/.venv-d1` | 34,561 个普通文件 | 仅 430 个独占 inode，6,280,626 B（5.99 MiB） |
| `records/d1_server.pid`、`records/d1_server.json` | 运行记录 | 8 B + 493 B |
| `agent/d1_service.py`、`agent/d1_planner.py`、`scripts/d1_server.sh`、`scripts/run_ab.py`、`tests/test_d1_planner.py` | 代码 | 由 git 历史保留 |
| `config/offline_calibration.json`、`records/offline/calibration_d1_v4.json` | d1 阈值校准集 | 已删除（阈值脚本只服务 d1） |

`.venv-d1` 的 `du` 显示 7296 MiB，但其中 34,131 个 inode 与 `.venv-openwam`
**硬链接共享**；`du` 会把共享块在两个 venv 各算一次。删除时只移除目录项，
共享块不释放，`.venv-openwam` 的 34,131 个文件仍在原位。这正是删除前逐 inode
核对、并在删除后复核的原因：**不能用 `du` 估算释放量，也不能用 `du` 验证完整性。**

### 15.3 实测释放量与完整性

| 指标 | 数值 |
| --- | --- |
| 预期释放（独占文件之和） | 6,271,372,764 B |
| 实测释放（`df` 空闲空间差，跨删除前后） | 6,287,941,632 B = **5.856 GiB** |
| 与预期差额 | +16.6 MB（目录项、块对齐、日志轮转） |
| `.venv-openwam` 删除后普通文件数 | 34,131（删除前同值） |
| 端口 14710 | 无监听 |
| d1 源文件 | 不存在 |
| GPU0 占用 | 4 MiB |

`.venv-d1/bin/python`、`bin/python3`、`bin/python3.10`、`lib64` 都是符号链接
（分别指向 `/usr/bin/python3.10` 与本目录 `lib`），`rm -rf` 不跟随符号链接，
因此系统 Python 与 `lib` 内容都不受影响。审计记录：
`records/acceptance/d1_cleanup_manifest.json`（含逐路径 inode 统计与删除后复核）。

保留的 d1 痕迹只有三类，均为**证据**而非可运行资源：
`records/ab/ab_frozen.json`（60 回合对照数据）、`records/ab/failure_evidence/`（拒绝回合的逐请求证据）、
`records/logs/d1_server.log`，以及 `experiment/d1-3b-20261027` 分支上的历史代码。

### 15.4 清理后的可运行性复核

删除后按顺序复核，任一失败即回滚：

1. Qwen `:8001/v1/models` 可枚举 → 通过；
2. Agent `:14700/health` 返回 `mode=semantic_router`，无 d1 字段 → 通过；
3. `scripts/test_lifecycle.py` 全链路（health → 场景加载 → 指令 → 执行 → 复位）→ 通过，见 §18.6；
4. 十任务完整回合（§18 的 20 回合）→ 通过，见 §18。

`config/engine.json`、`common/config.py`、`scripts/bench_speed.py` 中所有 d1 分支、
阈值加载、健康检查与测速入口同步移除；`page_ui` 删除了「Agent 模式」下拉框与「切换」按钮
（那正是 d1 的唯一入口），指令框改为自由文本（中英文皆可），并新增一行理解面板：
`原文 / 理解 / 标准任务 / 需要切换`，把「用户说了什么 → 路由到哪个键 → 最终送给策略的标准指令」
暴露在页面上，而不是只显示成功与否。`scripts/test_lifecycle.py` 新增断言：
health 响应中不得再出现 `d1` 字样，防止代码与文档再次漂移。

---

## 16. 十任务唯一目录（`config/tasks.json` v2）

### 16.1 为什么需要统一

此前的十任务信息散落在三处，各自可手改：`config/tasks.json`（给模型看的指令）、
LIBERO 安装包（`task_map` + BDDL + init）、以及文档里的手抄表格。
§14.3 的 task 4 / task 6 两行文字写错（写成 "open the bottom drawer..." 与
"put the cream cheese in the basket"）就是手抄层漂移的实例——**回合数据没错，
只是文档文字错**。本阶段把「哪个任务叫什么、目标是什么、指令原文是什么」
收敛到单一文件，其余位置只读取，不再手写。

### 16.2 权威来源与指纹

| 项 | 值 |
| --- | --- |
| 套件 | `libero_goal`（10 个任务） |
| 代码来源 | `https://github.com/RLinf/LIBERO.git` |
| commit | `a8323074d93a09e32bd898630a70531b1f51bc77`（`rpent` 构建） |
| 包路径 | `openwam_harness/envs/rpent/lib/python3.10/site-packages/libero/libero` |
| 任务顺序文件 | `benchmark/libero_suite_task_map.py` |
| 任务表指纹 | `tasks_info_sha256 = 547286ce…b479382` |
| 目录版本 / 指纹 | `2026-10-27.1` / `7eb62a5d…43ffecf`（`common/task_catalog.py` 计算） |

**官方指令不是 BDDL 的 `:language`。** LIBERO 的 `Task.language` 由
`libero/benchmark/__init__.py::grab_language_from_filename` 从文件名生成，
与环境里的 BDDL 文字并不一致，而评测对齐的是前者：

| task | 官方指令（评测口径） | BDDL `:language` | 差异 |
| --- | --- | --- | --- |
| 0 | open the middle drawer of the cabinet | Open the middle layer of the drawer | layer → drawer |
| 2 | put the wine bottle on top of the cabinet | Put the wine bottle on the top of the drawer | drawer ≠ cabinet |
| 3 | open the top drawer and put the bowl inside | Open the top layer of the drawer and put the bowl inside | layer → drawer |
| 4 | put the bowl on top of the cabinet | Put the bowl on the top of the drawer | drawer ≠ cabinet |
| 6 | put the cream cheese in the bowl | Put the cream cheese on the bowl | in → on |

若直接采信 BDDL 文案，task 4 会变成「放在抽屉顶上」（实际目标是
`wooden_cabinet_1_top_side`），task 6 会变成「放在碗上」。BDDL 在本项目里
只用于两件事：抽取 `goal_atoms` 供原生目标判定、以及审计哈希。

### 16.3 十任务表

| id | key | 官方指令 | 类型 | source → target | 原生目标原子 |
| --- | --- | --- | --- | --- | --- |
| 0 | `open_middle_drawer` | open the middle drawer of the cabinet | open_drawer | — → `wooden_cabinet_1_middle_region` | `Open wooden_cabinet_1_middle_region` |
| 1 | `put_bowl_on_stove` | put the bowl on the stove | pick_place | `akita_black_bowl_1` → `flat_stove_1_cook_region` | `On akita_black_bowl_1 flat_stove_1_cook_region` |
| 2 | `put_wine_bottle_on_cabinet_top` | put the wine bottle on top of the cabinet | pick_place | `wine_bottle_1` → `wooden_cabinet_1_top_side` | `On wine_bottle_1 wooden_cabinet_1_top_side` |
| 3 | `open_top_drawer_put_bowl_inside` | open the top drawer and put the bowl inside | composite | `akita_black_bowl_1` → `wooden_cabinet_1_top_region` | `In akita_black_bowl_1 wooden_cabinet_1_top_region` |
| 4 | `put_bowl_on_cabinet_top` | put the bowl on top of the cabinet | pick_place | `akita_black_bowl_1` → `wooden_cabinet_1_top_side` | `On akita_black_bowl_1 wooden_cabinet_1_top_side` |
| 5 | `push_plate_front_of_stove` | push the plate to the front of the stove | push | `plate_1` → `main_table_stove_front_region` | `On plate_1 main_table_stove_front_region` |
| 6 | `put_cream_cheese_in_bowl` | put the cream cheese in the bowl | pick_place | `cream_cheese_1` → `akita_black_bowl_1` | `On cream_cheese_1 akita_black_bowl_1` |
| 7 | `turn_on_stove` | turn on the stove | turn_on | — → `flat_stove_1` | `Turnon flat_stove_1` |
| 8 | `put_bowl_on_plate` | put the bowl on the plate | pick_place | `akita_black_bowl_1` → `plate_1` | `On akita_black_bowl_1 plate_1` |
| 9 | `put_wine_bottle_on_rack` | put the wine bottle on the rack | pick_place | `wine_bottle_1` → `wine_rack_1_top_region` | `On wine_bottle_1 wine_rack_1_top_region` |

注意 task 6：指令说 `in the bowl`，而这个构建的原生目标判定是 `On`。
**指令口径与判定口径本来就可以不同**，两者都以冻结文件为准，不互相改写；
§18 的「原生目标成立」指上表的谓词成立，不代表指令里的介词被逐字实现。

task 3 是唯一的复合任务，`semantics.subgoals` 记录「开上抽屉」与「放碗」两个子目标；
路由层不允许把它拆成两个单独能力（见 §17 提示词规则 3）。

### 16.4 单一来源与消费方

`common/task_catalog.py` 是唯一读取入口，对外提供
`task_by_id / task_by_key / task_for / capability_rows / scene_support / fingerprint`。
消费方全部改为调用它，不再自带副本：

| 消费方 | 用途 |
| --- | --- |
| `agent/semantic_router.py` | 候选键与能力描述（`capability_rows`）、场景实体校验、装配官方指令 |
| `engine/engine_service.py`、`/api/tasks` | 页面场景/任务下拉与健康信息 |
| `page_ui` | 「标准任务」显示路由后实际送给策略的指令 |
| `scripts/eval_offline.py`、`scripts/run_language_rounds.py` | 测试集校验、逐回合对照 |
| `tests/test_task_catalog.py` | 断言 key 唯一、10 条齐全、指纹与 LIBERO 安装一致、官方指令与文件名推导一致 |

因此「文档写一套、代码跑另一套」不再可能：指令文字只有 `tasks.json` 一处可改，
改了就改变指纹，测试与离线评测都会立刻发现。

---

## 17. Qwen 语义路由（替代整段规划）

### 17.1 旧链路的两个问题

旧实现让 Qwen 直接产出完整计划 JSON（`interpretation`、`success_condition`、`steps[]`），
再由代码校验后交给 OpenWAM。三套评测集上实测（`records/offline/semantic_eval_20261027*.json`）：

| 指标 | 旧链路（整段规划） |
| --- | --- |
| 请求中位耗时 | 4.93～5.92 s（三套集合） |
| p95 / 最大 | 6.53～6.77 s / **7.16 s** |
| prompt / completion token 中位 | 1305 / 118 |
| 首 token | 0.179 s |
| 指令与官方不一致 | 1/80（`t00_en1`） |

耗时几乎全在 decode（4.71 s / 4.93 s）：模型要把一百多个 token 的说明文字写完，
其中没有一个字节会进入策略输入。更严重的是第二点：`t00_en1`（"pull open the middle
drawer of the wooden cabinet"）被旧链路原样传给 OpenWAM，而不是目录里的官方指令
"open the middle drawer of the cabinet"——**改写指令等于改变策略输入分布**，
成功率无从保证，而且这种错误只会在回合跑完后才暴露。

### 17.2 新契约：只选键，其余交给代码

```
用户原文 ──► Qwen（非思考，≤24 token）──► {"choice":"<key>"} | {"choice":"clarify"} | {"choice":"unsupported"}
                                                    │
                     代码：目录查表 → 场景实体校验 → 已加载任务比对 → 装配官方指令 → plan_contract 校验
                                                    │
                                             {"skill":"whole_task","prompt":"<官方指令>"}
```

- **模型只看得到**：固定规则 + 十行能力描述（由 `tasks.json` 的 `capability` 生成）+ 用户原文。
  提示词里**没有** task_id、官方指令、BDDL 目标或场景快照（`snapshot_in_prompt=false` 写入遥测），
  所以不存在「答案泄漏」或「拿当前场景倒推任务」。
- **模型输出**：一个 12 选 1 的键（10 个任务 + `clarify` + `unsupported`），
  `max_tokens=24`，`temperature=0`，`response_format=json_object`，
  `chat_template_kwargs.enable_thinking=false`（非思考模式，避免先输出思维链）。
- **决策路径上没有任何关键词表或同义词词典**：同义句、中英文、语序变化全靠模型，
  代码只做确定性校验。这是本阶段刻意守住的红线——否则「中文能跑」只是查表跑通。
- **格式容错**：只允许 1 次修复回合（`max_repairs=1`），修复提示要求只返回修正后的 JSON。
  四套集合共 406 个请求，实际修复次数 **0**。
- 上限与模式来自 `config/engine.json` 的 `agent.semantic_router`（`max_tokens` / `max_repairs` / `model_mode`），
  `router_options()` 读取后写入 health 与响应遥测，避免常量与配置两处漂移。
- 返回给页面的 `understanding` 块包含：原文、选择键、标准指令、是否需切换场景、
  场景缺失实体、修复次数、`assembly_s`。

### 17.3 判定顺序（提示词的三条规则）

模型必须先判 `clarify`、再选键、最后才判 `unsupported`：

1. **`clarify`**：用户的话留下未填的槽位，且能力表能决定该槽位——未解析的代词/指示词
   （"put it on the stove"、"把它放到炉灶上"）、完全没有目标（"move the bowl"）、
   只给了容器没给层（"put the bowl inside the cabinet"）。
2. **选键**：对象、目标位置、关系、层数、以及复合任务的每个子目标都吻合，才可选；
   礼貌用语、语序、普通同义词不改变目标。
3. **`unsupported`**：请求足够具体、可判定，但没有任何能力吻合——不存在的物体、
   不存在的关系（"beside" / "near" / "behind" / "under"）、目录未收录的层数、
   复合任务的一半、或把两个能力拼在一起。

提示词里显式写明：**用户已经说出的关系或层数属于请求内容，不是缺失信息**，
"put the bowl next to the stove" 必须是 `unsupported` 而不是 `clarify`。
这条边界在两轮评测中分别错过两个方向，见 §17.5。

### 17.4 四套评测集结果（含一次回归重跑）

A / B / C 三套输入互不重叠（B′ 与 B 同一批输入，只作回归重跑），且与更早的回归集去重（生成脚本带重复项断言）：

| 集合 | 用途 | 规模 | 正例选择正确 | 负例判定正确 | 场景不匹配 | 修复 | wall 中位 / p95 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A `semantic_eval_20261027` | 开发集（写提示词时可见） | 104（80 正 / 22 负 / 2 错场景） | 80/80 | 17/22 | 2/2 | 0 | 0.624 / 0.822 s |
| B `…20261027b` | 首次未见 | 101（80 / 19 / 2） | 80/80 | 15/19 | 2/2 | 0 | 0.673 / 0.803 s |
| B′ `…20261027b_regression` | 改提示词后回归（与 B 同一批输入，不算独立样本） | 101 | 80/80 | 19/19 | 2/2 | 0 | 0.684 / 0.823 s |
| C `…20261027c` | **第二次未见，零重叠** | 100（80 / 18 / 2） | **80/80** | **18/18** | **2/2** | **0** | **0.681 / 0.823 s** |

正例 80 条 = 10 任务 ×（官方指令 + 英文改述 + 中文表达），覆盖：
"grab the black bowl and set it down on the plate"、"把酒瓶放到架子上面" 等。
负例分三类：目录外物体（apple / basket）、目录外关系（beside / behind / under）、
复合任务拆半（只开上抽屉）、缺槽位（"put it on the stove"、"打开柜子抽屉"）。

C 集的 `prompt_equals_official` 为 80/80：所有 `ok` 决定的 `whole_task` 输入
**逐字等于** `tasks.json` 官方指令，没有一条是用户原话或模型改写。

延迟拆解（C 集，100 请求）：

| 阶段 | 中位 | p95 |
| --- | --- | --- |
| 首 token（TTFT） | 0.179 s | 0.185 s |
| decode（≤24 token） | 0.457 s | 0.590 s |
| 解析 + 校验 + 装配 | 0.0002 s | — |
| 端到端墙钟 | **0.681 s** | **0.823 s** |
| 旧链路端到端（同 20 条样本） | 5.92 s | 6.77 s |

即端到端 **8.7×**，而 TTFT 与旧链路几乎相同（0.179 s vs 0.179 s）——
加速全部来自少写一百多个 token，而不是换了推理引擎。

### 17.5 两轮修正（真实错误，不做修饰）

- **A 集 8 条正例被拒**：全部是 task 7（stove）。模型选择键 80/80 全对，
  错在场景实体校验——炉灶本体只以 `flat_stove_1_cook_region`、
  `flat_stove_1_main` 等 site/前缀形式出现在快照里，旧校验只查对象表，
  于是判定 `required_entity_absent`。修正 `task_catalog.scene_support`：
  目标与对象都按「对象表 + site 表 + 前缀匹配」查找，并加回归测试
  `test_stove_is_found_through_its_cook_region_site`。
- **A 集 5 条负例判成 `clarify`**（"open the bottom drawer"、"put the apple in the basket"）。
  原因：提示词只说了「缺信息就 clarify」，没说「说出来了但目录没有」是 unsupported。
- **B 集 4 条负例反向出错**：补了上面的规则后矫枉过正，
  "put it on the stove"、"把碗放进柜子"（缺层）被判成 `unsupported`。
  最终提示词明确两条规则的先后：**未解析代词/指示词、缺层数的容器 → claridfy 优先**；
  **已说出的关系/层数 → 属于请求内容 → unsupported**。
- **B′ 与 C 均为 100%**，两套集合都是在提示词冻结之后才跑的；C 集与 B/B′ 零重叠。

这三轮改动只调提示词的判定顺序，没有加任何关键词或任务特判；
A 集的 5 条误判在 B′/C 上不再出现，B 集的 4 条误判在 B′/C 上也不再出现。

### 17.6 局限（必须与上面的数字一起读）

1. **测试集是自建的，没有人工标注金标。** C 集 100 条由我按任务目录生成并逐条核对，
   能证明「在这套分布上稳定」，不能证明「覆盖了真实用户的说法分布」。
2. **规模小。** 每任务 8 条正例；80/80 的 95% 置信下界约 96%，
   而不是「100% 正确」。
3. **负例边界是产品决策，不是客观真理。** 「缺层数的容器」判 `clarify`、
   「说的关系不存在」判 `unsupported`，这是本项目的取舍；换个团队可能反过来。
   规则已写进提示词与本文，避免以后把取舍当成 bug 反复改。
4. **没有测冷启动。** 以上延迟均在 vLLM 已加载、KV 缓存热的条件下测得；
   冷启动（进程首次加载、首次 CUDA graph 捕获）不在本轮数据内。
5. **没有并发测试。** 全部为单请求串行，多用户同时说话时的排队延迟未知。
6. **修复路径没有被真实触发。** 0/406 次修复意味着 `max_repairs` 的线上行为只有
   单元测试覆盖（用假模型注入坏 JSON），真实模型没有走到过。
7. **中文只测了 20 条/集。** 口音式表达、方言、同音错字（如「炉灶/炉罩」）未覆盖；
   模型对中文的鲁棒性仍是假设，不是结论。

---

## 18. 20 回合语言闭环（真实执行，2026-10-27）

### 18.1 协议

| 项 | 值 |
| --- | --- |
| 清单 | `config/language_rounds_20261027.json`，**首轮执行前冻结**（`frozen=true`） |
| 规模 | 20 回合 = 10 任务 ×（1 条英文改述 + 1 条中文表达） |
| 种子 | 0（每次 `env.reset()` 回到同一初始状态） |
| 栈 | Qwen3.5-9B 语义路由（`:14700`）+ OpenWAM-Alpha-Sim-LIBERO（`:14520`），`sync_step` |
| 目录 | `2026-10-27.1`，SHA256 `7eb62a5d…43ffecf` |
| 输入 | 只把**用户原话**发给 `/api/run`；官方指令由 Agent 回查目录后组装，测试脚本不传官方句子 |
| 无关变量 | 回合不重跑、不跳号；`max_recoveries=2` 是生产配置，失败回合同样保留 |
| 结果 | `records/acceptance/language_rounds_20261027.json`，日志 `records/acceptance/language_rounds_run.log` |

`mapping_ok` 的判定是五重与相等（不只看选了哪个键）：
`decision_status=ok`、`semantic_choice=expected_choice`、
`catalog_official_instruction`、`plan.steps[0].prompt`、`steps[0].verification.prompt`
后三者都**逐字等于** `tasks.json` 官方指令。

### 18.2 总表

| 维度 | 结果 |
| --- | --- |
| 语言映射正确 | **20/20**（英文 10/10，中文 10/10） |
| 官方指令逐字一致 | **20/20**（`plan_step_prompt` 与 `policy_verification_prompt` 均一致） |
| 原生目标成立 | **19/20**（英文 9/10，中文 10/10） |
| 释放证据 `observed` | **2/20**（task 6 英/中各一次，开度 0.0717 m） |
| 释放 `not_applicable` | 6（task 0 / 5 / 7 各两次，开抽屉、推、开炉灶无释放阶段） |
| 原生成立但夹爪仍闭合 | **11**（开度 0.0134～0.0418 m） |
| 失败 | 1（task 2 英文，`native_goal_not_reached`，1800 动作，519.9 s） |
| 路由器耗时 | 中位 **0.710 s**，p95 0.817 s（英文 0.712 / 中文 0.707） |
| 成功回合耗时 | 中位 **29.2 s**，p95 51.5 s；仿真动作中位 99.5 |
| 总墙钟 | 22.6 min（10:59:50 → 11:22:27），20 回合耗时之和 19.7 min |

### 18.3 逐回合

| # | task | 语言 | 用户原话 | 选择 | 原生 | 夹爪开度 (m) | 动作 | 耗时 (s) | 证据 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0 | en | please pull open the middle drawer of the cabinet | `open_middle_drawer` | ✅ | 0.0771 | 116 | 33.6 | 无释放阶段 |
| 2 | 0 | zh | 请把柜子中间的抽屉拉开 | `open_middle_drawer` | ✅ | 0.0797 | 135 | 38.8 | 无释放阶段 |
| 3 | 1 | en | place the bowl on the stove | `put_bowl_on_stove` | ✅ | 0.0139 | 85 | 26.6 | 仍夹持 |
| 4 | 1 | zh | 把碗放到炉灶上面 | `put_bowl_on_stove` | ✅ | 0.0136 | 82 | 25.4 | 仍夹持 |
| 5 | 2 | en | get the wine bottle onto the cabinet top | `put_wine_bottle_on_cabinet_top` | ❌ | 0.0785 | 1800 | 519.9 | 松手但未达目标 |
| 6 | 2 | zh | 请把红酒瓶摆到柜子上面 | `put_wine_bottle_on_cabinet_top` | ✅ | 0.0176 | 84 | 25.8 | 仍夹持 |
| 7 | 3 | en | open the top drawer and place the bowl inside it | `open_top_drawer_put_bowl_inside` | ✅ | 0.0347 | 178 | 51.0 | 仍夹持 |
| 8 | 3 | zh | 请先把顶层抽屉拉开，再把碗放进去 | `open_top_drawer_put_bowl_inside` | ✅ | 0.0134 | 182 | 51.5 | 仍夹持 |
| 9 | 4 | en | put the bowl on the cabinet's top surface | `put_bowl_on_cabinet_top` | ✅ | 0.0190 | 86 | 26.5 | 仍夹持 |
| 10 | 4 | zh | 把碗放到柜子顶部 | `put_bowl_on_cabinet_top` | ✅ | 0.0142 | 90 | 27.6 | 仍夹持 |
| 11 | 5 | en | push the plate to the front of the stove | `push_plate_front_of_stove` | ✅ | 0.0755 | 192 | 54.0 | 无释放阶段 |
| 12 | 5 | zh | 把盘子推到炉灶前方 | `push_plate_front_of_stove` | ✅ | 0.0756 | 147 | 42.5 | 无释放阶段 |
| 13 | 6 | en | put the cream cheese inside the bowl | `put_cream_cheese_in_bowl` | ✅ | **0.0717** | 98 | 29.2 | **已释放** |
| 14 | 6 | zh | 把奶油奶酪放到碗里 | `put_cream_cheese_in_bowl` | ✅ | **0.0717** | 101 | 30.3 | **已释放** |
| 15 | 7 | en | switch on the stove | `turn_on_stove` | ✅ | 0.0292 | 84 | 25.5 | 无释放阶段 |
| 16 | 7 | zh | 请把炉子打开 | `turn_on_stove` | ✅ | 0.0273 | 83 | 25.4 | 无释放阶段 |
| 17 | 8 | en | set the bowl on the plate | `put_bowl_on_plate` | ✅ | 0.0151 | 79 | 24.7 | 仍夹持 |
| 18 | 8 | zh | 请把黑碗搁到盘子上 | `put_bowl_on_plate` | ✅ | 0.0145 | 81 | 25.1 | 仍夹持 |
| 19 | 9 | en | place the wine bottle up on the wine rack | `put_wine_bottle_on_rack` | ✅ | 0.0417 | 129 | 37.6 | 仍夹持 |
| 20 | 9 | zh | 把酒瓶放到架子上 | `put_wine_bottle_on_rack` | ✅ | 0.0417 | 129 | 37.6 | 仍夹持 |

### 18.4 三个维度必须分开读

**（1）语言映射：20/20。**
十任务各一句英文改述、一句中文表达，全部经由模型选键，**没有任何关键词表**；
选出的键回查目录后，送给 OpenWAM 的 prompt 20/20 逐字等于官方指令。
中文与英文的路由耗时中位差 0.005 s（0.707 / 0.712），没有"中文更慢"的迹象。
这只能说明"在这 20 句冻结输入上映射正确"，样本量见 §18.7。

**（2）原生目标：19/20，但这不是成功率。**
唯一失败是 task 2 英文回合：3 次 600 步尝试（Qwen 恢复决策依次 `retry / retry / abort`，
共 11.4 s），1800 步走完仍未达到 `On wine_bottle_1 wooden_cabinet_1_top_side`。
关键证据是同任务的中文回合：

| | 回合 5（失败） | 回合 6（成功） |
| --- | --- | --- |
| 初始场景指纹 | `fc50366cc74015d5` | `fc50366cc74015d5`（**完全相同**） |
| 官方 prompt | put the wine bottle on top of the cabinet | 同左（逐字相同） |
| 结果 | 1800 动作，519.9 s，未达目标 | 84 动作，25.8 s，达标 |

**同一初始状态、同一官方指令、同一模型，结果相反**，说明 OpenWAM 的执行不是确定性的
（task 0 的两回合官方 prompt 也相同，却分别是 116 与 135 个动作）。因此「19/20」
是 20 个单样本的观测，不能当作成功率；本轮**没有**为了好看而重跑失败回合。

**（3）释放证据：只有 task 6 真正松手。**
达到原生目标的 13 个 pick_place / composite 回合里，11 个结束夹爪开度 0.0134～0.0418 m
（阈值 0.05 m），即**物体被放到目标位姿时手指仍然闭合**；只有 task 6 两次都在 0.0717 m，
是真正"放下并松手"。本轮失败的 task 2 英文回合反而是松手的（0.0785），说明
"松手"与"达成目标"互不蕴含。

这与 §11.4 的十任务单轮结果一致（task 1/3/4/8/9 当时就是 0.006～0.041），
不是本轮新出现的退化。结论：**本部署的 OpenWAM 多数放置回合是"按到目标位置但没松手"**，
原生目标成立只代表基准谓词成立，不代表一次完整的「放置并释放」。
页面与记录里因此把释放单列，不用它去反向修饰成功率。

### 18.5 耗时

- 路由只占 0.71 s，回合耗时主要是动作循环：中位 29.2 s、p95 51.5 s；
- 失败回合 519.9 s 由三次尝试构成，单次 600 步 = 156.5 s（3.83 Hz 墙钟，实时因子 0.192），
  其中 inference 80.0 s、`env_step` 132.2 s、observe 91.2 s、snapshot 17.4 s —— 与 §13 的
  瓶颈判断一致（不是模型前向，而是每步流水线）；
- 20 回合总墙钟 22.6 min；失败回合 519.9 s 占 20 回合耗时之和（19.7 min）的 44%。

### 18.6 生命周期与单元测试（本阶段收尾）

`python3 -m unittest discover -s tests -p "test_*.py"`：**29 passed**
（新增 `tests/test_task_catalog.py` 19 条，覆盖目录形状、LIBERO 一致性、指纹、场景实体校验）。

`scripts/test_lifecycle.py` 第一次运行**失败**，失败项是
`[fail] simulator advanced after stop ack: 30 -> 31`。追查后是两个真实缺陷：

1. `Engine.stop()` 用 `self.env.state_snapshot()` 读计数，而 `self.env` 是 `EnvManager`、
   没有该方法；`AttributeError` 被 `except` 静默吞掉，`sim_at_ack` 永远退回
   `sim_at_request`，所以"停止后是否还有动作"这条证据其实一直没被测准。
2. 停止回合的记录用 `prepare` 时的快照，`sim_action_count` 恒为 0 ——
   日志里"12 s 的回合、0 个动作"是错的。

修复：`_settled_sim_count()` 改为通过 `self.env.client` 连续取快照直到两次一致（上限 2 s，
取最大值），`in_flight_actions` 由此准确；`except Stopped` 分支在生成记录前重新取一次快照，
并在取不到时写 `stop_snapshot_failed` 日志而不是静默忽略。
修复后同一脚本**连续两次 PASS**，停止回合记录显示 31 个动作（此前为 0），
`in_flight_actions` 与实际一致。

这是**停止路径的记账缺陷**，不改变任何回合的执行结果；§18.1–18.5 的 20 回合数据
产生于该修复之前，未被改动，也未因此重跑。

### 18.7 局限

1. **每（任务，语言）只有 1 个样本。** 19/20 是观测值，不是成功率；task 2 的同初始状态
   一败一成已经证明方差极大。
2. **只用了 seed 0。** 初始物体位姿只有一套，"换初始化还能不能成"没有数据。
3. **失败回合独占 44% 的运行时间。** 两次 `retry` 让单回合从 26 s 变成 520 s，
   生产上是否值得两次重试，应该由"重试收益 / 时间成本"的独立实验决定，本轮未优化。
4. **恢复决策仍走旧的全量规划路径**（Qwen `max_tokens=300`，每次 3.6～3.9 s），
   不是语义路由；这部分延迟没有优化，也不在 0.71 s 的路由统计里。
5. **释放阈值 0.05 m 是自定标准。** "仍夹持"只说明手指没张开，不代表物体没放稳；
   反过来，张开也不等于放稳。两者需要各自的证据，本文没有把其中一个当作另一个。
6. **中文表达由我撰写**，没有母语者盲评，"自然说法"的覆盖度未知；
   这 10 句通过，不能推出任意中文都能通过。
7. **不含人机确认。** 全部回合 `human_interventions=0`，页面只展示，不介入。

## 19. 放置收尾：未松手的真实原因、修复与职责划分（2026-10-27）

### 19.1 未松手的代码位置（已读代码，不是推断）

三个事实此前被压成一个：LIBERO 的 `terminated`、服务端的 `can_continue`、策略 chunk 里
还剩什么动作。

| 事实 | 修复前 | 修复后 |
| --- | --- | --- |
| 首次 `native_goal=true` | `engine/openwam_rollout.py` 主循环直接 `break` | 进入 `_goal_decision()`，先看能否继续、关系是否成立 |
| 剩余 chunk（32 步 receding horizon） | 整块丢弃，**包括其中张开夹爪的命令** | Tier 1 在**本 chunk 内**继续执行，不重规划 |
| 服务端能否继续 | `mode="free"` 返回 `can_continue=true`，适配层清 `terminated` | 同上，收尾受同一回合取消标识约束 |
| chunk 耗尽仍没松手 | 无处理，回合记为完成 | Tier 2 `PlacementFinalizer`：清缓冲 → 原地张开 → `+z` 退离 → 3 个新样本验证 |

因此 §18.3 里"原生成立、夹爪仍闭合"的 11 个回合，**不是策略不会松手，而是回合在
策略松手之前被包装层结束掉了**：释放动作在 chunk 尾部，而首个 `native_goal=true`
出现在 chunk 中段。证据是修复后同样的任务在同一 chunk 内的第 5～15 个后目标动作里
测到开度 0.0658→0.0742 m 的张开。

修复后的判定不只看开度：

- `release_verified`：命令张开 + 实测开度 ≥0.05 m **连续 2 帧** + 物体已在手指跨度之外；
- `placement_verified`：释放之后、`sim_action_count` 更大的 3 个新样本上关系仍成立
  （finalizer 路径另外要求退离探针证明"物体不随手臂上升"）；
- 开度闭合本身**不**被当作"仍夹持"（`finalize.policy_release_block` 的注释与
  `test_closed_gripper_after_measured_release_is_not_held` 固定了这一点）。

### 19.2 三层职责与来源标记

| 层 | 谁来释放 | 标记 | 本轮回合数 |
| --- | --- | --- | --- |
| Tier 1 策略后目标窗口 | OpenWAM 自己的 chunk 动作 | `evidence_source="policy_chunk"`, `policy_owned=true` | 9 |
| Tier 2 执行层收尾 | `PlacementFinalizer`（21 个动作：12 张开 + 6 退离 + 3 采样） | `evidence_source="execution_finalizer"`, `policy_owned=false` | 2 |
| 不释放 | 原生目标未成立 | `evidence_source="none"`, `reason_code="not_at_release_candidate"` | 1 |

执行层收尾不是 OpenWAM 生成的，报告里不并入"自主成功率"：
**11 个原生成立的放置回合 = 9 个策略完成 + 2 个执行层辅助**。

### 19.3 三次同清单回归（同一冻结清单，代码版本不同）

清单 `config/placement_rounds_20261027.json`（`frozen=true`，seed 0，任务 1/2/3/8 × 3 回合，
语言顺序 en/zh/en，每回合强制 `/api/load`）。三次运行**都不是选择性重跑**，原因写在表里：

| 运行 | 时间 | 代码状态 | 映射 | 原生 | 释放+放置验证 | 说明 |
| --- | --- | --- | --- | --- | --- | --- |
| A | 12:37–12:46 | 收尾逻辑已就位；`engine_service.py` 在 A 结束后 12:47 又改过（补尝试摘要） | 12/12 | 12/12 | **11/12**（10 策略 + 1 执行层） | task 3 第 1 回合被正确拦下（`not_at_release_candidate`，碗不在抽屉内） |
| B | 13:04–13:14 | A 的同一逻辑 + 尝试摘要 | 12/12 | 12/12 | **12/12**（9 策略 + 3 执行层） | 为"同一最终代码版本"重跑；task 3 第 1 回合这次成立 |
| C | 13:17–13:32 | B + `final_verification` 补 `policy_owned`/`placement_reason` 两个只读字段 | 12/12 | **11/12** | **11/12**（9 策略 + 2 执行层） | 本轮验收数据集；task 2 中文回合失败 |

重跑的原因只有两个：**代码版本一致**和**补两个只读上报字段**，没有为了通过而重跑，
也没有调整任何阈值。三次结果并列保留，方差本身就是结论的一部分（同一冻结清单，
原生 12/12 → 12/12 → 11/12）。

### 19.4 最终回归逐回合（运行 C，验收数据集）

记录：`records/acceptance/placement_rounds_final2_20261027.json`，
原始回合只归档报告引用的 3 份（回合 3、5、11），见 `records/acceptance/placement_rounds_final2_runs/`。

| # | task | 语言 | 初始化指纹 | 首次 native goal（动作） | 后目标策略动作 | 释放来源 | 释放动作数 | 末次开度 (m) | 退离 | 动作 | 耗时 (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | en | `fcedc336` | 83 | 13 | `policy_chunk` | 13 | 0.0759 | skipped | 99 | 29.8 |
| 2 | 1 | zh | `fcedc336` | 84 | 12 | `policy_chunk` | 12 | 0.0759 | skipped | 99 | 29.7 |
| 3 | 1 | en | `fcedc336` | 84 | 12 | `execution_finalizer` | 21 (+12) | 0.0789 | completed | 120 | 31.5 |
| 4 | 2 | en | `fc50366c` | 78 | 12 | `policy_chunk` | 12 | 0.0742 | skipped | 93 | 28.4 |
| 5 | 2 | zh | `fc50366c` | — | 0 | `none` | 0 | 0.0499 | skipped | 1800 | **524.9** |
| 6 | 2 | en | `fc50366c` | 83 | 10 | `policy_chunk` | 10 | 0.0742 | skipped | 96 | 28.6 |
| 7 | 3 | en | `e4ebe088` | 170 | 10 | `policy_chunk` | 10 | 0.0759 | skipped | 183 | 48.6 |
| 8 | 3 | zh | `e4ebe088` | 174 | 5 | `policy_chunk` | 5 | 0.0742 | skipped | 182 | 51.1 |
| 9 | 3 | en | `e4ebe088` | 173 | 12 | `policy_chunk` | 12 | 0.0759 | skipped | 188 | 52.2 |
| 10 | 8 | en | `4060518b` | 81 | 14 | `policy_chunk` | 14 | 0.0742 | skipped | 98 | 29.4 |
| 11 | 8 | zh | `4060518b` | 84 | 12 | `execution_finalizer` | 21 (+12) | 0.0788 | completed | 120 | 31.1 |
| 12 | 8 | en | `4060518b` | 81 | 14 | `policy_chunk` | 14 | 0.0742 | skipped | 98 | 29.4 |

"释放动作数"列：`policy_chunk` 行是后目标窗口实际执行的动作数；`execution_finalizer`
行写成"21 (+12)"，意思是 finalizer 共 21 个动作，其中 12 个是张开夹爪，另有 6 个退离
和 3 个采样。`skipped` 不是失败：策略自己的 chunk 在松开后仍在移动手臂，
把退离单独再造一遍没有意义，记录里如实写"未执行"。

同一任务三回合的初始化指纹一致（`fcedc336` / `fc50366c` / `e4ebe088` / `4060518b`），
说明每回合确实从预冻结的 seed 0 起步。

### 19.5 收尾的额外代价

| 项 | 数值 |
| --- | --- |
| 策略后目标窗口 | 5–15 个动作，≈1.4–4.1 s（墙钟 0.25–0.27 s/动作） |
| 执行层 finalizer | 21 个动作，1.615 / 1.678 s，含 3 个新样本 |
| 回合耗时对比（§18 基线 → C） | 任务 1：26.6/25.4 → 29.8/29.7/31.5；任务 2 成功回合：25.8 → 28.4/28.6；任务 8：24.7/25.1 → 29.4/31.1/29.4；任务 3：51.0/51.5 → 48.6/51.1/52.2 |
| 12 回合总墙钟 | 运行 C 17.0 min（13:17:24–13:34:23，含每回合强制重载与一次 524.9 s 失败回合） |

任务 3 的后目标窗口只有 5–12 个动作，任务 1/8 有 12–14 个：抽屉任务的目标位姿在
更早的观测里就已成立，策略剩余动作少。这说明窗口长度由策略自己决定，不是固定预算。

### 19.6 停止与 reset（真实生命周期）

`scripts/test_lifecycle.py` 在最终代码上**通过**：

```
stop budget: {"reason": "lifecycle_check", "sim_at_request": 31, "sim_at_ack": 31,
              "in_flight_actions": 0, "thread_alive": false, "settle_agreements": 3,
              "cancel_barrier_s": 0.0007, "in_flight_after_cancel": 0,
              "inference_idle": true, "chunk_buffer_cleared": true}
no action issued after stop (in-flight at most 1); sim frozen at 31
model-side buffer cleared on stop (backend_resets +1)
reset: clean phase=ready scene_epoch 2 -> 3
second episode after reset: succeeded
lifecycle: PASS
```

判定不再只是"连续两次读到相同动作计数"：还要求

1. `cancel_barrier_s` 有值（取消屏障完成），
2. `in_flight_after_cancel=0`（取消后没有在途动作），
3. `inference_idle=true` 与 `chunk_buffer_cleared=true`（模型侧缓冲已清），
4. ack 之后 `sim_action_count` 不再变化（31 → 31），
5. reset 后 `scene_epoch` 递增，新回合能成功。

本轮另有一次中断运行时的真实 stop（1800 动作回合进行到 32 步时）：
`sim_at_request=32, sim_at_ack=32, in_flight_actions=0, settle_agreements=3`，
没有出现 ack 之后继续计步。

收尾期间被 stop 的情况**只在单元测试里验证**（`test_finalizer_stop_issues_no_further_action`：
第 3 个动作后置位取消，`execution_status="stopped"`，实际只发出 3 个动作），
真实回合里没有构造"恰好在 finalizer 中断"的时序；这是本节明确的证据缺口。

## 20. task 2 长时间失败：阶段分解与仍未确定的部分

### 20.1 两次真实失败的完整时间线

| | 失败 1（§18 回合 5） | 失败 2（本轮运行 C 回合 5） |
| --- | --- | --- |
| 时间 | 2026-10-27 11:02 | 2026-10-27 13:20 |
| 指令 | put the wine bottle on top of the cabinet | 把酒瓶放到柜子顶上（同任务中文） |
| 初始化指纹 | `fc50366cc74015d5` | `fc50366cc74015d5`（相同） |
| 动作 | 1800（3 × 600） | 1800（3 × 600） |
| 耗时 | 519.9 s | 524.9 s |
| 恢复 | `retry / retry / abort`，11.4 s | `retry / retry / abort`，10.7 s |
| 物体总位移 | 0.000 m（601 帧，从未碰到） | 0.0084 m（第 3 次尝试；前两次见下） |
| 保留的尝试记录 | 仅第 3 次（旧记录覆盖） | 第 1/2 次摘要 + 第 3 次完整记录 |

运行 C 的失败回合因为本轮新增的 `attempt_digests` 而第一次留下了三次尝试的对照：

| 尝试 | 动作 | chunk | 物体位移 | EEF 起点 → 终点 | 夹爪最小/最大开度 (m) | native 命中 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 600 | 19 | **2.89 cm**（z 0.89886 → 0.92159） | `[-0.211,-0.014,1.161]` → `[-0.071,-0.090,1.186]` | 0.0010 / 0.0798 | 0 |
| 2 | 600 | 19 | 0.08 cm | `[-0.071,-0.090,1.186]` → `[-0.263,-0.070,0.963]` | 0.0010 / 0.0797 | 0 |
| 3 | 600 | 19 | 0.84 cm | `[-0.263,-0.070,0.963]` → `[-0.042,-0.169,1.085]` | 0.0010 / 0.0499 | 0 |

关键事实：**第 2 次尝试的起点逐位等于第 1 次的终点，第 3 次的起点逐位等于第 2 次的终点**。
`engine/engine_service.py` 在重试时只做 `inference.reset("retry:…")` 清策略缓冲，
**不 reset 环境、不做归位**（`rehome` 只存在于 `engine/skills.py` 的子技能路径，
`whole_task` 不走它）。所以恢复是"从失败位姿继续"，而且位姿一次比一次远离物体
（EEF 从 x=-0.07 退到 x=-0.263）。

### 20.2 519.9 / 524.9 s 分阶段

| 阶段 | 失败 1 | 失败 2 | 结论 |
| --- | --- | --- | --- |
| 抓取 | 600 动作，物体 0 位移 | 600 动作，物体被抬起 2.89 cm（z +2.27 cm）但没送到柜顶 | 两次都不是"没接近"，而是抓/抬之后没完成放置 |
| 搬运 | 无（物体未离桌） | 抬到 0.9216 m 后停住，EEF 抬到 1.186 m | 未到达 `wooden_cabinet_1_top_side` |
| 定位放置 | 0 | 0 | 三次尝试 0 次 native 命中 |
| 目标判定 | 未触发 | 未触发 | 失败码 `native_goal_not_reached` |
| 恢复 | 3 次 Qwen 调用，10.7 s，retry/retry/abort | 同左 | 恢复只换了策略缓冲，没换状态 |
| 合计 | 1800 动作 / 519.9 s | 1800 动作 / 524.9 s | 单次尝试约 157–175 s |

耗时构成（失败 2）：执行 510.4 s = inference 82.4 + `env_step` 132.9 + observe 92.0 +
snapshot 17.3 + convert 1.3（这几项在时间上重叠，不能相加当墙钟），恢复 10.7 s，
规划 0.8 s，验证 0 s。

### 20.3 "同初始化、同 prompt、结果不同"的归因

先排除模型采样：`scripts/inference_determinism_probe.py` 对同一观测、同一 prompt、
缓冲清空后连续请求，两次 10 维动作 `max_abs_diff = 0.0`（`records/acceptance/inference_determinism.json`，
客户端不传 seed，服务端默认 42）。**模型对相同输入是确定的。**

再看输入是否相同。`scripts/render_determinism_probe.py`（本轮新增，任务 1，seed 0，3 次 reset）：

| 量 | 结果 |
| --- | --- |
| 同一状态连续渲染两次 | **0 像素差**（`max_abs_diff=0`） |
| 3 次 reset 的 EEF / 17 个关节 / 4 个物体 xyz / 夹爪开度 | **完全一致** |
| 3 次 reset 的 7 个物体 site 四元数 | **完全一致**（最大夹角 0.0°） |
| 3 次 reset 的图像两两对比 | 19.8% / 20.6% / 20.9% 像素不同，`max_abs_diff` 172–198，>16 灰阶的 2985–4811 像素 |
| 差异分布 | 集中在机械臂与物体所在区域，右侧空背景 8×8 分块全为 0 |

结论：**同一初始化、同一 seed，几何完全一致，但策略看到的图像不一样**（渲染在同一 reset
内可复现，跨 reset 不可复现）。所以"同初始化结果不同"首先是**输入不可复现**，不能归因于
模型采样；把这件事写成"模型随机"会掩盖真正该修的地方（观测/渲染）。

仍然没确定的是差异的物理来源：快照没有物体四元数（`sim.objects[*]` 只有 `pos`），
只能借 `site_geometry.quat_wxyz` 证明物体姿态一致；剩余的 20% 像素差异出在光照/阴影/
纹理或渲染上下文，本轮没有定位到具体机制。

### 20.4 可解释的"无进展"指标（新增，只报告不强制）

`engine/progress.py`：按固定窗口比较"手动了多少 / 物体动了多少"。

- 窗口 120 动作，物体位移阈值 5 mm，EEF 位移阈值 2 cm（冻结在代码里的默认值）；
- `never_touched`：整段物体位移始终 <5 mm；
- `stall_windows`：手在动、物体不动的窗口，带 `touched` 标记（接触前 = 接近/开抽屉，接触后 = 抓空）。

离线回放（`scripts/progress_probe.py` → `records/acceptance/progress_probe_20261027.json`）：

| 回合 | 物体总位移 | 首次物体运动 | 无进展窗口 | 接触后无进展动作 |
| --- | --- | --- | --- | --- |
| 运行 C 全部成功回合（11） | 0.167–0.378 m | 第 38–119 个动作 | 0 | 0 |
| 运行 C 回合 5（失败） | 0.0084 m | 第 10 个动作 | 4（共 480 动作） | 480 |
| 失败 1 第 3 次尝试 | **0.000 m** | 无（`never_touched`） | 5（共 600 动作） | 0（从未接触） |

**为什么只报告不强制**：任务 3 的成功回合里，物体第一次移动出现在第 **119** 个动作
（native goal 在第 170–174 个动作）。阈值敏感性直接算了：

| 无进展窗口 | 会被提前终止的成功回合 |
| --- | --- |
| 60 / 90 / 100 / 110 / 115 / 119 | **3/3**（任务 3 全部三个成功回合） |
| 120 | 0 |

也就是说，窗口从 119 改成 120，结论就从"杀掉三个成功回合"变成"一个都不杀"。这种
1 个动作的边界不是可用的判据，所以本轮**没有**把它接进执行路径，也没有为通过去调阈值。
真要用它做主早停，需要按任务类型分别标定，并且要报"本来能晚点成功却被终止"的代价。

## 21. 同场景两步组合：可行性检查与下一阶段入口（未执行）

### 21.1 资产事实（从已安装 BDDL 重新解析）

`scripts/composite_matrix.py` 每次运行都重新解析 10 个 BDDL。事实：

| 项 | 值 |
| --- | --- |
| 声明的 fixtures | `main_table`、`wooden_cabinet_1`、`flat_stove_1`、`wine_rack_1`（10/10 相同） |
| 声明的 objects | `akita_black_bowl_1`、`cream_cheese_1`、`wine_bottle_1`、`plate_1`（10/10 相同） |
| 语义 `:init` 区域 | 10/10 相同 |
| `pruned_init` 文件 | **10 个互不相同**（sha256 全不同） |
| 差异 | 只有 `:goal` 和采样出的初始关节状态 |

所以"两个目标是否同场景存在"对任意一对都成立；真正的约束是**一个 episode 只能用一个
`pruned_init`**，第二步必然从"不是自己官方初始状态"起步。

### 21.2 组合小表（判断列是人工标注，附理由）

| 对 | 判断 | 理由 | 第一步如何验证 | 风险 |
| --- | --- | --- | --- | --- |
| `0 → 8` 开中抽屉 → 碗放盘子 | **候选 1** | 第一步不动第二步要用的物体；两步目标无交集 | `drawer_open`（qpos < -0.14） | 暂未发现 |
| `1 → 7` 碗放灶台 → 开灶 | **候选 2** | `cook_region` 在按钮铰链 +x 方向 0.15 m，碗放灶眼不一定挡旋钮；唯一一对描述同一件"做饭"动作 | `object_on_target`（On bowl cook_region） | 碗可能挡在旋钮前方的接近路径上，需一次真实试探 |
| `7 → 1` 开灶 → 碗放灶台 | 允许但语义倒置 | 资产同上，场景没有热模型，物理上不冲突 | `stove_on` | 已开的炉灶在训练分布之外 |
| `6 → 8` 奶酪入碗 → 碗放盘 | 耦合 | 第二步要搬的正是第一步目标关系所依赖的碗 | `object_on_target` | 搬运中奶酪可能滑出，第一步目标会被悄悄破坏 |
| `8 → 4` 碗放盘 → 碗放柜顶 | 同物体二次搬运 | 两步只动碗，是同一技能的重复，不是两个技能的组合 | `object_on_target` | 仍需要释放/退离/新观测，收尾已具备 |
| `3 → 8` 抽屉内取碗 → 碗放盘 | 拒绝 | 第一步把碗放进抽屉，第二步要从抽屉深处抓取 | `object_in_region` | 正是 task 3 已暴露的失败模式（运行 A 回合 7） |
| `0 → 3` 开中抽屉 → 官方复合 | 拒绝 | task 3 本身就是官方"开抽屉+放入"复合，不能当作新增能力 | `drawer_open` | 会把已有能力重复计算 |

选出的后续候选：**`0 → 8`（第一个做）**、**`1 → 7`（第二个做，需先做旋钮可达性试探）**。
`1 → 7` 是唯一一对"两个目标合起来才构成一件有意义的家务"的组合，但它的可行性依赖
"碗在灶眼上时旋钮仍可达"，这一点还没有测量过。

### 21.3 下一阶段的执行入口（固定四条，不实现调度）

1. episode 的 BDDL 用**第二步**任务的，第二步的原生目标继续由 LIBERO 判定；
   第一步用引擎自己的谓词（`drawer_open` / `stove_on` / `object_on_target`）判定，
   **不看** `terminated`。
2. 第一步之后必须走完 §19 的收尾：释放验证 → 退离 → 新观测 → 清策略缓冲，第二步才允许
   请求策略，避免用第一步的旧观测生成第二步动作。
3. 组合目标分开记账（`step_1_verified`、`step_2_native`、`combo_verified`），
   不允许用第二步的 `terminated` 冒充整体完成。
4. 只用一个 `pruned_init`，并把"第二步从非官方初始状态起步"写进结果，不换 init 重试。

不做的部分：不 reset、不重载场景、不改 BDDL、不扩展路由输出、不实现通用组合调度。
旧 `openwam_harness/sequential` 里的 `task_scope="combo"` 是另一套执行栈的开关，
`task_catalog.single_task_block` 仍会拒绝多步计划，本轮没有把它接到本引擎上。

## 22. 产物与提交

| 产物 | 路径 |
| --- | --- |
| 收尾实现 | `engine/finalize.py`、`engine/grasp_geometry.py`、`engine/openwam_rollout.py`（后目标窗口）、`engine/engine_service.py`（编排） |
| 无进展指标 | `engine/progress.py`、`scripts/progress_probe.py` → `records/acceptance/progress_probe_20261027.json` |
| 确定性测量 | `scripts/inference_determinism_probe.py` → `records/acceptance/inference_determinism.json`；`scripts/render_determinism_probe.py` → `records/acceptance/render_determinism.json` |
| 组合矩阵 | `scripts/composite_matrix.py` → `records/acceptance/composite_matrix_20261027.json` |
| 三次回归清单 | `records/acceptance/placement_rounds_20261027.json`（A）、`…_final_…json`（B）、`…_final2_…json`（C） |
| 最终回归原始回合 | `records/acceptance/placement_rounds_final2_runs/`（只保留报告引用的 3 份：回合 3、11 执行层收尾，回合 5 失败；其余 9 份只留清单里的摘要行，≈2.1 MB） |
| 单测 | `tests/test_placement_finalize.py`（完成/释放/finalizer/窗口 16 条）、`tests/test_progress.py`（6 条）、`tests/test_attempt_digest.py`；`pytest tests` 共 **69 passed** |

证据缺口（如实列出）：

1. 运行 A/B 的原始回合记录在运行 C 前被 `records/runs` 的轮转清掉了，只剩冻结清单里的
   逐回合摘要字段（`records_rotated=true` 已写在清单里）；观测、采样与动作轨迹不可追回。
2. 收尾期间被 stop 只有单测覆盖，没有真实回合的时序样本。
3. `1 → 7` 组合的旋钮可达性没有测量。
4. task 2 失败的渲染差异来源（光照/阴影/纹理/渲染上下文）未定位。

## 23. 同场景两步组合 `0 → 8`：六回合冻结验收（真实执行，2026-10-27/28）

> 结论先行：**路由 6/6、第一步 6/6、交接 6/6，第二步原生目标 1/6，双目标同真 0/6。**
> `0 → 8` 不是一条已通过的能力；它是一条**执行机制已经跑通、物理上还没成立**的组合。
> 失败没有被重跑覆盖，也没有被包装成成功率。

### 23.1 协议

| 项 | 值 |
| --- | --- |
| 分支 | `feature/openwam-composite-0-8-20261027`（自 `8becbe2`） |
| 冻结清单 | `config/composite_rounds_20261027.json`（sha256 固定，运行前写入） |
| 回合 | 3 个初始化索引（0/1/2）×（英文/中文）= **6 回合**，不重试、不换 seed |
| 场景锚 | task 8 的 BDDL + 每个索引自己的 `pruned_init`；episode 内不 reset、不重载、不改 BDDL、不传送物体 |
| 第一步 | 向 OpenWAM 发 task 0 官方指令 `open the middle drawer of the cabinet`，由引擎 `drawer_open` 谓词判定（`qpos < -0.14`，连续 2 个新样本） |
| 第二步 | 向 OpenWAM 发 task 8 官方指令 `put the bowl on the plate`，由 LIBERO 原生目标判定；放置沿用 §19 的释放/稳定性验证 |
| Agent 模型 | `Qwen3.5-9B`（语义路由，非 thinking，`max_tokens=24`） |
| 动作模型 | `OpenWAM-Alpha-Sim-LIBERO`，`checkpoint_step_10690`，`action_dim=10`，`chunk=32` |
| 预算 | 第一步 240 动作、第二步 600 动作；handoff 有界（≤60 动作） |
| 记录 | `records/acceptance/composite_rounds_20261027.json` + `..._runs/`（6 份原始回合，逐份 1.8 MB） |

### 23.2 总表

| 回合 | 语言 | seed | 路由 | 第一步 动作 / s | 交接 动作 | 姿态 起点→终点 | 第二步 动作 / s | 总动作 | 总时长 s | 失败码 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `c08-s0-en` | en | 0 | ✅ | 121 / 31.6 | 36 | 103.5° → 2.8° | 600 / 157.3 | 757 | 195.7 | `composite_step_2_failed` |
| `c08-s0-zh` | zh | 0 | ✅ | 121 / 31.2 | 36 | 103.6° → 2.8° | 600 / 157.7 | 757 | 195.7 | `composite_step_2_failed` |
| `c08-s1-en` | en | 1 | ✅ | 115 / 30.2 | 32 | 105.5° → 2.7° | 600 / 156.0 | 747 | 193.0 | `composite_step_2_failed` |
| `c08-s1-zh` | zh | 1 | ✅ | 118 / 30.9 | 37 | 101.4° → 2.8° | 600 / 156.6 | 755 | 194.2 | `composite_step_2_failed` |
| `c08-s2-en` | en | 2 | ✅ | 114 / 30.0 | 45 | 105.4° → 1.1° | 479 / 126.2 | 641 | 164.2 | `composite_final_verification_failed` |
| `c08-s2-zh` | zh | 2 | ✅ | 123 / 32.3 | 31 | 102.3° → 2.6° | 600 / 156.7 | 754 | 195.6 | `composite_step_2_failed` |

汇总（`scripts/summarize_composite_rounds.py` 原样输出）：

```json
{"rounds": 6, "mapping_ok": 6, "step_1_ok": 6, "step_2_native": 1, "combo_verified": 0,
 "scene_epoch_continuous": 6, "handoff_ready": 6}
```

### 23.3 不变量核对（逐回合，全部通过）

| 回合 | `scene_epoch` | 动作计数 起→止 | 步间 reset | 步间 reload | BDDL 变更 | 物体传送 | 动作计数连续 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `c08-s0-en` | 2 → 2 | 0 → 757 | 0 | 0 | false | false | ✅ |
| `c08-s0-zh` | 4 → 4 | 0 → 757 | 0 | 0 | false | false | ✅ |
| `c08-s1-en` | 2 → 2 | 0 → 747 | 0 | 0 | false | false | ✅ |
| `c08-s1-zh` | 4 → 4 | 0 → 755 | 0 | 0 | false | false | ✅ |
| `c08-s2-en` | 2 → 2 | 0 → 641 | 0 | 0 | false | false | ✅ |
| `c08-s2-zh` | 4 → 4 | 0 → 754 | 0 | 0 | false | false | ✅ |

"动作计数连续"的判据：`第一步动作 + 交接退离动作 + 第二步动作 == episode_sim_actions`，
6/6 完全相等（例如 `c08-s0-en`：121 + 36 + 600 = 757）。

**"native 不得终止第一步"这条要求有了一次真实反例级证据**：`c08-s2-en` 的第一步里，
加载场景自己的原生标志在滚动过程中真的翻到了 `true`（`native_current=true`，
`native_ever=false`，即第二步目标的原子在此期间被某个瞬间满足过），第一步**没有**因此提前
结束，而是继续跑到 `drawer_open`（`qpos=-0.1484`）才停。记录里这条标志和"第一步的目标"
是分开的字段。

### 23.4 第一步与交接：机制通过

- 第一步 6/6 在 114–123 动作、30.0–32.3 s 内达到 `drawer_open`（`qpos` −0.1457…−0.1559），
  没有任何一回合用到 240 的预算上限，也没有任何一回合因场景原生标志提前结束。
- 交接 6/6 `status=ready`：第一步目标复验通过 → 抽屉/开关步**不套放置 finalizer**
  （`release.required=false`，理由写进记录）→ 有界退离 31–45 动作、
  **姿态从 101.4–105.5° 收敛到 1.1–2.8°**（不改这一步就是第二步"看着抽屉内壁抓碗"）→
  `scene_epoch`/`scene_generation` 不变 → 抽屉 `qpos` 在退离后仍在阈值内
  （−0.1457…−0.1559）→ `inference.reset("handoff:…")` 返回新 token → 新 `observation_id`
  （例如 `…:2:159:163` 相对交接前的 `…:2:121:124`）。
- 第二步第一块动作由新 token 生成，第一步的 receding-horizon 缓冲没有被第二步消费：
  6/6 回合 `chunks_used` 第一步 4 块、第二步 15–19 块，两段的行首请求都带各自的 `step_id`。

### 23.5 第二步失败：两类，都不粉饰

**A. 5 回合没到原生目标**（`native_goal_not_reached`，用满 600 动作预算）：

```json
{"ok": false, "method": "native_libero_goal", "native_goal": false, "samples": 601,
 "sim_actions": 600, "limit": "max_steps", "reason_code": "native_goal_not_reached",
 "prompt": "put the bowl on the plate"}
```

不是"差一点"，而是 30 s 内一次都没有出现 `On(bowl, plate)`。同一索引的单步对照
（§26）在同一台机器、同一策略上 26.7–29.3 s 就完成，所以这不是"这个索引太难"。

**B. 1 回合到了原生目标，但终局复验判定第一步目标已被破坏**（`c08-s2-en`）：

| 检查点 | 抽屉关节 `wooden_cabinet_1_middle_level` | 阈值 |
| --- | --- | --- |
| 第一步判定时 | −0.1484 | < −0.14 ✅ |
| 退离后 / 第二步开始 | **−0.1547** | < −0.14 ✅ |
| 第二步完成、终局复验 | **−0.0999** | < −0.14 ❌ |

第二步确实完成了它自己的任务（479 动作、原生目标成立、释放来源 `policy_chunk`、
`placement_verified=true`），但它在过程中把抽屉**推回了约 5.5 cm**，
于是"中间抽屉打开 ∧ 碗在盘子上并完成释放"的合取不成立。引擎报告
`composite_final_verification_failed` / `step_goal_regressed`，`combo_verified=false`。
这不是判据过严，是**组合的两个目标在物理上互相破坏**。

### 23.6 失败回合补测：破坏第一步状态的证据（修复后单回合，2026-10-28）

冻结的 6 回合是在"步骤失败证据还不含前序步骤状态"的代码上跑的，所以 5 个
`composite_step_2_failed` 回合**没有**记录失败瞬间的抽屉 `qpos`（这是本轮的证据缺口，见 §29）。
补上这段只读证据后，用同一句英文指令、同一 seed 0 跑了一回合
`config/composite_postfix_validation_20261027.json` → `records/acceptance/composite_postfix_validation_20261027b.json`：

| 项 | 值 |
| --- | --- |
| 路由 | `["open_middle_drawer", "put_bowl_on_plate"]`，`mapping_ok=true` |
| 第一步 | 通过（`drawer_open`） |
| 第二步 | 600 动作预算用尽，`native_goal_not_reached` |
| **失败瞬间的抽屉 `qpos`** | **−0.1167**（阈值 −0.14，已回到"没开够"） |
| 第二步动作数 | 763 总计 |

也就是说，两回合（`c08-s2-en` 的 −0.0999、本回合的 −0.1167）里凡是**在第二步结束时测过**
抽屉状态的，抽屉都被推回了 3.4–5.5 cm。第一步的开抽屉状态在这个场景布局下**撑不过第二步**。

严格说明这条证据的边界：它证明"第二步执行中抽屉被推回"，**没有**分离
"张开的抽屉挡住机械臂 → 第二步失败"与"机械臂在第二步里把抽屉碰回去 → 第一步失败"
这两个方向。要分离需要接触力或逐帧遮挡证据，本轮没有采集，所以不写成因果结论。

补充一条操作事实：这轮补测之前还有一次失败尝试，清单用的是临时文件、字段写成
`expected_choice`（脚本的复合分支读 `expected_keys`），于是报告里出现
`mapping_ok=false` 而实际 `semantic_choice` 与预期完全一致。那份记录
（`composite_postfix_validation_20261027.json`）**保留未改**，没有手工翻转任何判定；
正确做法是把清单写成 `config/composite_postfix_validation_20261027.json` 再跑一次，
那份才是本条引用的 `..._b.json`。

### 23.7 时延分解（引擎自报，秒）

| 回合 | 规划（路由） | 执行 | 交接 | 推理 Σ | `env.step` Σ | 观测 Σ | 快照 Σ | 总 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `c08-s0-en` | 0.824 | 188.9 | 2.762 | 33.6 | 48.3 | 31.5 | 7.1 | 195.7 |
| `c08-s0-zh` | 0.826 | 188.9 | 2.756 | 32.7 | 49.4 | 31.5 | 7.0 | 195.7 |
| `c08-s1-en` | 0.802 | 186.3 | 2.549 | 32.1 | 48.8 | 31.3 | 6.8 | 193.0 |
| `c08-s1-zh` | 0.833 | 187.5 | 2.938 | 32.0 | 49.5 | 31.5 | 6.9 | 194.2 |
| `c08-s2-en` | 0.838 | 156.1 | 3.613 | 26.9 | 41.5 | 25.9 | 5.7 | 164.2 |
| `c08-s2-zh` | 0.848 | 189.0 | 2.499 | 32.3 | 50.2 | 31.7 | 7.0 | 195.6 |

读法：总时长里 96–97% 是"两步的策略执行"，规划只占 0.8 s 左右，交接 2.5–3.6 s。
这组数字**不能**当作组合能用的证据——它只是把 6 个失败回合的时间花在哪里讲清楚。

## 24. 组合路由：Qwen 只选 key，代码查表（开发集 17 + 冻结验收集 16）

### 24.1 契约

模型输出**有序目录 key 列表**（`["open_middle_drawer", "put_bowl_on_plate"]`）或
`clarify` / `unsupported`；官方指令、task_id、BDDL 目标全部由代码查 `config/tasks.json`，
提示词里没有它们（也不含第一步的指令或结果）。识别与可执行是两件事：

| 情形 | 出口 | `reason_code` |
| --- | --- | --- |
| 目录里有、且是当前唯一放行的有序对 | 组装两步计划 | — |
| 目录里有、但不是放行的对（含倒序、其它组合） | `unsupported` | `unsupported_combo`（附放行对） |
| 目录里没有（关系/实体不存在） | `unsupported` | `key_not_in_catalog` / `step_goal_unsupported` |
| 超过两步 | `unsupported` | `step_limit_exceeded` |
| 缺槽（抽屉层级、代词、冠词不明确） | `clarify` | — |
| 说的是单任务 | 单任务路径（与既有行为一致，`task_scope="single"`） | — |

### 24.2 开发集（`config/composite_routes_20261027.json`，先固定）

| 指标 | 值 |
| --- | --- |
| 用例 | 17（中英各半：正序/同义/倒序/缺层级/仅单步/不支持对/三步/物体冲突/代词歧义/顺序歧义） |
| 通过 | **17 / 17** |
| 总时延 中位 / p95 / 最大 | 0.807 / 1.038 / 1.040 s |
| TTFT 中位 | 0.176 s |
| 输出 token 中位 | 14 |
| 请求数 / 修复次数 | 17 / 0 |

### 24.3 冻结验收集（`config/composite_routes_frozen_20261027.json`，开发集定稿后才写）

| 用例 | 语言 | 结果 | 总时延 s | TTFT s | token |
| --- | --- | --- | --- | --- | --- |
| `before-order` | en / zh | `ok`（两步） | 0.761 / 0.810 | 0.136 / 0.180 | 14 / 14 |
| `two-sentences` | en / zh | `ok` | 0.807 / 0.807 | 0.177 / 0.176 | 14 / 14 |
| `and-order` | en / zh | `ok` | 0.809 / 0.823 | 0.177 / 0.182 | 14 / 14 |
| `reversed-unsupported` | en / zh | `unsupported` | 0.825 / 0.816 | 0.177 / 0.179 | 14 / 14 |
| `other-pair` | en / zh | `unsupported` | 0.953 / 0.958 | 0.177 / 0.181 | 17 / 17 |
| `single-two-verbs` | en / zh | 单任务（`task_mismatch`） | 0.724 / 0.724 | 0.179 / 0.181 | 12 / 12 |
| `ambiguous-article` | en / zh | `needs_clarification` | 0.494 / 0.499 | 0.182 / 0.182 | 7 / 7 |
| `unsupported-relation` | en | `unsupported` | 0.444 | 0.178 | 6 |
| `three-actions` | zh | `unsupported` | 0.907 | 0.178 | 16 |

| 指标 | 值 |
| --- | --- |
| 通过 | **16 / 16** |
| 总时延 中位 / p95 / 最大 | 0.808 / 0.953 / 0.958 s |
| TTFT 中位 | 0.178 s |
| 输出 token 中位 | 14 |
| 请求数 / 修复次数 | 16 / 0 |

两组都达到本轮"median ≤1.5 s、p95 ≤2 s"的目标（实测 median ≈0.81 s、p95 ≈0.95–1.04 s），
而且是在**不牺牲判定正确性**的前提下达到的：倒序、其它对、三步、缺槽全部按规则拒绝或追问，
没有为了"能跑"把不支持的对降级成单任务。

## 25. 真实 stop 测试：四个同步点，四个都停住

同步不靠 `sleep` 猜时序：用例都等引擎**自己发布**的阶段再触发。

| 用例 | 触发条件（引擎发布） | 触发时动作数 | stop ack | ack 后是否继续动作 | 第二步是否开始 | 记录状态 |
| --- | --- | --- | --- | --- | --- | --- |
| `stop-step1` | `phase=running` 且 detail 含 `step 1/2`、动作 ≥25 | 25 | ok | 否 | 否 | `stopped` |
| `stop-handoff` | `phase=handoff`（第一步已判定、第二步未开始） | 119 | ok | 否 | 否 | `stopped` |
| `stop-step2` | `phase=running` 且 detail 含 `step 2/2`、动作 ≥20 | 160 | ok | 否 | 是（中断在第二步里） | `stopped` |
| `stop-finalizer` | 第一条 `finalizer_step` 日志（finalizer 正在执行自己的动作） | 97 | ok | 否 | 否（单任务对照） | `stopped` |

`stop-finalizer` 为什么用单任务：组合第二步的 finalizer 只有在碗成为"待释放候选"时才启动，
而本轮 5/6 回合第二步根本没到那一步，没有可同步的 finalizer 窗口。用它同一段
`PlacementFinalizer` 代码路径（task 4，seed 0）才能真的**在释放中途**打断，
报告里的这一条因此明确标注为"单任务对照，不是组合回合"。

补齐记录（中断瞬间的状态）：

```json
{"execution_status": "stopped", "reason_code": "user_stop", "attempted": true,
 "extra_actions": 2, "action_rows": [{"phase": "release", "n": 1, "gripper_opening_m": 0.0581},
                                    {"phase": "release", "n": 2, "gripper_opening_m": 0.0659}]}
```

**这两个 stop 用例各暴露了一个真实缺陷，都已修复并复测**（修复只动停止路径，不动组合判定）：

1. **`Stopped` 被 finalizer 吞掉**：`PlacementFinalizer.run` 内部捕获 `Stopped` 并写进
   placement 块，外层却没重新抛出，结果是"用户点了停止、释放被打断，回合却记为
   `completed` / `succeeded`"。修复前实测记录：`execution_status=completed`、
   `placement.execution_status=unknown`、`reason_code=user_stop`（自相矛盾）。
   修复后：`execution_status=stopped`、`user_goal_status=not_met`。
2. **中断后的 placement 证据被丢弃**：停止发生在步骤返回之前，步骤行从未生成，
   记录里于是写成 `reason_code=no_whole_task_step` / `attempted=false`，
   把"释放做了 2 个动作被打断"抹成"没尝试"。修复后保留
   `attempted=true`、`extra_actions=2` 与逐动作行。

## 26. 单步对照：把"第一步失败 / 交接失败 / 第二步分布变化"分开

同一策略、同一 3 个索引、同一句官方任务（task 8 单任务，无抽屉、无交接）：

| 回合 | 路由 | 原生目标 | 用户目标 | 动作 | 时长 s | 释放来源 | 推理 Σ | `env.step` Σ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `single-s0-en` | ✅ | ✅ | met | 98 | 29.3 | `policy_chunk` | 4.3 | 6.8 |
| `single-s1-en` | ✅ | ✅ | met | 88 | 26.7 | `policy_chunk` | 4.0 | 6.0 |
| `single-s2-en` | ✅ | ✅ | met | 92 | 27.6 | `policy_chunk` | 4.1 | 6.3 |

对照结论（只写对照能支撑的部分）：

- 三个索引上 task 8 **本身**是可完成的，用 88–98 动作、26.7–29.3 s，不需要执行层 finalizer。
  所以 5/6 的第二步失败**不是**"这个索引本来就做不出来"。
- 单步 `env.step` Σ 只有 6.0–6.8 s，组合第二步 600 动作回合 41.5–50.2 s：
  组合第二步的动作花在更长的轨迹上（更多块、更多次观测），不是单步同一路径的重复。
- 差异只可能来自"第二步从一个不是官方 init 的场景起步"（抽屉开着、机械臂带着第一步的历史）
  与"抽屉状态被第二步破坏"这两项组合固有因素。二者本轮**没有**被分离。

## 27. 十任务冒烟：组合代码上重跑，判定不变

同一份冻结清单 `config/smoke_10tasks_20261027.json`（每任务 1 回合、seed 0、自然语言指令），
在组合分支代码上重跑一遍，与 `8becbe2` 基线对比：

| task | 指令（冻结英文表述） | 路由 | native | 用户目标 | 释放来源 | 动作 | 时长 s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | pull open the middle drawer | ✅ | ✅ | met | 不适用（抽屉） | 125 | 36.3 |
| 1 | place the bowl on the stove | ✅ | ✅ | met | `policy_chunk` | 92 | 28.0 |
| 2 | get the wine bottle onto the cabinet top | ✅ | ✅ | met | `policy_chunk` | 94 | 28.5 |
| 3 | open the top drawer and place the bowl inside | ✅ | ✅ | met | `policy_chunk` | 183 | 51.3 |
| 4 | put the bowl on the cabinet's top surface | ✅ | ✅ | met | `policy_chunk` | 98 | 29.5 |
| 5 | push the plate to the front of the stove | ✅ | ✅ | met | 不适用（推物） | 142 | 41.2 |
| 6 | put the cream cheese inside the bowl | ✅ | ✅ | met | `policy_chunk` | 98 | 29.3 |
| 7 | switch on the stove | ✅ | ✅ | met | 不适用（开关） | 78 | 24.0 |
| 8 | set the bowl on the plate | ✅ | ✅ | met | `policy_chunk` | 99 | 29.5 |
| 9 | place the wine bottle up on the wine rack | ✅ | ✅ | **unknown** | 未测到（`not_at_release_candidate`） | 128 | 37.3 |

| 口径 | 基线 `8becbe2`（`smoke_10tasks_20261027.json`） | 组合分支（`smoke_10tasks_post_composite_20261027.json`） |
| --- | --- | --- |
| 路由正确 | 10 / 10 | 10 / 10 |
| 原生目标 | 10 / 10 | 10 / 10 |
| 用户目标 met | 9 / 10 | 9 / 10 |
| 用户目标 unknown | 1（task 9） | 1（task 9） |
| 回合失败 | 0 | 0 |

**task 9 的 `unknown` 是测量缺口，不是成功也不是失败**：原生目标与终局复验都成立，
但酒架在桌面上方，`relation + release_height` 前置条件不成立，释放无法测量，
于是 `evidence_source=none`、`placement.execution_status=unknown`。这条口径没有放宽。

冒烟只是"1 回合 × 10 任务"，**不构成稳定成功率**：单回合无法区分能力与运气，
本轮也没有用重跑去换一个更好看的数字。

## 28. `1 → 7` 旋钮可达性：只做了几何测量，动态可达性**未测**

`0 → 8` 没有通过，所以按本轮边界 `1 → 7` 不进入放行清单，也不跑完整回合。
只做了一个**只读**几何探针（`scripts/knob_reach_probe.py` →
`records/acceptance/knob_reach_probe_20261027.json`）：加载官方 task 7 场景（seed 0）后
只读状态，`env_steps=0`、`policy_calls=0`、动作计数与 `observation_id` 前后不变。

实测（世界坐标，来自实时仿真 + 安装的碰撞模型）：

| 量 | 值 |
| --- | --- |
| 灶台基座（旋钮铰点） | (−0.4067, 0.2145, 0.905) |
| 灶眼 `cook_region` 中心 | (−0.2567, 0.2145, 0.905)（= 基座 + (0.15, 0, 0)） |
| 灶眼半径（半边长） | 0.075 m |
| 旋钮外缘到灶眼近边的水平距离 | **0.0317 m** |
| 碗心在灶眼中心时，旋钮尖端到碗沿 | **0.0527 m** |
| 碗被推到灶眼最近的边缘时，旋钮尖端到碗沿（最坏） | **0.0223 m** |
| 高度区间 | 旋钮 0.9054–0.9550，碗 0.9066–0.9571（**重叠**） |

结论（严格限定在几何）：

- 没有直接穿插：即使把碗推到最靠近旋钮的灶眼边缘，也有约 2.2 cm 净空。
- 但两者的高度区间完全重叠，最坏净空只有 2.2 cm，**手指接触旋钮时可能扫到碗沿**。
- **动态可达性未测**：策略能不能在碗在场时按下旋钮、按下过程中会不会把碗推走，
  本轮没有执行，所以 `1 → 7` 继续留在"未验证、不放行"。
- 探针代码只在报告里提供几何，`verdict.dynamic_reachability="not_tested"` 是写死在产物里的。

## 29. 本轮结论与缺口（如实列）

### 29.1 结论

1. **执行机制通过**：一个 episode、一个 `scene_epoch`、动作计数连续、无 reset/reload/BDDL 变更/
   物体传送；两步各带自己的 `step_id`、各自的官方指令、各自的判定；交接把第一步目标复验、
   按类型免掉放置 finalizer、位姿闭环退离（姿态 105°→<3°）、清策略缓冲、换新观测都做到了。
2. **`0 → 8` 不是一个通过的能力**：第二步原生目标 1/6、双目标同真 0/6。
3. **失败位置已经定位到第二步**，而且有两条独立证据：(a) 同一策略在同一 3 个索引上单步 task 8
   全成；(b) 在测过抽屉状态的两回合里，第二步把第一步的开抽屉状态推回了 3.4–5.5 cm。
4. **自然语言组合规划通过**：开发集 17/17、冻结验收集 16/16，median ≈0.81 s、p95 ≈0.95–1.04 s，
   0 次修复重试；倒序/其它组合/三步/缺槽按规则拒绝或追问。
5. **真实 stop 覆盖通过**：第一步、步间、第二步、finalizer 中途四个同步点都停住且不泄漏动作；
   顺带修掉两个"停止被记账错"的缺陷。

### 29.2 缺口与未证实项

| # | 缺口 | 影响 | 为什么没做 |
| --- | --- | --- | --- |
| 1 | 第二步失败与抽屉状态的**因果方向**未分离 | 不知道是"抽屉挡臂"还是"臂撞抽屉" | 需要接触力/逐帧遮挡证据，本轮未采集 |
| 2 | 冻结 6 回合跑在 3 项"只改证据/只改停止路径"的修复之前 | 5 个失败回合缺失败瞬间的抽屉 `qpos` | 修复后补跑 1 回合（§23.6）复现同一失败，未重跑整组（重跑=覆盖证据） |
| 3 | `1 → 7` 动态可达性未测 | 该组合仍不可宣称可执行 | 边界要求 `0 → 8` 先通过，而它没通过 |
| 4 | task 9（酒架）用户目标 `unknown` | 十任务里 1 项无法判定 | 目标在桌面上方，释放高度前置不成立 |
| 5 | task 2 长时间失败的渲染差异根因未定位 | 未归因 | 与本节无关，沿用 §20 的结论 |
| 6 | 组合只有 6 回合、3 个索引 | 方差未知 | 本轮为冻结验收，不重试不扩样 |
| 7 | 无进展强制终止规则仍未接入 | 失败回合会跑满预算（600 动作） | 缺有效阈值，按边界不接入 |

### 29.3 本轮代码改动与证据的对应关系

| 改动 | 文件 | 为什么不能改变组合判定 |
| --- | --- | --- |
| 步骤失败时复验前序步骤目标（只读） | `engine/engine_service.py` | 只在失败分支追加证据字段，不改控制流 |
| finalizer 内 `Stopped` 重新抛出 | `engine/engine_service.py` | 只影响被 stop 的回合，正常回合不触发 |
| 停止回合保留 placement 块 | `engine/engine_service.py` | 只在 placement 为空（停止/失败）时兜底 |
| `eef_quat` 只读快照字段 | `adapters/env_runtime/episode_service.py` | 只读，不参与动作生成 |
| stop 测试脚本：第二步同步点、finalizer 提前退出、按任务重定向 | `scripts/composite_stop_tests.py` | 只影响测试脚本自身 |
| 旋钮几何探针 | `scripts/knob_reach_probe.py` | 只读探针，不接入执行 |

## 30. 本轮产物

| 产物 | 路径 |
| --- | --- |
| 组合契约 | `common/composite_plan.py` |
| 交接实现 | `engine/composite.py` |
| 执行编排 | `engine/engine_service.py`、`engine/openwam_rollout.py` |
| 组合路由 | `agent/semantic_router.py`、`agent/agent_service.py`、`config/engine.json` |
| 组合清单 | `config/composite_rounds_20261027.json`（6 回合）、`config/composite_control_single_20261027.json`（单步对照）、`config/composite_postfix_validation_20261027.json`（修复后单回合） |
| 路由集合 | `config/composite_routes_20261027.json`（开发）、`config/composite_routes_frozen_20261027.json`（冻结） |
| 组合回合 | `records/acceptance/composite_rounds_20261027.json` + `..._runs/`（6 份原始回合）、`composite_rounds_20261027.md` |
| 单步对照 | `records/acceptance/composite_control_single_20261027.json` + `..._runs/` |
| stop 测试 | `records/acceptance/composite_stop_tests_20261027.json`（三步）、`composite_stop_tests_finalizer_20261027.json`（finalizer 中途） |
| 路由评测 | `records/offline/composite_routes_dev_5.json`（17/17）、`composite_routes_test_frozen_2.json`（16/16） |
| 冒烟 | `records/acceptance/smoke_10tasks_20261027.json`（基线）、`smoke_10tasks_post_composite_20261027.json`（组合代码） |
| 旋钮探针 | `records/acceptance/knob_reach_probe_20261027.json` |
| 证据归档 | `records/acceptance/archive_20261027_8becbe2_before_composite/`（本轮开始前的验收清单与仍存在的原始记录；轮转已丢的只留清单摘要并标注，不重建） |
| 单测 | `tests/test_composite_plan.py`、`tests/test_composite_handoff.py`；`pytest tests` 共 **105 passed** |
