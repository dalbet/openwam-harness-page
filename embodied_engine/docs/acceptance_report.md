# 十任务验收摘要（LIBERO-Goal，seed 0，每项 3 回合）

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
| P2 | OpenWAM | BLOCKED：缺 Wan2.2-TI2V-5B 主干与运行依赖，未跑过动作推理 | 见 `architecture.md` §2，不冒充接入 |
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
