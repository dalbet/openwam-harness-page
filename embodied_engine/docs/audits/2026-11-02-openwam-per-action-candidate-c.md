# 2026-11-02 逐动作固定开销优化（候选 C）— OpenWAM / libero_goal task1

**状态：`PER_ACTION_OPT_PRELIM_PASSED`**

本轮把优化目标从"模型生成内核"移回"逐动作固定开销"，以候选 B（`torch.compile`
`reduce-overhead` + `TORCHINDUCTOR_EMULATE_PRECISION_CASTS=1` + CUDA Graph replay）
为固定基础，只增加一项有依赖证明的改动：**缓存出队步骤不构造、不编码、不传输、
不预处理模型图像**。20 个正式回合（10 初始化 × B/C）结果为 10/10 双方成功，配对
墙钟中位 **−10.245 s（−35.4 %）**，每动作缓存出队开销 **0.1146 s → 0.0016 s**，
生成时间保持不变（0.9658 s → 0.9638 s/回合）。

本轮是**引擎数据路径优化**，不计入 FlashRT 内核收益，也不改变生产服务。

---

## 1. 范围与不变项

| 项 | 值 |
|---|---|
| 任务 | `libero_goal` / task 1 / `put_the_bowl_on_the_stove` |
| 指令 | `put the bowl on the stove` |
| 检查点 | `OpenWAM-Alpha-Sim-LIBERO` / `checkpoint_step_10690.safetensors` |
| 精度 | bf16 |
| compile | `compile.enabled=true`，`self_attn.torch_mode=reduce-overhead`，`wan_blocks.enabled=false` |
| 精度模拟 | `TORCHINDUCTOR_EMULATE_PRECISION_CASTS=1` |
| denoise / horizon | `denoise_steps=10`，`inference_horizon=32`，执行 32/32 |
| 控制契约 | 20 Hz（1 个 sim action = 50 ms 仿真时间），预算 600 步 |
| 判定 | 原生 LIBERO goal 判定 + 释放窗口，未改动 |
| 计划 | 隔离计划 stub 固定计划，planner 不在环 |
| 其它 | 不批量下发动作、不引入异步预取、不调整观测反馈时序 |

未扩展组合任务，未重复完整静态审计，未做多任务大规模测试。

---

## 2. 现场保护与进程管理

- 独立栈：GPU0，端口 `14531`（PolicyServer）/ `14581`（env）/ `14611`（inference）/
  `14811`（engine）/ `14711`（plan stub），全部在 `/home/host/flashrt-exp/per_action`
  下，未触碰生产配置与生产端口。
- 未使用 `pkill -f`、`killall` 或任何模糊进程匹配。所有启停由 `tools/stack.sh` 与
  `tools/arm_switch.py` 执行：每个服务 `setsid` 起在自己的进程组，落账
  `run_id / PID / PGID / 启动时间 / 完整 cmdline / 端口` 到
  `raw/process_ledger.jsonl` 与 `raw/arm_switch_ledger.jsonl`。
- 停止前三重核验：`/proc/<pid>/cmdline` 必须同时命中 `per_action` 与本服务端口，否则拒绝停止。
- 本轮 run_id 序列：`PA-B1 → PA-C1 → PA-B2 → PA-C2 → PA-B3 → PA-C3 → PA-B4 → PA-C4 →
  PA-B5 → PA-C5 → PA-B6 → PA-C6`，每次切换都在 ledger 里留有前后两代 PID 与两个开关
  （`OPENWAM_CACHED_STEP_FASTPATH` / `PER_ACTION_SKIP_CACHED_IMAGES`）的实际取值。
- 生产服务全程未停止、未重置、未重启。上一轮误停事故在本报告中统一表述为
  **"引擎误停后恢复，模型后端未重启"**，本轮未新增任何此类事件。

---

## 3. 运行阶段 fallback 直接证据

钩子为进程内 CUDA Graph 追踪（`tools/cg_trace.py`），通过 `phase_ctl.json` /
`phase_ack.json` 做**阶段屏障**，在阶段边界读精确计数，而不是采样。计数钩子在
`tools/test_cg_trace.py` 中先用 PyTorch 2.7.1 的 `CUDAGraph` 树做单元验证。

阶段划分：**启动 → 编译预热（warmup 回合）→ 正式回合 → 收尾**。

每个 trace 目录的阶段计数（`replays` 为累计值，下表为阶段内增量）：

| 阶段 | 计数 | B（6 个 trace） | C（6 个 trace） |
|---|---|---|---|
| 编译预热（每 trace 1 个 warmup 回合） | `replays` 增量 | 10（首个 trace 为 12） | 10 |
| 编译预热 | `run_eager` 增量 | **1** | **1** |
| 编译预热 | `warmups` / `records` | 1 / 1 | 1 / 1 |
| 正式回合（每回合） | `replays` 增量 | **12**（3 次生成 × 4 个图区域） | **12** |
| 正式回合（每回合） | `run_eager` 增量 | **0** | **0** |
| 正式回合（每回合） | `silent_eager` | **0** | **0** |
| 收尾 | 无 `predict` 事件 | — | — |

**正式阶段回落的直接证据：B 臂 11 个正式回合（含 2 次判决实验重复）、C 臂 10 个正式回合，`run_eager`
增量逐回合为 0，`silent_eager` 为 0，`replays` 增量恒为 12。** 编译与图捕获只发生在
每个 policy 进程的 warmup 回合内；服务重启后每个臂各做一次 warmup，其余回合全部走
图重放。

这两对短闭环（init 0、init 1）**单独记录**在 `raw/trace_B_PA-B1`、`raw/trace_C_PA-C1`、
`raw/trace_B_PA-B2`、`raw/trace_C_PA-C2` 中，没有替代也没有追认上一轮 20 回合的缺失证据——
上一轮的原始记录里本来就没有阶段计数，本轮也没有给它补记。

---

## 4. 上一轮耗时解释的勘误与重算

### 4.1 勘误一：每回合生成时间是"次数 × 中位"，不是累计

上一轮 8.2 表写"每回合真实生成时间 = 生成次数 × 生成中位"：

| 项 | 上一轮写法 | 从原始记录重算（累计） |
|---|---|---|
| A 每回合生成时间 | 1.42 s | **1.7365 s**（S1+S3 共 10 回合累计 17.365 s ÷ 10） |
| B 每回合生成时间 | 0.65 s | **0.9385 s**（S2 共 10 回合累计 9.385 s ÷ 10） |

重算口径：每个请求贡献自己的 `engine_total_s`，逐请求求和（`tools/reconcile_old.py`）；
用分布里的 `n × mean` 复核，误差上界为 `n × 5e-5 s`，与引擎自记的
`timing.inference_s_sum` 残差 ≤ 0.006 s。

### 4.2 勘误二：`0.49 × 0.65 ≈ 0.7 s` 的分母用错了

`0.49` 是生成延迟降幅，`0.65` 是**优化后（B）**的每回合生成时间，两者相乘
= 0.318 s，与文中 0.7 s 不符。正确写法有两种，都必须标明分母：

- 理论上限（以优化前 A 为分母）：`0.489 × 1.7365 s = 0.849 s/回合`
- 实测累计差（A − B）：`1.7365 − 0.9385 = 0.798 s/回合（−46.0 %）`

上一轮实测模型等待 −0.60 s、墙钟 −0.87 s，与 0.798 s 同一量级；原句的结论方向没错，
算式不自洽。

### 4.3 勘误三：每回合缓存出队时间被低估约 8 倍

上一轮用"缓存出队中位 14.6 ms × 88 次"得到 B 每回合 1.33 s。逐请求累计（引擎层，
含编码与传输）实际是：

| 项 | 上一轮写法（中位 × 次数） | 累计重算（引擎层） | 累计重算（服务端层） |
|---|---|---|---|
| A 每回合缓存出队 | 1.56 s | **11.479 s** | 8.450 s / 5 回合 = 1.690 s |
| B 每回合缓存出队 | 1.33 s | **10.314 s** | 13.458 s / 10 回合 = 1.346 s |

差额落在**客户端 PNG 编码**：只看缓存出队这一项，B 的
`引擎层 − 客户端层 = 10.314 − 2.717 = 7.597 s/回合`，其中 7.6 s ≈ 0.083 s/动作
× 92 个缓存动作，与本轮实测的编码开销完全一致；而 `客户端层 − 服务端层 =
2.717 − 1.346 = 1.371 s/回合`（send + recv）。生成与缓存合计的 `引擎层 − 客户端层`
是 7.844 s/回合、合计 `客户端层 − 服务端层` 是 1.415 s/回合，两者不要混用。**这正是本轮候选 C 消除的部分**，也说明上一轮"每动作固定
开销约 0.25 s"的定性判断是对的，但把它拆成"缓存出队 1.33 s/回合"是错的。

### 4.4 三层嵌套关系（上一轮 B 臂，每回合均值）

| 层 | 生成 | 缓存出队 | 说明 |
|---|---|---|---|
| 服务端层（PolicyServer 内） | 0.648 s | 1.346 s | preprocess + 模型 / `popleft` |
| 客户端层（engine→inference→WS 往返） | 0.692 s | 2.717 s | send + recv |
| 引擎层（记账到 inference 阶段） | 0.939 s | 10.314 s | 另含 PNG 编码、序列化、回调 |

`inference_wall`（引擎记账）− 引擎层累计 = **1.169 s/回合**，是 engine 侧的请求构造、
JSON 序列化与回调开销；墙钟中未被任何阶段记账的部分只有 **0.028 s/回合**。

---

## 5. 缓存出队路径中确实无效的工作

### 5.1 依赖证明

服务端 `PolicyServer.predict()` 的实际数据流：

```
needs_generation = self._policy.needs_generation()   # = len(_action_buffer) == 0
if needs_generation:  obs = self._obs_preprocessor.preprocess(obs)
else:                 obs = {"prompt": ..., "state": ...}   # 图像从不读取
action = self._policy.predict_action(obs)            # 缓存命中时只 _action_buffer.popleft()
```

`predict_action` 在缓冲非空时**不读 `obs` 的任何图像字段**，因此：

- 客户端构造观测图像（180° 旋转 + LANCZOS resize）→ 编码为 base64 PNG → 写进 JSON，
- 服务端解码 PNG → crop → resize → 归一化，

在 31/32 的步骤上产出的中间结果**没有任何读者**。

同时确认未被跳过的部分：

| 检查项 | 结论 |
|---|---|
| 下一次生成所需状态/缓存 | 生成时仍用当次观测；缓存出队不更新任何生成状态 |
| 当前动作合法性 | 动作从缓冲 `popleft`，与图像无关；维度/有限性校验保留 |
| 成功、释放、停止判定 | 基于 `gripper opening` 与 sim 步数，与图像无关；判定代码未改 |
| 页面视频与快照 | 每步仍取 `raw_obs`、仍写 snapshot、仍更新 frame（见 8.3 实测） |
| 观测身份 | `observation_id` / `scene_epoch` / 指纹链路未改 |

### 5.2 逐请求实测（本轮，缓存出队路径）

| 分项 | B 每动作 | C 每动作 | 归属 |
|---|---|---|---|
| PNG 编码（`encode_s`） | **0.08297 s** | 0 | 客户端 |
| socket 写入（`send_s`） | 0.00997 s | 0.00022 s | 客户端 |
| 服务端处理（`server_latency_ms`） | 0.01565 s | 0.00003 s | 服务端 |
| 客户端往返合计（`inference_s`，send+recv） | 0.03146 s | 0.00142 s | 客户端 |
| **引擎记账合计（`inference_total_s`）** | **0.11464 s** | **0.00159 s** | 端到端 |

这里没有把 socket 写入、等待残差或不同实验中位数差值称为"纯网络传输"：上表每一列
都是该请求自己记录的字段；0.0100 s 的 `send_s` 是写入本地环回 socket 的实测时间，
`recv_s` 的残差在报告中不再拆分为"网络"。

---

## 6. 候选 C 的最小实现

C = B + 一个受服务端状态约束的开关，共三处改动，均在同一棵树内，由两个环境变量选择臂：

1. **服务端**（`openwam/deploy/server.py`）：`needs_generation()` 为假且
   `OPENWAM_CACHED_STEP_FASTPATH=1` 时跳过 `_obs_preprocessor.preprocess`，
   只透传 `prompt` 与 `state`；响应新增 `generated`、`obs_preprocessed`、
   `buffer_remaining`。开关关闭时行为与 B 逐字节相同。
2. **客户端**（`models/openwam_backend.py`）：仅当**上一次服务端响应**报告
   `buffer_remaining > 0` 时才省略图像；`reset()` 无条件清空该状态；服务端返回
   error 时立即带图重试一次并回到"总是发图"，直到服务端再次报告缓冲。
3. **引擎**（`engine/openwam_rollout.py` + `adapters/openwam_contract.py`）：
   `raw_obs` **永远获取**（proprio 与观测身份依赖它），只有旋转/resize 依
   `next_obs_images_required` 条件执行。

约束满足情况：

- 不猜 `buffer_index`：决策只来自服务端自报的 `buffer_remaining`。
- reset / 异常 / 缓冲耗尽 / 状态不同步：reset 清状态；error 触发一次带图重试；
  `buffer_remaining == 0` 时必然发图。
- 未改协议结构（只增加响应字段），未改调度与批量下发。

检查点、bf16、compile 配置、`emulate_precision_casts`、steps、seed、shift、
velocity cache、32/32、20 Hz、动作转换、预算、释放窗口与判定全部不变。

---

## 7. 语义与回放验证

同一观测序列（40 步，覆盖 chunk 边界 step 0 与 step 32）、同一 reset 顺序：

| 项 | B | C |
|---|---|---|
| 发图请求数 | 40 | **2**（仅两次真正生成） |
| 生成位置 | step 0, 32 | step 0, 32 |
| 动作逐位比较 | — | **`action_diffs = 0`** |
| 动作序列 SHA256 | `6abd88aef0f7de56` | `6abd88aef0f7de56` |
| 生成时刻观测摘要 | `938f0031ac37ed6f`, `11521b85fab3f91f` | 同左（逐位相同） |
| 编码累计 | 3.319 s | 0.139 s |
| 推理累计 | 20.43 s（含首次编译） | 0.49 s |

**输出逐位一致**（本优化不涉及数值计算，未沿用任何编译容差）。

失步恢复：人为把客户端缓冲状态设为 32（服务端实际为空），服务端按协议返回 error，
客户端带图重试 1 次，动作与参考**逐位一致**，`errors = 0`。

生成边界输入哈希已保存（`validation_c.json` → `replay.gen_obs_digest_at_generation`），
证明 C 在真正生成时用的是**当次**观测，而不是缓冲里的旧帧。

闭环内另有一致性证据：C 臂所有正式回合的服务端 `image_retries = 0`、
`errors = 0`，即没有任何一次"省略图像但服务端其实需要"的情况。

---

## 8. 20 回合闭环结果

### 8.1 配置与平衡

10 个初始化 × 2 臂 = 20 个正式回合，每对独立 reset 到同一初始化。顺序按批次交替，
使 **B→C 5 对、C→B 5 对**：

- B→C：init 0, 2, 4, 6, 8
- C→B：init 1, 3, 5, 7, 9

前两对（init 0、init 1）为冒烟，配置未变后继续余下八对。每臂每次重启后各做一次
warmup（编译预热），不计入正式统计。

### 8.2 配对完成时间（完整任务耗时，`record.duration_s`）

| init | B (s) | C (s) | 差 (s) | sim actions B/C | 停止 B/C |
|---|---|---|---|---|---|
| 0 | 27.91 | 17.88 | −10.03 | 98 / 98 | release / release |
| 1 | 27.30 | 17.11 | −10.19 | 95 / 94 | release / release |
| 2 | 27.03 | 17.44 | −9.59 | 93 / 95 | release / release |
| 3 | 28.31 | 17.66 | −10.65 | 98 / 98 | release / release |
| 4 | 27.41 | 17.73 | −9.68 | 95 / 98 | release / release |
| 5 | 28.50 | 19.64 | −8.86 | 99 / **120** | release / **chunk_exhausted** |
| 6 | 30.27 | 19.76 | −10.51 | 120 / 120 | chunk_exhausted / chunk_exhausted |
| 7 | 28.43 | 17.48 | −10.95 | 98 / 95 | release / release |
| 8 | 29.88 | 19.54 | −10.34 | 120 / 120 | chunk_exhausted / chunk_exhausted |
| 9 | 29.83 | 19.52 | −10.30 | 120 / 120 | chunk_exhausted / chunk_exhausted |

- 双方成功：**10/10**；失败、unknown、超时、后端异常：**0**
- 配对差中位 **−10.245 s（−35.4 %）**，均值 −10.11 s，范围 [−10.947, −8.858]
- **步数完全相同的 5 对**（init 0,3,6,8,9）：中位差 **−10.341 s**；把步数差按
  每动作中位折算后，这 5 对的差值与原始差一致
- 全 10 对剥离步数差后的中位差为 −10.32 s，与未剥离的 −10.245 s 基本一致：
  **收益不是靠少走步数换来的**

### 8.3 逐动作固定开销 / 生成时间 / 控制频率

| 指标（每回合均值） | B | C | 变化 |
|---|---|---|---|
| 缓存出队累计（引擎层） | 10.467 s | **0.145 s** | **−10.32 s（−98.6 %）** |
| 缓存出队每动作 | 0.11464 s | **0.00159 s** | −0.11305 s |
| 每动作总计（中位） | 0.2532 s | **0.1400 s** | −0.1132 s |
| 生成累计（3 次/回合） | 0.9658 s | 0.9638 s | **−0.002 s（不变）** |
| 观测构造 | 4.217 s | 4.228 s | +0.011 s |
| `env.step` | 6.592 s | 6.654 s | +0.062 s |
| 状态快照 | 0.891 s | 0.887 s | −0.004 s |
| 动作转换 | 0.063 s | 0.061 s | −0.002 s |
| 视频取帧 | 0.0002 s | 0.0004 s | 噪声 |
| 墙钟控制频率 | 3.851 Hz | 6.639 Hz | +72 %（仅执行更快） |

墙钟阶段记账（`openwam.wall`，每回合均值）：

| 阶段 | B | C | 差 |
|---|---|---|---|
| `inference_wall` | 12.685 s | **2.331 s** | **−10.35 s** |
| `observe` | 4.217 s | 4.228 s | +0.011 |
| `env_step` | 6.592 s | 6.654 s | +0.062 |
| `snapshot` | 0.891 s | 0.887 s | −0.004 |
| `convert` | 0.063 s | 0.061 s | −0.002 |
| 未归属 | 0.037 s | 0.062 s | 噪声 |

**节省的 10.35 s 全部落在 `inference_wall` 内，且其中 10.32 s 就是缓存出队；
其余阶段逐项不变。** 这排除了"只改善某个局部却增加其他开销"的情况。

控制频率从 3.85 Hz 升到 6.64 Hz 只是每步更快导致的执行节奏变化：20 Hz 契约是
"1 个 sim action 对应 50 ms 仿真时间"，没有改变；步数、轨迹与判定都由同一套
`WholeTaskRunner` 产生。

### 8.4 视频与快照路径未受影响

`snapshot`、`frame`、`convert` 三项在两臂之间差异均在测量噪声内（见 8.3），
说明候选 C 没有跳过页面视频、快照或观测身份记录。旁证视频：
`raw/videos/PA-VID-B_B_init3.mp4`、`raw/videos/PA-VID-C_C_init3.mp4`
（旁观取帧，不计入计时；C 的视频回合因冷启动含首次编译，耗时不可用于性能对比）。

---

## 9. 轨迹分叉、异常与重复执行的单列

### 9.1 init 5 的停止原因不同（唯一分叉对）

B 在 init 5 走 99 步并 `policy_release` 成功；C 走 120 步 `chunk_exhausted`
（预算用尽）也判定成功。**该对单列**：原始差 −8.86 s，按每动作中位剥离 21 步差后
为 **−12.98 s**。无论用哪个口径，C 都不慢于 B。

### 9.2 分叉的成因是环境重置的数值非确定性，不是 C

判决实验：**同一 B 臂、同一 init 1、同一场景/状态指纹**，连续两次正式回合：

| 次 | scene 指纹 | state 指纹 | sim actions | 第 1 步动作（前 3 维） |
|---|---|---|---|---|
| 1 | `a383e10beaab` | `3277a6c370d1` | 95 | 0.2307, 0.1135, 0.0142 |
| 2 | `a383e10beaab` | `3277a6c370d1` | 94 | 0.2765, 0.1282, 0.0169 |

两次 `eef` 也有 1e-4 级差异（`−0.21978` vs `−0.21958`）。**同一臂内部就会出现
第 1 步动作分歧**，说明闭环里的 1–3 步差异来自 LIBERO 重置的浮点非确定性，
与候选 C 无关；逐位一致性由第 7 节的固定观测序列回放证明。

### 9.3 一次真实失败（引擎侧，已保留）

C/init 5 的第一次尝试：`execution_status=failed`、`failure_code=engine_error`、
`failure_reason=URLError: <urlopen error [Errno 111] Connection refused>`、
`sim_actions=0`。原因是 engine 的 env 子进程在 load 阶段尚未就绪（该 env 进程由
engine 自己拉起，PID 与启动时间见 `raw/process_ledger.jsonl`）。这是一次**基础设施
瞬时故障**，不是 C 的语义问题：失败发生在 `execution_s=0`、没有任何 predict 之前。
该记录原样保留（`raw/records/PA-C3a_C_init5_ep-20261102-122931-2d9201.json`），
配对使用其后成功的重跑。

### 9.4 重复执行（采集瑕疵，如实记录）

`raw/pa_session.jsonl` 中有 5 条重复行：init 0 (B) 因驱动被启动两次；init 1 (B)
两条来自判决实验；init 4 (C) 与 init 5 (C) 各多一条，来自 engine 在 inference
重启期间的自动重放。配对统一使用"驱动发出的、成功的那一条"，重复项在
`pa_analysis.json → duplicate_runs` 全部列出，未删除任何原始记录。

---

## 10. 局限与未做的事

- 本轮的 C 只覆盖缓存出队路径（31/32 的步骤）。**生成步骤的 32 步动作中第 1 步仍是
  完整路径**（~0.32 s），这部分没有优化。
- `build_openwam_obs(images=False)` 的旋转/resize 跳过确实生效，但观测构造的
  4.22 s/回合 几乎全是 `env.raw_obs()`（~41 ms/步），因此该子项**没有可测收益**，
  不能算作本轮收益来源。
- 绝对时延（B 生成 225 ms、C 命中 0.03 ms 服务端处理）依赖单客户端、单 PolicyServer
  的部署形态；多客户端并发时服务端状态由谁驱动，本轮未测。
- 只测了 task 1。是否值得扩展到其他单任务，见第 11 节。
- 未做多任务大规模测试，未扩展组合任务。
- 视频 C 回合因冷启动含首次编译，不能作为性能证据。

---

## 11. 结论

**`PER_ACTION_OPT_PRELIM_PASSED`**

- 依赖关系成立：缓存出队时服务端的 `predict_action` 只 `popleft`，图像无人读取；
  跳过后动作与生成路径逐位不变。
- 收益稳定：10/10 配对均下降，中位 −10.245 s（−35.4 %），5 对步数完全相同的配对
  中位 −10.341 s，说明收益不是轨迹变短换来的。
- 生成时间保持：0.9658 s → 0.9638 s/回合。
- 其余阶段（观测、`env.step`、快照、转换、视频）逐项不变。
- 正式阶段 CUDA Graph 回落为 0，图重放每回合 12 次。

**是否值得扩展到其他单任务**：值得，但收益形态取决于"每动作观测载荷"占比。
本任务里编码占了每动作固定开销的 72 %（0.0830/0.1146），所以收益大。若换到
低分辨率/单相机或已经复用编码的任务，收益会按同一比例缩小。建议下一步只挑
一个**不同相机配置**的单任务做对照，而不是直接铺开到全部任务。

**未自动切换生产服务。** 本报告不构成生产切换建议。

---

## 12. 交付物与恢复命令

### 交付物

| 路径 | 内容 |
|---|---|
| `trees/OpenWAM/openwam/deploy/server.py` | 服务端缓存步快速路径（开关控制） |
| `trees/OpenWAM/openwam/deploy/executors/sync_executor.py`、`policy.py` | `needs_generation()` / `buffer_remaining()` |
| `trees/embodied_engine/models/openwam_backend.py` | 客户端按服务端状态省略图像 + 失步重试 |
| `trees/embodied_engine/adapters/openwam_contract.py` | `build_openwam_obs(images=False)` |
| `trees/embodied_engine/engine/openwam_rollout.py` | 引擎只跳过图像变换，`raw_obs` 始终获取 |
| `tools/stack.sh`、`tools/arm_switch.py` | 独立配置启停、臂切换、进程台账 |
| `tools/cg_trace.py`、`tools/test_cg_trace.py` | 阶段计数钩子与单元验证 |
| `tools/validate_c.py` | 语义/回放/失步验证 |
| `raw/pa_per_request.jsonl` | 1887 条逐请求记录 |
| `raw/pa_analysis.json`、`raw/pa_summary.json` | 逐回合与配对分析 |
| `raw/errata_old_round.json` | 上一轮重算勘误 |
| `raw/videos/*.mp4` | B/C 对照视频（补充） |

### 启动（B 臂）

```bash
cd /home/host/flashrt-exp/per_action
PER_ACTION_RUN_ID=PA-B1 bash tools/stack.sh start policy B
PER_ACTION_RUN_ID=PA-B1 bash tools/stack.sh start inference B
```

### 切换到 C 臂（含就绪探测与台账）

```bash
/home/host/page-dmh/embodied_engine/.venv-openwam/bin/python tools/arm_switch.py C PA-C1
```

### 跑一个批次

```bash
python tools/run_episodes.py --arm C --tag PA-C1a \
  --trace raw/trace_C_PA-C1 --inits 2,3 --out pa_session.jsonl --run-id PA-C1
```

### 复核

```bash
python tools/analyze_pa.py && python tools/pa_summary.py
python tools/reconcile_old.py
/home/host/page-dmh/embodied_engine/.venv-openwam/bin/python tools/validate_c.py all
```

### 恢复（只停本轮自己的进程组）

```bash
PER_ACTION_RUN_ID=PA-C6 bash tools/stack.sh stop policy
PER_ACTION_RUN_ID=PA-C6 bash tools/stack.sh stop inference
```

`stack.sh stop` 在停止前会校验 `/proc/<pid>/cmdline` 命中 `per_action` 与本服务端口；
不匹配则拒绝并报告，不扩大匹配范围。

### 本轮收尾实际结果（`raw/cleanup.json`）

| 服务 | PID | 结果 |
|---|---|---|
| policy | 737597 | 已停（cmdline 命中 `per_action` + 14531） |
| inference | 739407 | 已停（cmdline 命中 `per_action` + 14611） |
| stub | 389741 | 已停（cmdline 命中 `per_action` + 14711） |
| engine | 390164 | **未停**：其 cmdline 不含端口串，三重校验无法通过，守卫拒绝 |
| env（engine 子进程） | 730208 | 随 engine 保留 |

守卫拒绝后**没有绕过、没有强制结束**，engine 与其 env 子进程原样保留并在
`raw/cleanup.json` 中报告。生产服务全程未被发送任何信号。
