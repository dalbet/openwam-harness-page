# OpenWAM 引擎执行侧优化：PyTorch 编译 + CUDA Graph（2026-11-02）

- 代码版本：`/home/host/page-dmh` @ `33f18148d27654f7223e925248fe9b26b23410e4`（本轮未提交改动，仅新增 `embodied_engine/docs/audits/`、`embodied_engine/records/acceptance/`）
- 隔离副本：`/home/host/flashrt-exp/src/openwam_iso`（**未修改一个字节**；本轮 `find src/openwam_iso -newermt` 为空）
- 实验设备：**GPU0 独占**（开始 2 MiB / 结束 2 MiB，未启用持久模式）
- 生产侧未触碰：GPU2 OpenWAM（pid `1380923`，`14520` LISTEN，24768 MiB）、GPU1 Pi0.5（pid `2784215`，`14510` LISTEN，7164 MiB）、GPU3 vLLM EngineCore（pid `2843863`，22098 MiB）全程未发信号、未重启
- 本轮自建进程（`exec_opt_*`、`serve_instrumented.py --port 14521`）均按 cmdline 与 PID 身份确认后自行停止
- 上一轮遗留的一个只读采样进程（pid `51918`，`nvidia-smi --query-gpu=... -lms 500`，始于 `00:44:47`，仅写 `logs/allgpu_300s.csv`，不做任何计算）在本轮全部测量期间持续存在且对两臂完全相同，收尾时按 cmdline 确认身份后停止；不构成 A/B 偏置

> 本轮成果全部来自 **PyTorch `torch.compile` + CUDA Graph**。**未涉及 FlashRT FA2 切片**，不主张任何 FlashRT 收益。`decode_video` 全程为 `false`，也未把关闭它计入收益。

---

## 0. 结论摘要

```
执行侧优化：        EXECUTION_OPT_PASSED
推荐候选：          B = inductor 编译（默认模式）+ emulate_precision_casts + CUDA Graph 重放
完整生成时延：      294.2 ms -> 166.9 ms（配对中位数 -43.3%，跨 3 个测量块、2 个独立进程）
WS 请求往返：       307.6 ms -> 174.2 ms（配对中位数 -43.4%）
p95：               295.9 ms -> 167.7 ms
动作一致性：        9/9 冻结观测通过（回传 7D 命令）| 缓存决策轨迹 9/9 相同 | 夹爪符号翻转 0
```

四条关键事实（都有本轮实测支撑，不是模式名称推断）：

1. **编译边界只覆盖 30 层 MoT 联合循环**，不动图像/prompt 预处理、VAE 首帧编码、噪声采样、velocity cache 判断、反归一化与 WS 协议（§2）。
2. **CPU 提交量下降 95.3%**（每次生成 CPU 事件 113532 → 5289），内核数 19672 → 7571，GPU 忙碌占比 26.9% → 72.7%——瓶颈从"CPU 发射受限"翻转到"GPU 受限"（§6）。
3. **收益不是模式名带来的**：CUDA Graph 的录制/重放次数是被直接计数并配合 `TORCH_LOGS=cudagraphs` 日志核对的（`records=1`、`replays=610`）（§6.2）。
4. **默认 inductor 数值口径会越过验收线**（夹爪维最大 0.1094），加 `emulate_precision_casts=1` 后降到 0.0352 才通过；这个开关不是"看到误差后放宽容差"，而是官方提供的、把 inductor 省略掉的 bf16 下降-上升对补回来的口径（§7.2）。

另外交付一个与本轮结论同等重要的运维事实：**本机没有 Python 开发头文件，`torch.compile` 会因此静默回退到 eager，而入口代码把异常吞掉了**（§4）。

---

## 1. 保护现场

| 项 | 状态 |
|---|---|
| 隔离副本源码 | 未修改（本轮 0 个 `.py` 的 mtime 变化落在 `src/openwam_iso` 内） |
| 主仓库 | 仅新增报告与验收 JSON，未改 `config/engine.json`、未改部署默认值 |
| 生产 GPU2/GPU1/GPU3 | 未触碰（PID、端口、显存逐项复核不变） |
| GPU0 | 仅本轮实验；结束时 2 MiB |
| 编译缓存 | 独立目录 `cache/inductor_A`、`cache/inductor_Aemul`、`cache/inductor_B`（各自 `triton_*` 并行），不写共享缓存 |
| 依赖 | 未升级任何 pip 包；未安装系统包（Python 头文件通过只读环境变量引用 uv 自带 CPython，见 §4） |
| 历史报告/数据 | 未覆盖；本轮所有原始数据为新文件 |

保持不变的契约：检查点 `checkpoint_step_10690.safetensors`、bf16、`denoise_steps=10`、shift、seed、`inference_horizon=32`/动作段 `(32, 10)`、图像与状态编码、`min-max` 归一化、EEF10→7D 转换、DiT velocity cache 配置（`cosine_threshold=0.99`、`max_consecutive_skips=3`）、`decode_video=false`、控制频率与成功判定。

---

## 2. 编译边界

### 2.1 已有入口实际覆盖什么

`DualSystemSelfAttnArchitecture.apply_compile_optimizations()`（`openwam/model/architectures/dual_system/joint_self_attn.py`）做两件事：

```
self._compiled_mot_run_joint_loop = torch.compile(_run_joint_loop, mode=..., dynamic=...)
        _run_joint_loop = driver.run_joint_loop(vstate, astate)   # 30 层联合循环
super().apply_compile_optimizations(compile_cfg)                   # -> Wan 逐块 lazy 编译
```

- **覆盖**：`DualSystemMoTDriver.run_joint_loop` 的全部 30 层（视频 DiT 块 + 动作分支 cross-attn + 共享 self-attn）。每次 `generate` 调用 4 次（= DiT 重算次数），每次展开 30 层。
- **不覆盖**：`ObsPreprocessor`、prompt/文本编码、首帧 VAE 编码、初始噪声采样、`DiTVelocityCache.should_recompute` 的分支判断、`extract_prediction`、反归一化、`SyncInferenceExecutor` 动作块缓存、WS 协议层。这些都是每**次生成**一次而非每层一次的操作，而且正是"动态控制流"所在。
- **坑**：`--compile-enabled true` 单独使用时还会**同时**打开 `optimization.compile.wan_blocks`（默认 `enabled: true`、`torch_mode: default`），即对 30 个 block 各自 lazy 编译一次。那会在已编译的外层循环里再嵌一层编译对象，既增加启动成本又改变融合边界。本轮候选显式关掉它（`wan_blocks.enabled=false`），只保留外层整循环编译。

实测确认（`exec_opt_configcheck_B.json`）：
```
compile.enabled=True  self_attn.torch_mode=reduce-overhead  wan_blocks.enabled=False
compiled_loop=True    wan_blocks_compile_enabled=False
```

### 2.2 为什么选这块

上一轮已经把生成定性为 CPU 发射受限：单次生成 19386 个 kernel、~88500 次 CPU op，GPU 忙碌只占 26%。`run_joint_loop` 是其中**形状完全稳定（`dynamic=False` 可锁死）、纯张量、无数据相关分支**的重复计算区，且每层边界清晰。整段 `generate` 含噪声采样与缓存分支，不适合整体捕获。

---

## 3. 三个候选与实施结果

| 候选 | 内容 | 生成中位数 | 正确性 | 结论 |
|---|---|---|---|---|
| A | `self_attn.torch_mode=default`（inductor 融合，无 CUDA Graph） | 177.9 ms | **不通过**（夹爪 0.1094 > 0.05） | 保留证据，停止性能验收 |
| A′ | A + `TORCHINDUCTOR_EMULATE_PRECISION_CASTS=1` | 177.9 ms | 通过（夹爪 0.0352） | 可用，但被 B 支配 |
| **B** | A′ + `self_attn.torch_mode=reduce-overhead`（CUDA Graph 重放） | **166.9 ms** | **通过** | **推荐候选** |
| C | 手工 CUDA Graph | 未实施 | — | 自动图模式已真实生效（`records=1`/`replays=610`），按规则不实施 |

- A′ 的 177.9 ms 是单次抽样（出现在正确性核对里），**不是**配对基准，仅用于说明"编译本身"与"图重放"各自的量级：编译约 −39.5%，图重放再叠加约 −5.5%（11 ms）。正式性能验收只对 B 做配对测量。
- C 未实施的理由是实测的：见 §6.2。规则要求"仅当自动图模式未生效、且仍有明确 CPU 提交瓶颈时"才手工捕获，两条都不成立。

---

## 4. 本轮真正的第一道墙：编译静默回退（必须记录）

第一次跑候选 A 时看到 `177 ms` 级的提升是**假的**，实际情况是：

```
/tmp/tmpXXX/main.c:4:10: fatal error: Python.h: No such file or directory
dual self-attn compiled MoT loop failed; falling back to eager: CalledProcessError: ... gcc ...
```

即 Triton 的 C launcher 编译失败 → `apply_compile_optimizations` 内部 `try/except` 记录 warning → `_compiled_mot_run_joint_loop` 置空 → `forward()` 走 eager。**首轮"候选"的 301.3 ms 与 eager 的 301.2 ms 完全一致，说明那一次测的是 eager**。

本机 `python3.10-dev` 未安装，`/usr/include/python3.10/` 不存在。解法不改系统、不改 venv，只用构建期环境变量指向 uv 自带 CPython 的头文件：

```
C_INCLUDE_PATH=/home/host/.local/share/uv/python/cpython-3.10.22-linux-x86_64-gnu/include/python3.10
```

（3.10.22 头文件 + 3.10.12 解释器，同一 minor 版本；先用最小扩展模块验证可编译再投入使用。）

从此以后，所有验收运行都要求 `fallbacks_detected == 0`：

- `Arms.set_arm()` 检测钩子是否被后端自己置空并计数；
- `exec_opt_perf` 记录 `measurement_clean`；
- 每份 JSON 都带 `fallbacks_total`。

这条被迫加的闸门当场抓到了一次真实事故：我自己的 cudagraph 计数器签名写错（`record_function(self, function_id)` 少一个位置参数），被入口的 `try/except` 吞掉，那一轮 300 次生成全部退化成 eager（`fallbacks_total=111`）。该轮数据整份作废，未进入本报告的任何一个数字。

> 结论：**任何声称 `torch.compile` 收益的测量，必须先证明编译没被吞掉。** 

---

## 5. 动态状态的处理

| 动态要素 | 处理方式 | 证据 |
|---|---|---|
| DiT velocity cache 的"重算/跳过" | **完全留在 eager**：`should_recompute` 在 `generate()` 循环里，编译区只接收"这一步要不要算"的外层结果 | 9 个观测的 10 步决策轨迹全为 `1100010001`（4 重算 / 6 跳过），与 eager 逐一相同 |
| 初始噪声 / seed | 在编译区之外采样，编译区是 `(vstate, astate)` 的纯函数 | 同一观测重复生成逐位相同；不同观测输出互不相同（9/9 唯一） |
| 图像特征、proprio、timestep、latent、prompt 条件 | 每次调用重新构造 `vstate/astate` 并原位进入编译区 | 改 prompt、改观测都会改变输出；`aot_eager` 追踪与 eager **逐位相同** |
| 缓存读写与每次生成的 reset | reset 不触碰编译区，只清执行器动作块 | `reset` 后与 reset 前逐位相同；块边界不变 |
| 输出缓冲区 | 动作张量在编译区外由 `squeeze/float/cpu/numpy` 新建 | 保留 9 个返回数组的视图引用，全部调用结束后逐个复查：无一个被后续 replay 覆写 |
| 输入地址稳定（CUDA Graph 要求） | 交给 cudagraph trees 的 copy-in 机制，不手工固定 | 同上：切换输入/prompt 后结果按预期改变，未出现"固化第一次" |

---

## 6. 实际执行路径证据

### 6.1 编译是否真的发生

`exec_opt_correct_B.json`（最终一次）：

```
dynamo_unique_graphs = 1     graph_break = 0     （整段 30 层循环 = 1 张图）
inductor_generated_kernels: 已生成
fallbacks_detected = 0        compiled_loop_alive = True（切回 eager 前快照）
```

没有 graph break、没有重编译（30 层展开后仍是单图），所以不需要"只针对某个 break 做最小修改"。

### 6.2 CUDA Graph 是否真的启用（不靠模式名）

三重证据：

1. `TORCH_LOGS=cudagraphs` 日志（`records/logs/candB_torchlogs.log`）：
```
[__cudagraphs] recording cudagraph tree for graph without symints
[__cudagraphs] Running warmup of function 0
[__cudagraphs] Recording function 0 of graph recording id 0
```
2. 运行期直接给 `CUDAGraphTreeManager.record_function` / `CUDAGraphNode.run_graph` / `CUDAWarmupNode.run` / `run_eager` 挂钩计数。
   性能 run1/run2 均为 `warmups=1, records=1, replays=610, eager_fallbacks=1`（1 次预热 + 1 次录制；610 = 150 次生成 × 4 次循环 + 预热）。
   形状桶只有 1 个（形状固定），无跳过原因（`cudagraph_skips` 为空）。
3. 无"退化到 eager"：`fallbacks_total=0`。

### 6.3 CPU 提交 vs GPU 执行（profiler，各 2 次代表性生成）

| 指标 | eager | 候选 B | 变化 |
|---|---|---|---|
| kernel 数 / 生成 | 19672 | 7571 | **−61.5%** |
| CPU 事件 / 生成 | 113532 | 5289 | **−95.3%** |
| GPU 忙碌 / 生成 | 161.2 ms | 144.7 ms | −10.3% |
| stream 跨度 / 生成 | 600.3 ms | 198.9 ms | −66.9% |
| GPU 忙碌占比 | 26.9% | **72.7%** | — |

CPU 发射量降了两个数量级，kernel 数下降来自融合（点对点链合成单个 Triton kernel），GPU 忙碌时间也降了 10%（同样的数学少了重复的中间物化）。**生成不再是 CPU 发射受限。**

---

## 7. 正确性

### 7.1 口径

- 参考实现：未改动的 eager 后端，同一进程内切换（不复制模型，避免两份 24.8 GB 常驻）。
- 比较对象：**完整 `(32, 10)` 动作段**，不是首动作。
- 判定一律作用在**真正回传给控制器的量**上：`native_eef10_to_libero7d` 之后的 7 维命令（含 rot6d→轴角 与 `-gripper` 与 `clip(-1,1)`）。
- 预声明容差（测量前写死在 `exec_opt_correct.py` 的 `TOL`，未按观测到的误差调整）：
  - 位置 0.02 归一化单位 = 1.0 mm（环境 0.05 m/单位）
  - 旋转 0.02 归一化单位 = 0.01 rad ≈ 0.57°（环境 0.5 rad/单位）
  - 夹爪 0.05 归一化单位 = 全行程 5%（连续量，无阈值）
  - 语义硬约束：缓存决策轨迹必须相同；夹爪不得出现符号翻转
- 原始 EEF10 上的 clip 边界事件**只作诊断**：它可能标记两臂在 ±1 边界附近取值不同，但若回传命令仍在容差内，控制量没有变化（被 clip 抹平）。这一点是口径修正（原先错误地把中间量当判据），**不是**容差放宽；三个数值容差一个都没动。

### 7.2 数值差异的来源定位（先定位，再谈容忍）

| 对照 | 结果 | 含义 |
|---|---|---|
| `aot_eager`（dynamo 追踪 + 分解，不融合）vs eager | **9/9 逐位相同** | 追踪语义无误：没有固化状态、没有漏输入、没有把分支写死 |
| eager + `flash`/`mem_efficient` SDPA 互换 | **9/9 逐位相同** | attention 后端不是差异源 |
| eager + `allow_bf16_reduced_precision_reduction` 翻转 | 逐位相同 | 该口径在本模型上无影响 |
| inductor 默认 vs eager | 最大 0.1094（夹爪） | 差异来自 inductor 代码生成 |
| inductor + `emulate_precision_casts=1` vs eager | 最大 0.0352 | 补回 bf16 下降-上升对后收敛到约 1 个 bf16 量化步长 |

PyTorch 对该开关的官方说明（`torch/_inductor/config.py`）正好解释了方向性：

> PyTorch eager computes bf16/fp16 by upcasting inputs to fp32 and downcasting after. When two low precision operators are fused together, Inductor will elide the downcast-upcast pairs … Typically, Inductor's behavior should be closer to fp64 ref numerics. However, with this knob you can ensure the downcast-upcast are preserved so that you can emulate the eager numerics.

也就是说：默认融合**更接近精确值**，但它与"生产参考实现"的数值口径不同。本轮验收的对象是"与现状等价"，因此选择 `emulate_precision_casts=1`，把口径对齐而不是把容差放宽。

### 7.3 九个冻结观测（候选 B）

| 观测 | 位置 max | 旋转 max | 夹爪 max | 原始 EEF10 max | 缓存轨迹一致 | 夹爪符号翻转 | 判定 |
|---|---|---|---|---|---|---|---|
| task1_p0 | 0.0055 | 0.0078 | 0.0078 | 0.0078 | ✓ | 0 | PASS |
| task1_p1 | 0.0073 | 0.0041 | 0.0078 | 0.0078 | ✓ | 0 | PASS |
| task1_p2 | 0.0073 | 0.0077 | 0.0352 | 0.0352 | ✓ | 0 | PASS |
| task3_p0 | 0.0073 | 0.0039 | 0.0078 | 0.0078 | ✓ | 0 | PASS |
| task3_p1 | 0.0092 | 0.0041 | 0.0078 | 0.0092 | ✓ | 0 | PASS |
| task3_p2 | 0.0073 | 0.0039 | 0.0039 | 0.0078 | ✓ | 0 | PASS |
| task8_p0 | 0.0073 | 0.0039 | 0.0078 | 0.0078 | ✓ | 0 | PASS |
| task8_p1 | 0.0073 | 0.0078 | 0.0078 | 0.0078 | ✓ | 0 | PASS |
| task8_p2 | 0.0073 | 0.0042 | 0.0156 | 0.0156 | ✓ | 0 | PASS |

尺度参照：动作量级 `|a|` 均值 0.4156、最大 1.0；上表最大值基本等于该量级下 1 个 bf16 量化步长（`2^-8 ≈ 0.0039`，在 1.0 附近为 0.0078）。夹爪 0.0352 出现在 `task1_p2`，是约 9 个步长，且该维度在两臂都处于 ±1 附近（`-0.9922` vs `-0.9922` 等），符号与阈值均未变。

单位换算（7 维命令是归一化空间，环境再乘 0.05 m / 0.5 rad）：位置误差 ≤ 0.46 mm，旋转 ≤ 0.26°，夹爪 ≤ 3.5% 行程；`clip` 使用情况与真值参考相同（被裁剪到 ±1 的元素集合差异只发生在边界上，回传值差异仍在 0.0078 内）。

### 7.4 其它检查

```
候选自身重复生成：逐位相同（9/9 观测）
eager 自身重复：  逐位相同
切回 eager 后：    与切换前参考 逐位相同（3/3 观测，无残留包装/缓存/路径污染）
返回数组被覆写：   无（9 个视图引用在全部调用结束后复查）
不同观测输出唯一： 9/9
输入切换 A->B->A： 与首次 A 逐位相同，且 B 不同于 A
prompt 切换：      eager 与候选都随之改变（新输入确实进入计算）
reset 后：         与 reset 前逐位相同
候选的第一次生成（编译/捕获触发点）：与 eager 参考差 0.0352 / 0.0073 / 0.0077（在容差内）
```

A（默认数值口径）的失败证据保留在 `exec_opt_correct_A.json`：`task8_p2` 夹爪 0.1094（约 14 个量化步长），超过预声明容差，故**未对其做任何性能验收**。

---

## 8. 性能对照

### 8.1 方法

- 同进程交替：唯一差异是 `architecture._compiled_mot_run_joint_loop` 是否存在；切回 eager 后逐位复现参考动作（§7.4）。
- 每个 block：50 个 `(观测, 配对)` 单元，每单元按预先生成、固定种子的随机顺序决定 `AB` 或 `BA`（各占一半），两条样本相邻、同观测。
- 每个 block 每臂 50 次真实生成，共 3 个 block = 150 次/臂。
- 观测在 9 个冻结观测间轮换。
- CPU 状态：进程外独立速度计（独立进程、绑核 100、固定工作量纯 Python 循环），每 ~35–105 ms 一个采样，全程共 ~1890 个采样；每个样本按时间戳就近（±1 s）取档。
- 分档规则固定且与生成时延无关：把上一轮独立测得的量化档位 `{8.8, 11.4, 17.0, 26.3} ms` 按本次运行最快采样缩放，在 log 空间取最近邻。
- 慢样本不删除；跨档配对单独列出。
- 候选在正式计时前完成编译与图捕获（预热 3 次生成）。
- `measurement_clean` 必须为 true（`fallbacks_total == 0`）。

### 8.2 结果（run1，进程 A）

```
3 block × 50 对，eager 150 次 / 候选 150 次
eager  median 294.15 ms   p95 295.94 ms
cand   median 166.93 ms   p95 167.69 ms
paired same-band  n=150   ratio(中位) 0.5671   Δ(中位) −127.28 ms   Δ(均值) −128.31 ms
```

按 block：

| block | eager n / 中位 / p95 | cand n / 中位 / p95 | 中位比 |
|---|---|---|---|
| 0 | 50 / 294.3 / 318.3 | 50 / 166.5 / 167.5 | 0.5656 |
| 1 | 50 / 294.2 / 295.0 | 50 / 166.9 / 167.5 | 0.5672 |
| 2 | 50 / 294.1 / 295.3 | 50 / 167.1 / 167.8 | 0.5682 |

### 8.3 独立进程重启复验（run2，不同 PID、不同配对种子）

```
eager  median 294.01 ms   p95 296.02 ms
cand   median 166.89 ms   p95 167.62 ms
paired same-band  n=149   ratio(中位) 0.5677   Δ(中位) −127.16 ms
paired cross-band n=1     ratio 0.5668
```

| block | eager 中位 / p95 | cand 中位 / p95 | 中位比 |
|---|---|---|---|
| 0 | 294.2 / 296.6 | 166.5 / 167.5 | 0.5658 |
| 1 | 294.1 / 296.0 | 166.8 / 167.6 | 0.5670 |
| 2 | 292.1 / 295.0 | 167.0 / 167.6 | 0.5717 |

两次独立进程的中位比 0.5671 / 0.5677，块间 0.5656–0.5717，**收益跨块、跨进程保持**。

### 8.4 CPU 状态覆盖

| 运行 | 档位 | eager n | cand n | eager 中位 | cand 中位 | 同档配对 n | 配对中位比 |
|---|---|---|---|---|---|---|---|
| run1 | 8.78 ms | 126 | 126 | 294.19 | 167.00 | 126 | 0.5674 |
| run1 | 11.37 ms | 24 | 24 | 294.11 | 166.19 | 24 | 0.5655 |
| run2 | 8.81 ms | 126 | 126 | 293.79 | 166.91 | 126 | 0.5684 |
| run2 | 11.41 ms | 24 | 23 | 294.52 | 166.46 | 23 | 0.5651 |
| run2 | 17.02 ms | 0 | 1 | — | 166.10 | 0 | — |

**未覆盖**：上一轮的慢档（435–490 ms，速度计 17.0/26.3 ms）在这两个测量窗口里没有出现，因此**没有慢档结论**。上一轮观察到的"档位变化"，在本轮两个窗口内对生成时延没有可测量影响（eager 在 8.8 与 11.4 档的中位都是 294 ms 级），说明该量化档位未必与本次所用的执行核集合直接对应——本轮不做进一步推断。

### 8.5 CPU process time

| 运行 | eager 中位 | cand 中位 | 变化 |
|---|---|---|---|
| run1 | 422.12 ms | 294.81 ms | −30.2% |
| run2 | 414.63 ms | 287.70 ms | −30.6% |

CPU 足迹下降幅度小于 CPU **事件数**下降幅度（−95%），因为剩下的主要是 torch 层 Python 开销与内核启动的固定成本，已经不是可再靠融合削掉的量。

### 8.6 WS 请求往返

同一服务进程内，每个配对：切执行臂 → reset（清动作块缓存，保证一定触发真实 generate）→ 发同一观测；`AB`/`BA` 各半、固定种子。客户端 `compression=None`、`ping_interval=None`（与生产 `WSPolicyClient` 一致）。

```
n=30 对
客户端 wait：  eager 307.56 ms (p95 310.88) -> cand 174.24 ms (p95 178.55)   中位比 0.5665
服务端阶段：   generate 301.2 -> 167.8 ms；preprocess 4.7 -> 4.8 ms；dump+send 0.09 -> 0.11 ms
```

即 WS 链路本身没有新增代价，收益完整穿过协议层（比进程内多出的约 7 ms 是 preprocess + 序列化，两臂相同）。

---

## 9. 冷启动、编译成本与资源

| 项 | 数值 |
|---|---|
| 进程装载模型 | 106–122 s（eager，与上轮一致） |
| 首次生成（含 inductor 编译 + 图捕获），冷缓存 | A：77.7 s；B：81.3 s |
| 首次生成，热缓存（`cache/inductor_B` + `cache/triton_B` 已存在） | 13.7–18.9 s |
| 图捕获本身 | 1 个图、1 次录制；`cudagraph` 私有池 |
| 稳态显存（allocated） | 23681.4 → 23693.8 MiB（**+12.4 MiB**） |
| 稳态显存（reserved） | 23868.0 → 24232.0 MiB（**+364 MiB**，含图池） |
| 单卡峰值 | 23933.5 MiB / 32109 MiB |
| 重编译次数 | 0（单一形状桶，`dynamic=False`） |
| 图跳过原因 | 无 |

编译成本只在服务启动时付一次；热缓存下 14–19 s，冷缓存约 80 s。若部署侧要求秒级启动，应把 `cache/inductor_B`、`cache/triton_B` 一起预热进镜像（或把 `TRITON_CACHE_DIR`、`TORCHINDUCTOR_CACHE_DIR` 指向持久目录）。

---

## 10. 结论与下一阶段

### 10.1 是否达到进入闭环的门槛

门槛是"完整生成时延下降 ≥ 10%"。实测配对中位 −43.3%（跨 2 个独立进程、3 个 block 稳定），p95 295.9 → 167.7 ms，WS 往返同幅下降。**达到门槛，且是机制可解释的收益**（CPU 事件 −95%、GPU 忙碌占比 27%→73%）。

### 10.2 推荐候选与最小配置

推荐 **B**。启用方式全部走现有 deploy 配置入口，**不需要改任何源码**：

```bash
# 构建期（本机缺 python3.10-dev，仅编译期需要；不安装任何包）
export C_INCLUDE_PATH=/home/host/.local/share/uv/python/cpython-3.10.22-linux-x86_64-gnu/include/python3.10
# 新 torch 版本的数值口径对齐（运行期）
export TORCHINDUCTOR_EMULATE_PRECISION_CASTS=1
# 独立编译缓存（可选，但强烈建议，便于预热与回滚）
export TORCHINDUCTOR_CACHE_DIR=/home/host/flashrt-exp/cache/inductor_B
export TRITON_CACHE_DIR=/home/host/flashrt-exp/cache/triton_B

python -m openwam.deploy.server \
  --ckpt-dir  /home/host/page-dmh/openwam_harness/checkpoints/OpenWAM-Alpha-Sim-LIBERO \
  --ckpt-name checkpoint_step_10690.safetensors \
  --device cuda:0 --host 127.0.0.1 --port <PORT> \
  --inference-mode sync --denoise-steps 10 --inference-horizon 32 \
  --compile-enabled true \
  optimization.compile.self_attn.torch_mode=reduce-overhead \
  optimization.compile.wan_blocks.enabled=false
```

回滚：`--compile-enabled false`（或去掉后三条），并 `unset TORCHINDUCTOR_EMULATE_PRECISION_CASTS`。两条 override 的意义：

- `self_attn.torch_mode=reduce-overhead`：默认配置里该值是 `default`，不覆盖就**没有图重放**（只拿 −39.5% 的编译收益）。
- `wan_blocks.enabled=false`：**必须**，否则 `--compile-enabled true` 会额外对 30 个 block 各自编译，启动更慢且融合边界不同（本轮未评估该组合）。

这条命令本身已被验证（`exec_opt_configcheck_B.json`）：`compiled_loop=True`、`wan_blocks=False`、`records=1`、`replays=78`、生成中位 171.3 ms、动作与已保存的 eager 参考差 0.0092/0.0078/0.0352。

### 10.3 尚未做 / 不应据此下结论的部分

- **未做**闭环、成功率、量化；本报告只声明延迟与数值等价。
- 慢档（435–490 ms）未覆盖；不能说"候选消除了双档"，只能说"在两个已覆盖档位上收益稳定"。
- `wan_blocks` 逐块编译组合、`mode=max-autotune`、`cross_attn`/`idm` 分节编译均未评估。
- 手工 CUDA Graph（候选 C）未实施：自动图模式已真实生效。
- 建议的下一轮最小闭环：单任务（如 `task1`）用 `B` 与 `eager` 各跑固定 epizodes，比成功率与完成时间，作为"控制等价"的行为级证据。

---

## 11. 复测与恢复

```bash
cd /home/host/flashrt-exp
PY=/home/host/page-dmh/embodied_engine/.venv-openwam/bin/python
UVINC=/home/host/.local/share/uv/python/cpython-3.10.22-linux-x86_64-gnu/include/python3.10
export C_INCLUDE_PATH=$UVINC TORCHINDUCTOR_EMULATE_PRECISION_CASTS=1
export TORCHINDUCTOR_CACHE_DIR=$PWD/cache/inductor_B TRITON_CACHE_DIR=$PWD/cache/triton_B
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1

# 1) 九观测完整动作段正确性（含缓存轨迹、reset、输入切换、覆写检查）
$PY tools/exec_opt_correct.py --candidate B_compile_reduce_overhead \
    --out records/raw/exec_opt_correct_B.json --save-arrays records/raw/exec_opt_chunks_B.npz

# 2) 定位数值差异来源（aot_eager 逐位对照）
$PY tools/exec_opt_localize.py --candidate A_compile_default --out records/raw/exec_opt_localize_A.json

# 3) 配对性能（进程外速度计 + 3 block × 50 对）
$PY tools/exec_opt_perf.py --candidate B_compile_reduce_overhead --blocks 3 --pairs 50 \
    --speedo-out records/raw/bg_speedo_run1.jsonl --out records/raw/exec_opt_perf_B_run1.json --tag run1

# 4) profiler 归因
$PY tools/exec_opt_profile.py --candidate B_compile_reduce_overhead --gens 2 \
    --out records/raw/exec_opt_profile_B.json

# 5) 边界情形（首次生成 / prompt / reset / A->B->A）
$PY tools/exec_opt_edges.py --candidate B_compile_reduce_overhead --out records/raw/exec_opt_edges_B.json

# 6) 配置路径（= 交付给部署的启动参数）
$PY tools/exec_opt_configcheck.py --gens 20 --out records/raw/exec_opt_configcheck_B.json

# 7) WS 配对
$PY tools/serve_instrumented.py --port 14521 --warmup 12 \
    --candidate B_compile_reduce_overhead --no-instrument &
$PY tools/ws_exec_opt_client.py --port 14521 --pairs 30 --out records/raw/exec_opt_ws_B.json
```

恢复/清理：

```bash
pkill -f "serve_instrumented.py --port 14521"
pkill -f "exec_opt_perf.py"
pkill -f "bg_cpu_monitor.py"
pkill -f "nvidia-smi --query-gpu=index,power.draw"   # 上一轮遗留的只读采样（本轮已停止）
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader   # GPU0 应回到 2 MiB
```

注意：`serve_instrumented.py` 必须带 `--no-instrument`，否则 `probe_generation.install_instrumentation()` 会把计数包装装进执行路径（本轮已验证该组合不应混入性能数字）。

---

## 12. 原始数据与复现工具

| 文件 | 内容 |
|---|---|
| `records/raw/exec_opt_correct_B.json` / `exec_opt_chunks_B.npz` | 最终候选九观测正确性 + 完整 32×10 数组 |
| `records/raw/exec_opt_correct_A.json` | 候选 A（默认数值口径）失败证据（夹爪 0.1094） |
| `records/raw/exec_opt_correct_Aemul.json` / `exec_opt_chunks_Aemul.npz` | 候选 A′（编译 + 口径对齐，无图） |
| `records/raw/exec_opt_localize_A.json` | `aot_eager` 逐位一致、eager 数值开关包络 |
| `records/raw/exec_opt_envelope_A.json` | SDPA 后端互换包络（逐位相同） |
| `records/raw/exec_opt_perf_B_run1.json` / `_run2.json` | 配对性能原样本与分档 |
| `records/raw/bg_speedo_run1.jsonl` / `run2.jsonl` | 进程外速度计原始采样 |
| `records/raw/exec_opt_ws_B.json` / `ws_server_stages_B.jsonl` | WS 往返与服务端阶段 |
| `records/raw/exec_opt_profile_B.json` | profiler 归因 |
| `records/raw/exec_opt_edges_B.json` | 首次生成 / prompt / reset / 臂切换 |
| `records/raw/exec_opt_configcheck_B.json` | 部署配置路径验证 |
| `records/logs/candB_torchlogs.log` | `TORCH_LOGS=cudagraphs` 捕获证据 |
| `tools/exec_opt_*.py`、`tools/ws_exec_opt_client.py` | 本轮新增工具（均在 `tools/`，未改 `src/`） |
