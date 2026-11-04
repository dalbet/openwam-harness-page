# OpenWAM 生成时延双档与请求链路归因（2026-11-02）

- 代码版本：`/home/host/page-dmh` @ `33f18148d27654f7223e925248fe9b26b23410e4`（未提交改动：`embodied_engine/docs/audits/`、`embodied_engine/records/acceptance/2026-10-31-flashrt-openwam-r2-poc.json`）
- 隔离副本：`/home/host/flashrt-exp/src/openwam_iso`（上一轮已校验的原后端副本，本轮未改源码）
- 实验设备：**GPU0 独占**（未启用持久模式，实验前 5 MiB / 实验后 2 MiB）
- 生产侧未触碰（收尾逐项核验）：GPU2 OpenWAM（pid `1380923`，端口 `14520` LISTEN，24768 MiB）、GPU1 Pi0.5（pid `2784215`，端口 `14510` LISTEN，7164 MiB）、GPU3 vLLM EngineCore（pid `2843863`，22098 MiB，仅内部 ZMQ 端口）。14561 上的 vLLM API 前端在收尾时未监听；本轮**从未向任何 vLLM 进程发送信号**，只 signal 过 3 个按 cmdline 校验为 flashrt-exp 工具的自建进程
- 本轮自建进程（`serve_instrumented.py` pid `3413459`/`50377`、`regime_longprobe.py` pid `58087`）均已按 cmdline 身份确认后自行停止，GPU0 归零
- 复现配置与生产完全一致：`--compile-enabled false --inference-mode sync --denoise-steps 10 --inference-horizon 32`（生产进程 cmdline 逐字核对）

本轮**不**涉及 FlashRT FA2 切片，也不主张任何 FlashRT 收益。

---

## 0. 结论摘要

| 问题 | 本轮状态 | 一句话结论 |
|---|---|---|
| 302 ms 与 440–490 ms 的生成时延双档 | `CAUSE_CONFIRMED_NO_FIX` | 额外时间 **100% 落在 CPU 提交侧**；GPU 执行时间、SM 时钟、缓存跳步、动作数值全部不变。生成是 CPU 发射受限（GPU 内核仅占 26–35%），因此整机 CPU 交付吞吐的量化步进被 1:1 放大到时延上。触发源在本进程之外，本轮未找到任何进程内可操作的开关。 |
| 进程内 predict 与 WebSocket 的 ~170 ms 差额 | `CAUSE_CONFIRMED_NO_FIX_NEEDED` | **该差额不存在**。它是"快档的进程内测量"减"慢档的 WS 测量"造成的跨档比较。同一时间窗口内配对测量：`ws` 505.6 ms vs `inproc` 513.5 ms（同一输入、同一 chunk 边界）。真实 WS 代价 = 传输 0.32–0.45 ms + 冷字符串预处理 2.9 ms。 |
| 最小修复 | 未交付修复 | 唯一可动的杠杆（降低每次生成的 CPU 发射量）本轮未实施，理由与前置条件见 §7。 |

额外两条可复用的结论（不是收益）：

1. `optimization.decode_video` 在生产默认已是 `false`；**若被打开，每次生成多 174 ms（+58%）且动作数值逐位不变**——即纯粹的浪费。已用 A/B 定量（§6.3）。
2. `websockets` 的 `compression` 默认值是 `'deflate'`。生产客户端 `WSPolicyClient` 显式关掉了它；任何新写的客户端若漏掉，134.6 kB 观测的单次 `send` 会从 0.22 ms 变成 17 ms，服务端出队再额外 +6.8 ms（§6.2）。

---

## 1. 先修正上一轮的测量口径（否则双档无法解释）

上一轮"三个 block 汇总"掩盖了双档的真实结构。逐次样本（`records/raw/ab_generation.json`）显示：**档位是按连续时间段成块出现的，不是按 arm/路径出现的**。

```
按执行顺序的档位切换点（阈值 340 ms）：
  #  0 block1 sdpa       i= 0  437.4 -> SLOW
  #  5 block1 sdpa       i= 5  310.1 -> FAST
  #150 block2 sdpa       i= 0  385.9 -> SLOW     ← 同一 arm，重新进入慢档
  #182 block2 sdpa       i=32  293.8 -> FAST     ← 同一 arm，自行回到快档
  #248 block2 fa2        i=48  435.6 -> SLOW
  #250 block2 split_sdpa i= 0  303.7 -> FAST
  #423 block3 split_sdpa i=23  466.2 -> SLOW
```

`block2 sdpa` 一个 block 内部就跨越了慢→快。同一 arm 的同一 block 前 12 次全部 366 ms、后段全部 292 ms，说明**不是 attention 实现差异，也不是 arm 之间的共享状态残留**。

四条口径修正（都已在本轮工具中落实）：

| 口径问题 | 后果 | 本轮做法 |
|---|---|---|
| 跨档位比较 | 把 302 与 480 相减，得出"170 ms 未归因" | 所有对照必须在**同一时间窗口内交错**，并同时记录 CPU 速度计 |
| 进程内复用同一 `payload` 对象 | `preprocess` 被低估：5.9 ms（热字符串）vs 8.8 ms（WS 收到的冷字符串） | WS 与 inproc 分别记录，不混算 |
| 采样线程与工作循环共享亲和性掩码 | 造出 9.2/25.1 ms 的假双档（`machine_state_idle.jsonl` 首版） | 速度计进程与采样线程分开绑核，工作循环内不做 `/proc` 扫描 |
| `websockets` 默认 `compression='deflate'` | 测试客户端 `send` 从 0.22 ms 虚增到 17 ms | 客户端显式 `compression=None`，与生产 `WSPolicyClient` 一致 |

另外核实一处命名风险：本机 `nvidia-smi -i 0` **不受** `CUDA_VISIBLE_DEVICES` 重映射（已用 3 组环境变量验证，`-i 0` 恒为物理 GPU0）。因此运行中出现过的 `24376 MiB / 46℃ / 2880 MHz` 读数就是 GPU0 上**我们自己的**实验进程，不是外部负载。

---

## 2. 双档是否真实复现

**复现。** 跨 15 个以上独立进程、1592 次已落盘生成样本、多组输入、多种线程配置均出现；呈现三个离散带：

| 带 | 生成时延 | 出现实例 |
|---|---|---|
| 快档 | 288–306 ms | `regime_repro` i≥68、`regime_long_pin8` 中段、`freq_inline`、`decode_ab` 的 `decode=False` 块 |
| 中档 | 350–380 ms | `regime_repro` i=0..16、`threads_ab` seq0–seq3、`ab_generation` block2 sdpa |
| 慢档 | 435–490 ms | `regime_repro` i=17..108、`regime_cpu_probe` 全程、`ws_pair` 全部 |

逐次样本（`records/raw/regime_repro.json`，237 次真实生成，`threads=64`）：

```
fast n=163  wall=302.0  dev=302.0  cpu=1585.0  rqwait=0.12 ms
slow n= 74  wall=437.9  dev=437.9  cpu=1740.5  rqwait=0.15 ms
```

`dev` 是 CUDA Event 区间，`wall` 与 `dev` 逐次只差 0.05 ms → 整个生成过程都在 stream 时间轴内，CPU 侧不存在"生成之外"的隐藏开销。`rqwait`（`/proc/self/task/*/schedstat` 第二列，就绪队列等待）在两档都是 0.1 ms 量级 → **没有被调度出去**。

**出现条件**：与输入无关（task1/task3/task8 九个真实观测轮换，同一时刻全部同档）；与进程生命周期无关（每个进程都是从加载开始，慢/快都可能先出现）；与缓存历史无关（见 §3）。

---

## 3. 执行条件是否真的一致

上一轮已确认"attention 次数与 cache skip 一致"，本轮进一步给出**每次生成**的增量，而不只是 block 汇总（`records/raw/regime_repro.json`，236 个相邻差分）：

```
逐次增量集合 (rec, skip, joint) = {(4, 6, 120)}   ← 唯一解
```

即 10 步去噪里恒为 **4 次 DiT 重算 + 6 次缓存跳过 + 120 次 joint 调用**，快慢档完全相同。因此双档**不是**缓存命中差异、不是 DiT 前向次数差异。

动作数值：`task1_p2` 固定输入下首动作恒为 `0.40649411`，60 次跨 6 个 block 一致；`decode_ab` 中 60 次生成的完整动作段 `(32,10)` SHA256 全部相同（`438dbe6ed731f488`）。**慢档不改变控制语义。**

---

## 4. 额外时间落在哪个阶段

### 4.1 首先排除 GPU

同一进程内同时跑两个与生成互不重叠的固定工作量探针（`records/raw/regime_long_pin8` 打印、`regime_hw_probe.json`）：

| 量 | 快档 | 慢档 | 比 |
|---|---|---|---|
| 固定 4096² bf16 GEMM（CUDA Event） | 0.757 ms | 0.762 ms | **1.007** |
| 单线程纯 Python 固定工作量 | 3.10 ms | 4.05 ms | **1.31** |
| SM 时钟（`nvidia-smi -lms 500` 全程） | 2872–2895 MHz | 2872–2895 MHz | 1.00 |
| 温度 / 功耗 | 50±3 ℃ / 313–328 W | 同区间 | — |
| `clocks_throttle_reasons.active` | `0x0` | `0x0` | — |

GPU 计算吞吐、显存时钟（恒 13801 MHz）、SM 时钟、限频原因全程不变。**慢档不是 GPU 变慢。**

### 4.2 生成是 CPU 发射受限

`records/traces/trace_slow.json`（PyTorch Profiler，单次生成，含 profiler 开销）：

```
主 stream=7：kernel 数 19386，内核累计 161.2 ms，stream 跨度 611.6 ms
→ GPU 忙碌占比 26%；缺口 129.3 ms，1740 处，最大单处 1.9 ms
cpu_op 次数 88556；CPU 侧 self 时间合计 862.4 ms
   aten::linear 108.4 / aten::addmm 81.1 / aten::to 65.3 / aten::_to_copy 61.3 /
   aten::scaled_dot_product_attention 57.1 / aten::mul 47.1 / aten::copy_ 42.1
```

GPU 在 74% 的时间里是空的——CPU 每发射一个 kernel 要 2–9 µs，GPU 早就跑完了。**这条链是串行发射链，整机 CPU 交付吞吐的变化被 1:1 放大到时延上。**

一致性检验（8 线程，交错 A/B，`records/raw/threads_ab.json`）：

```
seq0 threads=64 wall=377.1 cpu=1667.3   seq1 threads= 8 wall=370.8 cpu= 491.8
seq2 threads=64 wall=379.6 cpu=1681.8   seq3 threads= 8 wall=370.1 cpu= 494.3
seq4 threads=64 wall=298.9 cpu=1587.1   seq5 threads= 8 wall=296.9 cpu= 417.3
```

线程数从 64 降到 8，进程 CPU 时间降 73%，wall 只差 1.6% —— 说明瓶颈是**串行发射线程**，不是并行区。64 线程时多出来的 1.2 s CPU 是 OpenMP 自旋等待。

### 4.3 整机 CPU 吞吐是"量化步进"的，且与本进程无关

一个**完全独立**的进程（无采样线程、单线程、绑核 100）跑固定工作量，三轮各 300–500 s：

```
run1: 8.9 ms ──t≈190s──> 17.0 ms        (1.91x)
run2: 16.9 ms ──t≈232s──> 8.8 ms        (1.92x)
run3: 11.4 ms ──t≈90s──> 8.8 ms         (1.30x)
```

离散水平集合 ≈ `{8.8, 11.4, 17.0, 26.3}` ms，比值 **1 : 1.30 : 1.93 : 2.99**。

run3 期间**没有任何实验在跑**，4 张 GPU 全部 `util=0%`、功耗 3–11 W、每核 busy 0%，速度计仍然自行切换。→ **触发源在进程外，与我们的模型/负载无关。**

同一时刻的旁证：`steal=0`（含每核 steal）、`nr_periods=0`（无 cgroup 限流）、非自愿上下文切换 ≈1 次/生成、`/proc/stat` 全机 busy ≈0.8–4%。

`intel_pstate` 上报忙碌核主频恒为 2100 MHz（快慢档都一样，`records/raw/freq_inline.json`），而吞吐变化 1.3–3.0x → **上报频率无法解释吞吐变化**，交付频率在 guest 不可见。

### 4.4 可证伪的定量模型

设 `wall = G（GPU 部分）+ C·η（CPU 提交部分，η 为整机 CPU 速度因子）`，用 `{302（η=1）, 350–380（η=1.30）, 440–470（η=1.93）}` 三带拟合：

```
C ≈ 155 ms，G ≈ 148 ms
预测：η=1.00 → 303 ms   η=1.30 → 349 ms   η=1.93 → 447 ms
实测：      288–306          350–380            435–490
```

`G ≈ 148 ms` 与 profiler 独立测出的 kernel 累计 161 ms 同量级。模型同时解释了"为什么慢档比例不是 2x 而是 1.5x"——GPU 那 148 ms 是下限。

---

## 5. 请求链路分解（同一 request_id 对齐）

服务端在隔离副本中逐阶段打点（`tools/serve_instrumented.py`，仅 monkeypatch，未改源码），客户端在 `inproc` / `ws` 之间**交错**发同一种 32 步批次（`tools/ws_pair_client.py`）。

### 5.1 生成请求（每 32 次 predict 触发 1 次 generate）

| 阶段 | inproc | ws |
|---|---|---|
| `preprocess` | 15.29 ms | 15.49 ms |
| `policy.predict_action`（含 generate） | 496.80 ms | 489.88 ms |
| `predict_total` | 513.52 ms | 505.64 ms |
| 服务端 `server_total` | — | 506.46 ms |
| 客户端 `wait` | — | 506.91 ms |
| 客户端传输差额（wait − server_total） | — | **0.45 ms** |

`ws` 与 `inproc` 的生成耗时差 **−7.9 ms（ws 更快）**，在档内噪声范围内。**生成请求没有 WS 代价。**

### 5.2 出队请求（缓存里的其余 31 次）

| 阶段 | inproc | ws |
|---|---|---|
| `preprocess` | 5.92 ms | 8.82 ms |
| `policy` | 0.01 ms | 0.01 ms |
| `predict_total` | 5.95 ms | 8.86 ms |
| `dump` + `send`（服务端） | 0 | 0.18 ms |
| `server_total` | 5.95 ms | 9.29 ms |
| 客户端 `wait` | — | 9.62 ms |
| 传输差额 | — | **0.32 ms** |

**2.9 ms 的 `preprocess` 差来自字符串冷热**：inproc 路径复用同一个 `payload` dict，90 kB base64 字符串始终在缓存里；WS 路径每次 `json.loads` 分配新字符串。所以**进程内 predict 的预处理数字是下界，不能当作真实成本**。

### 5.3 `compression` 默认值是一个真实陷阱（非生产路径）

测试客户端若用默认参数（`compression='deflate'`），同一 134.6 kB 观测：

| | `compression='deflate'`（默认） | `compression=None`（生产 `WSPolicyClient`） |
|---|---|---|
| 客户端 `serialize` | 1.02 ms | 1.02 ms |
| 客户端 `send` | **17.00 ms** | **0.21 ms** |
| 服务端 `preprocess`（出队） | 15.29 ms | 8.82 ms |
| 服务端 `server_total`（出队） | 16.07 ms | 9.29 ms |

生产客户端已显式关闭（`benchmarks/utils/transport.py:59`），因此**当前生产不受影响**；但这是任何新客户端/新压测脚本会踩的坑，单步控制周期会白付 ~20 ms。

---

## 6. 定向对照（每次只改一个因素）

### 6.1 线程数（64 → 8）：不是双档成因

`threads_ab.py` 交错 A/B/A/B，同一进程、同一输入、同一时间窗口（§4.2 表）。结论：**时延跟随时间窗口，不跟随线程数**；线程数只改变 CPU 足迹（−73%）。

上一轮 `thread_control.json` 里"64→16 使 374→294 ms"是**块顺序与档位切换重合**造成的假因果，本轮已用交错 A/B 推翻。动作数值在 1/4/8/16/64 线程下逐位相同（`act0_max_abs_vs_64 = 0`）。

### 6.2 线程池重建：不是双档成因

`pool_rebuild_control.py`：不改变线程数（恒 64），只调用 `torch.set_num_threads(64)` 重建池，A/B/C 三块 305.8 / 303.2 / 303.0 ms。排除"线程池放置/亲和性残留"。

### 6.3 `decode_video`：验证测量方法，同时暴露一个"开了就白付 174 ms"的开关

`phase_split.py`（单次生成内部阶段）：

```
wall 481.1 ms = pre 14.5 + loop 288.2 + decode 174.4 + post 0.2
decode 占比 36.3%
```

`decode_ab.py` 交错 A/B（同一进程、同一窗口，CPU 速度计全程 9.07–9.09 稳定）：

```
seq0 decode=True  469.1 ms    seq1 decode=False 294.6 ms
seq2 decode=True  470.0 ms    seq3 decode=False 296.3 ms
seq4 decode=True  470.5 ms    seq5 decode=False 295.4 ms
动作段 SHA256：on/off 全同 = 438dbe6ed731f488   actions_identical=True
```

两件事：

1. `phase_split` 归因出的 174.4 ms 与 A/B 实测的 175 ms 一致 → **阶段计时方法可信**。
2. 但解析后的 deploy 配置里 `optimization.decode_video` **已经是 `False`**（`resolve_deploy_args` 实测输出），生产进程 cmdline 也核对过。所以这不是本轮新发现的可优化项，而是：**一旦有人把它打开，每次生成白付 174 ms（+58%），只为了产出被执行器丢弃的 video**——两个执行器都只取 `result["actions"]`（`sync_executor.py:54`、`async_executor.py:264`），WS 契约只返回 `action/step/latency_ms`（`server.py:214`）。

### 6.4 未做的对照

`--compile-enabled true`（`torch.compile` + `reduce-overhead`）是唯一能实质降低 CPU 发射量的、语义不变的杠杆，且配置段已存在（`optimization.compile.{idm.torch_mode: reduce-overhead, video_loop, action_cache}`）。本轮**未实施**，理由见 §7。

---

## 7. 是否有修复

**没有交付修复。**

- 双档本身在进程内不可修：它来自整机 CPU 交付吞吐的量化步进，与本进程无关（§4.3 的 run3 是空载下自行切换）。任何"锁频/调功耗/停其他业务"的做法都被本轮规则排除，也没有使用。
- 唯一可能的应对方向是**降低每生成的 CPU 发射量**，把 155 ms 的 CPU 分量压下去；但那属于架构级改动（CUDA Graph / `torch.compile`），前置条件至少两条，本轮都不满足：
  1. 必须先有可用的动作等价性判据（本轮已建立：`(32,10)` 动作段 SHA256 + 最大绝对差）；
  2. 必须在**快、中、慢三档各自**都有交错对照，否则又会重演上一轮的跨档误判。
- 线程数 64→8 是**可交付但不该当收益宣传**的改动：CPU 足迹 −73%，时延 −1.6%（交错 A/B，同档内）。它的价值是给同机其他服务让出 CPU，不是降延迟。

---

## 8. 交付状态

```
双档归因：        CAUSE_CONFIRMED_NO_FIX
请求链路差额：    CAUSE_CONFIRMED_NO_FIX_NEEDED（差额不存在，已解释）
最小修复：        未实施
```

### 回答六个问题

1. **双档是否真实复现，出现条件是什么？**
   真实复现。三个离散带（288–306 / 350–380 / 435–490 ms）。成块出现，块长几十次到数分钟，与输入、prompt、噪声、缓存历史、线程数、进程重启顺序均无关；同一 block 内可自行切换。

2. **额外时间发生在哪个阶段？**
   100% 在 **CPU 提交（kernel 发射）阶段**。GPU 内核累计时长、SM 时钟、显存时钟、限频原因、GPU 微基准吞吐全部不变；进程 CPU 时间与 wall 的相关系数 `+0.997`；就绪队列等待 0.1 ms、steal 0、无非自愿切换。生成是 CPU 发射受限（GPU 忙碌占比 26%）。

3. **哪些有因果证据，哪些只是相关？**
   - 因果证据：CPU 侧吞吐 ↔ 生成时延（同一进程内交错 A/B 与整段相关性）；`decode_video` ↔ 174 ms（交错 A/B + 动作逐位相同）；`compression` ↔ 17 ms `send`（同客户端双配置）；线程数 ↔ CPU 足迹（交错 A/B）。
   - 仅为相关：晶粒/封装温度与档位（温度同区间内两档都出现）；GPU 利用率与档位（GPU 全程 0–60% 无对应关系）；上报的 `scaling_cur_freq` 与吞吐（2100 MHz 恒定而吞吐变 1.3–3.0x）。
   - **未解决（unresolved）**：整机 CPU 吞吐为什么会量化步进。离散比值 `1 : 1.30 : 1.93 : 2.99` 很像"同一物理核被 k 个执行流复用"的 `1/(k+1)` 形势，但 guest 侧 `steal=0`、每核 busy≈0、无 cgroup 限流，无法在该命名空间内直接证实是宿主层 P-state/功耗预算还是核超分。**不把这条当结论。**

4. **请求链路差额解释到什么程度？**
   完全解释，且差额为零。同一窗口配对：`ws` 505.6 ms vs `inproc` 513.5 ms；传输 0.32–0.45 ms；服务端 `dump+send` 0.18 ms。上一轮的 ~170 ms 是"快档 inproc 减慢档 ws"的跨档相减。另外进程内 `preprocess` 因复用同一 base64 对象被低估 2.9 ms，这个偏差方向恰好会**夸大** WS 成本。

5. **是否完成修复，收益能否跨测量块和进程重启保持？**
   未完成修复，因此没有可保持的收益。本轮唯一测到的"消除 174 ms"是关掉一个**生产已经关闭**的开关，不计为收益。

6. **是否值得进入单任务闭环验收？**
   **不值得**。本轮没有产出可跨块保持的延迟收益；闭环验收只会重复测量同一个受外部 CPU 状态支配的时延分布。进入闭环的前置条件是先在 §6.4 那条线上拿到"动作等价 + 三档交错对照均成立"的 CPU 发射量下降。

---

## 9. 下一步（按优先级）

1. **建一个时延 A/B 的强制口径**：任何生成时延对照必须在同一窗口内交错，并并列记录进程外 CPU 速度计（`tools/bg_cpu_monitor.py`）与固定 GEMM。没有速度计的同档读数不可比。
2. **只在 CPU 发射量上想办法**：目标是把 155 ms 的 CPU 分量压下来，而不是动 GPU。候选是 `--compile-enabled true`（配置段已就绪），必须给出：动作段最大绝对差、三档各自的交错对照、跨进程重启复测。
3. **加两条回归闸门**（成本极低、收益确定）：
   - 断言 `optimization.decode_video` 为 `False`（打开即 +174 ms 且动作不变）；
   - 任何 WS 客户端/压测脚本断言 `compression=None`（漏掉即 +17 ms/请求 + 服务端 +6.8 ms）。
4. **若要把双档本身纳入验收标准**，先向宿主侧确认 CPU 交付频率/核分配策略；在 guest 内无法观测，也无法控制。

---

## 10. 复现方式

```bash
cd /home/host/flashrt-exp
PY=/home/host/page-dmh/embodied_engine/.venv-openwam/bin/python

# 双档 + 四类时钟（快慢档逐次时间线）
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 $PY tools/regime_repro.py --out records/raw/regime_repro.json

# CPU 提交 vs GPU 执行拆分
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 $PY tools/regime_split.py --out records/raw/regime_split_run1.json

# 8 线程下的 CPU 微基准 + 逐次 wall
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 $PY tools/regime_cpu_probe.py --threads 8 --gens 150 --out records/raw/regime_cpu_probe_t8.json

# 进程外 CPU 速度计（必须独立进程 + 独立绑核）
$PY tools/bg_cpu_monitor.py --core 100 --seconds 330 --out records/raw/bg_speedo_pair.jsonl

# 线程数交错 A/B
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 $PY tools/threads_ab.py

# generate 内部阶段
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 $PY tools/phase_split.py --out records/raw/phase_split_decode.json --decode-video 1

# decode_video 交错 A/B + 动作等价
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 $PY tools/decode_ab.py --out records/raw/decode_ab_p1.json

# WS/inproc 配对（先起服务，再跑客户端，端口 14521）
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 $PY tools/serve_instrumented.py --port 14521 --warmup 30
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 $PY tools/ws_pair_client.py --port 14521 --rounds 3 --per 32
```

粗体约束：`serve_instrumented.py` 与 `ws_pair_client.py` 同时校验两条设置——服务端 `ping_interval=None`、客户端 `compression=None`。少任何一条，数字就不可比。
