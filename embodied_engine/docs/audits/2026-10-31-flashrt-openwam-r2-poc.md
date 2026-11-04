# 2026-10-31 FlashRT × OpenWAM R2 注意力切片验证（隔离 POC）

**最终裁决：`R2_POC_NO_GAIN`**

- 切片**实现正确**：FlashRT FA2 路径数值与 eager SDPA 等价（cos = 0.99999958，动作段 worst max_abs = 0.03125，即 bf16 精度粒度）。
- 切片**没有性能收益**：注意力切片本身慢 60%（15.59 ms vs 9.77 ms / 生成），整模型 A/B 慢 5.5%（311.3 ms vs 295.2 ms，3 块 × 50 次真生成）。
- 根因是**可加速面太小**：整模型每生成一次，注意力内核只占 4.80 ms（device 215 ms/gen 的 2.2%、稳态墙钟 302 ms 的 1.6%）。即便注意力降为零，生成时延上限收益也只有约 1.6%，远低于 ≥10% 的下一轮候选目标。
- 本轮不做闭环、不动 planner、不改步数/动作窗口/控制频率，也不宣称闭环成功率。

详细裁决依据与全部原始记录见第 11、12 节。

---

## 0. 本轮结论速览

| 项目 | 结果 |
| --- | --- |
| C1 `torch.compile` 状态 | **未编译**。`--compile-enabled false`，`dynamo_counters = {}`（0 帧被编译）；`compile_worker`（`--workers=32` 池）是 `torch._inductor.config` 在 import 期创建的，父进程即本进程、早于模型加载 |
| C2 计时冲突 | **严格嵌套成立，原"不嵌套"现象由"字段语义 + 空闲后首个生成"解释**：`send_s` 只是 socket 写入，`recv_s` 是残差；73,379 条历史记录与 70 条本轮隔离请求 100% 满足 `server_latency_ms < client_total_ms` |
| 新发现（重要） | 生成时延存在 **~302 ms / ~440–490 ms 双档**，同进程同输入可复现；慢档 SM 时钟实测 2887 MHz（上限 3090），**排除 DVFS**；两档的注意力调用数与 DiT cache skip 完全一致。已标记为未解释项，**绝对时延只在同进程内可比** |
| 冻结观测 | 3 任务 × 3 阶段 = 9 个真实观测（真实相机 + 真实 EEF10 proprio），动作段 **32 × 10**，非 96 |
| 原始生成基线 | 稳态 **292.9–305 ms**（30 次连续生成 median 292.95；9 观测中 7 个 302–305），稳态块 p95 ≈ 315–325 ms；DiT velocity cache 每次生成 **4 次 recompute + 6 次 skip** |
| 原始后端重复性 | 同一观测连续生成动作段 **逐位相同**（9/9 观测 `sdpa_repeat_bitwise_equal = true`） |
| 注意力真实内核 | `fmha_cutlassF_bf16_aligned_64x128_rf_sm80`（`PyTorchMemEffAttention`），**240 次/生成，4.80 ms/生成** |
| 切片数值 | 通过（max_abs 0.03125 = bf16 粒度，cos 0.99999958） |
| 切片性能 | **慢 60%**（fa2 15.59 ms vs baseline 9.77 ms / 生成）；注：FA2 单次无 mask 392×392 只需 6.64 ms（比基线快 1.47x），但**语义错误**，正确拆分后反而更慢 |
| 整模型 A/B | fa2 **0.9482x**（-5.47%），split_sdpa **0.9687x**（-3.23%） |
| 请求往返 | 历史/隔离 WS 服务 482–504 ms（属慢档，含 36.4 ms 图像预处理 + ~21.7 ms RPC 开销），进程内 `predict` 生成档 312 ms、出队档 4.6 ms |

---

## 1. 现场保护与隔离

### 1.1 现场复核（未回退、未修改）

| 项 | 历史基线 | 本轮实测 | 结论 |
| --- | --- | --- | --- |
| `page-dmh` HEAD | `33f18148d27654f7223e925248fe9b26b23410e4` | 同 | 无变化 |
| OpenWAM HEAD | `4d7f642b841913ea3c176010b692b1de3d5ef180` | 同 | 无变化 |
| `git status`（page-dmh） | — | 1 条未跟踪：`embodied_engine/docs/audits/`（本轮与上轮报告目录） | 未修改任何受跟踪文件、未改 `config/engine.json` |
| GPU0（实验用） | 空闲 | 实验前 4 MiB；实验后 **5 MiB** | 已完全释放 |
| GPU2（生产 OpenWAM） | 24778 MiB | 未接触、未发任何 obs/reset | 未干扰 |
| GPU1（Pi0.5）、GPU3（vLLM） | 7174 / 23078 MiB | 未接触 | 未干扰 |

所有实验资产位于 `/home/host/flashrt-exp/`，与工作区、`page-dmh` 主配置、FlashRT 原安装完全分离。

### 1.2 隔离手段（针对"启动脚本会写回原工作区"的风险）

- `tools/run_isolated_server.sh`：独立端口 `14521`，`CUDA_VISIBLE_DEVICES=0`，日志 `logs/openwam_iso_14521.log`，PID `pids/openwam_iso_14521.pid`，`TORCHINDUCTOR_CACHE_DIR`/`JOBLIB_TEMP_FOLDER` 全部指向 `/home/host/flashrt-exp/cache/`，`cd` 到隔离副本 `src/openwam_iso` 后启动，不回写 `page-dmh`。
- 环境服务：`14571/14572/14573` 三个独立 LIBERO 实例（task 1/3/8，seed 0），PID 单独记录，日志 `logs/env_1457*.log`；本轮结束已全部停止。
- 生产端口 `14510`（Pi0.5）、`14520`、`14561` 全程未触碰，仅作为"非本轮进程"记录。
- 源码隔离：`src/openwam_iso` 是 OpenWAM @`4d7f642` 的逐字节拷贝（去 `.git`/`__pycache__`），480 个文件哈希记录在 `provenance.json.files.openwam_iso`；隔离环境服务代码 6 个文件哈希记录在 `provenance.json.files.env_iso`。
- FlashRT：`/home/host/flashrt-store/page_engine/FlashRT` 仅作**只读快照引用**，未进行任何原地重编译，未覆盖现有 CPython-3.12 扩展；本轮最小扩建设在 `flashrt-exp/build/` 下，独立 `.so`。

### 1.3 运行环境（未升级、未硬凑）

```
Python 3.10.12
torch 2.7.1+cu128   (CUDA 12.8)
checkpoint: OpenWAM-Alpha-Sim-LIBERO / checkpoint_step_10690.safetensors
device: cuda:0 (RTX 5090, SM120), 独立 .venv-openwam（复用主环境解释器，未 pip 升级）
```

---

## 2. C1：`torch.compile` 状态

**结论：本轮基线全程未启用 `torch.compile`，模型图没有被编译。**

证据（`records/raw/c1_compile_state_false.json`、`c1_compile_false.json`）：

1. **配置层**：`cfg.optimization.compile.enabled = false`；架构实际类型 `DualSystemSelfAttnArchitecture`，`wan_blocks_compile_enabled = False`；`apply_compile_optimizations`/`Module.compile` 只是存在（bound method），未被调用。
2. **运行时层**：`torch._dynamo` 计数器 `dynamo_counters = {}`（空），即没有任何帧被 Dynamo 拦截/编译；`first_predict_s = 1.981 s`（纯 eager 首帧，若触发编译会是数十秒级）。
3. **`compile_worker` 进程的真实来源**：本进程子进程列表中确实有 `torch/_inductor/compile_worker/__main__.py --workers=32 --kind=fork --parent=<本进程 pid>`，其产生原因是 `torch._inductor.config` 的**import**（记录中有 51 条 `inductor_import_events`，栈为 `importlib._bootstrap`），与该配置的 `--workers=32` 默认值一致，**早于模型加载**。
4. 全场 `compile_worker_processes_total` 共 67 条，**包含其他长期运行的 python 实例**（例如 pid 1380923 的 14520 服务），因此"worker 数量"完全不能作为本进程是否启用 compile 的判据。

结论：本轮基线是**纯 eager**；正式性能运行未开启任何高开销编译日志（`TORCH_LOGS`/`TORCHINDUCTOR_*` 全部未设），诊断运行（`c1_*`）与性能运行（`perf_*`、`ab_*`）分文件保存，未混用；也未做"静默启用 compile 再当基线"的偷换。

---

## 3. C2：计时字段冲突

### 3.1 字段语义核实

| 字段 | 起止点 | 时钟域 | 单位 | 备注 |
| --- | --- | --- | --- | --- |
| `send_s` | `await ws.send(payload)` 前后 | 客户端单调 | 秒 | **只计 socket 写入**，不含服务端处理 |
| `recv_s` | 由 `elapsed - send_s` 计算 | 客户端单调 | 秒 | **是残差，不是独立收包腿**；把它当"接收耗时"是不可信的 |
| `client_total_ms` | 请求发出前 → 收到响应后 | 客户端单进程单调 | ms | 完整往返主指标 |
| `server latency_ms` | `predict()` 内 `t0 = time.monotonic()` → 组响应字典前 | 服务端进程单调 | ms | 含 obs 预处理 + policy + executor + generate；**不跨进程相减绝对时间戳** |
| `inference_s` | `engine.generate()` 前后 | 服务端单调 | 秒 | 仅模型生成，不含预处理 |
| GPU 计算时间 | CUDA Event / `torch.cuda.synchronize()` 边界 | device | ms | 未把 launch 时间当完成时间；CPU 侧 `perf_counter` 一律加 sync |

### 3.2 嵌套性与"不嵌套"现象

- 历史 73,379 条记录：`server_latency_ms < client_total_ms` 满足率 **100%**。
- 本轮隔离 WS 单连接、以严格请求序号标注（`records/raw/c2_ws_timing_clean.json`，70 条：3 次生成 + 67 次缓存出队 + 1 次 reset ack）：**violations = 0**。
  - 生成档：client 504.4 ms / server 482.6 ms，overhead 21.7 ms；
  - 出队档：client 37.4 ms / server 15.5 ms，`json.dumps` 1.0 ms，overhead 21.9 ms。
- 早前"`send+recv` 小于 `server_latency_ms`"的观感来自：`send_s` 只覆盖写入、`recv_s` 是残差，两者相加恒等于 client 总时长（构造使然），**不能**与 server 内部耗时按"阶段"比较。
- **仍然无法完全归因的部分**：进程内 `srv.predict()` 生成档为 312 ms，而隔离 WS 服务同路径为 482 ms，差 170 ms 无法用 RPC（21.7 ms）与图像预处理（36.4 ms）解释。经第 5.2 节的档位实验，该差距与"空闲后首个生成落入慢档"一致，但**慢档的物理原因未查明**；因此本报告**不把请求级绝对时延（482/504 ms）作为性能结论依据**，只使用同一进程内交替测量的生成时延与 CUDA 内核计时。该字段标记为"条件可信（仅相对可比）"。

---

## 4. 冻结观测与完整动作段

工具：`tools/capture_obs.py`（采集）、`tools/probe_generation.py`（shapes/perf/profile/requests 四种模式）、`tools/baseline_sweep.py`（9 观测扫描 + 请求路径同进程对比）。

### 4.1 观测来源（不用截图、不用零状态）

- 从**三个独立 LIBERO 环境服务**（task 1 / task 3 / task 8，seed 0）在**真实 episode 的 p0/p1/p2 三个阶段**读取同一时刻的原始相机与真实 EEF10 状态：
  - `agentview_image` 256×256、`wrist_image` 256×256、`main_images` 256×320、`wrist_images` 128×160（uint8，原始像素）；
  - EEF10：位置 3 + 旋转 6D 3 + 夹爪 1（真实 proprio，非零填充）。
- 严格沿用既有 **`build_openwam_obs` 契约**转换为模型输入，未自造字段。
- 每份观测存 `.npz`（原图 + proprio 原始数组）与 `.json`（prompt、配置、形状），sha256 记录在 `records/manifest.json`。
- 冻结条件：prompt 固定（task1 "put the bowl on the stove" / task3 "open the top drawer" / task8 "put the cube in the basket"）、`seed=42`、`steps=10`、`shift=5.0`、`horizon=32`、bf16、20 Hz 契约不变。

### 4.2 动作段（实测，不看配置字段）

- `engine.generate()` 返回 `actions` 形状 **`(32, 10)`**，即 **32 步 × 10 维**，全部步骤保存为 `.npy`/`.npz`（`records/raw/actions_*.npy`、`segment_*_*.npz`），9 个观测逐一保存并参与比较。
- 提醒：`bench_speed.py --openwam-actions 96` 是**动作请求数**（96 次单步请求），不是 96 次生成；本报告所有"生成次数"均为真实 `generate()` 调用计数。

### 4.3 原始后端自身重复性

- 同一观测、同一 seed、同进程连续生成：动作段**逐位相同**——9/9 观测的 `sdpa_repeat_bitwise_equal = true`（见 `ab_generation.json.action_segment_comparison`）。这是原始后端重复性的直接证据，也是后续"A 臂 vs 优化臂"数值比较的对照基准。
- 这不等于跨实现逐位一致：bf16 + 不同 attention 内核会带来 ≥1 ulp 差异（见 7.2，FA2 臂 `bitwise_equal = false`，max_abs 0.0078–0.03125）。
- DiT velocity cache 实际行为已记录：每次生成 **120 次 joint attention 调用、4 次 recompute、6 次 skip**（`threshold=0.99`, `max_skips=3`）；`reset()` 清空 executor 的 action buffer，**但不清空 prompt embed cache**（LRU，按 prompt 键控，属期望行为）。本轮所有对照使用**同一段连续生成历史**（不中途 reset），保证 cache 轨迹一致；接入 FA2 后 recompute/skip 仍为 4/6，说明数值扰动**没有改变 cache 命中轨迹**。

---

## 5. 原始生成性能基线

### 5.1 正式基线（预热 → 30 次真生成）

`records/raw/perf_task1_p0_30gen.json`（预热 5 次后连续 30 次真生成）：

- median **292.95 ms**（该次运行的汇总输出），逐次样本未写入该记录文件，仅保存了聚合计数与形状直方图；分布与波动以下面的 9 观测扫描和 A/B 的逐次样本为准（已披露该记录缺陷）。
- 每生成：`sdpa_calls_total = 133`、`joint_calls = 120`、DiT `recompute = 4` / `skip = 6`；
- 注意力形状直方图（每 30 次生成）：`q(1,24,392,128)|mask(392,392)` 1330 次、`q(1,24,360,128)|mask(1,1,360,513)` 1329 次、`q(1,24,32,128)|mask(1,1,32,513)` 1329 次、`q(1,1,480,640)|maskNone` 12 次。

### 5.2 9 观测扫描（`records/raw/baseline_sweep.json`）

每个观测预热 3 次 + 连续 8 次真生成：

| 观测 | median ms | min | max | joint/gen | recompute/skip per gen |
| --- | --- | --- | --- | --- | --- |
| task1_p0 | 449.1 | 438.9 | 473.3 | 120 | 4 / 6 |
| task1_p1 | 317.6 | 302.9 | 445.9 | 120 | 4 / 6 |
| task1_p2 | 302.1 | 301.1 | 305.6 | 120 | 4 / 6 |
| task3_p0 | 402.2 | 304.9 | 443.3 | 120 | 4 / 6 |
| task3_p1 | 303.5 | 302.1 | 487.0 | 120 | 4 / 6 |
| task3_p2 | 303.0 | 301.9 | 318.3 | 120 | 4 / 6 |
| task8_p0 | 303.4 | 302.2 | 304.8 | 120 | 4 / 6 |
| task8_p1 | 303.5 | 300.8 | 315.0 | 120 | 4 / 6 |
| task8_p2 | 302.2 | 301.0 | 305.5 | 120 | 4 / 6 |

**关键观察**：慢样本（402–487 ms）与快样本（302–305 ms）的**注意力调用数、recompute/skip 计数完全一致**，因此慢档不是"cache 命中变差"或"注意力变多"导致的。

### 5.3 档位实验（`records/raw/idle_clock_probe.json`）

单独进程：稳态连续 6 次生成 → 插入 0.5/1.6/3.0/6.0 s 空闲后再生成：

- 该进程整段落在慢档：438–493 ms（首次冷启 2013 ms）；
- 生成期间 SM 时钟 **2887–2895 MHz（上限 3090 MHz）**，空闲 3 s 后采样到 2392 MHz，但**紧随其后的生成仍为 482 ms**；
- 即：慢档发生在时钟接近满频时，**排除 DVFS/时钟回落**作为主因。

结论：绝对时延有 ~1.45x 的**档位差异（进程级爬升 + 整块漂移，A/B 中已复现整块 50 次落入慢档）**，机制未查明（作为威胁项披露，见 9.4）。因此：
1. 所有正式对照在**同一进程内交替测量**，并对每臂前 5 次启动样本做剔除复算；
2. 历史 WS 482 ms 与本轮 WS 504 ms 归入慢档，不与稳态 302 ms 直接比较；
3. 慢档不影响本轮裁决方向：微基准与档位无关，FA2 在正确语义下慢 60%；端到端 FA2 的最快块（309.76 ms）仍慢于基线最快块（292.94 ms）。

### 5.4 请求路径分解（同进程，`baseline_sweep.predict_requests`）

- `srv.predict()`（含 obs 预处理 + policy + executor）生成档：**312.0 ms**（n=2）；
- 缓存出队档：**4.6 ms**（n=31，与 WS 服务端 15.5 ms 的差主要是 WS 消息编解码 + 事件循环）；
- 图像侧独立测量（`tools/bench_preprocess.py`，4 相机、0.29 MB base64 PNG）：base64 解码 + PNG 解码 + resize 到 640×480 + 归一化 = **36.4 ms**（median，p95 36.8）。
- 结论：请求路径中**非生成部分至少 41 ms（预处理 + 出队）**，RPC 层 ~21.7 ms；注意力优化对其最多只能贡献 <1%。

---

## 6. R2 适配价值（先评估，后编译）

### 6.1 `WAM_ATTENTION_IMPL` 的真实覆盖范围

`openwam/model/action_backbone/components.py:372` 的 `get_attention_fn()` 是唯一读取该环境变量的位置，覆盖 **action backbone（ActionDiT）的注意力**：

- 自/交叉注意力：`q(1,24,360,128) × kv(1,24,513,128)`（视频 token 流对文本条件）、`q(1,24,32,128) × kv(1,24,513,128)`（动作 token 对文本条件），mask `(1,1,S,513)`；
- 该路径是**无 mask 语义的密集注意力**（mask 仅 padding），本来就适合 FA2，但它是 `q_len ≠ kv_len` 的 cross-attention。

**主视频 DiT 走的是另一套实现**：`openwam/model/video_backbone/wan/shared/core/attention/attention.py`（有自己的 backend 校验，不读 `WAM_ATTENTION_IMPL`）。

**R2 真正关心的模块**是 `openwam/model/architectures/*/joint_self_attn.py` 的 **joint self-attention**：

- `q/k/v`：`(1, 24, 392, 128)`，bf16，连续，`S = Sv + Sa = 360 + 32`，`head_dim=128`，`scale=1/√128`，非 causal；
- mask：`bool (392,392)`，`mask_mode=mutual` + `video_attention_mask_mode=first_frame_causal`；实测 `true_frac = 0.7876`，`rows 0..119` 只可见 `cols 0..119`（首帧 1×12×10=120 token 的 Isolated 块），其余 272 行全见；
- 调用量：**120 次/生成**（30 层 × 4 次 recompute），与 `joint_calls = 120/生成` 实测一致；其中落到 **392×392 带 mask SDPA** 的约 **44 次/生成**（SDPA 形状直方图口径）。
  - 口径说明（已披露）：另一部分注意力走 video DiT 的独立 `AttentionModule` 入口，`F.scaled_dot_product_attention` 直方图统计不到；因此**形状级归因只用于定位"哪些调用可被 R2 覆盖"，总耗时一律以 profiler 内核计数为准**（见 6.2：240 次 FMHA 内核/生成）。

### 6.2 基线 SDPA 实际 dispatch 的内核

不能把 "SDPA" 当朴素实现。`torch.profiler`（`records/raw/profile_task1_p0.json`，3 次生成，device 总计 645.5 ms = 215.2 ms/gen）：

```
 33.90 ms/gen  aten::addmm (4912 calls)
 12.56 ms/gen  cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x128...
  9.12 ms/gen  cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x128...
  6.48 ms/gen  cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x...
  4.82 ms/gen  aten::_efficient_attention_forward (722 calls)
  4.80 ms/gen  fmha_cutlassF_bf16_aligned_64x128_rf_sm80 (PyTorchMemEffAttention) (720 calls)
  3.24 ms/gen  aten::mul
  3.13 ms/gen  aten::copy_
```

- SDPA 已 dispatch 到 **Cutlass MemEff FMHA**（`PyTorchMemEffAttention::AttentionKernel`），**240 次内核启动/生成，4.80 ms/生成**，占 device 时间 **2.2%**、占稳态墙钟（302 ms）**1.6%**。
- **更正上一轮记录**：先前"~14.4 ms/gen"实为 **3 次生成合计 14.39 ms**；正确单次值为 **4.80 ms/gen**，注意力占比比最初估计更小。
- 剩余 device 时间以 GEMM 为主（`addmm` 33.9 + 三个 cutlass GEMM ReLU 合计 28.2 = 62.1 ms/gen，28.9%）。

### 6.3 FlashRT 快照中的兼容入口与约束

- `FlashRT/csrc/attention/fa2_wrapper.cu` 暴露 `fvk_attention_fa2_fwd_bf16`，底层是 vendored FA2（cutlass 3.x）的 `flash_fwd_hdim128_bf16_sm80`。
- 兼容性约束（逐项核对，均为实测或源码确认）：
  1. **无 mask 参数**：FA2 入口只接受 q/k/v + causal flag，**不能表达 block-structured bool mask**；
  2. **要求 `seqlen_q == seqlen_k`**：实测当 `q_len ≠ kv_len`（如 272×392）时输出数值损坏（cos 0.54–0.84，392×128 出现 NaN）；这是 vendored kernel 的假定，**不是 OpenWAM 的契约**；
  3. 需要 `--expt-relaxed-constexpr --expt-extended-lambda`（cutlass 3.x 头文件要求）；
  4. 需要显式链接 `libcudart.so.12`（隔离环境没有 `nvcc` 对应的 cudart 开发包）；
  5. 固定 `head_dim=128`、bf16、SM80 指令路径（在 SM120 上以兼容模式运行）。
- 覆盖评估：**只有 joint self-attention 的 392×392 一个形状**可以尝试；mask 无法表达 → 必须拆成两次等长无 mask 调用（`392×392` 全量 + `120×120` 首帧块）再拼接。这直接导致每次注意力变成 **2 次内核启动**。

---

## 7. 最小构建、切片正确性与性能

### 7.1 最小构建（out-of-tree，未动原安装）

`tools/build_fa2_slice.sh`，产物 `build/libflashrt_fa2_slice.so`：

```
nvcc -std=c++17 -O3 --expt-relaxed-constexpr --expt-extended-lambda --cudart none
     -shared -Xcompiler -fPIC -gencode arch=compute_120,code=sm_120
     -DFA2_HAS_HDIM_128=1 -DFA2_HAS_BF16=1
     -I<isolated env>/nvidia/cuda_cccl/include
     -I<snapshot>/csrc -I<snapshot>/csrc/attention/flash_attn_2_src -I.../cutlass/include
     -l:libcudart.so.12 (PyTorch 2.7.1+cu128 自带)
     fa2_wrapper.cu + flash_fwd_hdim128_bf16_sm80.cu + flash_fwd_split_hdim128_bf16_sm80.cu
```

构建仅从**只读快照**取源码；未运行 `build_flashrt.sh`，未写 `/home/host/flashrt-store/page_engine/FlashRT/build`，原有 CPython-3.12 扩展未被覆盖。构建日志：`logs/fa2_build*.log`、`fa2_build_console*.log`。

### 7.2 切片正确性（同 GPU、同精度、同输入、正确计时）

`tools/bench_attention.py`，输入为真实采集的 `records/raw/attn_task1_p0.npz`（真实 q/k/v + 真实 mask）：

| 实现 | 数值 | 说明 |
| --- | --- | --- |
| `split_sdpa`（392×392 + 120×120，SDPA） | max_abs = 0.03125，cos = 0.99999958 vs baseline | 拆分本身无损（0.03125 是 bf16 粒度） |
| `fa2_split`（392×392 + 120×120，FlashRT FA2） | max_abs = 0.03125，cos = 0.99999958 vs baseline | **数值通过**，全部有限，无 NaN |
| 尝试单次 `272×392`（非等长） | cos 0.54–0.84，出现 NaN | **不可用**，已弃用 |

- 全部代表性形状均验证（392×392、120×120、360×513 cross、32×513 cross）；不是只挑有利形状。
- 动作段级（9 个观测，整模型接入后，同一观测的原始臂已确认逐位可重复）：FA2 臂 `bitwise_equal = false`，
  - `max_abs` 逐观测为 0.0078 / 0.0078 / **0.03125** / 0.0078 / 0.0078 / 0.0078 / 0.0078 / 0.0078 / 0.0117；
  - `mean_abs` 0.00063–0.00120；
  - `rel_max` 与 `max_abs` 同量级（首动作基线约 1.0，故相对误差 ≈ 0.8%–3.1%）；
  - 维分解：位置 max 0.0073 / rot6d max 0.0078 / **夹爪 max 0.03125**（夹爪维在 bf16 下量化步长即 0.03125，属精度粒度而非算法误差）。
- 动作单位：反归一化后的物理量（米 / 弧度 / 无量纲夹爪命令）。**不做"控制等价"的宣称**：`1e-3` 筛查线在夹爪维按 bf16 粒度为 0.03125，其余维 0.0078，均属同量级量化扰动，不能据此推断闭环成功率保持。
- 真实调用证据：`records/raw/bench_attention_fa2.json` 的 `variants.fa2_split` 分支 n=200 完整统计（per-call median 0.1299 ms），且 FA2 分支出错会抛异常而不是静默回退（`sdpa`/`split_sdpa` 分支不加载 `.so`）。

### 7.3 切片性能（完整切片，含转置/连续化/拼接）

每生成 120 次调用的累计耗时（真实张量，CUDA Event + sync）：

| 实现 | ms / 生成（120 次调用） | per-call median | 相对基线 | 语义 |
| --- | --- | --- | --- | --- |
| `baseline_masked_sdpa` | **9.77** | 0.0814 | 1.00x | 正确（eager SDPA + bool mask） |
| `unmasked_sdpa_full` | 5.74 | 0.0478 | 1.70x | **错误**（丢掉首帧 causal 隔离，仅作参考） |
| `fa2_single_unmasked_full` | 6.64 | 0.0553 | 1.47x | **错误**（同上，FA2 单次 392×392） |
| `split_unmasked_sdpa` | 15.14 | 0.1262 | 0.65x | 正确（392×392 + 120×120） |
| `fa2_split` | 15.59 | 0.1299 | 0.63x | 正确（392×392 + 120×120） |

- **关键结论**：FA2 核心内核本身并不慢（单次无 mask 392×392 为 6.64 ms，比带 mask 的 SDPA 快 1.47x），但"首帧 120 token 只能看自己"的 block mask 无法用 FA2 表达，必须拆成两次等长调用；拆分带来的额外 launch、`120×120` 计算与拼接拷贝（+9.4 ms）**远超**内核省下的时间，净结果为慢 60%。
- 计时包含 `permute/contiguous/cat` 与数据搬运，未只展示核心 kernel（n=200，每实现独立预热）。

---

## 8. 整模型接入与端到端 A/B

### 8.1 接入方式（显式、可恢复、零侵入）

- 在隔离副本内通过 `tools/probe_generation.py` 的 `install_instrumentation()` 以 **显式 monkey-patch** 方式替换 `JointAttention.mixed` 与 `F.scaled_dot_product_attention`，只影响实验进程；`src/openwam_iso` 源代码**未被修改**（哈希与 `provenance.json` 一致）。
- 分支由命令行显式选择（`sdpa` / `split_sdpa` / `fa2`），无环境变量全局替换、无静默回退；FA2 不可用会直接报错。
- 两臂在**同一进程、同一 GPU、同一环境**内**交替**运行（block1: sdpa → block2: fa2 → block3: split_sdpa → …），每组 3 个 block × 50 次真生成。

### 8.2 结果（`records/raw/ab_generation.json`，每臂 150 次真生成，逐次样本已保存）

| 臂 | n | median | mean | p95 | 分块 median（块1/2/3） | 逐次求和 | 相对基线 median |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `sdpa`（基线） | 150 | **295.18 ms** | 316.13 | 383.5 | 295.54 / 366.05 / 292.94 | 47,419.3 ms | 1.0000x |
| `fa2`（FlashRT） | 150 | 311.32 ms | 313.70 | 326.1 | 312.24 / 311.57 / 309.76 | 47,054.3 ms | **0.9482x（-5.47%）** |
| `split_sdpa` | 150 | 304.71 ms | 319.54 | 379.3 | 303.77 / 305.05 / 377.19 | 47,931.5 ms | 0.9687x（-3.23%） |

逐块分布（暴露档位问题，必须与结论一起读）：

```
block1 sdpa        med=295.54  sum=15546.6  前5次 = 437 427 458 445 388   ← 进程启动爬升
block1 fa2         med=312.24  sum=15675.2  前5次 = 311 311 325 312 311
block1 split_sdpa  med=303.77  sum=15209.7
block2 sdpa        med=366.05  sum=17125.5  ← 整块落入慢档
block2 fa2         med=311.57  sum=15861.7
block2 split_sdpa  med=305.05  sum=15337.3
block3 sdpa        med=292.94  sum=14747.2  ← 最快档
block3 fa2         med=309.76  sum=15517.4
block3 split_sdpa  med=377.19  sum=17384.5  ← 整块落入慢档
```

- **档位混淆确实存在**：`sdpa` block2（366 ms）与 `split_sdpa` block3（377 ms）整块落在 5.3 节描述的慢档，而 FA2 三块极稳（309.8–312.2，spread 2.5 ms）。
- 因此**不能**用 pooled median 简单相减下结论（block2 甚至会让 FA2 "赢" 15%）。稳健的读法有三条，结论一致：
  1. **同档位比较**：FA2 最快块 309.76 ms，仍慢于基线最快块 292.94 ms（-5.4%）、也慢于 split_sdpa 最快块 303.77 ms；
  2. **剔除每臂前 5 次启动样本**：sdpa 295.04 / fa2 311.32 / split_sdpa 304.78 → FA2 0.9477x；
  3. **微基准（与档位无关）**：正确语义下 FA2 切片慢 60%，见 7.3。
- 三臂 `recompute/skip` 均为 4/6（cache 轨迹未被数值差异改变）；FA2 臂动作段非逐位一致（0.0078–0.03125），见 7.2。
- 口径：本结果是**同进程、同 GPU、交替块**对照，能排除"GPU2 在线 vs GPU0 实验"的跨机污染，但受 9.4 节档位问题限制，**不宣称跨进程绝对时延**，也**不宣称闭环成功率保持**。

---

## 9. 收益归因、累计核对与本轮边界

### 9.1 三层收益

| 层级 | 基线 | FlashRT 切片 | 变化 | 说明 |
| --- | --- | --- | --- | --- |
| 注意力切片（120 次调用 / 生成） | 9.77 ms | 15.59 ms | **-59.6%** | 拆分 + SM80 兼容路径 |
| 完整 chunk 生成（32×10 动作） | 295.2 ms | 311.3 ms | **-5.47%** | 同进程交替 3×50 |
| 完整请求往返（WS） | 482–504 ms（慢档） | 未接入 WS 服务 | — | 本轮未在服务端接入，属慢档且绝对时延不可比，不作为结论 |

### 9.2 真实生成耗时累计（不用"次数 × 中位数"）

- 各轮真生成次数：`perf` 30 + `sweep` 9×8=72 + `A/B` 3 臂 ×150 = 450 → **本轮共 552 次真生成**（另有预热/诊断生成不计入统计）。
- 累计实际耗时（**按 450 条逐次样本求和**，不是"次数 × 中位数"）：

```
sdpa       n=150  sum=47,419.3 ms  median=295.18  p95=383.5
fa2        n=150  sum=47,054.3 ms  median=311.32  p95=326.1
split_sdpa n=150  sum=47,931.5 ms  median=304.71  p95=379.3
三臂合计 142.4 s
```

- 累计求和不重复计入嵌套阶段：`server_latency_ms`、`inference_s`、CUDA device 时间三者分属不同测量域，只作为互相校验，不叠加。
- 注意：`fa2` 的求和最短（47.05 s）**不代表更快**——它是因为基线臂在 block2 整块落入慢档（366 ms）而 FA2 恰好三块都在 311 ms；这正是必须看同档位比较（8.2）与微基准（7.3）的原因。

### 9.3 加速上限（明确假设下的估算，非闭环实测）

- 假设注意力耗时降为 0、其余不变：generation 上限收益 = 4.80/302 ≈ **1.6%**（若按切片自测的 9.77 ms 计，也仅 ≈ 3.2%）；
- 请求级上限则更低（<1%，因为 482 ms 里还有 36.4 ms 预处理 + 21.7 ms RPC + ~150 ms 未归因的服务端差异）；
- 与"≥10% 生成时延下降"的下一轮候选门槛相差一个数量级。

### 9.4 威胁项与不可信字段（据实披露）

1. **档位双峰（未解释）**：同输入同配置，生成时延在 ~292–305 ms 与 ~440–490 ms 两档之间跳变；既表现为进程启动爬升（A/B block1 基线前 5 次 437–458 ms），也表现为**整块 50 次落在慢档**（基线 block2 = 366 ms、split_sdpa block3 = 377 ms）。慢档出现在满频附近（2887 MHz / 上限 3090），已排除 DVFS 与 cache 轨迹差异（两档 recompute/skip 均为 4/6、joint 调用 120/生成）。影响：**任何绝对时延只在同进程同块内有效**；历史 482 ms 与本轮 WS 504 ms 不作加速比分母。档位幅度（±25%）大于本轮任何内核级差异（±5%），这本身说明"内核级微优化"不是当前系统的瓶颈。
2. **请求级 170 ms 差异未归因**：进程内 `predict` 312 ms vs WS 服务 482 ms（预处理 36.4 + RPC 21.7 只能解释 58 ms）。该字段标记为"条件可信"。
3. **采样矩阵差异**：`sdpa_calls` 计数（133/生成）与 profiler 内核计数（240/生成）不一致，原因是 instrumentation 包裹层级不同；本报告以 **profiler 内核计数与耗时**为准，切片微基准单独标注口径。
4. 本轮为**离线回放**，未做 60 回合闭环，不代表闭环成功率。

---

## 10. 后续方向建议（不作为本轮结论）

R2 在"首帧 causal 块 mask + 392 token 小序列"这一形态上**不值得继续**（可加速面 1.6%，且 FA2 在该形态反而更慢）。按收益排序的独立方向：

1. **GEMM 与整体 device 利用率**：`addmm` 33.9 ms/gen + cutlass GEMM ReLU 28.2 ms/gen（28.9%），是注意力的 13 倍；值得评估 fp8/更优 tile 或避免重复投影。
2. **DiT velocity cache 已经是最大杠杆**（60% skip）；若能提高 skip 率或降低 recompute 步成本，收益远大于注意力。
3. **请求路径非 GPU 部分**：PNG/base64 解码 + resize 36.4 ms（8% 请求时延）、逐动作 RPC 每次 21.7 ms（20 Hz 下若有 32 次单步 RPC，累计影响很大）。这些与 FlashRT 内核无关，**不得记作 FlashRT 收益**。
4. **档位双峰本身**：若能查明并稳定到快档，等效收益 ~45%，远超任何内核级优化。

---

## 11. 复现与清理

```bash
# 0) 环境（不改动主环境）
PY=/home/host/page-dmh/embodied_engine/.venv-openwam/bin/python   # Python 3.10.12 / torch 2.7.1+cu128

# 1) 启动隔离环境服务（GPU0，端口 14571-14573）
cd /home/host/page-dmh/openwam_harness/envs
nohup rpent/bin/python .../adapters/libero_env_server.py --transport http --host 127.0.0.1 \
  --port 14571 --suite libero_goal --task 1 --seed 0 --cuda-device 0 ...   # task 3/8 -> 14572/14573

# 2) 采集冻结观测（真实相机 + 真实 EEF10）
CUDA_VISIBLE_DEVICES=0 $PY /home/host/flashrt-exp/tools/capture_obs.py ...

# 3) 构建最小 FA2 切片（out-of-tree）
bash /home/host/flashrt-exp/tools/build_fa2_slice.sh          # -> build/libflashrt_fa2_slice.so

# 4) 切片正确性 / 性能
CUDA_VISIBLE_DEVICES=0 $PY /home/host/flashrt-exp/tools/bench_attention.py --impl all

# 5) 原始基线（多观测 + 请求路径）
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 $PY /home/host/flashrt-exp/tools/baseline_sweep.py

# 6) 端到端 A/B（3 块 × 50 次真生成，同进程交替）
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 $PY /home/host/flashrt-exp/tools/ab_generation.py

# 7) 清理（本轮实验进程全部停止；生产 14510/14520/14561 不要动）
kill $(cat /home/host/flashrt-exp/pids/openwam_iso_14521.pid) 2>/dev/null
pkill -f "libero_env_server.py --transport http --host 127.0.0.1 --port 1457" 
nvidia-smi --query-gpu=memory.used --format=csv,noheader -i 0    # 复核：5 MiB
```

**清理状态（本轮结束时实测）**：GPU0 = 5 MiB（无 compute 进程）；隔离 WS 服务与 3 个环境服务已停止；`page-dmh` HEAD 未变、受跟踪文件未改。

---

## 12. 修改文件清单与记录索引

### 12.1 工作区（`page-dmh`）新增，仅文档

- `embodied_engine/docs/audits/2026-10-31-flashrt-openwam-r2-poc.md`（本报告）
- `embodied_engine/records/acceptance/2026-10-31-flashrt-openwam-r2-poc.json`（机器可读验收记录）
- 未修改 `embodied_engine/docs/audits/2026-10-31-flashrt-openwam-acceleration-audit-r2.md`、`2026-10-29-*`、`config/engine.json` 及任何历史实验数据。

### 12.2 实验目录 `/home/host/flashrt-exp/`（新增，独立）

| 路径 | 内容 |
| --- | --- |
| `provenance.json` | 源码/环境来源与逐文件 sha256（480 + 6 个文件） |
| `records/manifest.json` | 本轮全部记录文件 sha256（25 个） |
| `records/obs/*.npz/.json` | 9 个冻结真实观测 |
| `records/raw/perf_task1_p0_30gen.json` | 正式基线与注意力形状直方图 |
| `records/raw/baseline_sweep.json` | 9 观测扫描 + 请求路径同进程对比 |
| `records/raw/idle_clock_probe.json` | 档位/时钟实验 |
| `records/raw/preprocess_bench.json` | 图像解码预处理 36.4 ms |
| `records/raw/c1_*.json` | C1 compile 状态证据 |
| `records/raw/c2_ws_timing*.json` | C2 计时嵌套证据（70 条） |
| `records/raw/bench_attention*.json` | 切片正确性/性能 |
| `records/raw/ab_generation.json` | 端到端 A/B（3 臂 × 150 次） |
| `records/raw/segment_*_*.npz`、`actions_*.npy` | 完整 32×10 动作段 |
| `tools/*.py`、`tools/build_fa2_slice.sh`、`tools/run_isolated_server.sh` | 本轮实验工具 |
| `build/libflashrt_fa2_slice.so` | 最小 FA2 扩展 |
| `logs/`、`pids/` | 隔离日志与 PID |

原始记录不复制进仓库（体积大、含 14 MB 张量），以路径 + sha256 引用，见验收 JSON 的 `evidence` 字段。
