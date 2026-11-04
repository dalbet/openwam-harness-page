# OpenWAM 候选 B：task1 单任务闭环验收（2026-11-02）

- 任务：`task_id=1` / `libero_goal` / `put_the_bowl_on_the_stove`（pick_place）
- 对照：**A = eager 基线**（生产同款 `--compile-enabled false`） vs **B = 候选 B**
- 规模：10 个数据集初始化 × 2 模式 = 20 个正式回合（另有 3 个预热回合、3 个补充回合）
- 隔离：GPU0 / 独立端口 / 独立配置副本 / 独立记录目录；生产 GPU1/2/3 与 5 个业务服务未参与实验
- 判定：**CLOSED_LOOP_PRELIM_PASSED**（并附一条必须知悉的运行事故，见 §9.1）

本轮**只**回答三件事：候选 B 在连续变化的新观测下能否正常跑闭环、是否出现行为退化、模型加速有多少转成完整任务收益。上一轮的静态审计与冻结观测性能测试不再重复。

---

## 0. 三问三答

**1）候选 B 在连续变化的新观测下能否正常执行 task1？——能。**
20 个正式回合全部跑到终态（`completed`），无超时、无 `unknown`、无后端异常、无引擎重试/重规划。B 组执行路径闸门逐项成立：编译循环存在、`wan_blocks` 逐块编译关闭、CUDA Graph 真实录制并重放（`records=1`，`replays≈4×生成次数`）、无回退 eager、无形状桶失控。每回合的观测都是当步新观测（引擎按 20 Hz 控制契约每动作重新取 `raw_obs` 并重构模型输入），不是缓存回放。

**2）是否出现抓取、夹爪、释放、停止或 reset 行为退化？——本轮样本未观察到。**
20/20 回合夹爪都闭合到物体上（最小开口 0.0010–0.0110 m），20/20 回合出现 ≥0.05 m 的阈值穿越，20/20 回合放置验证通过；策略自行释放 11 次（A 5 / B 6），策略未在窗口内释放时由既有执行层收尾器接管 9 次（A 5 / B 4），两种收尾在两组间的分布相当，没有任何一组出现"未松手""提前打开""反复修正"。每回合之间的 reset 都按既有契约执行（`stop → env.reset → 模型侧缓冲 reset`），未观察到动作跨回合串用。A 组 init4 出现 216 步/51.8 s 的离群回合：引擎 `recovery_count=0`（不是引擎重试），是策略自己生成了 6 个动作块、native goal 到第 190 步才命中；B 同 init 为 4 块 120 步。

**3）模型加速有多少转化为完整任务收益？——单次生成时延降 49%，但只换到约 0.9 s（−2.9%）的回合墙钟中位改善。**
在闭环真实条件下单次生成 425.3 → 215.7 ms（配对中位比值 0.512）。但一次生成要摊到 32 个控制动作上，每回合只有 3 次真实生成：生成时间只占回合墙钟的 4.6%（A）/ 2.3%（B）。其余时间被每动作固定开销吃掉（观测构造约 4.0 s、`env.step` 约 6.9 s、WS/服务传输约 9–10 s、状态快照约 0.9 s）。配对结果：墙钟中位差 −0.87 s（−2.9%），模型等待中位差 −0.60 s——即**模型侧省下的时间基本全额落到回合墙钟上，但模型本身只占墙钟很小一块**，所以任务级收益是"与占比相称的小改善"，不是数量级改善。

---

## 1. 现场与隔离

### 1.1 实验前现场

| 资源 | 状态 |
|---|---|
| GPU0 | 2 MiB / 0%，空闲 → 本轮使用 |
| GPU1 | 7174 MiB，Pi0.5 `:14510`（pid 2784215） |
| GPU2 | 24778 MiB，生产 OpenWAM `:14520`（pid 1380923） |
| GPU3 | 22108 MiB，planner vLLM `:8001` + `VLLM::EngineCore`（pid 2843863） |
| 业务服务 | `:14600` inference、`:14700` agent、`:14800` engine、`:14900` page_ui、`:14561` 生产 env |

### 1.2 隔离方式

- 复制一份代码树到 `/home/host/flashrt-exp/closed_loop/embodied_engine`（8.7 MB，排除 `.venv-openwam`/`records`/`__pycache__`），改动**只有**端口与设备：

| 配置键 | 生产 | 本实验 |
|---|---|---|
| `services.engine.port` | 14800 | 14801 |
| `services.inference.port` | 14600 | 14601 |
| `services.agent.port` | 14700 | 14701（冻结计划 stub） |
| `services.env.port` | 14561 | 14571 |
| `services.openwam.port` / `server_url` | 14520 | 14521 |
| `action_model.openwam.gpu_index` | 2 | 0 |
| `env.cuda_device` | 3 | 0 |

对 `config/engine.json` 做逐键 diff 的结果就是上表 10 个键，**其余全部逐字节相同**：`max_steps=600`、`generate_horizon=32`、`execute_horizon=32`、`denoise_steps=10`、`control_frequency_hz=20`、`post_goal.window_actions=31`、`release_open_m=0.05`、`release_confirm_samples=2`、`finalizer` 配置、bf16、检查点与归一化均未改动。主 `config/engine.json` 未被修改（sha256 前缀 `9235ad36…` 未变）。

- 记录写入隔离树自己的 `records/`，主仓库 `records/` 未被写入（生产 `records/runs` 最后一个文件仍是 10-28）。
- 未升级任何依赖；推理/引擎用生产同款 `envs/rpent/bin/python`，模型服务用生产同款 `.venv-openwam/bin/python`。

### 1.3 planner 冻结

本轮不请求生产 planner，也不让 planner 变化混进执行对照：用一个只返回固定计划的本地 stub（`:14701`）替代 agent 层，返回结构与真实 `/plan` 响应同构（字段、嵌套层级取自历史真实单任务记录），只接受 `task_id=1` 且 `backend=openwam`，否则 400。

26 次调用（= 26 个回合）全部 `status=ok`、`backend=openwam`、指令均为 `put the bowl on the stove`、计划哈希全部为 `068d66aae4b0ad82`。原始日志：`raw/plan_calls.jsonl`。

> **范围声明**：本报告验收的是"固定任务计划下的动作执行闭环"，**不是**"自然语言 → 规划 → 完成任务"的全链路性能验收。planner 延迟（真实约 0.6 s）在两侧都被 stub 抹平，不在结论里。

---

## 2. 两个对照模式的冻结参数

### 2.1 原始组 A（eager）

生产同款入口与参数，只换设备/端口：

```bash
cd /home/host/page-dmh/openwam_harness/OpenWAM
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 \
/home/host/page-dmh/embodied_engine/.venv-openwam/bin/python scripts/deploy.py \
  --ckpt-dir /home/host/page-dmh/openwam_harness/checkpoints/OpenWAM-Alpha-Sim-LIBERO \
  --ckpt-name checkpoint_step_10690.safetensors \
  --device cuda:0 --host 127.0.0.1 --port 14521 \
  --inference-mode sync --denoise-steps 10 --inference-horizon 32 \
  --compile-enabled false
```

### 2.2 优化组 B（候选 B）

同一入口 + 上一轮验收通过的三个开关与两个环境变量：

```bash
cd /home/host/page-dmh/openwam_harness/OpenWAM
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 \
TORCHINDUCTOR_EMULATE_PRECISION_CASTS=1 \
C_INCLUDE_PATH=/home/host/.local/share/uv/python/cpython-3.10.22-linux-x86_64-gnu/include/python3.10 \
TORCHINDUCTOR_CACHE_DIR=/home/host/flashrt-exp/closed_loop/cache/inductor \
TRITON_CACHE_DIR=/home/host/flashrt-exp/closed_loop/cache/triton \
CLOSED_LOOP_CG_OUT=/home/host/flashrt-exp/closed_loop/raw/cudagraph_B.json \
/home/host/page-dmh/embodied_engine/.venv-openwam/bin/python \
  /home/host/flashrt-exp/closed_loop/tools/closed_loop_serve.py \
  --ckpt-dir /home/host/page-dmh/openwam_harness/checkpoints/OpenWAM-Alpha-Sim-LIBERO \
  --ckpt-name checkpoint_step_10690.safetensors \
  --device cuda:0 --host 127.0.0.1 --port 14521 \
  --inference-mode sync --denoise-steps 10 --inference-horizon 32 \
  --compile-enabled true \
  optimization.compile.self_attn.torch_mode=reduce-overhead \
  optimization.compile.wan_blocks.enabled=false
```

`closed_loop_serve.py` 是薄包装：与 `scripts/deploy.py` 同样装配 `sys.path`，随后原样调用 `openwam.deploy.server.main()`；唯一附加物是 cudagraph 树"录制/重放/退回 eager"的计数（每生成 4 次调用），**不包装** preprocess / predict / generate，因此不改每请求路径。启动日志确认：

```
openwam.model.video_backbone.wan_backbone: Wan block compile disabled by config; running eager.
openwam.model.architectures.dual_system.joint_self_attn: Enabled dual self-attn MoT compile
    with torch.compile kwargs={'dynamic': False, 'mode': 'reduce-overhead'}
```

未启用 `wan_blocks` 逐块编译，未追加任何其他优化。

### 2.3 启停脚本

`/home/host/flashrt-exp/closed_loop/tools/arm.sh {start A|start B|stop|status}`：`stop` 先校验 cmdline 含 `14521` 再发信号，避免误杀；日志按会话带时间戳保留。

---

## 3. 接入的是真实单任务控制链

每回合由隔离栈里**真实的** `engine/engine_service.py` 驱动，链路与生产完全一致：

```
POST /api/run
  → Engine._prepare_env           EnvClient.ensure(task,seed) + env.reset()
  → POST {agent}/plan             冻结计划 stub（仅此一处与生产不同）
  → WholeTaskRunner.run()         observe(raw_obs→build_openwam_obs) → InferenceClient.infer
                                  → native_eef10_to_libero7d → env.step → state_snapshot
                                  每动作重取新观测；32/32 重规划；释放窗口与停止判定原样
  → Engine._finish_placement      finalize.policy_release_block + 执行层收尾（若策略未释放）
  → records/runs/<episode_id>.json
```

- 动作转换、滑动缓冲、停止标志、成功判定、释放窗口、收尾器全部来自仓库原有代码，本轮没有改写、没有简化脚本替代。
- 每个回合先 `POST /api/load`（指定 init seed，确认场景起好、`scene_epoch` 就绪）再 `POST /api/run`；回合结束 `POST /api/reset`，它会 `stop()` + `env.reset()` + `InferenceClient.reset()`（清模型侧动作缓冲），因此动作不会跨回合串用。
- 单一引擎实例、单控制客户端，没有共享动作 buffer。

---

## 4. 预热与执行路径闸门

### 4.1 预热（不计入正式统计）

每个服务会话前先跑 1 个预热回合（init 0），把编译、图捕获、首调用全部吃掉：

| 会话 | 预热回合墙钟 | 首个真实生成 | 说明 |
|---|---|---|---|
| A（eager） | 29.5 s / 30.2 s | — | 无编译，仅模型装载 |
| B（冷 inductor 缓存） | 111.9 s | 85.85 s | 首次 `torch.compile` + 图捕获 |
| B（热 inductor 缓存） | 44.1 s | 16.19 s | 复用 `cache/inductor` |

模型装载：A 约 110 s（08:38:00→08:39:52），B 约 115 s（08:46:25→08:48:20，冷缓存那次约 124 s）。这些数字**没有**进入任何正式回合的计时；正式计时也没有剔除运行中的真实停顿。

### 4.2 B 组闸门证据（每回合成立，非按模式名推断）

| 闸门 | 证据 | 结果 |
|---|---|---|
| compiled loop 存在 | 启动日志 `Enabled dual self-attn MoT compile ... mode='reduce-overhead'` | 成立 |
| `wan_blocks` 逐块编译关闭 | 日志 `Wan block compile disabled by config; running eager.` | 成立 |
| 图真实录制 | `cudagraph_B_final.json`: `installed=true, warmups=1, records=1` | 成立 |
| 图真实重放 | `replays=130`，同会话真实生成 33 次 → 4.0 次/生成，与"每生成 4 次重算"一致 | 成立 |
| 正式回合内无回退 eager | 进程级计数为 `eager_fallbacks=1`（**不是 0**）：该计数落盘不带时点，无法自证发生在哪一刻；判定依据是运行日志无 `falling back to eager`，且 33 次生成的时延全部落在编译路径档（≈215 ms），没有出现 eager 档（≈425 ms）的离群点 | 正式回合成立，计数本身非零（见 §10 限制） |
| 无异常重编译/形状桶失控 | 全会话 `records` 保持 1（未出现第二次录制） | 成立 |
| 生成确实走编译路径 | 生成中位 215.7 ms，而 eager 同条件为 425.3 ms；若回退 eager 不可能出现该分布 | 成立 |

`cudagraph_B_final.json` 是从 B 会话（pid 188895，跑完 10 个正式回合）周期性落盘的计数，取值为该进程最后一帧，`replays` 比理论值 132 少 2，是 5 s 落盘周期与进程结束之间的尾巴。

正式运行期间**没有**挂任何逐请求计时钩子；计数钩子每生成只触发 4 次，量级可忽略。

---

## 5. 20 回合清单与顺序

### 5.1 init 选择

LIBERO `libero_goal/put_the_bowl_on_the_stove.pruned_init` 共 50 个初始化；env 侧 `rid = Σ(前序任务 init 数) + seed % 50`，本轮 **seed = init_id ∈ [0,9]**，因此 10 个 init 互不相同。任务预算 `max_steps=600`，本轮实际最多用到 216 步。

### 5.2 顺序（预注册后未改动）

| 会话 | 进程 | 内容 | 顺序类型 |
|---|---|---|---|
| S1 | A（pid 137389） | 预热 + init 0,1,2,3,4 | AB（A 先） |
| S2a | B（pid 188895） | 预热 + init 0,1 | AB（A 先）——**冒烟门槛在此** |
| S2b | B（同上进程） | init 2,3,4,5,6,7,8,9 | A 先（pair 2–4）/ B 先（pair 5–9） |
| S3 | A（pid 274815） | 预热 + init 5,6,7,8,9 | BA（B 先） |

- 平衡：pair 0–4 为 AB，pair 5–9 为 BA，共 5/5。
- 冒烟：前 2 对（4 个正式回合）完成后检查——4/4 成功、无后端异常、无输入失效、无控制异常，才继续余下 8 对。冒烟配置未变，这 4 个回合按预注册并入同一清单，未拼接不同版本。
- 每对都用 `/api/load` 独立 reset 回同一数据集初始化，不从上一组结束状态继续。

---

## 6. 逐回合记录

### T1 逐回合（完整记录见 `raw/records/<tag>_<arm>_init<N>_<episode>.json`）

| # | arm | init | session | 墙钟 s | 动作数 | 控制 Hz | 真实生成 | 缓存出队 | 生成中位 ms | 停止原因 | 策略释放 | 收尾来源 | 结果 |
|---|-----|------|---------|--------|--------|---------|----------|----------|-------------|----------|----------|----------|------|
| 1 | A | 0 | S1 | 27.55 | 98 | 3.89 | 3 | 92 | 427.7 | policy_release | 是 | policy_chunk | 成功 |
| 2 | B | 0 | S2a | 25.55 | 94 | 4.09 | 3 | 88 | 213.7 | policy_release | 是 | policy_chunk | 成功 |
| 3 | A | 1 | S1 | 25.88 | 95 | 4.10 | 3 | 89 | 433.3 | policy_release | 是 | policy_chunk | 成功 |
| 4 | B | 1 | S2a | 25.93 | 98 | 4.21 | 3 | 92 | 216.6 | policy_release | 是 | policy_chunk | 成功 |
| 5 | A | 2 | S1 | 25.87 | 95 | 4.04 | 3 | 89 | 424.5 | policy_release | 是 | policy_chunk | 成功 |
| 6 | B | 2 | S2b | 25.14 | 92 | 4.11 | 3 | 86 | 217.5 | policy_release | 是 | policy_chunk | 成功 |
| 7 | A | 3 | S1 | 26.35 | 97 | 4.05 | 3 | 91 | 466.5 | policy_release | 是 | policy_chunk | 成功 |
| 8 | B | 3 | S2b | 28.44 | 98 | 3.84 | 3 | 92 | 221.1 | policy_release | 是 | policy_chunk | 成功 |
| 9 | A | 4 | S1 | 51.77 | 216 | 4.10 | 6 | 186 | 425.6 | chunk_exhausted | 否 | execution_finalizer | 成功 |
| 10 | B | 4 | S2b | 30.04 | 120 | 3.85 | 3 | 93 | 221.2 | chunk_exhausted | 否 | execution_finalizer | 成功 |
| 11 | A | 5 | S3 | 30.45 | 120 | 3.81 | 3 | 93 | 427.2 | chunk_exhausted | 否 | execution_finalizer | 成功 |
| 12 | B | 5 | S2b | 27.63 | 99 | 4.01 | 3 | 93 | 191.8 | policy_release | 是 | policy_chunk | 成功 |
| 13 | A | 6 | S3 | 30.91 | 120 | 3.80 | 3 | 93 | 422.5 | chunk_exhausted | 否 | execution_finalizer | 成功 |
| 14 | B | 6 | S2b | 30.02 | 120 | 3.87 | 3 | 93 | 215.9 | chunk_exhausted | 否 | execution_finalizer | 成功 |
| 15 | A | 7 | S3 | 28.86 | 98 | 3.79 | 3 | 92 | 421.8 | policy_release | 是 | policy_chunk | 成功 |
| 16 | B | 7 | S2b | 28.01 | 98 | 3.86 | 3 | 92 | 218.1 | policy_release | 是 | policy_chunk | 成功 |
| 17 | A | 8 | S3 | 30.50 | 120 | 3.80 | 3 | 93 | 417.2 | chunk_exhausted | 否 | execution_finalizer | 成功 |
| 18 | B | 8 | S2b | 30.04 | 120 | 3.85 | 3 | 93 | 216.7 | chunk_exhausted | 否 | execution_finalizer | 成功 |
| 19 | A | 9 | S3 | 30.91 | 120 | 3.78 | 3 | 93 | 425.0 | chunk_exhausted | 否 | execution_finalizer | 成功 |
| 20 | B | 9 | S2b | 30.03 | 120 | 3.86 | 3 | 93 | 224.7 | chunk_exhausted | 否 | execution_finalizer | 成功 |

每回合记录的其它字段（`release_evidence`、`post_goal.rows` 的逐动作夹爪开度、`observations`、`action_trace`、`recovery_log`）都在 `raw/records/*.json` 原样保存。

### 6.1 行为判据的检查结果

| 检查 | 判据 | 结果 |
|---|---|---|
| 是否抓到物体 | 最小夹爪开口 < 0.02 m 且随后被抬升 | 20/20 成立（最小开口 0.0010–0.0110 m） |
| 是否提前打开 | 原生目标之前出现 ≥0.05 m 开口 | 0 次 |
| 是否松手 | 停止时开口 ≥0.05 m（策略或收尾器） | 20/20 成立 |
| 是否未松手 | `release_evidence.state == verified` | 20/20 `verified` |
| 放置是否成立 | `placement.placement_verified` | 20/20 true |
| 策略是否反复修正 | `recovery_count` / `recovery_log` | 20/20 为 0 |
| 动作是否卡顿 | 动作间隔 p95 / 最大间隔 | A 0.26 s、B 0.26 s；无 >0.5 s 的停顿（预热除外） |
| 动作是否跨回合串用 | 每回合 `sequence_id`/`chunk_id` 从 0 起且 `/api/reset` 清缓冲 | 无串用 |
| 停止是否正常 | 终态 `completed`、`stop_reason` ∈ {policy_release, chunk_exhausted} | 20/20 |

`chunk_exhausted` 的含义是：原生目标命中时，本块 32 个动作只剩 ≤1 个，释放窗口无动作可用，于是由**既有执行层收尾器**（`evidence_source="execution_finalizer"`，开夹爪 12 步 + 撤离探针）完成释放。两组都出现（A 5 次 / B 4 次），属于该策略 32 步块边界与目标时刻的相位差，**不是**候选 B 引入的行为变化，也不需要改释放窗口。

### 6.2 起点一致性

| 检查 | 结果 |
|---|---|
| `initial_scene_fingerprint`（对象位姿 + EEF 位姿）逐对相等 | 10/10 |
| `initial_state_fingerprint`（对象位姿 + 全部关节角 + EEF 位姿 + 夹爪开度）逐对相等 | 10/10 |
| 指令、init seed、数据集选择规则逐对相同 | 10/10 |
| 初始像素观测独立哈希 | **未取到，已作废该指标**（见下） |

> 说明：第一版实现里"初始帧哈希"取的是 `/api/frame`，但 `/api/frame` 只返回引擎缓存的上一帧；`_set_frames` 在 `obs.main_images is None` 时直接返回，因此 `load()` 之后取到的那一帧可能是**上一回合余帧**——冒烟阶段两对 4 个回合都出现了"指纹相等但帧哈希不等"的现象，正是这个原因。该指标因此作废，报告中不据此下任何结论；起点一致性以引擎自己写进记录的两个指纹为准（它们覆盖对象位姿、全部关节角、EEF 与夹爪，比"只比 qpos"更强，但确实**不含像素**）。

---

## 7. 公平分析

### 7.1 宿主状态（独立 CPU 探针）

`tools/bg_cpu_monitor.py`（进程外、绑核、固定工作量）全程运行，62 574 个样本，中位 8.89 ms，最快 8.83 ms。按上一轮同一定档规则（档位 {8.78, 11.37, 16.96, 26.24} ms，scale=1.006）：**20 个正式回合全部落在最快档 8.78 ms**，10/10 配对同档。本轮没有出现需要单列的跨档样本，也没有为等档位而人为等待。

### 7.2 T2 配对结果

| init | A 墙钟 | B 墙钟 | Δ墙钟 | A 动作 | B 动作 | A 生成中位 | B 生成中位 | 生成比 | Δ模型等待 | 同档 | 起点状态一致 |
|------|--------|--------|-------|--------|--------|------------|------------|--------|-----------|------|--------------|
| 0 | 27.55 | 25.55 | -2.00 | 98 | 94 | 428 | 214 | 0.500 | -1.25 | 是 | 是 |
| 1 | 25.88 | 25.93 | +0.05 | 95 | 98 | 433 | 217 | 0.500 | -0.38 | 是 | 是 |
| 2 | 25.87 | 25.14 | -0.73 | 95 | 92 | 424 | 217 | 0.512 | -0.99 | 是 | 是 |
| 3 | 26.35 | 28.44 | +2.09 | 97 | 98 | 467 | 221 | 0.474 | -0.37 | 是 | 是 |
| 4 | 51.77 | 30.04 | -21.73 | 216 | 120 | 426 | 221 | 0.520 | -13.55 | 是 | 是 |
| 5 | 30.45 | 27.63 | -2.81 | 120 | 99 | 427 | 192 | 0.449 | -1.44 | 是 | 是 |
| 6 | 30.91 | 30.02 | -0.89 | 120 | 120 | 422 | 216 | 0.511 | -0.58 | 是 | 是 |
| 7 | 28.86 | 28.01 | -0.85 | 98 | 98 | 422 | 218 | 0.517 | -0.59 | 是 | 是 |
| 8 | 30.50 | 30.04 | -0.46 | 120 | 120 | 417 | 217 | 0.519 | -0.60 | 是 | 是 |
| 9 | 30.91 | 30.03 | -0.88 | 120 | 120 | 425 | 225 | 0.528 | -0.48 | 是 | 是 |

- 全部回合的成功矩阵：**两组都成功 10 / 仅 A 成功 0 / 仅 B 成功 0 / 两组都未成功 0**。
- 失败、`unknown`、后端异常：各 0。
- 配对墙钟差：中位 **−0.87 s**，均值 −2.82 s（均值被 init4 拉大）；比值中位 **0.9714**。
- **init4 的 −21.7 s 不能算作加速收益**：A 用了 216 步、B 用了 120 步，是策略自身轨迹分叉（A 在该 init 反复修正），不是"更快完成同一轨迹"。剔除 init4 后的 9 对：中位 −0.879 s，均值 −0.72 s。
- 生成比全部落在 0.449–0.528（中位 0.512），说明**单次生成收益在 20 个回合里稳定复现**，与墙钟差的小幅波动（含 +2.09 s 的反向样本）来自动作数差异 ±3 的环境抖动，而非生成路径不稳定。

### 7.3 T4 计数

| 项 | A | B |
|----|---|---|
| 正式回合 | 10 | 10 |
| 原生目标成功 | 10 | 10 |
| 策略自行释放已验证 | 5 | 6 |
| 执行层收尾释放 | 5 | 4 |
| 放置验证通过 | 10 | 10 |
| 真实生成次数 | 33 | 30 |
| 缓存出队次数 | 1011 | 915 |
| 引擎重试/重规划 | 0 | 0 |
| 超时 | 0 | 0 |
| `unknown` 终态 | 0 | 0 |

生成次数差异（33 vs 30）完全来自动作数差异（A 有 216 步的离群回合）。

---

## 8. 时间归因：加速为什么只换来小收益

### 8.1 T3 每回合均值

| 项 | A | B |
|----|---|---|
| 墙钟 s | 30.91 | 28.08 |
| 模型调用等待 s（含缓存出队与传输） | 14.45 | 12.42 |
| `env.step` s | 7.07 | 6.65 |
| 观测构造 s | 4.08 | 3.94 |
| 状态快照 s | 0.89 | 0.84 |
| 单次生成中位 ms | **429.1** | **215.7** |
| 缓存出队中位 ms | 15.40 | 14.60 |
| 动作间隔中位 s | 0.24 | 0.25 |
| 仿真动作数（均值） | 117.9 | 105.9 |
| 控制频率 Hz | 3.92 | 3.96 |
| 模型等待/墙钟 | 46.7% | 44.2% |

### 8.2 真实生成时间 vs 每动作固定开销

| 项 | A | B |
|---|---|---|
| 每回合真实生成时间（生成次数 × 生成中位） | 1.42 s | 0.65 s |
| 占墙钟 | 4.6% | 2.3% |
| 每回合缓存出队时间 | 1.56 s | 1.33 s |
| 每回合其它（观测/传输/`env.step`/快照/转换） | 约 27.9 s | 约 26.1 s |

拆开看就清楚了：

1. 一次生成摊到 32 个控制动作，每回合只有 3 次生成（B 平均 3.0 次/回合）。
2. 每动作固定开销约 0.25 s（其中服务端到推理服务的 HTTP 往返 + 观测序列化约 0.10 s，`env.step` 约 0.07 s，`raw_obs` 约 0.04 s，快照约 0.01 s），生成快慢几乎不影响它。
3. 因此"生成 −49%"落到回合上的理论上限就是"生成时间的 49%"，即约 0.7 s/回合；实测模型等待中位改善 −0.60 s、墙钟中位 −0.87 s，与上限同量级。

**这不是把收益藏起来**：本轮的运行条件（`execute_horizon=32`、20 Hz 控制契约、`decode_video=false`、单 PolicyServer 单客户端）全部沿用既有配置，没有提高控制频率、没有减少动作、没有改变回放速度。反过来也说明：**在当前闭环形态下，模型生成已经不是主要瓶颈**；要拿到更大的任务级收益，需要动的是每动作固定开销（观测构造/传输）或执行视界，而不是继续压生成延迟。

### 8.3 与上一轮"294.2 → 166.9 ms"的差异（必须知道）

本轮的绝对生成时延是 **425 → 216 ms**，比上一轮台架上的 294 → 167 ms 高。差异来自测量制度，不是候选 B 变了：

| 差异项 | 上一轮台架 | 本轮闭环 |
|---|---|---|
| `torch` 线程数 | 工具里显式 `torch.set_num_threads(8)`（记录中 `threads: 8`） | deploy 服务端默认（未设置 `OMP_NUM_THREADS` 等，宿主 128 vCPU / 64 物理核） |
| GPU 上的邻居 | GPU0 只有模型 | GPU0 上同时有 LIBERO 的 EGL 渲染上下文 |
| 客户端 | 性能工具直接调用 | 真实 `engine → inference → WS` 控制链 |

eager（CPU 派发重）从 294 → 425 ms，候选 B（CUDA Graph 重放，受 CPU/邻居影响小）从 167 → 216 ms，**相对收益方向与量级一致（台架 −43.3% vs 闭环 −48.9%）**。本轮没有为归因这两个因素单独做实验（那需要另起服务进程、改动服务端线程设置，已超出本轮范围），因此上面两行只是差异清单，**不作为因果结论**。

本轮所有对照都在**同一制度**下取得，配对比较不受影响。

---

## 9. 交付、状态与限制

### 9.1 ⚠ 运行事故与恢复（必须知悉）

- **事件**：全部 20 个正式回合与补充视频跑完之后、清理实验进程时，我用了一条过宽的 `pkill -f "...|engine/engine_service.py"`，正则的第二个分支同时命中了**生产引擎 `:14800`（pid 3456863）**，把它一并停掉。生产 env 服务 `:14561` 随引擎退出而停止。
- **影响**：生产引擎进程内状态（已加载场景、`last_result`、metrics 计数）被清空；下一次使用需要重新装载场景（约 16 s）。
- **恢复**：按仓库 `scripts/restart_engine.sh` 的同款命令立即重启：
  ```bash
  cd /home/host/page-dmh/embodied_engine
  nohup /home/host/page-dmh/openwam_harness/envs/rpent/bin/python engine/engine_service.py \
    > records/logs/engine_service.log 2>&1 &
  ```
  新 pid 347662，`:14800/health` 正常，`records/engine_service.pid` 已更新。随后 `POST /api/load {"task_id":1,"seed":0}` 验证 env 能重新拉起（16.6 s），`:14561` 恢复监听，引擎回到 `phase=ready`。
- **未受影响**：GPU2 生产 OpenWAM（pid 1380923，未重启）、GPU1 Pi0.5（pid 2784215）、GPU3 vLLM（pid 2843863）、`:14600` / `:14700` / `:14900` 全部原进程在跑；GPU 显存回到 7174 / 24778 / 23078 MiB。
- **对结论的影响**：无。事故发生在数据采集**之后**，20 个正式回合都在隔离端口上完成，与生产无关；但它是一次真实的运维事故，如实记录在此，且后续同类清理必须按 PID + cmdline + 端口三元确认（`arm.sh stop` 就是这么做的）。

### 9.2 分母与完成度

| 项 | 计划 | 实际 |
|---|---|---|
| 正式回合 | 20（10 init × 2 模式） | **20** |
| 不同初始化 | 10 | **10**（init 0–9，无重复） |
| 预热回合（排除计时） | — | 3 |
| 补充回合（视频，排除计时） | — | 3（A/B 各 1 个计时回合 + 1 个 B 预热） |
| 回合记录 | 20 | 20（+ 6 个非正式，全部原样保留） |
| 视频 | 至少 1 对 | 2 段（`videos/VID_A_init0.mp4`、`videos/VID_B_init0.mp4`） |

### 9.3 最终状态

**`CLOSED_LOOP_PRELIM_PASSED`**：控制链正常，20/20 回合完成，本轮样本未观察到行为退化。

明确限定：
- 10 个初始化属于**初步行为验证**。即便成功数相同，也只能说"本轮样本未观察到成功率下降"，**不能**宣称统计意义上的控制等价或成功率非劣（10 对样本、全成功、无失败事件，无法给出有意义的置信区间）。
- 仍然只覆盖单任务 `task1`，未扩展到其他单任务或组合任务。
- 本轮收益是"模型生成侧"的收益，**不是** FlashRT 内核收益，两者不可混记。
- 未自动切换生产服务；生产仍跑 eager（`--compile-enabled false`）。

### 9.4 结论与建议

1. **可以**把候选 B 用于其他单任务的行为验证（推荐扩到 2–3 个 `libero_goal` 任务、每任务 10 init × 2 模式），因为闭环控制链与仿真侧都没有出现与候选 B 相关的异常。
2. **不建议**现在就按"任务会明显变快"来立项：在本闭环形态下生成只占墙钟 2.3%–4.6%，单任务墙钟收益约 −2.9%（中位）。若目标是把任务完成时间压下来，下一步应该量化每动作固定开销（观测构造 ~0.12 s/动作、服务间传输 ~0.10 s/动作），而不是继续压生成。
3. 释放窗口、动作预算、成功判定阈值**未改动**，也不建议为了"让策略自己释放"而改：`chunk_exhausted` 在两组中同等出现，收尾器行为一致。

### 9.5 未覆盖/未验证

- 单个 GPU0 同时承载模型与 EGL 渲染，与生产"模型 GPU2 / 渲染 GPU3"的放置不同；绝对时延不可直接外推到生产，配对差异不受影响。
- 本轮未做"线程数"与"GPU 邻居"的归因实验（见 §8.3）。
- 未验证 `execute_horizon` 变化、多客户端并发、`decode_video=true`、planner 真实延迟参与等条件。
- A 组第一个会话与 B 组主会话的进程日志在重启时被覆盖（脚本当时用 `: >` 截断）；这两段的权威证据是逐回合 run record + cudagraph 计数快照，`arm.sh` 已改为按会话保留日志，重跑不会再丢。
- 初始像素观测未独立哈希（§6.2）。
- **`eager_fallbacks` 进程级计数为 1、不是 0**（§4.2）。落盘计数不带时点，因此"发生在启动预热、正式回合内为 0"是用运行日志与 33 次生成时延分布推出来的，不是计数直接给出的。若要硬性拿到"正式回合 0 回退"，需要给计数钩子加阶段标签后重跑。
- 单次生成在闭环里是 215.7 ms、台架是 166.9 ms（eager 侧 429.1 vs 294.2 ms）。差异来自线程池与 GPU0 上的 EGL 上下文，本轮未做归因；因此"−48.9%"只适用于台架口径，闭环口径是 −43.3%。

---

## 10. 复现命令

```bash
ROOT=/home/host/flashrt-exp/closed_loop

# 0) 冻结计划 stub（替代生产 planner，端口 14701）
/home/host/page-dmh/openwam_harness/envs/rpent/bin/python $ROOT/tools/plan_stub.py --port 14701

# 1) 宿主 CPU 探针（进程外，供档位判定）
cd /home/host/flashrt-exp && /home/host/page-dmh/embodied_engine/.venv-openwam/bin/python \
  tools/bg_cpu_monitor.py --out closed_loop/raw/bg_speedo.jsonl --seconds 10800

# 2) 原始组（eager）或优化组（候选 B）
$ROOT/tools/arm.sh start A     # 或 start B
$ROOT/tools/arm.sh status
$ROOT/tools/arm.sh stop

# 3) 隔离推理服务 + 隔离引擎（读隔离树的 config/engine.json）
cd $ROOT/embodied_engine
nohup /home/host/page-dmh/openwam_harness/envs/rpent/bin/python inference/inference_service.py \
  --port 14601 > $ROOT/logs/inference_14601.log 2>&1 &
nohup /home/host/page-dmh/openwam_harness/envs/rpent/bin/python engine/engine_service.py \
  > $ROOT/logs/engine_14801.log 2>&1 &

# 4) 跑回合（每个会话一条；--no-warmup 表示复用已预热的进程）
/home/host/page-dmh/openwam_harness/envs/rpent/bin/python $ROOT/tools/run_episodes.py \
  --arm A --tag S1 --inits 0,1,2,3,4 --warmup-init 0 --out session_S1.jsonl
/home/host/page-dmh/openwam_harness/envs/rpent/bin/python $ROOT/tools/run_episodes.py \
  --arm B --tag S2 --inits 0,1 --warmup-init 0 --out session_S2a.jsonl
/home/host/page-dmh/openwam_harness/envs/rpent/bin/python $ROOT/tools/run_episodes.py \
  --arm B --tag S2 --inits 2,3,4,5,6,7,8,9 --no-warmup --out session_S2b.jsonl
/home/host/page-dmh/openwam_harness/envs/rpent/bin/python $ROOT/tools/run_episodes.py \
  --arm A --tag S3 --inits 5,6,7,8,9 --warmup-init 0 --out session_S3.jsonl

# 5) 汇总
/home/host/page-dmh/embodied_engine/.venv-openwam/bin/python $ROOT/tools/analyze.py

# 6) 收尾（按 PID + cmdline + 端口三元确认后再杀；勿用过宽 pkill）
$ROOT/tools/arm.sh stop
for p in $(ss -ltnp | grep -E ":14571|:14601|:14701" | grep -o 'pid=[0-9]*' | cut -d= -f2 | sort -u); do
  tr '\0' ' ' < /proc/$p/cmdline     # 先确认身份
done
```

### 产物索引

| 内容 | 路径 |
|---|---|
| 逐回合原始记录（引擎原样输出） | `closed_loop/raw/records/*.json`（26 个） |
| 会话级收集（含 load/reset/健康快照/视频索引） | `closed_loop/raw/session_S1.jsonl`、`session_S2a.jsonl`、`session_S2b.jsonl`、`session_S3.jsonl`、`session_video_A/B.jsonl` |
| 汇总数字 | `closed_loop/raw/closed_loop_summary.json` |
| 冻结计划调用日志 | `closed_loop/raw/plan_calls.jsonl` |
| cudagraph 计数 | `closed_loop/raw/cudagraph_B_final.json`（B 主会话 pid 188895） |
| CPU 探针（62 574 样本） | `closed_loop/raw/bg_speedo.jsonl` |
| 服务启动命令台账 | `closed_loop/raw/openwam_cmd.jsonl` |
| 引擎 journal（隔离侧） | `closed_loop/raw/isolated_run_journal.jsonl` |
| 会话日志 | `closed_loop/logs/*.log` |
| 视频 | `closed_loop/videos/VID_A_init0.mp4`、`VID_B_init0.mp4` |
| 本轮工具 | `closed_loop/tools/{arm.sh,plan_stub.py,closed_loop_serve.py,run_episodes.py,analyze.py}` |
| 验收 JSON | `embodied_engine/records/acceptance/2026-11-02-openwam-closed-loop-task1.json` |
| manifest 登记 | `/home/host/flashrt-exp/records/manifest.json`（新增 55 条 closed_loop 产物；隔离代码树副本与可重建编译缓存不登记） |
