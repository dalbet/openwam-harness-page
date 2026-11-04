# OpenWAM 三层具身引擎 × FlashRT 加速专项审计（只调查，不实施）

- 日期：2026-10-29
- 范围：`/home/host/page-HK`（FlashRT 工程）、`/home/host/page-dmh/embodied_engine`（三层引擎）
- 约束：本轮不修改业务源码、不安装/升级依赖、不启动新 GPU 模型、不停止现有服务、不执行压测、不改 planner / 动作窗口 / 控制频率 / 停止条件 / 成功判定
- 证据标记：
  - `[S]` 源码证据（读到的代码与配置）
  - `[R]` 运行证据（进程、端口、GPU、日志、历史 run 记录）
  - `[H]` 历史线索（用户提供或既有报告，仅作线索）
  - `[I]` 推断（未被直接证据证实，已标注需如何确认）
- 未测量到的数字一律写“待测”，不把推断写成实测。

---

## 0. 结论摘要

1. **`/home/host/page-HK` 不是 git 仓库，FlashRT 也不是 git checkout**。`page_engine/FlashRT` 是一个**没有 `.git` 的源码快照**（`flash-rt` 0.2.0），其打包元数据指向 `https://github.com/flashrt-project/FlashRT`，与 `Infini-AI-Lab/FlashRT` 无关。`[S][R]`
2. **现场实际在跑的模型是 OpenWAM-Alpha-Sim-LIBERO（step 10690），不是 π0.5**；`config/engine.json` 里 `action_model.name` 仍写着 “Pi0.5 LIBERO-130 SFT”，这是遗留字段，真正的后端由 `action_model.backend=openwam` + `action_model.openwam.*` 决定。`[S]`
3. **FlashRT 当前不具备 OpenWAM 能力**：全仓只有一处 “WAM-style loops” 的文档提及（`docs/subgraph_stage_plans.md:270`），没有任何 OpenWAM 前端、权重映射、双流 MoT 注意力或动作头。**π0.5 与 Wan2.2 的既有支持都不构成兼容性证明**。`[S]`
4. **两套环境不兼容**：FlashRT 侧是 **Python 3.12.14 + torch 2.14.1+cu130**，编译产物是 `cpython-312` 的 `.so`；OpenWAM 侧是 **Python 3.10.12 + torch 2.7.1+cu128**。直接 `import flash_rt` 复用不可行，必须重编译或走跨进程协议。`[R]`
5. **端到端收益上限必须说清**：在最近一回合真实记录里，模型生成只占 wall 的约 7.9%（`609.8 ms × 4 次 = 2.44 s`，总 `30.90 s`），每个动作的**固定开销**（客户端 PNG 编码 + WebSocket + 环境 step）才是主项。把生成时间压到 0，闭环也只快约 8%（3.82 Hz → 约 4.14 Hz）。因此“加速前后对比”必须分两层展示：**策略服务级（生成延迟，收益显著）** 与 **闭环级（wall / Hz / RTF，收益有限）**。`[R][I]`
6. 本轮建议的第一接入方案不是 IR，也不是“整模型移植”，而是 **R2 专用内核路线的最小切片**：在 OpenWAM 的 py3.10 / torch2.7.1+cu128 / SM120 环境里，用 FlashRT 的 `csrc` 重编译 attention/GEMM 算子，先替换 `torch_sdpa` 热路径，用固定观测回放做数值对齐，再决定是否走 R1（IR/harness 全量接入）。`[S][I]`

---

## 1. 现场核实

### 1.1 目录与文档

| 路径 | 实际内容 | 证据 |
|---|---|---|
| `/home/host/page-HK` | **不是 git 仓库**，只有 `models/`、`page_engine/` 两个目录 | `[R]` `git rev-parse` 报 `not a git repository` |
| `/home/host/page-HK/page_engine` | **符号链接农场**，指向 `/home/host/flashrt-store/page_engine`（`FlashRT`、`.venv`、`env.sh`、`build_flashrt.sh`、`bench_pi05` 等均为软链） | `[R]` `ls -la` |
| `/home/host/page-HK/page_engine/FlashRT` | FlashRT 源码快照，无 `.git` | `[R]` |
| `/home/host/page-HK/page_engine/build.log` | 本地构建日志（39 KB，10-26） | `[R]` |
| `/home/host/page-HK/page_engine/qwen3_4b_bench.json` | 本地 Qwen3-4B 基准结果（非 OpenWAM） | `[R]` |
| `/home/host/page-dmh` | **git 根**（`embodied_engine` 的上层），remote `git@github.com:dalbet/openwam-harness-page.git` | `[R]` |
| `/home/host/page-dmh/embodied_engine` | 三层引擎；文档只有 `docs/architecture.md`、`docs/acceptance_report.md` | `[R]` |
| `/home/host/page-dmh/openwam_harness/OpenWAM` | 独立 git 仓库，remote `https://github.com/OpenWAM-Official/OpenWAM.git`，**detached HEAD** `4d7f642b841913ea3c176010b692b1de3d5ef180`，工作区干净 | `[R]` |
| `/home/host/page-dmh/openwam_harness/RPent` | 环境侧依赖，`envs/rpent/bin/python` = 3.10.12 | `[R]` |

> 说明：`page-HK` 下没有 `AGENTS.md`；`page-dmh` 仓库根也没有 `AGENTS.md`。适用的架构说明是 `embodied_engine/docs/architecture.md` 与 `openwam_harness/OpenWAM/README.md`。

### 1.2 git 身份（准确值）

| 工程 | 根目录 | remote | 分支 | HEAD | 工作区 |
|---|---|---|---|---|---|
| 三层引擎 | `/home/host/page-dmh` | `git@github.com:dalbet/openwam-harness-page.git` | `feature/openwam-pi05-review-prep` | **`33f18148d27654f7223e925248fe9b26b23410e4`** | **干净**（`git status --porcelain` 为空） |
| OpenWAM | `.../openwam_harness/OpenWAM` | `https://github.com/OpenWAM-Official/OpenWAM.git` | detached | `4d7f642b841913ea3c176010b692b1de3d5ef180` | 干净 |
| FlashRT | `/home/host/flashrt-store/page_engine/FlashRT` | **无**（无 `.git`） | — | — | — |

- `[H]` 用户提供的冻结基线 `abb75dbc…`、审查提交 `33f1814`：现场 HEAD 正好是 `33f1814`，其父链包含 `abb75db`（`git log` 可见）。**未做任何 checkout/回退**。
- `[S]` `config/engine.json` 中记录 `action_model.openwam.repo_commit = 4d7f642b…`，与现场 HEAD 一致。
- `[S]` FlashRT 版本来自打包元数据：`pyproject.toml:6-7` `name = "flash-rt"`、`version = "0.2.0"`；`flash_rt.egg-info/PKG-INFO:6-8` 的 Homepage/Repository/Issues 全部是 `https://github.com/flashrt-project/FlashRT`。

### 1.3 运行态（端口 / PID / GPU / 环境 / 模型路径）

`[R]` `ss -ltnp` + `nvidia-smi` 现场抓取：

| 端口 | 服务 | PID | GPU | 显存 | 说明 |
|---|---|---|---|---|---|
| 8001 | vLLM（planner LLM） | 2843152（子进程 2843863） | 3 | ~22.1 GB | Qwen3.5-9B，OpenAI 兼容 `/v1` |
| 14510 | π0.5 常驻服务（对照基线） | 2784215 | 1 | 7.16 GB | 当前**未被使用**（`backend=openwam`） |
| **14520** | **OpenWAM 常驻策略服务（实际使用）** | **1380923** | **2** | **24.77 GB** | WebSocket `ws://127.0.0.1:14520` |
| 14561 | LIBERO 环境服务 | 2219370 | 3 | 0.5 GB | 每回合由引擎拉起 |
| 14600 | 推理服务（多后端） | 3900284 | — | — | `/infer` 队列 + 后端复用 |
| 14700 | Agent 服务 | 3904106 | — | — | `/plan` 语义路由 |
| 14800 | 引擎服务 | 1854309 | — | — | `/api/run`、`/api/frame`、`/api/stop` |
| 14900 | page_ui | 2784912 | — | — | 页面与双路 JPEG 推流 |

GPU 0 空闲（4 MiB），GPU 1/2/3 分别被 π0.5、OpenWAM、vLLM+env 占用（`[R]`）。

启动脚本（全部读取 `config/engine.json`，无硬编码）：

- `scripts/openwam_server.sh start|stop|restart|status`：按 `action_model.openwam.*` 拉起 `scripts/deploy.py`，并写 `records/openwam_server.json` 启动记录（含 `checkpoint_file` / `generate_horizon` / `execute_horizon` / `denoise_steps` / `gpu_index` / `compile_enabled:false`）。
- `scripts/start_all.sh`、`scripts/stop_all.sh`、`scripts/status.sh`、`scripts/restart_engine.sh`。

`[R]` OpenWAM 实际进程命令行（已脱敏，没有凭据）：

```
CUDA_VISIBLE_DEVICES=2 /home/host/page-dmh/embodied_engine/.venv-openwam/bin/python scripts/deploy.py \
  --ckpt-dir /home/host/page-dmh/openwam_harness/checkpoints/OpenWAM-Alpha-Sim-LIBERO \
  --ckpt-name checkpoint_step_10690.safetensors --device cuda:0 --host 127.0.0.1 --port 14520 \
  --compile-enabled false --inference-mode sync --denoise-steps 10 --inference-horizon 32
```

模型与权重（`[S]` `config/engine.json`）：

- OpenWAM：`/home/host/page-dmh/openwam_harness/checkpoints/OpenWAM-Alpha-Sim-LIBERO`，`checkpoint_step_10690.safetensors`，HF revision `110b07e4…`
- 对照 π0.5：`.../checkpoints/RLinf-Pi05-LIBERO-130-fullshot-SFT`，revision `6222623f…`
- 环境：`libero_goal`，seed 0，`env.cuda_device=3`，`env.python=.../envs/rpent/bin/python`

### 1.4 证据分级小结

- 源码证据：调用链、契约、配置语义（第 3 节全部带行号）。
- 运行证据：端口/PID/GPU、启动命令行、`records/logs/openwam_server.log`、`records/acceptance/**` 的 run 记录。
- 历史记录：`records/acceptance/speed_*.json`（10-27 02:50 与 02:52 两次测量）。
- 推断：`torch.compile` 是否在稳态生效、视频分支耗时占比（第 4 节标注“待测/待确认”）。

---

## 2. FlashRT 身份确认

### 2.1 是哪一个 FlashRT？

| 判据 | 结论 |
|---|---|
| 无 `.git`、无 remote | 无法用 git 判断；只能看元数据 |
| `flash_rt.egg-info/PKG-INFO:6-8` | `https://github.com/flashrt-project/FlashRT` → **不是** `Infini-AI-Lab/FlashRT` |
| `pyproject.toml:6-7` | `flash-rt` 0.2.0，`Requires-Python >=3.10` |
| 目录结构 | `flash_rt/{hardware,models,frontends,pipelines,subgraphs,runtime,cpp,amd,npu}` + `csrc/` + `CMakeLists.txt` + 预编译 `flash_rt_kernels.cpython-312-*.so`、`flash_rt_fa2.cpython-312-*.so` |
| 特性指纹 | SM110 Thor / SM120 RTX 5090 / SM89 / AMD CDNA3/4 / Ascend NPU 的 `detect_arch()`（`flash_rt/hardware/__init__.py:32-50`）与该仓库公开文档一致 |

**判定：`page-HK/page_engine/FlashRT` 是 `flashrt-project/FlashRT` v0.2.0 的本地源码快照（非 git clone），不是自研，也不是 Infini-AI-Lab 版本。** `[S][R]`

### 2.2 已支持什么（与 OpenWAM 相关的部分）

`[S]` `_PIPELINE_MAP`（`flash_rt/hardware/__init__.py:124` 起）注册的模型包括：π0、π0.5、π0-FAST、GROOT N1.6/N1.7、Hy-Embodied、Cosmos3、Qwen3-VL、Wan2.2-TI2V-5B、motus、lingbot 等。

与本次直接相关的两条：

- π0.5：`flash_rt/hardware/__init__.py:140-151`，`Pi05TorchFrontendRtx`（SM120）。
- Wan2.2：`flash_rt/hardware/__init__.py:230-231`，`Wan22TorchFrontendRtx`；实现只在 `flash_rt/models/wan22/pipeline_rtx.py`，**是视频生成（T2V/I2V）前端，没有动作分支、没有 ActionDiT、没有双流联合自注意力**。

OpenWAM 相关代码：

- 全仓搜索 `openwam|WAM` 只有 1 处文档字样：`docs/subgraph_stage_plans.md:270` “For diffusion or **WAM-style loops**, register a factory” —— 讲的是**如何注册一个参数化切分**，不是实现。
- **没有任何** OpenWAM 的 `WEIGHT_SPEC`、pipeline、frontend、算子、测试或基准结果。
- 结论：**OpenWAM 在 FlashRT 中完全未适配**，既没有可运行实现，也没有占位接口。`[S]`

### 2.3 已有的实测结果（可引用的历史数据）

`[R]` `/home/host/flashrt-store/bench_pi05/*.json`（同一台 RTX 5090、同一观测、20 次、shape `[10,7]`）：

| 实现 | 精度 | median |
|---|---|---|
| openpi（原生 torch eager，bf16 selected params） | fp32/bf16 | **52.36 ms** |
| FlashRT `Pi05TorchFrontendRtx` | fp16 | **36.24 ms** |
| FlashRT `Pi05TorchFrontendRtx` | fp8 | **23.66 ms** |

> 这只证明 **π0.5** 的加速比（约 2.2×），**不能**外推到 OpenWAM：模型结构、动作头、去噪调度、分辨率、文本编码器全都不同。

`[S]` FlashRT 官方文档 `docs/benchmark_comparison.md:13-24` 另有 Thor/5090 的公开数字，属于外部参考，不作为本项目实测。

### 2.4 关键环境错配（接入的第一道硬约束）

| 项 | FlashRT 侧 | OpenWAM 侧 |
|---|---|---|
| Python | **3.12.14** | **3.10.12** |
| torch | **2.14.1+cu130** | **2.7.1+cu128** |
| 编译产物 | `flash_rt_kernels.cpython-312-*.so`、`flash_rt_fa2.cpython-312-*.so` | 无法加载（ABI/CPython tag 不符） |
| venv | `/home/host/page-HK/page_engine/.venv` | `/home/host/page-dmh/embodied_engine/.venv-openwam` |

`[R]` 现场验证：`.venv-openwam` 内 `import flash_attn` / `flash_attn_interface` / `xformers` **全部为 False**。

→ 任何“直接把 FlashRT 的 `.so` 拷进 OpenWAM venv”的方案都不可行；必须**在 py3.10 + torch2.7.1+cu128 + SM120 下重编译 `csrc`**，或保持两个进程、用现有 WebSocket 协议通信。

---

## 3. 实际调用链（页面 → env.step）

### 3.1 总览

```
浏览器 (14900 page_ui)
  └─ POST /api/load, /api/run                page_ui/server.py:329,341
       └─ POST 14800 /api/run                engine/engine_service.py:2622,2704
            └─ WholeTaskRunner.run()          engine/openwam_rollout.py:639
                 ├─ Agent /plan (14700)       engine/engine_service.py:1272 → agent/agent_service.py:276
                 │     └─ vLLM 8001 /chat/completions   agent/planner.py:168-180
                 ├─ observe → build_openwam_obs        openwam_rollout.py:506-513
                 │     └─ adapters/openwam_contract.py:175-200
                 ├─ Inference 14600 /infer             inference/inference_service.py:203
                 │     └─ OpenWAMBackend.predict       models/openwam_backend.py:146
                 │          └─ ws://127.0.0.1:14520    openwam/deploy/server.py:246
                 │               └─ JointInferenceEngine.generate  openwam/deploy/engine.py:252
                 ├─ native_eef10_to_libero7d           adapters/openwam_contract.py:133
                 └─ env.step(action7)                  engine/openwam_rollout.py:673
                      └─ LIBERO env server 14561       engine/engine_service.py:456-468
```

### 3.2 逐段细节（文件:行号）

**(a) 页面 → 引擎**：`page_ui/server.py:90-93` 通过 `/api/frame` 拉引擎帧；`:329` 发 `/api/load`，`:341` 发 `/api/run`；`:193-282` 是单 poller，`frame_seq` 变化才推帧。引擎侧路由在 `engine/engine_service.py:2701-2708`，run 处理器 `:2622`，构造 `WholeTaskRunner` 在 `:1017`。

**(b) Agent（planner）**：`engine/engine_service.py:1272` 向 `{agent_url}/plan` POST；`agent/agent_service.py:276` 接收；语义路由 `agent/semantic_router.py:187-218`（`max_tokens=24`，`model_mode=non_thinking`）；真正请求 vLLM 在 `agent/planner.py:168-180`。planner 只产出字符串指令，不参与动作；本轮不改。

**(c) 观测构造（客户端）** `[S]` `adapters/openwam_contract.py`：

- 输入：LIBERO 原始 obs 的 `agentview_image`、`robot0_eye_in_hand_image`（`uint8`，256×256）、`robot0_eef_pos`、`robot0_eef_quat`、`robot0_gripper_qpos`（`:181-186` 强制校验）。
- 图像：`rotate_180`（`:150`）→ `resize_slot`（`:158`）→ head `(320, 256)`、wrist `(160, 128)`（`:33-38` 常量 `HEAD_SLOT`/`WRIST_SLOT`）。
- 本体：`eef10_from_raw`（`:119-131`）输出 **10 维 float32** = `pos(3) + rot6d(6) + grip(1)`；`grip = 2*width/0.08 - 1`，`width = qpos[0]-qpos[1]`（`:109-116`）。
- 返回字段：`main_images`、`wrist_images`、`states`、`raw_proprio`（`:193-200`）。

**(d) 传输（客户端）** `[S]` `models/openwam_backend.py:146-195`：

- 图像用 **PNG + base64**：`_png_b64`（`:30`），字段 `head_camera` / `left_wrist_camera` / `right_wrist_camera`。
- `prompt` 原样透传（服务端不做模板包装，`openwam/deploy/server.py:31`）。
- `state` 为 10 维 float 列表，长度不符直接抛错（`:169-172`）。
- 一次请求一收一发：`json.dumps` → `sock.send` → `sock.recv(timeout=300)`（`:175-191`）。

**(e) 服务端入口** `[S]` `openwam/deploy/server.py:246-263`：单连接循环，`type` 分派 `obs` / `reset` / `ping`；`reset` 走 `:220-223`；`ping_interval=None`（`:301-305`，注释说明首次 compile warmup 会阻塞事件循环）。

**(f) 推理引擎与去噪** `[S]` `openwam/deploy/engine.py:252-360`：

- `denoise_steps` 默认取 `inference.denoise_steps`（`:286`），现场 = **10**。
- 采样噪声：`seed = conditions.get("seed", 42)`（`:350`）→ `base.py:1494` `torch.Generator(device).manual_seed(seed)`。**引擎从不传 seed，所以每次都是 42** → 同 `(obs,prompt)` 理论上可复现（`scripts/inference_determinism_probe.py` 就是验证这件事）。
- 视频/动作双流：`base.py:1372-1510` 一次前向同时跑 video DiT（Wan 5B）与 ActionDiT（1B），由 `DualSystemSelfAttnArchitecture.forward` 的 MoT 联合循环驱动（`openwam/model/architectures/dual_system/joint_self_attn.py:160-262`）。
- 视频分支**照常去噪**，`optimization.decode_video=false` 只跳过 VAE 解码（`engine.py:211-214`），不省 DiT。
- DiT velocity cache：`enabled=true, cosine_threshold=0.99, max_skips=3`（`configs/deploy.yaml`，`engine.py:186-201`），日志确认已启用（`records/logs/openwam_server.log:19`）。
- prompt embedding LRU 32（`configs/deploy.yaml`）。

**(g) 动作返回与缓冲** `[S]`：

- 服务端 `SyncInferenceExecutor`（`openwam/deploy/policy.py:48`）持有 32 步 chunk，前 1 次请求真算，后续 31 次只出队。
- 响应里带 `chunk_generation` / `cache_hit` / `chunk_id` / `buffer_index` / `server_step`（`records/acceptance/speed_qwen_openwam.json` 的 `openwam.first_generation` 可见完整字段）。
- 客户端按 `[actions][0]` 取第一条，并附带 chunk 元数据（`models/openwam_backend.py:238`）。

**(h) 动作转换** `[S]` `adapters/openwam_contract.py:133-147`：

- 10 维 native EEF10 → 7 维 LIBERO OSC：`d_xyz(3) + axis_angle(3) + gripper(1)`。
- `rot6d → quat → axis_angle`（`:105`），夹爪 `open_scale → -open_scale`（`:114`），三路各自 `clip(-1,1)`。
- 非有限值直接抛错（`:140-141`）。

**(i) 执行** `[S]` `engine/openwam_rollout.py:660-675`：`observe → infer → convert → env.step`；`:673` 是 `env.step(action7)`；`:685-687` 处理 stop 在 step 中抬起的竞态。`env_step_s`、`observe_s`、`infer_wall_s`、`snapshot_s`、`frame_s` 全部逐动作记录（`:604-630`）。

**(j) 生命周期契约** `[S]`：

- `reset`：客户端 `openwam_backend.py:131-144` 发 `{"type":"reset"}` 等 `reset_ack`；服务端 `server.py:220-223` → `engine`/`executor` 清空 chunk 与 DiT cache（`engine.py:321-322`）。
- `stop`：`inference_service.py:122`、`engine_service.py` `/api/stop`（`:2705`）；rollout 在 `:649` 与 `:685` 两处检查，保证不发出新动作。
- 超时：`action_model.openwam.timeout_s=300`，inference 队列 `queue_size=1`（`config/engine.json`），`max_episode_steps=10000`，`timing.stop_join_s=5.0`。
- 会话隔离：inference 服务单队列 + 单 backend 锁（`inference_service.py:203-222` 的 `with self.lock`），同一时刻只有一个 in-flight 请求。

### 3.3 生成/执行长度与控制频率（现场值）

| 项 | 值 | 来源 |
|---|---|---|
| `generate_horizon`（模型一次产出） | 32 | `config/engine.json`、启动参数 `--inference-horizon 32` |
| `execute_horizon`（执行多少步后重规划） | 32 | 同上 |
| `denoise_steps` | 10（sync） | 启动参数 + `engine.py:286` |
| `action_dim` / 空间 | 10 / `native_eef10` | `config/engine.json` |
| 控制频率（配置） | 20 Hz | `action_model.control_frequency_hz=20` |
| 实测 wall 控制频率 | **3.82 Hz** | run 记录 `wall.control_hz_wall` |
| 实测 sim RTF | **0.191** | run 记录 `wall.sim_real_time_factor` |

### 3.4 已启用的优化 / 特性（现场事实）

| 特性 | 状态 | 证据 |
|---|---|---|
| DiT velocity cache | **启用**（0.99 / 最多跳 3 次） | 日志 `:19`，`engine.py:186-201` |
| prompt embed LRU | 启用，32 | `configs/deploy.yaml` |
| VAE 解码 | **关闭**（`decode_video=false`） | `configs/deploy.yaml:26` |
| torch.compile | **命令行关闭，但存在可疑进程** | 见下 |
| CUDA Graph | 未启用 | 无相关代码/日志 |
| 注意力实现 | **`sdpa`（auto-detected）** | 日志 `records/logs/openwam_server.log:1`；`action_backbone/components.py:352-363` |
| `flash_attn` / `xformers` | **未安装** | venv 探测 False |
| Wan 侧注意力 | 优先 `flash_attn`，缺失回退 `torch_sdpa` | `model/video_backbone/wan/models/dit.py:13-42` |
| 混合精度 | **bf16** | `model_loader.py:210-211`、checkpoint `config.yaml:114 mixed_precision: bf16` |
| 异步去噪 / prefetch | 服务端 `--inference-mode sync`；引擎侧 `async_chunk_prefetch` 仅对缓存出队 | 启动参数、`config/engine.json inference.mode` |

**`torch.compile` 的矛盾证据（必须列为待确认）**

- `[S]` `scripts/openwam_server.sh` 固定传 `--compile-enabled false`；`openwam/deploy/server.py:104-110, 642` 会把它写回 `optimization.compile.enabled`；架构侧 `joint_self_attn.py:134-142` 有 `if not compile_enabled(...): return` 守卫。
- `[R]` 但现场存在 **32 个** `torch/_inductor/compile_worker` 子进程（PPID = 1380923，启动时间 02:50:04，正好是首次推理）。
- `[R]` 然而：日志中没有 “Enabled dual self-attn MoT compile …” 这一行；`/tmp/torchinductor_host`（332 MB）里**没有任何**包含 `run_joint_loop`/`action_backbone` 符号的缓存文件；进程 fd 也没有 torchinductor 句柄。
- `[I]` 判断：**编译池被初始化过，但模型图没有真正编译进稳态**，实际仍是 eager。第一轮测量时用 `TORCH_LOGS=recompiles` 在新实例上复核（不要动现有服务）。

---

## 4. 适配判断与瓶颈

### 4.1 兼容性结论（不假设）

| 问题 | 结论 |
|---|---|
| FlashRT 是否支持 OpenWAM？ | **否**。没有前端/权重映射/双流 MoT/动作头，只有文档里一句 WAM-style 的说法。`[S]` |
| 能否用 π0.5 的支持类推？ | **不能**。π0.5 是单流 VLA + 5 步 chunk；OpenWAM 是 Wan5B 视频 DiT + ActionDiT1B 联合自注意力 + 10 步双流去噪 + 32 步 chunk。`[S]` |
| 能否用 Wan2.2 的支持类推？ | **不能**。FlashRT 的 Wan2.2 只有视频 T2V/I2V pipeline，没有动作分支。`[S]` |
| 能否直接加载 FlashRT 的 `.so`？ | **不能**。cp312 + torch2.14 vs cp310 + torch2.7。`[R]` |
| 属于哪条路线？ | **两条都沾**：完整接入属于“新模型注册（IR/harness + 专用 kernel）”；最小切片属于“专用内核替换”。见 4.3。 |

### 4.2 现有实测耗时分解（只用真实记录，不估算）

**记录 A：单任务真实闭环** `records/acceptance/composite_state_restore_12_20261028_runs/ep-20261028-100255-e4e68d.json` `[R]`

- 总 wall `30.8958 s`，`118` 个动作，`3.82 Hz`，RTF `0.191`
- 分项：observe `5.0649 s`、inference `16.6906 s`、env_step `7.8872 s`、snapshot `1.1452 s`、convert `0.0751 s`
- 每动作 median `0.2446 s`（p95 `0.2554 s`）
- chunk 生成 **4 次**：服务端 median `510.7 ms`、客户端 `inference_s` `526.0 ms`、引擎总 `609.8 ms`（含 PNG 编码 `83.6 ms`）
- 缓存出队 **114 次**：服务端 median `15.26 ms`、客户端 `29.9 ms`、**引擎总 `112.2 ms`**

**记录 B：固定观测的独立测量** `records/acceptance/speed_qwen_openwam.json`（10-27 02:52，96 动作）`[R]`

- 生成 3 次：服务端 median `426.69 ms`，客户端 `442.5 ms`，引擎总 `528.3 ms`
- 缓存 93 次：服务端 median `15.26 ms`，引擎总 `113.5 ms`

**由此得到的三条硬结论：**

1. **模型生成不是每动作成本**：32 步才发生一次，摊到每动作约 `510/32 ≈ 16 ms`。
2. **每动作固定开销 ~112 ms**（客户端 PNG 编码 84 ms + WS 往返 + 队列），比模型生成摊薄值大一个量级。
3. **同模型、同 checkpoint 的两次测量差 20%**（426.7 vs 510.7 ms）→ 任何“加速前后”对比必须**同机、同卡、同空闲状态、同 warmup**，否则噪声会吞掉结论。

`[I]` 端到端上限：把生成降到 0，本回合 wall 从 30.90 s → 约 28.46 s，Hz 3.82 → 约 4.14（**约 8%**）。这是闭环维度的天花板，必须提前和合作方对齐预期。

**模块级分解（视频 DiT vs ActionDiT vs 文本编码）：待测。** 现有日志只有整段 `server_latency_ms`，没有分模块计时；`WAM_PROFILE=1`（`engine.py:217`）可以打开 profiling，但需要新实例，本轮未执行。

### 4.3 三条接入路线与排序

| 排名 | 路线 | 做什么 | 预期收益依据 | 工作量 | 正确性风险 |
|---|---|---|---|---|---|
| **1（推荐先做）** | **R2 专用内核最小切片** | 在 py3.10/torch2.7.1+cu128/SM120 下重编译 FlashRT `csrc`（attention/GEMM/RMSNorm 融合），替换 `torch_sdpa` 热路径；保持模型结构、步数、chunk、seed 完全不变 | 现场注意力是 **sdpa**，`flash_attn` 未装（`[R]`）；FlashRT 在 π0.5 上把 eager 52.36 → 23.66 ms（`[R]`），说明其 kernel 在 SM120 上有实效；但 OpenWAM 能吃到多少**待测** | 中：编译 + 逐算子对齐；不改业务源码 | 中低：可逐层 cosine 对齐，失败可回退 sdpa |
| 2 | R1 IR/harness 全量接入 | 把 OpenWAM 注册成 FlashRT 新模型：`AttentionSpec` + pipeline + `WEIGHT_SPEC` + frontend + `_PIPELINE_MAP` | FlashRT 自己的文档给出工作量基线：pipeline 200–400 行/硬件、WEIGHT_SPEC 60–120 行、frontend 700–1000 行（`docs/adding_new_model.md:190,236,294`） | **大**：双流 MoT 交织无法用现有 stage plan 表达（`docs/subgraph_stage_plans.md:270` 只支持链式/参数化切分） | 高：数值漂移、调度语义变化 |
| 3 | R3 服务层“提速” | 加大 chunk / 异步 prefetch / 跳过视频分支 | 会改动作窗口或控制时序 | — | **本轮明确禁止** |

**不推荐但必须说明**：R3 里“把 `denoise_steps` 从 10 降到 5”“关掉视频分支”收益最大（近似线性），但**改变模型行为**，与合作方“同模型加速前后”的前提冲突，只能作为负对照。

### 4.4 优化分类（哪些改推理、哪些改闭环、哪些改行为）

| 类别 | 手段 | 影响面 |
|---|---|---|
| **只改善推理** | FlashRT kernel 替换、attention 后端、CUDA Graph、bf16→fp8 | 只影响 `server_latency_ms`；闭环收益按 4.2 摊薄 |
| **可能改善闭环** | 生成停顿变短（每 32 动作一次）、客户端编码改 JPEG/共享内存（不属于模型加速） | 影响 wall / Hz / RTF，但不改动作语义 |
| **改变行为或时序** | 降去噪步数、改 chunk 长度、FP8/FP4 量化、DiT cache 阈值、跳过视频分支 | 动作分布变化，必须重新做任务成功率验收 |

### 4.5 最小接入方案（M1）与改动清单

**核心思路：不动业务代码，新增一个“同协议”的 FlashRT 策略服务进程。**

现有架构已经允许这么做：`OpenWAMBackend` 只认 `action_model.openwam.server_url`（`models/openwam_backend.py:45-78`，`config/engine.json:72`），协议是 `obs/reset/ping` 三种消息（`openwam/deploy/server.py:246-263`）。因此：

| 改动 | 文件 | 性质 |
|---|---|---|
| 新增 FlashRT 版策略服务（实现同一 WS 协议） | 新目录 `embodied_engine/flashrt_policy_server/`（或 `page-HK` 侧独立包） | 新增，不改旧文件 |
| 切换后端地址 | `config/engine.json` → `action_model.openwam.server_url`（临时指向新端口，如 14521） | 配置 |
| 启动/停止脚本 | 新增 `scripts/flashrt_policy_server.sh`（照抄 `scripts/openwam_server.sh` 的结构） | 新增 |
| 数值对齐 | 复用 `scripts/inference_determinism_probe.py` 与 `scripts/bench_speed.py` | 不改 |
| 编译 FlashRT 扩展 | `bash /home/host/page-HK/page_engine/build_flashrt.sh`（在**独立 venv**里，py3.10 + torch2.7.1+cu128） | 环境隔离 |

**必须保持不变的量**（否则不是“同模型”）：checkpoint 文件与 revision、`generate_horizon=32`、`execute_horizon=32`、`denoise_steps=10`、`seed=42`、`shift=5.0`、图像 resize/rotate 契约、EEF10→7D 转换、planner、控制频率、停止与成功判定。

---

## 5. 后续验证方案

三套测试，从便宜到贵，全部使用**现有脚本**，不新增接口。

### 5.1 A. 固定观测回放（策略服务级，最先做）

- 入口：`python scripts/inference_determinism_probe.py`（现有；同一观测发两次，中间 reset，比较逐元素差异）
- 入口：`python scripts/bench_speed.py --out records/acceptance/speed_<tag>.json --skip-planner --skip-episode --openwam-actions 96`（现有参数见 `scripts/bench_speed.py:374-380`）
- 公平条件：
  - 同一台机、同一张卡、同一 checkpoint（比对 sha256）、同一 `config/engine.json`
  - 每次测量前 `reset`，warmup ≥ 5 次不计入
  - 先 `bash scripts/openwam_server.sh status` 记录基线，测量期间不跑其它 GPU 任务
  - 记录 `nvidia-smi --query-gpu=clocks.sm,power.draw,temperature.gpu` 作为环境备注
- 指标：`server_latency_ms`（median/p95）、`client_inference_s`、`encode_s`、`send_s`、`recv_s`、`action_raw` 向量
- 数值校验（建议阈值，需与合作方确认）：
  - 逐元素 `max|Δ| ≤ 1e-3`，余弦相似度 `≥ 0.9999`（纯 kernel 替换、同精度）
  - 若上 FP8：`cosine ≥ 0.999`，且必须跑 5.3 的任务验收
- 通过标准：**同一输入下两次请求 bit-identical**（现有 probe 已验证 seed=42 的前提），且新旧实现差异在阈值内

### 5.2 B. 策略服务基准（生成延迟，展示“加速前后”的主指标）

- 与 A 同一脚本，但只统计 `chunk_generation=true` 的行（`bench_speed.py` 已区分 `generation` / `cache`）
- 报告：`generation.server_latency_ms` 的 median / p95 / min / max，附 `n`
- 必须同时报告 `cache.server_latency_ms`，否则读者会把出队当成推理
- 建议样本：≥ 30 次真实生成（即 `--openwam-actions ≥ 960`，耗时较长，放到测量阶段再决定）

### 5.3 C. 单任务闭环（任务效果，最终验收）

- 入口：`python scripts/run_rounds.py`（现有；manifest 驱动的冻结回合，含 `/api/load` 与 `/api/run`）
- 公平条件：
  - 同一 `task_id`、同一 `seed`、同一 `initial_scene_fingerprint`（run 记录里有）
  - 同一 warmup、同一 GPU 空闲状态；A/B 交替执行（ABAB），不是先跑完 A 再跑 B
  - 不改 `max_steps`、`control_frequency_hz`、`execute_horizon`、停止条件
- 指标（run 记录里已有）：`user_goal_success` / `native_task_success`、`wall.control_hz_wall`、`wall.sim_real_time_factor`、`per_action_total_s` 的 median/p95、`chunk_stats.generation`、`first_action_s`
- 样本：沿用现有验收习惯 **12 回合**（见 `records/acceptance/composite_state_restore_12_20261028*`）
- 数值校验：成功率不下降；`wall` 与 `control_hz_wall` 的改善方向与 A/B 的生成延迟改善方向一致

---

## 6. 后续实施命令与涉及文件（不编造）

### 6.1 只读检查（随时可跑，不影响服务）

```bash
bash /home/host/page-dmh/embodied_engine/scripts/status.sh
bash /home/host/page-dmh/embodied_engine/scripts/openwam_server.sh status
ss -ltnp | grep -E ':(8001|14510|14520|14561|14600|14700|14800|14900)\b'
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader
```

### 6.2 测量（本轮之后再做；会真实执行，不属本轮）

```bash
# A/B 固定观测回放（不跑 planner、不跑闭环）
cd /home/host/page-dmh/embodied_engine
/home/host/page-dmh/openwam_harness/envs/rpent/bin/python scripts/bench_speed.py \
  --out records/acceptance/speed_<tag>.json --skip-planner --skip-episode --openwam-actions 96

# 逐元素可复现性
/home/host/page-dmh/openwam_harness/envs/rpent/bin/python scripts/inference_determinism_probe.py
```

### 6.3 M1 实施时才会新建/修改的文件

| 文件 | 动作 | 说明 |
|---|---|---|
| `embodied_engine/flashrt_policy_server/`（新目录） | 新增 | 实现 `obs/reset/ping` 同协议，绑定 `127.0.0.1:14521` |
| `embodied_engine/scripts/flashrt_policy_server.sh` | 新增 | 参照 `scripts/openwam_server.sh` 的 start/stop/status 结构 |
| `embodied_engine/config/engine.json` | 修改 1 行 | `action_model.openwam.server_url` 指向新端口（可随时改回） |
| `/home/host/flashrt-store/page_engine/`（或独立 venv） | 新增环境 | py3.10 + torch2.7.1+cu128，`bash build_flashrt.sh` 重编译 `csrc` |
| `embodied_engine/docs/audits/` | 新增 | 本报告与后续测量报告 |

**明确不会修改**：`agent/*`、`inference/inference_service.py`、`engine/openwam_rollout.py`、`adapters/openwam_contract.py`、`models/openwam_backend.py`、`page_ui/*`、以及 `records/` 下任何历史文件。

---

## 附录 A：待测清单（本轮明确没有测的）

1. OpenWAM 单次生成内部拆解：文本编码 / 视频 DiT / ActionDiT / 联合注意力 / 采样器各占多少（需 `WAM_PROFILE=1`）。
2. 视频 DiT 与 ActionDiT 的 FLOPs 占比（结构上 5B vs 1B，但实际占比随 token 数变化，**不估算**）。
3. FlashRT `csrc` 在 py3.10 + torch2.7.1+cu128 + SM120 下能否编译通过、性能如何。
4. 现有 sdpa 是否已走 cuDNN/flash 后端（PyTorch 2.7 内部选择），需 `torch.backends.cuda` 日志确认。
5. `torch.compile` 是否在稳态生效（4.2 的矛盾证据）。
6. 客户端 PNG 编码 84 ms 的可优化空间（不属于模型加速，仅作为闭环上限的解释项）。

## 附录 B：本轮未做（按约束）

- 未修改任何业务源码；未安装/升级依赖；未启停任何服务；未跑压测；未 checkout/回退到 `abb75dbc` 或 `33f1814`；未新增 GPU 模型；未改 planner / 动作窗口 / 控制频率 / 停止与成功判定。
- 未改动 `records/` 下任何历史实验数据；本报告是新增文件。

## 附录 C：关键证据索引

| 主题 | 位置 |
|---|---|
| 后端选择 | `config/engine.json:57`、`inference/inference_service.py:40-62` |
| 推理入口 | `inference/inference_service.py:203-236` |
| 客户端协议 | `models/openwam_backend.py:146-195` |
| 观测契约 | `adapters/openwam_contract.py:119-200` |
| 动作转换 | `adapters/openwam_contract.py:105-147` |
| rollout 主循环 | `engine/openwam_rollout.py:639-700` |
| 服务端入口 | `openwam/deploy/server.py:220-263, 301-305` |
| 去噪与 seed | `openwam/deploy/engine.py:252-360`、`model/architectures/base.py:1372,1494` |
| 编译守卫 | `model/architectures/dual_system/joint_self_attn.py:134-170` |
| 注意力后端 | `model/action_backbone/components.py:352-363`、日志 `records/logs/openwam_server.log:1` |
| FlashRT 身份 | `flash_rt.egg-info/PKG-INFO:6-8`、`pyproject.toml:6-7` |
| FlashRT 能力表 | `flash_rt/hardware/__init__.py:124, 140-151, 230-231` |
| FlashRT 接入工作量 | `docs/adding_new_model.md:190,236,294`、`docs/subgraph_stage_plans.md:270` |
| 既有 π0.5 实测 | `bench_pi05/{openpi,flashrt,flashrt_fp16}.json` |
| 真实闭环耗时 | `records/acceptance/composite_state_restore_12_20261028_runs/ep-20261028-100255-e4e68d.json` |
| 独立生成基准 | `records/acceptance/speed_qwen_openwam.json` |
