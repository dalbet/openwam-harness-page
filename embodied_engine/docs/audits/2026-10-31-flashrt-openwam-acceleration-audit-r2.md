# OpenWAM 三层具身引擎 × FlashRT 加速专项审计（第二轮 / 只调查不实施）

- 日期：2026-10-31
- 范围：`/home/host/page-HK`（FlashRT 工程）、`/home/host/page-dmh/embodied_engine`（三层引擎）
- 约束（本轮全程遵守）：不修改业务源码、不安装/升级依赖、不启动新 GPU 模型、不停止现有服务、不执行压测、不改 planner / 动作窗口 / 控制频率 / 停止条件 / 成功判定
- 与既有报告的关系：`docs/audits/2026-10-29-flashrt-openwam-acceleration-audit.md` **只读引用，未做任何修改**。本文件是第二轮补充：补齐调用链行号、模型内部形状、采样与随机性、缓冲/reset 契约、以及本轮新发现的计时口径冲突。
- 证据标记：
  - `[S]` 源码证据（读到的代码/配置）
  - `[R]` 运行证据（进程、端口、GPU、日志、历史 run 记录）
  - `[H]` 历史线索（仅作线索）
  - `[I]` 推断（标注了确认方式）
- 未测到的数字一律写“待测”，不把推断写成实测。

---

## 0. 结论摘要（对应六项交付）

1. **身份**：`/home/host/page-HK` **不是 git 仓库**；`page_engine/` 是符号链接农场，真实内容在 `/home/host/flashrt-store/page_engine`；`FlashRT/` 是**无 `.git` 的源码快照**，以 editable 方式装入 `page_engine/.venv`（`direct_url.json` = `file:///home/host/page-HK/page_engine/FlashRT`），版本 `flash-rt 0.2.0`，包元数据指向 `https://github.com/flashrt-project/FlashRT`，**不是** `Infini-AI-Lab/FlashRT`。三层引擎的 git 根是 `/home/host/page-dmh`，分支 `feature/openwam-pi05-review-prep`，HEAD `33f18148d27654f7223e925248fe9b26b23410e4`，工作区干净。`[R][S]`
2. **现场在跑的模型是 OpenWAM-Alpha-Sim-LIBERO step 10690**（GPU2，24.8 GB），不是 π0.5；`config/engine.json` 的 `action_model.name="Pi0.5 LIBERO-130 SFT"` 是遗留字段，真正后端由 `action_model.backend="openwam"` 决定。`[S][R]`
3. **FlashRT 不具备 OpenWAM 能力**：全仓仅有 1 处 “WAM-style loops” 文档提及；无 OpenWAM 前端 / 权重映射 / 双流 MoT / 动作头。π0.5 与 Wan2.2 的既有支持**都不构成兼容性证明**（Wan2.2 在 FlashRT 里是纯视频生成 `Wan22TorchFrontendRtx`，没有动作分支）。`[S]`
4. **环境硬错配**：FlashRT 侧 `Python 3.12.14 + torch 2.14.1+cu130`，内核是 `cpython-312` ABI；OpenWAM 侧 `Python 3.10.12 + torch 2.7.1+cu128`。进程内 `import flash_rt` 不可行，必须重编译或走跨进程协议。`[R]`
5. **瓶颈结构（实测，非估算）**：一次真实回合（118 动作 / 4 次 chunk 生成）里，生成中位 `510.7 ms`、缓存出队中位 `15.3 ms`；生成总耗时 `4 × 0.6098 s = 2.44 s`，仅占 wall `30.90 s` 的 **7.9%**。把生成压到 0，闭环最多从 **3.82 Hz → 4.14 Hz**。真正的大头是每动作固定开销（observe 5.06 s + env_step 7.89 s + 缓存出队的 112 ms×114）。`[R]`
6. **推荐路线**：先做 **R2 专用内核最小切片**（在同版本 py3.10/torch2.7.1 里重编译 FlashRT `csrc`，先替换 `sdpa` 注意力热路径，固定观测回放对齐），再做 R1 IR/harness 全量注册。理由见 §4.4。**本轮不进入实施。**

---

## 1. 现场核实

### 1.1 文档与约定

| 检查项 | 结论 |
|---|---|
| `AGENTS.md` | `/home/host/page-HK`、`/home/host/page-dmh`、`embodied_engine`、`FlashRT` 四处**均不存在** `[R]` |
| 适用架构文档 | `embodied_engine/docs/architecture.md`（三层端口/来源/状态）、`FlashRT/docs/architecture.md`（八组件+分层）、`FlashRT/docs/adding_new_model.md`（接新模型硬规则）、`OpenWAM/README.md` |
| 既有审计 | `embodied_engine/docs/audits/2026-10-29-flashrt-openwam-acceleration-audit.md`（本轮只读） |
| 历史线索核对 | 用户给的 `abb75db…`、`33f1814` 与现场 HEAD/父链一致；**未 checkout、未回退** `[R]` |

### 1.2 git 身份（准确值）

| 工程 | 真实 git 根 | remote | 分支 | HEAD | 工作区 |
|---|---|---|---|---|---|
| 三层引擎 | `/home/host/page-dmh` | `git@github.com:dalbet/openwam-harness-page.git` | `feature/openwam-pi05-review-prep` | `33f18148d27654f7223e925248fe9b26b23410e4` | 干净（`status --porcelain` 空） |
| OpenWAM | `/home/host/page-dmh/openwam_harness/OpenWAM` | `https://github.com/OpenWAM-Official/OpenWAM.git` | **detached** | `4d7f642b841913ea3c176010b692b1de3d5ef180` | 干净 |
| FlashRT | `/home/host/flashrt-store/page_engine/FlashRT` | **无 remote、无 `.git`** | — | — | 不可判定（无版本控制） |
| `/home/host/page-HK` | **不是 git 仓库** | — | — | — | — |

- `[S]` `config/engine.json` 的 `action_model.openwam.repo_commit = 4d7f642b…` 与现场一致；`hf_revision = 110b07e4d5581c81520a77419cbb0e9371df231b`；`checkpoint_file = checkpoint_step_10690.safetensors`。
- `[R]` 子模块：三层引擎无 `.gitmodules` 声明 → 无子模块；OpenWAM 为独立 clone。

### 1.3 运行态（本轮现场抓取）

| 端口 | 服务 | PID | cwd | 解释器 | GPU / 显存 |
|---|---|---|---|---|---|
| 8001 | vLLM `Qwen3.5-9B`（planner） | 2843152 | `embodied_engine` | `openwam_harness/envs/vllm/bin/python` | GPU3 / 22.1 GB |
| 14510 | Pi0.5 常驻服务（对照） | 2784215 | — | RPent | GPU1 / 7.2 GB |
| 14520 | **OpenWAM 常驻策略服务** | 1380923 | `openwam_harness/OpenWAM` | `.venv-openwam/bin/python`（3.10.12 / torch 2.7.1+cu128） | GPU2 / 24.8 GB |
| 14561 | LIBERO 环境 RPC | 2219370 | `embodied_engine` | `envs/rpent/bin/python` | GPU3 / 0.5 GB |
| 14600 | 推理服务 | 3900284 | `embodied_engine` | `envs/rpent/bin/python` | — |
| 14700 | Agent 服务 | 3904106 | `embodied_engine` | `envs/rpent/bin/python` | — |
| 14800 | 引擎服务 | 1854309 | `embodied_engine` | `envs/rpent/bin/python` | — |
| 14900 | page_ui | 2784912 | `embodied_engine` | `envs/rpent/bin/python` | — |

OpenWAM 真实启动命令行（`/proc/1380923/cmdline`）：

```
.venv-openwam/bin/python scripts/deploy.py \
  --ckpt-dir .../checkpoints/OpenWAM-Alpha-Sim-LIBERO \
  --ckpt-name checkpoint_step_10690.safetensors \
  --device cuda:0 --host 127.0.0.1 --port 14520 \
  --compile-enabled false --inference-mode sync \
  --denoise-steps 10 --inference-horizon 32
```

GPU 现场：0 号卡空闲（4 MiB），1/2/3 分别被 Pi0.5、OpenWAM、vLLM+env 占用。**本轮不新增、不停止任何进程。**

### 1.4 证据冲突（必须在第一轮测量中解决）

| # | 冲突 | 现状 | 解决方式 |
|---|---|---|---|
| C1 | `torch.compile` 是否生效 | `[S]` CLI `--compile-enabled false` → `server.py:104-110,642` 写回配置 → `joint_self_attn.py:134-142` 直接 return；日志无 “Enabled dual self-attn MoT compile”。`[R]` 但存在 32 个 `torch/_inductor/compile_worker` 子进程（PPID 1380923，02:50:04 启动） | 新实例开 `TORCH_LOGS=recompiles,dynamo` 复核；不要动现有服务 |
| C2 | 服务端 `latency_ms` 与客户端 `send→recv` 不能嵌套 | 同一请求：服务端 `511.7 ms`，客户端 `send+recv = 10.8 + 431.7 = 442.5 ms`，服务端跨度**大于**客户端跨度，物理上不可能 | 以客户端单一时钟为唯一权威；服务端数字只作相对量；回放探针里同时打印两侧时间戳 |
| C3 | 生成延迟两个记录不一致 | `speed_qwen_openwam.json`（96 动作）生成中位 `426.7 ms`；`composite_state_restore_12` 回合生成中位 `510.7 ms` | 差异来自任务/观测与冷热态，第一轮用同一固定观测重复测 |

---

## 2. FlashRT 身份确认

### 2.1 结论：`flashrt-project/FlashRT` 的上游快照，不是 fork，也不是自研

- `[S]` `flash_rt.egg-info/PKG-INFO:6-8` Homepage / Repository / Issues 全为 `https://github.com/flashrt-project/FlashRT`；`pyproject.toml:6-7` `name="flash-rt"`、`version="0.2.0"`。
- `[S]` `docs/COMMUNITY_AND_CREDIT.md` 链接同一仓库；`CITATION.cff` 存在。
- `[R]` 安装形态：`page_engine/.venv/lib/python3.12/site-packages/flash_rt-0.2.0.dist-info/direct_url.json` = `{"url":"file:///home/host/page-HK/page_engine/FlashRT","dir_info":{"editable":true}}`。
- `[R]` 本地新增物（相对上游）：`build/`、`flash_rt/*.cpython-312-x86_64-linux-gnu.so`、`flash_rt.egg-info/`、`page_engine/build.log`、`bench_pi05` 符号链接。**没有** `.git`、`CHANGELOG`、本地补丁文件，所以“相对上游是否改动过”**无法用 git 判定** `[I]`（可用上游相同 tag 做 diff 确认，需联网，本轮不做）。
- 与 `Infini-AI-Lab/FlashRT` 无关：该仓库的任何标识（包名/module 布局/引用）在本机源码中都不存在 `[S]`。

### 2.2 已支持能力（与本案相关的部分）

| 类别 | 内容 | 状态 |
|---|---|---|
| 硬件后端 | `rtx_sm120`（RTX 5090 / GB10）、`rtx_sm89`、`rtx_sm87`、`thor`(SM110)、`npu`、`amd_cdna3/4`、`amd_rdna35/4`、`apple_mps` | 可运行（`hardware/__init__.py:32-50`） |
| 已注册模型（`_PIPELINE_MAP`，`hardware/__init__.py:124+`） | pi0 / pi05 / pi0fast / groot n16·n17 / groot_n17 amd / hyvla / lingbot / cosmos3(video, edge, reasoner) / wan22_ti2v_5b / qwen3·qwen36·qwen3_vl / spark_x25 / flint / dm05 / ltx25 / motus / nexn2 / alpamayo15 / higgs_audio_v3 / melband_roformer / chameleon / omnivoice / minimax_remover | 已注册；**其中只有 pi05、wan22 与本任务相关** |
| π0.5 实测 | FP8 `23.66 ms`、FP16 `36.24 ms`、OpenPI eager 参考 `52.36 ms`（`/home/host/flashrt-store/bench_pi05/*.json`，`runs=20`） | **已实测**，可引用 |
| Wan2.2 | `Wan22TorchFrontendRtx`（`frontends/torch/wan22_rtx.py:27`）+ `models/wan22/pipeline_rtx.py`；默认 832×480/81 帧/20 步/unipc，带 TeaCache | **仅视频生成**；无动作分支、无 proprio、无动作头 |
| 结构目录（IR 素材） | `catalog/structures/`：`video_generation_pipeline`（含显式 `denoise` stage + `graph_buckets` + parity gate）、`vla_tick_pipeline`、`attention_core`、`qkv_pack`、`qk_norm_rope`、`adaln_producer`、`norm_fused`… | 结构规格存在，**binding 只覆盖已有模型**（`catalog/bindings/*.yaml` 里无 openwam） |
| 运行时/IR | `flash_rt/runtime/{export,exec,rtc,vlash}.py`、`subgraphs/{capture,stage_plan}.py`；“kernel library + 8 个基础设施组件”，**不是编译器/图重写器**（`docs/architecture.md` TL;DR） | 可用，但需要按模型手写 pipeline |
| 参考执行 | `hardware/torch_reference.py`（CPU/GPU 纯 torch 参考 provider） | 可作数值基准 |
| 测试 | `flash_rt/tests/`（Jetson PI 三件套）+ `benchmarks/`（14 个脚本，含 `pi05_production_profile.py`、`pi05_amd_latency.py`） | 无 OpenWAM 相关 |

### 2.3 OpenWAM 相关代码：**零**

- `grep -rni "openwam"` 全仓 → 0 命中。
- `grep -rni "WAM"` → 仅 `docs/subgraph_stage_plans.md:270` 一句 “For diffusion or WAM-style loops, register a factory”。
- 因此不存在“能跑的 OpenWAM 代码 / 仅接口 / 占位”三类中的任何一类，**全部缺失**。`[S]`

### 2.4 缺失清单（R1 全量接入所需，`docs/adding_new_model.md` 口径）

| # | 缺失项 | 上游要求/参考行数 |
|---|---|---|
| 1 | `AttentionSpec`（双流 MoT 的 joint self-attn，视频 24×128 + 动作 N×64，mask=`mutual`/`first_frame_causal`） | 15–35 行/模型 |
| 2 | pipeline forward（10 步 schedule 交织，`fixed_shape`，swap 缓冲） | 200–400 行/硬件 |
| 3 | `WEIGHT_SPEC`（`WanModel` 30 层 + `ActionDiT` 30 层 × `video_projs` + UMT5-XXL + Wan2.2 VAE） | 60–120 行/组合 |
| 4 | frontend（`load_model` 入口、proprio normalizer、`normalization_stats.npy`、binary dim 投影） | 700–1000 行 |
| 5 | `_PIPELINE_MAP` 注册 + config YAML + 精度测试 | 4 + 30 + 80 行 |
| 6 | 数值基准：同 seed 的 `torch_reference` 对照（`matched_noise`） | 文档 §3 |

---

## 3. 实际调用链（页面 → env.step）

### 3.1 总览

```
page_ui/server.py:329 /api/load  → engine_service.py:748 load()
page_ui/server.py:341 /api/run   → engine_service.py:2622 run() → start_run()
        ↓
engine_service.py:1272  POST {agent}/plan  → agent_service.py:276 plan()
        ↓ semantic_router.decide() → config/tasks.json 官方指令 → plan_contract 校验
engine_service.py:1017  WholeTaskRunner(...)
        ↓
openwam_rollout.py:639  run()  ── 每动作循环 ──────────────────────────────
  :506 _observe_model_input() → env.raw_obs() → openwam_contract.build_openwam_obs()
  :513 _infer() → inference_service.py:203 infer() → models/openwam_backend.py:146 predict()
        ↓ WebSocket ws://127.0.0.1:14520  {type:"obs", images{head_camera,left_wrist_camera}, prompt, state}
  OpenWAM: server.py:246 ws_handler → :252 json.loads → :189 predict()
        ↓ obs_preprocess.py:120 preprocess() → 384×320 L 形拼图 + 归一化 state
        ↓ policy.py:53 predict_action() → executors/sync_executor.py:43 predict_action()
            ├─ 缓冲非空 → popleft()（≈15 ms 路径）
            └─ 缓冲空   → :51 _generate_and_enqueue() → engine.py:252 generate()
                            → base.py:1361 generate() 10 步去噪（≈510 ms 路径）
        ↑ {"type":"action","action":[10],"step":N,"latency_ms":...}
  :671 native_eef10_to_libero7d(action) → 7 维
  :673 env.step(action7)
  :706 _snapshot() / frame_sink
```

### 3.2 逐段行号（关键文件）

| 段 | 文件:行 | 关键内容 |
|---|---|---|
| 页面→引擎 | `page_ui/server.py:305,329,341` | `/api/tasks`、`/api/load`、`/api/run` 转发（`engine_get/post` → 14800） |
| 引擎入口 | `engine/engine_service.py:2622` | `async def run()` → `start_run(task_id, instruction, seed)` |
| Agent 调用 | `engine/engine_service.py:1272` | `POST {agent_url}/plan` |
| Agent 入口 | `agent/agent_service.py:276` | `async def plan()`；`:175 semantic_router.decide()`、`:181 assemble_plan()` |
| 语义路由 | `agent/semantic_router.py:187-218` | `max_tokens=24`、`model_mode=non_thinking`、`temperature=0.0` |
| rollout 构造 | `engine/engine_service.py:1017` | `WholeTaskRunner(...)` |
| **主循环** | `engine/openwam_rollout.py:639` | `run()`；`:645` 起每动作一次迭代 |
| 观测 | `engine/openwam_rollout.py:506` | `raw = self.env.raw_obs()` → `build_openwam_obs(raw)` |
| 推理调用 | `engine/openwam_rollout.py:513-530` | `self.inference.infer(request, obs, timeout_s=...)`，`np.float32` 断言 |
| 推理服务 | `inference/inference_service.py:40-62` | `_build_backend()`：`pi05`→`Pi05Backend(service_url("pi05"))`、`openwam`→`OpenWAMBackend(...)` |
| 推理队列入队 | `inference/inference_service.py:203-236` | `infer()`：`lock` 串行，`backend.predict(obs, prompt)` |
| 后端客户端 | `models/openwam_backend.py:146-236` | 编码 PNG → `json.dumps` → `send` → `recv(timeout=300)` → 解析 |
| 编码 | `models/openwam_backend.py:30` | `_png_b64()`（base64 PNG，实测 ≈83.6 ms/次） |
| reset | `models/openwam_backend.py:131` | 发 `{"type":"reset"}`，等 `reset_ack`（timeout 60 s） |
| 服务端路由 | `openwam/deploy/server.py:246-263` | `reset` / `obs`(→`predict`) / `ping` 三分支；`:305` `websockets.serve(..., ping_interval=None)` |
| 服务端预测 | `openwam/deploy/server.py:189-216` | `t0=monotonic()` → `preprocess` → `policy.predict_action` → `latency_ms` |
| 预处理 | `openwam/deploy/obs_preprocess.py:67,120` | `ObsPreprocessor.preprocess()`：多视角 L 形拼图 `canvas=384x320`，`camera_layout=['observation.images.image','observation.images.image2',None]` |
| 策略门面 | `openwam/deploy/policy.py:48,53,76` | sync 用 `SyncInferenceExecutor`；`predict_action` 尾部做二值维度 ±1 投影 |
| 缓冲 | `openwam/deploy/executors/sync_executor.py:22,43,51,69` | `deque` 队列；空则生成，`extend(actions[:inference_horizon])`；`reset()` 清空 |
| 生成入口 | `openwam/deploy/engine.py:252-370` | `generate()`；`:286 denoise_steps`、`:321 dit_cache.reset()`、`:350 seed=42` |
| 去噪主循环 | `openwam/model/architectures/base.py:1361,1454-1490,1505` | `generate()`；`torch.randn(..., generator=manual_seed(seed))`；`for i in range(len(schedule)-1)` |
| 动作→环境 | `engine/openwam_rollout.py:671-673` | `native_eef10_to_libero7d(raw_action)` → `self.env.step(action7)` |

### 3.3 张量契约（本轮实测/源码值）

| 量 | 形状 / dtype | 生成处 | 说明 |
|---|---|---|---|
| `agentview_image` / `robot0_eye_in_hand_image` | `(256,256,3) uint8` | LIBERO env | 原始 |
| `main_images` | `(320,256,3) uint8`（W,H） | `openwam_contract.py:175-207`，`rotate_180` → LANCZOS `resize` | `HEAD_SLOT=(320,256)` |
| `wrist_images` | `(160,128,3) uint8` | 同上 | `WRIST_SLOT=(160,128)` |
| 服务端拼图 | `(384,320,3)` RGB | `obs_preprocess.py:120+`（`crop_and_resize` / L 形） | 日志 `canvas=384x320` |
| `states` | `(10,) float32` | `eef10_from_raw()` `:119` | `[eef_pos(3), rot6d(6), grip_scale(1)]`，`grip=2*width/0.08-1` 夹到 ±1 |
| 服务端 `state` | `(10,)` list → tensor | `server.py:189+` | 模型内部 `state_dim=80`（`unify_state_map` 补齐） |
| 采样 dt / 步数 | `denoise_steps=10`，`shift=5.0`（动作）/`shift_video=5.0` | `engine.py:286-312` | `make_schedule(...)` |
| 初始动作噪声 | `(1, 31, 80)` bf16 | `base.py:1490` | `action_num_frames-1 = 32-1`，`action_dim=80` |
| 动作输出 | `(1, 31, 80)` → 取 `<unify_action_map 0-9>` 维 → `(10,)` | `policy.py:53` + 归一化反变换 | 线上 10 维 `native_eef10` |
| 发回客户端的动作 | `(10,) float32`，每维 ∈[-1,1] | `server.py:214` `action.tolist()` | 客户端再 `clip` |
| LIBERO 指令 | `(7,) float32` | `openwam_contract.py:133` | `[d_xyz(3), axis_angle(3), -open_scale(1)]` |
| 模型内部宽度 | 视频 DiT `dim=3072, ffn=14336, heads=24, layers=30, in_dim=48`；ActionDiT `dim=1024, ffn=4096, layers=30`（必须等于视频层数）；文本 `UMT5-XXL, text_dim=4096, seq_len=512` | checkpoint `config.yaml` | 权重 `24.81 GB` safetensors |

### 3.4 采样、随机性与确定性

- `[S]` 每次生成都新建 generator：`base.py:1490` `torch.Generator(device).manual_seed(seed)`，`seed` 来自 `engine.py:350` 的 `conditions.get("seed", 42)`；客户端**从不传 seed**（`openwam_backend.py:146-190` 无 seed 字段）→ 同观测同 prompt 应逐位一致。
- `[S]` 视频 latent 由 `preprocess_input_for_inference(..., seed=seed, ...)` 生成，同一 seed 同形状同噪声。
- `[R]` 既有 `scripts/inference_determinism_probe.py` 专门验证这一点（两次请求 + reset 后比对），可直接进入第一轮。
- `[S]` DiT velocity cache 会**跳过**视频 DiT 计算（阈值 0.99 / 最多连续 3 次），这是**改变数值路径**的优化，已启用；做“加速前后”对比时必须报告它是否开启。

### 3.5 缓冲 / reset / 超时 / 会话隔离契约

| 契约 | 实现 | 影响 |
|---|---|---|
| chunk 生成 | `generate_horizon=32`（`inference.num_frames=33` → 32 个动作） | 每 32 次请求生成 1 次 |
| 执行长度 | `--inference-horizon 32` → `SyncInferenceExecutor.inference_horizon=32` | 32 个动作全执行，第 33 次请求重新生成 |
| 缓冲语义 | `deque`；空才生成；生成后 `clear()` 再 `extend` | 无跨 chunk 拼接、无重叠执行 |
| reset | 客户端 `{"type":"reset"}` → `PolicyServer.reset()` `server.py:219-226` → `WAMPolicy.reset()` `policy.py:76` → `executor.reset()` 清 `deque`；`_request_count` 归零 | 回合/组合步骤切换必须 reset（`composite.handoff.reset_policy_buffer=true`） |
| 超时 | 客户端 `recv(timeout=300)`（`action_model.openwam.timeout_s=300`）；引擎侧 `inference.openwam_timeout_s=300` | 首次编译/冷启可能超时，故 `ping_interval=None` |
| 会话隔离 | 单 WS 监听、单 `PolicyServer` 实例、`_policy`/`_action_buffer` 均进程级；`infer()` 在 `lock` 内串行 | **不支持多会话并发**；第二个客户端会共享同一 buffer（协议层面无 session id） |
| 异常回退 | 传输异常 → `_reconnect_locked()`；动作非有限值 → 抛错（不静默） | `openwam_backend.py:178-213` |
| 停止 | 引擎 `stop_event` 在 `env.step` 前后各检查一次（`openwam_rollout.py:670,676-687`） | 已发出的动作不可撤回 |

### 3.6 已启用 / 未启用的优化（现场事实）

| 特性 | 状态 | 证据 |
|---|---|---|
| 混合精度 bf16 | **已启用** | ckpt `config.yaml` `mixed_precision: bf16`；`model_loader.py:210-211` |
| 注意力实现 | **`sdpa`（auto）** | `records/logs/openwam_server.log:1` “Using attention backend: sdpa (auto-detected)”；`action_backbone/components.py:352-360` 优先级 `flash3→flash2→sage→xformers→sdpa`，前四者均未安装 `[R]` |
| DiT velocity cache | **已启用**（0.99 / 3） | 日志 `:19`；`optimizations/dit_cache.py` |
| prompt embedding LRU | 已启用（maxsize 32） | `configs/deploy.yaml` |
| VAE 解码 | **已关闭**（`decode_video:false`） | `configs/deploy.yaml:26` |
| torch.compile | **CLI 明确关闭**（`--compile-enabled false`）；配置默认值 `enabled: true` 被 CLI 覆盖 | 见 C1 冲突 |
| CUDA Graph | **未启用**（无手工捕获代码；仅 `compile_options.py:183` 提到 `reduce-overhead` 会用到） | `grep CUDAGraph` 仅 1 处注释 |
| 异步去噪 / prefetch | 服务端 `sync`；引擎 `async_chunk_prefetch` 只作用于缓存出队 | 启动参数 + `engine.json` |
| 客户端图像编码 | base64 **PNG** | `openwam_backend.py:30`（JPEG 可显著降低，但会改输入比特） |

---

## 4. 适配判断与瓶颈

### 4.1 兼容性判定（逐项，不用类比）

| 问题 | 判定 | 依据 |
|---|---|---|
| FlashRT 能直接加载 OpenWAM checkpoint？ | **不能** | 无 `WEIGHT_SPEC`；`_PIPELINE_MAP` 无 `openwam` 键 |
| Wan2.2 支持可复用？ | **仅结构参考** | FlashRT 的 Wan2.2 是 `diffusion video only`，无 `ActionDiT`、无 joint attention、无 proprio 通道 |
| π0.5 支持可复用？ | **仅工程参考** | 单流 VLA，`pi05` pipeline 无双流交织 |
| 同进程 import flash_rt？ | **不能** | py3.12/torch2.14 vs py3.10/torch2.7（ABI + 算子集都不兼容） |
| 跨进程协议复用？ | **可行** | `OpenWAMBackend` 只依赖 `obs/reset/ping` 三种消息 + `server_url`；协议在 `server.py:246-263` |
| IR/stage-plan 能表达双流 MoT？ | **不能直接表达** | `video_generation_pipeline` 是「单 latent + 单 scheduler」；`subgraph_stage_plans.md:270` 的参数化切分只做链式/分块，无「双流同层交织」语义 |
| 专用 kernel 能嵌入现服务？ | **可行（推荐）** | FlashRT `csrc` 是独立 `.so`，可在 py3.10 环境重编译；替换点是 `torch_sdpa` 与 GEMM/RMSNorm |

### 4.2 实测耗时（只用真实记录）

来源：`records/acceptance/composite_state_restore_12_20261028_runs/ep-20261028-100255-e4e68d.json`

| 指标 | 值 | 说明 |
|---|---|---|
| 回合 wall | `30.90 s` / 118 动作 | `control_hz_wall=3.82`、`sim_real_time_factor=0.191` |
| 生成（server） | n=4，median `510.7 ms`，min 507.69，max 511.7 | 含预处理 + 10 步去噪 |
| 生成（engine_total） | median `609.8 ms` | 含 PNG 编码 83.6 ms + 传输 |
| 缓存出队（server） | n=114，median `15.3 ms` | 预处理 + deque |
| 缓存出队（engine_total） | median `112.2 ms` | 编码 83.6 + 传输 ~13 + 服务端 15 |
| 单动作总耗时 | median `0.2446 s`（p95 0.2554） | 决定 3.82 Hz |
| 相位合计 | observe 5.06 / inference 16.69 / convert 0.075 / env_step 7.89 / snapshot 1.15 | `unattributed 0.03 s` |
| 首次动作 | `0.685 s` | 首 chunk 生成 |

生成占比：`4 × 0.6098 = 2.44 s / 30.90 s = 7.9%`。
→ 生成时间归零的上限：`30.90 - 2.44 = 28.46 s` → `4.15 Hz`（现 3.82 Hz，**+8.6%**）。
→ 生成时间减半（如 510→255 ms）：`29.68 s` → `3.98 Hz`（**+4.2%**）。

### 4.3 为什么“模型加速”在闭环里被摊薄

- 生成只在每 32 个动作发生 1 次（`chunk_generations=4 / requests=118`）。
- 其余 114 次请求仍是**完整一次网络往返**：客户端 base64 PNG（83.6 ms）+ WS + 服务端预处理（15 ms）。
- 因而闭环的主项是 `observe + env_step + 每次往返固定开销`，不是 DiT。
- **结论**：对外展示要分两层——「策略服务级生成延迟」（模型加速的真实收益，510.7 ms 是基线）与「闭环级 Hz/RTF」（收益有限，且会被非模型开销淹没）。

### 4.4 三条路线与排序

| 排名 | 路线 | 内容 | 收益依据 | 工作量 | 正确性风险 |
|---|---|---|---|---|---|
| **1** | **R2 专用内核最小切片** | 在 `py3.10 + torch2.7.1+cu128 + SM120` 独立 venv 里重编译 FlashRT `csrc`；先接 `attention`（FA2/Sage2 或 FlashRT 自有 attention）替换 `sdpa`，再考虑 fused RMSNorm/GEMM；模型结构、步数、chunk、seed 全不变 | 现场注意力是 `sdpa` 且 `flash_attn/xformers/sage` 全未装 → 存在明确可替换项；FlashRT 在 π0.5 上把 eager 52.36→23.66 ms 说明其 kernel 在 5090 有效；**OpenWAM 能吃到多少 = 待测** | 中（重编译 + 逐层对齐） | 中低（可逐层 cosine 对齐，失败回退 sdpa） |
| 2 | R1 IR/harness 全量注册 | `AttentionSpec` + pipeline + `WEIGHT_SPEC` + frontend + `_PIPELINE_MAP`，把 OpenWAM 做成 FlashRT 一等公民 | 上游文档工作量基线：pipeline 200–400 行、WEIGHT_SPEC 60–120 行、frontend 700–1000 行；`video_generation_pipeline` 结构只覆盖单流 | **大** | 高（双流交织语义、数值漂移、调度） |
| 3 | R3 服务层“提速” | 加 chunk / 异步 prefetch / 跳过视频分支 / 降步数 | 收益大但**改行为或时序** | 低 | **本轮明确禁止**（降步数只能作负对照） |

**最小切片的具体切口（三个，按风险从低到高）**：

1. 只换注意力实现（`WAM_ATTENTION_IMPL` 环境变量已存在，`components.py:363-385`），不碰权重与结构；
2. 换注意力 + 融合 RMSNorm/AdaLN 的小算子；
3. 动作侧 `ActionDiT` 单独走一个优化 pipeline（视频侧仍 eager），风险最高，因为 joint attention 需要两侧同层同步。

### 4.5 优化分类

| 类别 | 手段 | 影响面 |
|---|---|---|
| **只改善推理** | attention 后端替换、fused kernel、CUDA Graph、bf16→fp8 | 只降 `server_latency_ms`；闭环按 §4.3 摊薄 |
| **可能改善闭环** | 生成停顿变短；客户端 PNG→JPEG/共享内存；服务端复用解码缓冲 | 改 wall / Hz / RTF，不改动作语义（图像编码会改输入比特，必须做数值对齐） |
| **改变行为或时序** | 降 `denoise_steps`、改 chunk/horizon、FP8/FP4 权重量化、改 DiT cache 阈值、跳视频分支 | 动作分布变化 → 必须重跑任务成功率验收 |

---

## 5. 后续验证设计（三套，全部用现有脚本）

### A. 固定输入回放（策略服务级，最便宜，先做）

- 入口：`scripts/inference_determinism_probe.py`
- 做法：先用 `--save-npz` 抓一份真实观测（`main_images/wrist_images/states`），之后所有对比都用 `--input-npz` 回放同一份。
- 公平条件：同一 `--prompt`、同一 `--state`、同一 checkpoint、`reset` 后再请求；客户端不传 seed（服务端固定 42）。
- 指标：`identical`（逐位）、`max_abs_diff`、`mean_abs_diff`、`server_latency_ms`、客户端 `send→recv`。
- 通过标准（建议）：`max_abs_diff ≤ 1e-3`（bf16 同路径应接近 0）；延迟只做相对比较。

### B. 策略服务基准（对外展示“加速前后”的主指标）

- 入口：`scripts/bench_speed.py --skip-planner --skip-episode --openwam-actions 96`
- 指标：`generation.server_latency_ms` 的 median/p95、`cache.server_latency_ms`、`encode_s`、`send_s`、`recv_s`、`chunk_generations/cache_hits`。
- 公平条件（必须写进报告）：同 checkpoint、同 `denoise_steps=10`、同 `generate/execute_horizon=32`、同 seed 42、同 bf16、同卡（GPU2 独占）、warmup 至少 5 次生成、测量 ≥20 次生成、报告 median 与 p95 而不是单次。
- 已知必须披露的混淆项：DiT velocity cache 是否命中（命中会跳视频 DiT）；`torch.compile` 是否开启；是否有其它客户端占用 GPU2。

### C. 单任务闭环（任务效果，最终验收）

- 入口：`scripts/run_rounds.py --manifest <新清单> --out <新记录>`
- 清单格式（`rounds` 数组，字段见 `run_rounds.py:369-448`）：`{"round":1,"task_id":<n>,"lang":"en","instruction":"<原始指令>"}`
- 公平条件：同任务、同 `seed`、同初始场景（比对 `initial_scene_fingerprint`）、同 `max_steps`、同停止/成功判定；加速前后各 N 回合（建议 ≥12，与既有 12 回合验收口径一致）。
- 指标：`native_task_success`、`user_goal_success`、`sim_action_count`、`timing.*`、`openwam.wall.control_hz_wall`、`sim_real_time_factor`、`chunk_stats`。
- 数值校验：动作级比对（同一观测下 10 维动作的 max/mean abs diff）+ 轨迹级比对（`episode_sim_actions`、成功与否）。

---

## 6. 第一轮测量与实验计划（本轮之后执行，含命令）

> 全部命令都是仓库里**已存在**的脚本；不新增接口、不编造路径。执行前先确认没有别的实验在跑。

### 步骤 0：只读基线快照（不产生负载）

```bash
cd /home/host/page-dmh/embodied_engine
bash scripts/status.sh
curl -s --max-time 5 http://127.0.0.1:14600/health | python3 -m json.tool
curl -s --max-time 5 http://127.0.0.1:14800/api/status | python3 -m json.tool
cat records/openwam_server.json          # 启动记录（pid/ckpt/horizon/steps/compile_enabled）
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
```

### 步骤 1：确认权重与加载契约（CPU，不占 GPU）

```bash
cd /home/host/page-dmh/embodied_engine
.venv-openwam/bin/python scripts/openwam_probe.py --device cpu \
  --out records/acceptance/openwam_weight_contract_20261031.json
```

### 步骤 2：固定观测回放（策略服务级，抓一份观测并冻结）

```bash
cd /home/host/page-dmh/embodied_engine
# 第一次：抓观测并保存
RP=openwam_harness/envs/rpent/bin/python
$RP scripts/inference_determinism_probe.py \
  --engine http://127.0.0.1:14800 --url ws://127.0.0.1:14520 \
  --prompt "put the bowl on the stove" \
  --save-npz records/acceptance/replay_obs_20261031.npz \
  --out records/acceptance/determinism_baseline_20261031.json
# 之后每次都只回放这份
$RP scripts/inference_determinism_probe.py \
  --input-npz records/acceptance/replay_obs_20261031.npz \
  --url ws://127.0.0.1:14521 \
  --out records/acceptance/determinism_flashrt_20261031.json
```

### 步骤 3：策略服务基准（加速前后各一次，同一 frozen 观测）

```bash
cd /home/host/page-dmh/embodied_engine
RP=openwam_harness/envs/rpent/bin/python
$RP scripts/bench_speed.py --skip-planner --skip-episode \
  --openwam-actions 96 \
  --out records/acceptance/speed_openwam_baseline_20261031.json
# 加速后（新服务在 14521，不改 config/engine.json 的长期值，用临时 override 或 bench 脚本参数）
```

> 注意：`bench_speed.py` 当前读的是引擎 `config/engine.json` 里的 `server_url`。切换到新端口时**只做临时改动并立即还原**，或新增一个只读 `--server-url` 的薄包装脚本（本轮不写）。

### 步骤 4：闭环单任务（N≥12 回合）

```bash
cd /home/host/page-dmh/embodied_engine
cat > records/acceptance/manifest_flashrt_ab_20261031.json <<'JSON'
{"rounds":[
 {"round":1,"task_id":8,"lang":"en","instruction":"open the top drawer of the cabinet and put the bowl inside"},
 {"round":2,"task_id":8,"lang":"en","instruction":"open the top drawer of the cabinet and put the bowl inside"}
]}
JSON
RP=openwam_harness/envs/rpent/bin/python
$RP scripts/run_rounds.py \
  --manifest records/acceptance/manifest_flashrt_ab_20261031.json \
  --out records/acceptance/rounds_openwam_baseline_20261031.json
```

> 指令文本必须从 `config/tasks.json` 的官方 `instruction` 取，不要自拟（`run_rounds.py:373` 用该目录做校验）。

### 通过 / 失败判据

| 层级 | 通过 | 失败即回退 |
|---|---|---|
| 权重加载 | key 全命中、无 missing/unexpected | 停止接入 |
| 回放数值 | `max_abs_diff ≤ 1e-3` | 保留 sdpa |
| 服务延迟 | median 生成时间有可复现下降（≥10%），p95 不恶化 | 保留原实现 |
| 闭环 | 成功率不低于基线（同样本量），`sim_action_count` 同量级 | 保留原实现 |

---

## 7. 最小改动清单（下一轮实施时；本轮不动手）

| # | 改动 | 文件（新增/配置，不改旧源码） | 备注 |
|---|---|---|---|
| 1 | 新增 FlashRT 版策略服务（同 WS 协议 `obs/reset/ping`） | 新目录 `embodied_engine/flashrt_policy_server/`（或 `page-HK` 侧独立包） | 协议照 `openwam/deploy/server.py:246-263` |
| 2 | 新服务启动脚本 | 新增 `scripts/flashrt_policy_server.sh`（照 `scripts/openwam_server.sh` 结构） | 端口 14521，GPU0（当前空闲） |
| 3 | 重新编译 FlashRT 扩展 | `bash /home/host/page-HK/page_engine/build_flashrt.sh`，**在 py3.10/torch2.7.1 的独立 venv** | 现有 `build/` 是 py3.12 产物，不能复用 |
| 4 | 环境隔离 | 新增 venv（如 `embodied_engine/.venv-flashrt`） | 不动 `.venv-openwam` |
| 5 | 切换后端地址（仅实验期间） | `config/engine.json` → `action_model.openwam.server_url` | 实验后还原；长期值不变 |
| 6 | 数值对齐 | 复用 `scripts/inference_determinism_probe.py`、`scripts/bench_speed.py` | 不改 |
| 7 | 新的测量记录 | `records/acceptance/*_20261031.json` | 不改历史记录 |

**必须保持不变**（否则不是“同模型”）：checkpoint 文件与 `hf_revision`、`generate_horizon=32`、`execute_horizon=32`、`denoise_steps=10`、`seed=42`、`shift=5.0`、图像 rotate+LANCZOS+384×320 拼图契约、EEF10→7D 转换、planner、控制频率 20 Hz、停止与成功判定。

---

## 8. 未决问题与风险

1. **C1（compile 真相）**：必须先在新实例确认，否则“加速前基线”可能被误标。
2. **C2（计时口径）**：服务端 `latency_ms` 与客户端跨度不能嵌套，差 ~69–80 ms；对外只用一个时钟。
3. **`page-HK` 无版本控制**：任何就地修改都不可追溯。接入必须在独立 venv + 独立目录进行，并记录文件哈希。
4. **无 AGENTS.md**：约定只存在于 `docs/architecture.md` 与本审计，实施前应补一份规则文件（本轮不创建）。
5. **单会话协议**：新旧服务不能同时对同一 `PolicyServer` 实例发起请求；A/B 必须串行，且每轮 `reset`。
6. **DiT cache 的非确定性感知**：它按余弦相似度跳步，同一观测在不同上下文下可能走不同步数 → 回放对比必须固定“冷 buffer + reset”。

---

## 附录 A：本轮新增证据索引

| 主题 | 文件:行 | 值 |
|---|---|---|
| FlashRT 版本 | `pyproject.toml:6-7`、`flash_rt.egg-info/PKG-INFO:6-8` | `flash-rt 0.2.0`，`flashrt-project/FlashRT` |
| FlashRT 安装形态 | `page_engine/.venv/.../flash_rt-0.2.0.dist-info/direct_url.json` | editable，`file:///home/host/page-HK/page_engine/FlashRT` |
| 硬件分派表 | `flash_rt/hardware/__init__.py:32-50,124+` | 9 类 arch，π0.5 在 `:140-151` |
| Wan2.2 仅视频 | `flash_rt/frontends/torch/wan22_rtx.py:27-56` | 默认 832×480/81 帧/20 步 |
| 结构目录 | `flash_rt/catalog/structures/video_generation_pipeline/structure.yaml` | 单 latent + 显式 denoise stage |
| 接新模型工作量 | `FlashRT/docs/adding_new_model.md:139,190,236,294,338` | 15–35 / 200–400 / 60–120 / 700–1000 行 |
| OpenWAM 配置 | `config/engine.json` `action_model.openwam.*` | horizon 32/32、steps 10、dim 10、gpu 2、timeout 300 |
| 模型结构 | `.../checkpoints/OpenWAM-Alpha-Sim-LIBERO/config.yaml` | DiT 3072/30 层/24 头/in_dim 48；ActionDiT 1024/4096；action_dim 80；bf16 |
| 缓冲 | `openwam/deploy/executors/sync_executor.py:22,43,51,69` | `deque` + `inference_horizon` |
| 采样 | `openwam/model/architectures/base.py:1361,1490,1505` | `manual_seed(42)`、`for i in range(len(schedule)-1)` |
| 服务端计时 | `openwam/deploy/server.py:189-216` | `t0` 在 `predict` 内部 |
| 客户端计时 | `models/openwam_backend.py:146-236` | `recv_s = elapsed - send_s` |
| 注意力后端 | `records/logs/openwam_server.log:1` | `sdpa (auto-detected)` |
| 实测耗时 | `records/acceptance/composite_state_restore_12_20261028_runs/ep-20261028-100255-e4e68d.json` | 510.7 ms / 15.3 ms / 3.82 Hz / RTF 0.191 |
| π0.5 基准 | `/home/host/flashrt-store/bench_pi05/{flashrt,flashrt_fp16,openpi}.json` | 23.66 / 36.24 / 52.36 ms |
