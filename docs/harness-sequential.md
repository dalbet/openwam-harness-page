# 阶段 A/B：顺序 Harness 最小可运行版本

顺序页在 `8766`。`HARNESS_LIVE=1` 才创建 LIBERO。干跑会标明自己，不会用占位帧进入确认，也不会在环境或 VLA 缺失时偷偷改成干跑。模型计划两次失败返回 `PLANNING_FAILED`，不会偷偷换成人工计划。人工计划只有用户显式选择，`plan_source=manual`。

## 现场位置

- Git 根目录是 `/root/Page_dmh`，远程是 `git@github.com:dalbet/page.git`。
- 嵌套上游 `openwam_harness/RPent` 不进入本仓库。现场改动保持为可重放补丁 `openwam_harness/baseline/runtime_patches.py`（robosuite `default_base is None` 时使用 `NullMount`）。
- 原 Page UI 在 `/root/Page_dmh/page_ui`，协议不变：`load` / `start` / `restart`，二进制双相机帧。它仍是 ManiSkill 路径。
- 顺序模式走薄适配层 `openwam_harness/sequential/adapter.py`，端口 `8766`。视频帧格式与 Page UI 相同。

## 保留的原模式

原 Harness 仍是 `rpent --robot libero --planner api`。Planner 仍是已下载的 Qwen3.5-9B，VLA 仍是 Pi0.5，分割仍是 SAM3。不接 FlashRT，不换 VLA，不训练。

`openwam_harness/baseline/start_planner.sh` 保持正在运行的服务参数：`max-model-len 32768`，`gpu-memory-utilization 0.48`。不要把它改成 65536。

## 已保存的失败

分类在 `openwam_harness/sequential/records/failure_catalog.json`。原始日志在 `openwam_harness/runs/baseline_original/`（不上传）。

| episode | 类别 | Pi0.5 `pi0_pick` |
|---|---|---|
| `libero_10_t0_s0_r0` | planner 超时 1200s | 未调用。调用了 SAM3 `segment` |
| `libero_10_t0_s0_r1` | context 错误，输入至少 24577 + 请求输出 8192 > 32768 | 未调用 |
| `libero_10_t0_s0_r2` | 同上 | 未调用 |
| `libero_10_t1_s0_r0` | 同上 | 未调用 |
| `libero_10_t1_s0_r1` | 本轮 SIGINT，transcript 未写完 | 未调用 |

没有操作失败记录，因为这几回都没有进入抓取。未启动的 `t1` 重复回合已取消。

## 预算和短记忆

配置：`openwam_harness/sequential/budget_config.json`。

- Planner 输出额度 2048。
- 上下文上限 32768，与当前 vLLM 一致。
- 每次请求前把任务、技能约束、当前失败证据、工具 schema 和短历史分开计数。
- 没有 tokenizer 时按每 2 个字符至少 1 个 token 高估。超过上限在本地抛 `BudgetError`，不发送。
- 不裁任务、技能约束和当前失败证据。不把整份 memory corpus 或 guide 放进请求。
- 初始规划一次。格式不合法时最多修复一次。仍不合法则用人工计划。
- 同一工具失败后只重试一次，再失败则 `no_progress` 退出。

## 顺序执行

人工计划：`openwam_harness/sequential/plans/two_object.json`。

- 任务 0：alphabet soup 和 tomato sauce 放入 basket。
- 任务 1：cream cheese box 和 butter 放入 basket。
- 允许的动作只有 `view_env_state`、`segment`、`pi0_pick`、`move_to`、`set_gripper`、`pause_for_confirm`。
- `pi0_doubled`、任意代码和子任务间 `reset` 会被 schema 拒绝。
- `pi0_pick` 的返回只记为 `grasp_signal`。子任务在确认前是 `unknown`。整体成功只看 LIBERO `env.terminated`（toolkit `solved()`）。
- 每个物体放下后暂停，等用户确认成功、失败或未知。确认前不进入下一个物体。
- 动作之后只取新观测，不因此再请求 LLM。
- `episode_id`、`scene_epoch`、`plan_version` 变化或取消后，旧确认被丢弃。

`HARNESS_LIVE=1` 时适配层调用现有 `LiberoToolkit`，复用 Pi0.5 和 SAM3。默认 `HARNESS_LIVE=0`，只跑契约和 UI 协议，不占第二套模型。

## 启动

Planner 已经在 `127.0.0.1:8000`。不要再开一个 vLLM。

```bash
# 原模式 Planner（仅当 8000 没在听时）
bash /root/Page_dmh/openwam_harness/baseline/start_planner.sh

# 干跑只检查协议，页面会显示干跑并拒绝执行。
HARNESS_LIVE=0 bash /root/Page_dmh/openwam_harness/sequential/start_adapter.sh

# 真实闭环。脚本若发现本项目的旧 adapter 占着 8766，会停掉那个 pid 再启动，不会复用旧进程。
# 不杀其他端口上的进程。显存不够就先停 8000 上的 vLLM，不要再开第二个。
HARNESS_LIVE=1 bash /root/Page_dmh/openwam_harness/sequential/start_adapter.sh
```

原 Page UI 仍用 `page_ui` 的 8765。顺序页是 8766。

## Windows 转发

容器里的 `SSH_CONNECTION` 是 `127.0.0.1`，公网 SSH 端口不在环境变量中。当前实例区域是 `bj-B2`。仓库里原 Page UI 记录的例子来自另一台 `bjb1`：

```powershell
ssh -N -L 8766:127.0.0.1:8766 -p 14263 root@connect.bjb1.seetacloud.com
```

请用 AutoDL 控制台里这一台 `bj-B2` 机器的 SSH 命令，只把本地转发改成上面这一行的 `-L`。原 Page UI 把 `8766` 换成 `8765`。浏览器打开 `http://127.0.0.1:8766`。

## 手动测试

1. 打开顺序页，选 `libero_10_t0` 或 `t1`，点「加载场景」。
2. 点「开始」。干跑会立刻走到第一个放下边界并显示「等待确认」。
3. 点「确认未知」「确认成功」或「确认失败」。未确认不会抓第二个物体。
4. 点「停止」结束。点「重新开始」增加 `scene_epoch` 并丢弃旧确认。
5. `HARNESS_LIVE=1` 时同样操作会走真实 env、SAM3 和 Pi0.5，子任务之间不 reset。这一步留给手动测试。自动诊断若要跑完整回合，设 `SEQUENTIAL_DIAGNOSTIC_S=180`，到时保存现场并退出。本轮没有启动这个诊断。

## 日志

- 失败分类：`openwam_harness/sequential/records/failure_catalog.json`
- 原回合目录：`openwam_harness/runs/baseline_original/libero_10_t*_s0_r*/`
- 原回合包装日志：`openwam_harness/tmp/baseline_run.log`
- Planner 服务日志：`openwam_harness/tmp/planner/vllm.log`
- 短请求记录：`openwam_harness/sequential/records/planner_check.json`
- 真实顺序回合会写到 `openwam_harness/runs/sequential/<episode_id>/`，该目录不上传。

## 回退

```bash
git checkout feat/official-show-harness-sim
```

该分支停在 `fc9d499`。原模式命令仍是：

```bash
cd /root/Page_dmh/openwam_harness/RPent
/root/Page_dmh/openwam_harness/envs/rpent/bin/rpent \
  --robot libero --suite libero_10 --task 0 --seed 0 \
  --libero-type standard --planner api \
  --model openai:Qwen3.5-9B --base-url http://127.0.0.1:8000/v1
```

## 本轮验证

- CPU：`sequential/test_contract.py` 8 项，`baseline/test_episode_log.py` 与 `test_runtime_patches.py` 7 项通过。
- UI：浏览器工具无法打开容器内的 `127.0.0.1:8766`（页面停在 `chrome-error`）。同一服务上的 WebSocket 客户端完成了加载、开始、`soup_place` 暂停、`sauce_place` 暂停和停止，结果 `UI_PROTOCOL_OK`。
- Pi0.5：复用已有检查，`libero_10_t0_s0_r0/vla_server.log` 有 `Loaded norm stats` 和 `model ready in 32.6s`。本轮不再并发加载第二套 VLA。
- Planner 短请求：本地 tokenizer 预检输入 265 token，加上输出额度 2048 后 projected 2313，低于 32768。服务返回两次用量：第一次 prompt 327 / completion 266；格式不合法后的唯一一次修复 prompt 627 / completion 338。两次都没通过 schema，执行器改用 12 步人工计划。记录在 `records/planner_check.json`。没有第三次请求。
