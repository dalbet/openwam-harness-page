# 第一阶段验收

标记：**PHASE1_COMPLETE**

> PHASE2A 修订：本文三次运行只证明 LIBERO 原生 `On` 可达。当时 `execution_status=failed`、release/retreat 未执行，`user_goal_status=met` 只是按几何关系判定，不能当作放置完成。修复和复验见 `docs/phase2a-acceptance.md`。

清理后的服务已重新启动，Pi0.5、SAM3 和 LIBERO 在加载场景后实际拉起，代表任务连续 3 次达到 LIBERO 原生目标。生命周期、双相机和 145 项单元测试通过。计划里的 release / retreat 在原生目标成立后没有执行，这是清理前基线里已有的行为，记在文末，留给第二阶段。

## 版本与链路

| 位置 | 提交 |
| --- | --- |
| 本仓库 `main` | `885e64a`，比 `origin/main` 的 `df7e083` 超前 1 个提交 |
| `openwam_harness/RPent` | `a8c5311`，detached，editable，`import rpent` 指向这份源码 |
| `openwam_harness/OpenWAM` | `4d7f642`，detached，源码保留，本阶段执行链不调用 |

链路：顺序页 `http://127.0.0.1:8766` → `sequential/adapter.py` → Planner `127.0.0.1:8000` 的 Qwen3.5-9B → 本地顺序执行 → `live.py` → RPent 的 Pi0.5、SAM3、LIBERO。规划输入是场景元数据，`vision=false`。页面结构未改。远程浏览器访问不到只监听 `127.0.0.1` 的服务，控件用本机 `GET /` 核对，双相机用 WebSocket 二进制帧核对。

启动、状态、停止：

```bash
bash /home/host/page-dmh/scripts/start_planner.sh
bash /home/host/page-dmh/scripts/start_adapter.sh
bash /home/host/page-dmh/scripts/status.sh
bash /home/host/page-dmh/scripts/stop.sh
```

## 清理前基线

任务 `libero_goal` task 1，`put the bowl on the stove`。无人工确认。

| 项 | 结果 |
| --- | --- |
| episode / scene_epoch / run_id | `dc319597bf67` / 2 / 2 |
| 耗时 | 84.7s |
| reset 前后 `sim_action_count` | 65 → 0，结束后 64 |
| 技能 | 碗定位 `sim_state`，炉灶定位 `mujoco_site`，`GRASP_HELD`，`TRANSPORTED`，`NATIVE_SUCCESS`（`libero_terminated`） |
| LIBERO 原生目标 | `native_task_success=true` |
| 用户目标 | `user_goal_status=met` |
| 执行状态 | `execution_status=failed`，`first_error=episode_not_continuable`，`retreat_error=null` |
| 退离 | release / retreat 未执行 |

## 清理

保留：Qwen3.5-9B（19G，revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`）、Pi0.5（7.0G，revision `6222623f635769bfc73c9472e29fab9b7fd8e027`，不含 `optimizer.pt`）、SAM3（3.3G，`facebook/sam3` 的 `sam3.pt`）、`envs/vllm`、`envs/rpent`、OpenWAM 源码、LIBERO 资产。

已删除且确认无当前依赖：`page_ui/`、`scripts/page/`、未完成的空虚拟环境、`evdev-build`（约 16MB）、`openwam_harness/tmp` 里的旧下载和日志。checkpoints 里没有另一套可删的大权重。实际腾出的空间主要是这约 16MB，不是模型目录。

证据不足所以保留：OpenWAM 源码及其未初始化的 `third_party/cosmos-predict2.5`。SAM3 下载脚本没有钉 revision。

## 清理后回归

服务由新入口拉起。Planner pid `1586039`，权重已加载。adapter pid `1587463`。加载场景后 `vla_server`、`sam3_server`、`env_server` 均为 alive。三次运行之间都先 reset，`sim_action_count` 回到 0，episode 彼此不同。无人工确认。

| 次 | episode | epoch / run | sim 0 → | 原生 On | 用户目标 | 执行状态 | 最后完成步骤 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `d5b8bae4c0c2` | 2 / 2 | 58 | `inside_xy`、`supported`，delta 约 `[-0.0004, -0.0001, 0.0289]` | met | failed，`episode_not_continuable` | `descend_bowl_transfer` |
| 2 | `c425f9ee29e1` | 3 / 3 | 60 | 同上，delta 约 `[-0.0009, -0.0012, 0.0275]` | met | 同上 | 同上 |
| 3 | `2b1b9c6deb42` | 4 / 4 | 59 | 同上，delta 约 `[-0.0005, -0.0015, 0.0275]` | met | 同上 | 同上 |

三次都有 `GRASP_HELD`（`sim_state`）和 `NATIVE_SUCCESS`（`libero_terminated`）。`held_at_end=true`。`retreat_error=null`。release / retreat 没有跑。`sim_action_count` 来自 env server，不是日志行号。

场景切换：加载 `put the bowl on the plate`，task 8，episode `343fba894236`，epoch 5，`sim_action_count=0`。这次只验证切换，没有把该任务当成功样本。

执行中停止：

- 计划刚接受、第一个工具之前停止：episode `b7192a71f0d7`，`execution_status=stopped`，`sim_action_count` 保持 0。reset 后新 episode `d074d8f64f0e`，epoch 7，阶段 idle，计步 0。
- 抓取之后停止：episode `2025c38519bb`，停止时 `sim_action_count=10`，2 秒后仍是 10。终态 `stopped`，原生目标 false，用户目标 unmet。reset 后 episode `0fb8804c04b1`，epoch 9，idle，原生字段为空，计步 0。旧回合的 ended 没有写进新回合。

双相机：一帧同时包含两张 JPEG，都是 512×512，文件头 `ffd8ff`，agent 33792 字节，wrist 36706 字节。页面 `GET /` 仍包含 suite、load、task、planner-profile、start、stop、reset、agent、wrist。

单元测试：`python -m unittest discover -s openwam_harness/sequential -p 'test_*.py'`，145 项通过，1 项跳过。跳过的是依赖历史 journal 锚点、而本机 journal 已轮转的用例。

## 第二阶段断点

自由模式在 LIBERO `On` 成立后，环境不再 `continuable`。`native_execution_gate` 以 `episode_not_continuable` 结束，`execution_status` 仍是 `failed`，夹爪仍握着碗，release 和 retreat 不会执行。下一阶段先处理这个收尾，而不是更换 Planner、Pi0.5 或 SAM3。

PHASE2A 已处理该断点：自由模式的原生目标改为事件，只允许当前 transfer 在既有预算内完成释放和退离，用户目标要求释放后证据。复验见 `docs/phase2a-acceptance.md`。
