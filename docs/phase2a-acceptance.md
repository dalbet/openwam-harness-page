# 第二阶段 A 验收

标记：**PHASE2A_COMPLETE**

运行时提交 `a9c559e`。本阶段只改顺序系统的原生目标收尾、释放后验证和状态收口，不改 Planner、Pi0.5、SAM3、LIBERO 资产、页面布局、按钮、双相机或成功阈值。三次真实回归在同一份工作区代码上完成（运行期间尚未提交，提交内容与运行文件一致）；停止、重置、任务切换和单元回归通过。未推送远程。

## 根因

LIBERO 的 `On` 只做几何/接触判定，不要求松手：

- `openwam_harness/envs/rpent/lib/python3.10/site-packages/liberopro/liberopro/bddl_files/libero_goal/put_the_bowl_on_the_stove.bddl` 的目标是 `(On akita_black_bowl_1 flat_stove_1_cook_region)`。
- `.../liberopro/.../envs/object_states/base_object_states.py` 的 `ObjectState.check_ontop`（78-94）和 `SiteObjectState.check_ontop`（178-198）只看高度、接触和 xy。
- `.../envs/objects/site_object.py` 的 `SiteObject.under`（63-76）是炉灶区域的几何网格。
- `.../envs/problems/libero_tabletop_manipulation.py` 的 `_check_success`（135-143）直接使用上述状态。

因此原生 `terminated` 可以在夹爪仍握着碗时成立。修复前，拒绝继续的第一层不是 env server，而是顺序系统的 episode 包装：

- `episode_service.py:51-90` 的 `classify_step` 把自由模式的原生 `terminated` 原样发布为终止；`_publish`（498）把它写给客户端。
- RPent 的 `env_client.py:49-65` 用 `self.terminated |= term` 锁存，后续 `step` 断言失败；env server 本身没有锁存（`env_server.py:67-68`，`auto_reset=False`、`ignore_terminations=False`），仍可继续 step。
- `live.py:166-174` 的 `_ended_reason` 和 `executor.py:928-954` 的 `native_execution_gate` 因此返回不可继续，release/retreat 不执行。
- `free_task.py:147-235` 的 `judge_user_goal` 只按几何算 `held_at_end`；`finalize.py` 直接采信它，没有释放证据，于是抓着碗的 `On` 也被写成 `user_goal_status=met`。finalize 本身只在终态运行一次，提前的是“原生即终态”这个判断，不是 finalize 的调用时机。

## 修复

- `episode_service.py:51-90`：自由模式的原生目标发布为 `native_goal=true, terminated=false, episode_continuable=true` 事件；benchmark 的终止语义和 `truncated` 不变。
- `executor.py:928-954`、`961-978`、`1556-1564`：原生成立后只允许当前 transfer 剩余的 `descending`/`releasing`/`retreat` 继续，预算取同一子任务剩余阶段的既有预算（本任务为 place 30s + retreat 30s，共 60s），记录 `native_finish_scope/budget_s/start_sim_action`；其他步骤和其他事务不运行，stop/reset/取消/真实终止立即生效。
- `executor.py:1513-1553`：终态显式给出 `release{command_executed, verified, observation_id, sim_action_count, gripper_opening_m}`、`retreat_complete` 和 `native_finish_window`。
- `finalize.py:91-148`、`154-183`、`186-253`：用户目标要求 `PLACED` 收据带 `released_required`、夹爪开度不低于 `gripper_open_min_m=0.05`、释放后新观测带 `sim_action_count`、最终观测计数不小于释放后观测、观测 ID 不同、最终夹爪仍打开且关系成立。抓着碗的 `On` 返回 `unknown`，关系不成立返回 `unmet`。
- `adapter.py:42-80`：自由模式只有在原生成功、用户目标 `met` 且 release `verified` 时才算 `completed`；原生成立但缺少释放后验证是 `incomplete`。用户目标 met 且退离失败时保留 met 和 `retreat_error`，`finalization=user_goal_met_retreat_error`，执行不记为完成。

没有用“翻转 terminated 字段”了事：那会丢掉原生信号，允许无界继续执行，仍会把抓着碗的几何当成放置完成，也会改变 benchmark 和 truncation 的语义。当前做法保留原生证据、限定收尾范围和预算，并把用户目标与释放证据分开。

### done / success / terminated / can_continue

这条链没有单独使用 gym `done`；以 env server 的 `terminated`/`truncated`、`episode_service` 发布值和 RPent 客户端锁存为准：

| 层 | 原生 `On` 时的行为 |
| --- | --- |
| LIBERO 成功检查 | `terminated=true`；不检查夹爪 |
| env server | 返回 `term=true`，自身不锁存，仍可继续 step |
| episode_service（自由） | 发布 `native_goal` 事件，`terminated=false`、`continuable=true`；benchmark 保持 `terminated=true`、`continuable=false`；`truncated` 始终终止 |
| RPent 客户端 | 只对发布后的 `term/trunc` 锁存；“不可继续”由包装层决定 |
| 顺序执行 | 自由模式进入当前 transfer 的有限收尾；benchmark 按原生成功结束 |
| finalize | 原生成功、用户目标 met、release verified 三者齐全才 completed |

## 真实回归

任务 `libero_goal` task 1，`put the bowl on the stove`，加载 `task_id=1, seed=0`，与 phase1 相同。三次独立回合，每回合先 reset，无人工确认，无自动重跑，不挑回合。计划均为 8 步，`recovery_events` 为空，`first_error`/`verification_error` 为空。

| 次 | episode / run / epoch | 总耗时 | 首次原生 On | 原生时最后完成步骤 | 释放与退离 | 释放后观测 | 最终状态 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `29060dc1a9a2` / 1 / 1 | 99.34s | +86.14s，sim 60 | `descend_bowl_transfer`，`TRANSPORTED`，relative delta 0.0013（仍夹持） | `set_gripper` 执行，原生后 8+4+4 步；`PLACED`；`RETREATED` 后退离 6 步 | `query-f6b72d78860a402e99f90e574fb210a0:2:76:19` @76，开度 0.07763；最终 `:2:82:22` @82，0.07823 | native true / met / completed / retreat null |
| 2 | `542a2d17c341` / 2 / 2 | 90.95s | +77.70s，sim 62 | `descend_bowl_transfer`，`TRANSPORTED`，delta 0.0013 | 8+4+4 步；`PLACED`；`RETREATED` 后退离 6 步 | `...:3:78:42` @78，0.07763；最终 `:3:84:45` @84，0.07823 | 同上 |
| 3 | `fc74a2a5aeaf` / 3 / 3 | 97.45s | +84.20s，sim 60 | `descend_bowl_transfer`，`TRANSPORTED`，delta 0.0035 | 8+4+4 步；`PLACED`；`RETREATED` 后退离 6 步 | `...:4:76:67` @76，0.07763；最终 `:4:82:70` @82，0.07823 | 同上 |

补充证据：

- 首次原生 On 的 `sim_action_count` 取原生事件前最后一次 `descend_bowl_transfer` 观测的编号计数（60/62/60），释放后观测编号从 68/70/68 开始；不是日志行号或帧号。
- 原生事件为 `NATIVE_GOAL_FINISH`，`native_finish_scope=bowl_transfer`，`native_finish_budget_s=60`；原生到 `RETREATED` 分别 13.05s/13.09s/13.10s，均在预算内。
- 释放后最终关系均为 `inside_xy=true`、`supported=true`，`delta_site` 分别为 `[-0.0014,-0.0004,0.0239]`、`[-0.0005,-0.0016,0.0239]`、`[0.0006,-0.0019,0.0239]`；释放证据 `status=verified`，最终开度 0.07823，均高于 0.05。
- 三次 `sim_action_count` 从 0 到 82/84/82；每次 reset 后回到 0，episode 和 `scene_epoch` 递增（2/3/4），没有旧回合完成信号进入新回合。

## 停止与重置

执行中停止（任务 1，pick 阶段）：

- 停止时 `sim_action_count=1`，终态 `execution_status=stopped`，`native_task_success=false`，`user_goal_status=unmet`，release 证据 `missing/release_not_executed`，`subgoal_results` 中 `On=false`，最终夹爪开度 0.01288。
- 没有 `PLACED` 或 `RETREATED` 事件。停止后 in-flight 的 pick 动作块把计数从 1 推到 11，3 秒后仍是 11，没有继续释放或退离。
- reset 后新 episode `4a082ad943b0`，`scene_epoch=5`，`sim_action_count=0`，idle；旧完成字段为空。

三次回合间 reset：

| reset | 新 episode | epoch | sim | 阶段 | 旧完成字段 |
| --- | --- | --- | --- | --- | --- |
| 1 后 | `542a2d17c341` | 2 | 0 | idle | 空 |
| 2 后 | `fc74a2a5aeaf` | 3 | 0 | idle | 空 |
| 3 后 | `ea66f5e08f44` | 4 | 0 | idle | 空 |

任务切换：加载 task 8（`put the bowl on the plate`）后 episode `8395a0b22492`，epoch 6，`sim_action_count=0`，阶段 idle，双相机 512×512，首帧 `fid=0`。探针最初抓到的是上一个 reset 的缓存帧（eid `4a082ad943b0`），另开 WebSocket 连接复核后确认新 episode 的帧正确；这是探针缓存，不是页面或 env 的状态错误。

## 页面

只做协议核对：`GET /` 返回 200，包含套件、加载、任务、边缘/云端、开始、停止、重置和双相机；套件列表是四个标准 LIBERO 套件；帧仍是二进制双相机消息，agent/wrist 均 512×512。页面布局、按钮、双相机和主要交互未改，`ended`/`state` 消息只追加字段、没有删除页面读取的字段。本机没有做真实浏览器点击验收，因此不声称页面级验收。

## 测试

`python -m unittest discover -s openwam_harness/sequential -p 'test_*.py'`：153 项通过，1 项跳过。跳过的是依赖历史 journal 锚点、而本机 journal 已轮转的用例。新增 `test_phase2a_contract.py` 8 项，覆盖本轮修过的组合：

1. 原生 `On` 成立但仍夹持时不判用户放置完成（`unknown`）；
2. 释放后关系成立才判 `met`；
3. 释放后关系不成立判 `unmet`；
4. 缺少释放后观测计数判 `unknown`；
5. env 不可继续时不执行任何动作；
6. 用户目标 met 且退离失败时同时保留两者；
7. 停止发生在释放前时不执行 release/retreat；
8. reset 后旧原生结果不进入新回合。

同时更新了自由模式原生事件/可继续、`completion_phase` 需要 release verified、释放门禁等既有测试。

## 已知限制与下一阶段入口

- 成功结束消息的内联 `ended` 仍不带 `release`、`retreat_complete`、`native_finish_window`；这些证据在 `subgoal_results[*].release_evidence`、事件流和执行结果里，失败/取消路径已带字段。下一步可把成功路径对齐 `_ended_from` 后再复验，不应在复验前改变判据。
- 只有 task 1 做了三次完整回归；其他任务尚未做同类三回合验证。
- 自由模式的有限收尾是本顺序系统契约；benchmark 和真实 `truncated` 仍按终止处理。
- 收尾预算沿用现有 place/retreat 预算，任务 1 为 60s；没有为单次运行放宽阈值。
- adapter `/health` 的 `commit` 是进程启动时的 HEAD，运行中提交后 `disk_head` 会变化，不能把 `running_commit_matches_disk_head=false` 单独当成代码不同的证据。
