# “使用旧观测”根因

当前启动和运行入口见 `docs/harness-sequential.md`。本文只保留当时的根因记录。

这份报告只使用磁盘上的代码和已保存记录。它不把磁盘 `HEAD` 当成当时正在跑的进程。

## 版本

| 来源 | 能确定的值 | 不能当成 |
| --- | --- | --- |
| 写这份报告时的磁盘 `HEAD` | `01a8df6fbad4f5e3c3274aed7335e61de61e43bc`，分支 `feat/harness-sequential-demo` | 正在服务的进程版本 |
| 上次保存的 `/health` | `.checkpoints/20261020-demo-select/health.txt`：`pid 3158`，`commit 04191e5ed048fcb9f0608fa62e846ba7a66879af`，`static_version view12` | 磁盘 `HEAD` |
| 本机当前监听 | 写报告时没有适配器或 env server 在监听 | 不能把“现在没进程”写成“当时没跑过 04191e5” |
| 适配器启动版本 | `adapter.py` 在 import 时执行 `git rev-parse HEAD`，记入 `STARTUP_COMMIT`，`/health` 的 `commit` 用这个值 | 请求 `/health` 时再读一次磁盘并覆盖 |
| 前端资源 | 该次 health 写明 `static_version=view12`。当前源码 `STATIC_VERSION` 已是 `view14`，所以 view12 是旧进程的静态资源标记 | 当前工作区的 `view14` |
| env server | 保存的 health 没有 env server 的 commit、pid 或启动参数。`start_adapter.sh` 在 `HARNESS_LIVE=1` 时由适配器拉起 LIBERO，env 的代码版本没有单独记进那次 health | 用适配器 commit 代替 env server commit |

`01a8df6` 已经推送。checkpoint 说明曾经有进程以 `04191e5` 启动。两次失败回合的记录落在该进程仍可能存活的时间段，但记录文件本身没有写入当时的 `commit`。因此下面的失败链可以确定到 request、run、episode 和观测字段，不能再额外声称“这条记录的字节一定来自 04191e5 的某个行号”。判断逻辑以当前源码 `verify.py` 的 `_epoch_mismatch` 为准，因为它仍会把下面的字段判成 `stale_observation`。

## 代码上的比较

`verify._transport` 在对象位姿之前调用 `_epoch_mismatch`：

- 条件：`before.scene_epoch != after.scene_epoch`，或者两边的 `env_step` 都是 int 且 `after.env_step < before.env_step`。
- 失败文案固定为「搬运失败：使用了旧观测」，`reason_code=stale_observation`。
- `missing` 被写成 `["current_observation"]`，三个证据来源都被写成 `stale_observation`。

这和字段是否真的缺失无关。`_gap` 会把 `missing` 里的名字再写成「current_observation 缺失」。

两个写入方把不同的数放进同一个 `env_step`：

| 观测 | 代码 | `observation_id` | 写入 `env_step` 的数 | `step_clock` |
| --- | --- | --- | --- | --- |
| 抓取后的 hold | `executor._observations_from_samples` | `hold-<env_action_count>` 或 `env-actions-<before>:<after>` | `sample["env_action_count"]`，来自适配器进程里的 `_env_action_count` | `env_action_count` |
| 动作后的 `observe()` | `live.LiveBackend.observe` | `live-<record.step_idx>-<time_ns>` | `toolkit.state.latest_record().step_idx`，注释写明这是工具记录序号 | `tool_record` |

`_judge` 把上一步的 `last_obs` 当作 `before`。抓取稳定窗口的最后一条就是 `hold-*`。下一步 `move_to` 的 `outcome.raw` 没有 `samples` 时，`post = [observe()]`，`after` 就是 `live-*`。

`observe()` 不读仿真动作计数。`move_to` 的诊断把缺失的 `raw["env_action_count"]` 用 `or 0` 写成 0。这个 0 不是测到的仿真步数。

## 已对齐的失败回合

关联键是记录里的 id，不是时间接近。

`run_journal.jsonl` 用同一个 `run_id` + `client_request_id` 写到 `planner_response.detail = seq-...`。`records/runs/<episode>-<run_id>.json` 的 `execution.request_id` 等于这个 `seq-...`，并带上 `episode_id`、`scene_epoch`、`primitive`、`before_id`、`after_id`。

没有保存 env server 的 RPC 报文。动作侧只能确定到运行记录里的 `primitive=move_to`、`return_summary.executed=true`、`error=null`。不能编造 RPC id。

| client_request_id | planner request_id | run_id | episode_id | scene_epoch | 用户句 | 步骤 | before_id | after_id | before_epoch | after_epoch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `req-1792474268057-c7a708` | `seq-a46e4c6971c0` | 2 | `cc86e2a1c136` | 2 | 把碗放在炉灶上 | `move_bowl_transfer` / `move_to` | `env-actions-155:160` | `live-7-1792474336803610363` | 2 | 2 |
| `req-1792474323` 之后的 task 1 另两次 | `seq-a416bd8adfb9`、`seq-0532d0232a9f` | 4、6 | `eab269c5f9e1`、`72aa624ea3cc` | 3、6 | 把碗放到/放在炉灶上 | `move_bowl_transfer` | `env-actions-270:275`、`env-actions-389:394` | `live-12-...`、`live-22-...` | 与 run 的 epoch 相同 | 相同 |
| `req-1792484025646-193f12` | `seq-da3f2488b559` | 9 | `4e7feda11a72` | 9 | 把碗放在盘子上 | `move_bowl_to_plate` | `hold-577` | `live-35-1792484094658090361` | 9 | 9 |
| `req-1792484151709-1d724e` | `seq-827e720d6554` | 10 | `0127f5ae3ff1` | 10 | 把奶油奶酪放到碗里 | `move_step1` | `hold-636` | `live-38-1792484223025563721` | 10 | 10 |

task 6、task 8 各有一条完整 id 链，对应人工两次里的失败记录。task 1 至少三条 `stale_observation`，也有用户目标完成的记录（例如 `6f0d2539538c-4`、`25239e56fe01-5`、`c8c6e5670311-7`）。成功记录没有 `before_id`，不能反推它们当时的 `env_step`。

journal 的 `accepted.detail` 在 task 6/8 这两次仍是「把碗放到炉灶上。」加载场景的官方句是 task 1。用户句在 `task_check` 里。开始时没有换成 task 6 或 task 8 的场景。

## 每条过期错误实际比较了什么

以 `0127f5ae3ff1` / `seq-827e720d6554` 为例，其余 `hold-*` 或 `env-actions-*` 对 `live-*` 的记录相同。

- 失败步骤：`move_step1`，`completion_check=object_transported`，`transfer_phase=transporting`。
- 比较条件：`_epoch_mismatch`。epoch 两边都是 10，没有跨场景。接下来用 `env_step` 的整数比较。
- expected：后观测的 `env_step` 大于等于前观测的 `env_step`。
- received：前 id `hold-636` 的数字来自动作计数；后 id `live-38-...` 的 38 是 `record_index`。记录里的 `execution.record_index` 就是 38。38 < 636，于是判失败。
- 状态来源：前观测是 hold 采样写进执行器的内部观测，物理量来自 `sim_sample`（工具返回后的仿真读取）。后观测是 `observe()`，又读了一次当前仿真，但版本号用的是工具记录序号。都不是只读 RPC 返回的仿真动作版本。`named_poses` 才会走 `env.named_poses`。
- 动作开始前版本：hold 的动作计数 636，存在前观测里，字段名却是 `env_step`。
- 动作结束版本：没有写入。`execution.env_action_count` 为 0，是缺失后的默认值。
- 检查器所用版本：后观测的工具记录序号 38。
- 对象、末端、目标区域：失败发生在读位姿之前，所以这三项没有各自的测量值，被统一标成 `stale_observation`。
- `native_terminated=false`，`native_truncated=false`，`producer_alive=true`。动作生产者当时还在。记录里没有 RPC 超时。

`4e7feda11a72` 是同一条件：`hold-577` 对 `live-35`，epoch 都是 9。

task 1 的 `env-actions-155:160` 对 `live-7` 也是动作计数对工具记录序号。

## 六类里它属于哪一类

| 类 | 这些记录 |
| --- | --- |
| a. 真的使用动作前状态 | 不是。后 id 是动作之后的 `live-*`，epoch 相同 |
| b. 相同状态被重复读取 | 不是。两边 id 不同 |
| c. 缺少第二个稳定样本 | 不是。失败在搬运比较，还没进入稳定窗口 |
| d. 静态区域缓存 | 不是。site 缓存不在这条比较里 |
| e. 跨 reset / 回合 | 不是。`before_epoch == after_epoch` |
| f. RPC 超时后动作仍在执行 | 记录不支持。`executed=true`，`error=null`，没有 timeout 文本 |

实际原因是把两种时钟的整数放进 `env_step` 再比大小，然后用「使用旧观测」和「current_observation 缺失」盖住。`current_observation` 是有的，只是它的版本字段不是仿真动作计数。

另外，`14ed1ab4fe62-10` 在用户目标已经完成时，退离步骤的 `return_summary.native_terminated=true`，`root_code=TOOL_EXCEPTION`。这和「原生成功后继续 step」是另一条问题，不是上面的时钟比较。`env_client.step` 在客户端已经记下 term/trunc 时会断言 `env.step called after the episode signaled term/trunc`。自由模式的 `classify_step` 会把 terminated 清成 false，使客户端不锁存，从而还能再 step。
