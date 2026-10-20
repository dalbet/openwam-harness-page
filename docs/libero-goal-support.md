# LIBERO Goal 十任务支持

数据来自本机 `fixtures/libero_task_catalog.json`、`libero_goal/*.bddl`、资产 XML，以及 `skills.py`。CPU 测试通过不等于机器人任务成功。没有真机或仿真人工验收的条目只标“已实现，待人工验收”。

场景都是 `libero_goal`。官方成功仍是环境 `terminated` 且未 `truncated`。几何检查只作诊断，不代替这个标志。

## 优先任务

| 任务 | 英文 | 中文 | 技能 | 对象 / 区域 | 官方谓词 | 代码 | 验收 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | put the bowl on the stove | 把碗放到炉灶上。 | segment、pi0_pick、move_to、set_gripper | `akita_black_bowl_1` → `flat_stove_1_cook_region` | `(On akita_black_bowl_1 flat_stove_1_cook_region)` | 已实现 | 人工曾成功，不稳定。本轮改了状态契约和落点后尚未再验收 |
| 6 | put the cream cheese in the bowl | 把奶油奶酪放进碗里。 | 同上 | `cream_cheese_1` → `akita_black_bowl_1` | BDDL 目标是 `(On cream_cheese_1 akita_black_bowl_1)`，英文原文是 Put the cream cheese on the bowl | 已实现 | 人工两次都失败于“使用旧观测”。尚未验收 |
| 8 | put the bowl on the plate | 把碗放到盘子上。 | 同上 | `akita_black_bowl_1` → `plate_1` | `(On akita_black_bowl_1 plate_1)` | 已实现 | 人工两次都失败于“使用旧观测”。当时加载的仍是 task 1。尚未验收 |
| 2 | put the wine bottle on top of the cabinet | 把酒瓶放到柜子顶上。 | 同上 | `wine_bottle_1` → `wooden_cabinet_1_top_side` | `(On wine_bottle_1 wooden_cabinet_1_top_side)` | 已实现，待人工验收 | 未真实测试 |
| 4 | put the bowl on top of the cabinet | 把碗放到柜子顶上。 | 同上 | `akita_black_bowl_1` → `wooden_cabinet_1_top_side` | `(On akita_black_bowl_1 wooden_cabinet_1_top_side)` | 已实现，待人工验收 | 未真实测试 |

完成检查：

- 炉灶：`cook_region` site，半尺寸 0.075 × 0.075 × 0.0025，`SiteObject.under`。下降高度是 site 顶面减去碗碰撞底（底 z=-0.012），不是 body 中心加一个 `z_offset`。接近高度仍是区域参考加 0.12。
- 盘子：中心碰撞顶面约 body z+0.00535，水平限制用官方 `check_ontop` 的 0.03。`top_site` z=0.04 不是盘面。
- 碗：内部半径 0.025，碗沿相对 z=0.04，碗底碰撞 z=-0.012。奶酪底要落在这个腔内。xy=0.08 的“靠近”不算在碗里。接触力没有读到时记 `not_provided`，不编造。
- 柜顶：`top_side` 半尺寸 0.12534 × 0.09438 × 0.00147，朝向为单位四元数。碗和酒瓶用各自碰撞底，不共用一个高度。抽屉的 `top_region` / `middle_region` / `bottom_region` 带旋转，不是柜顶，也不当固定区域缓存。只有官方柜顶句子会选定 `top_side`；泛泛说“放到柜子上”仍然是区域含糊。

## 其余任务

### task 0：open the middle drawer of the cabinet

- 中文：打开柜子的中间抽屉。
- 官方谓词：`(Open wooden_cabinet_1_middle_region)`。
- 关节：`WoodenCabinet.is_open` 要求该抽屉 `qpos < -0.14`。打开区间 `[-0.16, -0.14]`，关闭区间 `[0.0, 0.005]`。
- 代码：未实现。`open_drawer` 不在 Planner 技能表。
- 下一批：读 `wooden_cabinet_1` 中间抽屉关节；参数是关节名、目标 qpos 和步数预算；完成条件是 `qpos < -0.14`；失败退出和取消不得再 `env.step`；通过后再加入技能表。

### task 5：push the plate to the front of the stove

- 中文：把盘子推到炉灶前方。
- 官方谓词：`(On plate_1 main_table_stove_front_region)`。这是桌面 range，不是炉灶 site。
- 代码：未实现。`push` 不在技能表。`move_to` 是夹持搬运，不是推动。
- 下一批：定义接触方向、终点 range、推进进度和动作预算。完成前不得把 `pi0_doubled` 放进技能表。

### task 7：turn on the stove

- 中文：打开炉灶。
- 官方谓词：`(Turnon flat_stove_1)`。`FlatStove.turn_on` 为旋钮 `qpos >= 0.5`，开启区间 `[0.5, 2.1]`。
- 代码：未实现。`turn_on_stove` 不在技能表。
- 下一批：读旋钮关节；完成条件用官方 `turn_on`；必须有步数预算、失败退出和取消。`pi0_doubled` 目前只是 RPent 工具，没有这套契约。

### task 3：open the top drawer and put the bowl inside

- 中文：打开上层抽屉，并把碗放进去。
- 官方谓词：`(In akita_black_bowl_1 wooden_cabinet_1_top_region)`。
- 依赖：先完成 task 0 的开抽屉契约。`top_region` 的四元数不是单位四元数，平移 `move_to` 不能放进腔体。还要有插入腔体的完成条件，不能复用柜顶 `top_side`。

### task 9：put the wine bottle on the rack

- 中文：把酒瓶放到架子上。
- 官方谓词：`(On wine_bottle_1 wine_rack_1_top_region)`。
- 依赖：酒架 `top_region` 带旋转。`move_pose` 能同时伺服位置和 pitch/yaw，但没有顺序执行的完成检查、失败退出和取消接线，不能当作已适配技能。瓶子碰撞从 z≈0.001 延伸到约 0.158，原点靠近瓶底。接近路径和姿态都未验收。

## 未开放的原语

| 原语 | 位置 | 现状 |
| --- | --- | --- |
| `pi0_pick` | 顺序执行已包装 | 抓取技能。checkpoint、归一化、采样、chunk、控制频率本轮未改 |
| `move_to` | 顺序执行已包装 | 位置伺服，保持当前姿态 |
| `pi0_doubled` | `RPent/robots/libero/tools.py` | 接触类 Pi0.5 循环，最多 20 个 chunk。返回值只镜像官方终止。没有独立完成条件、取消和失败退出，不进入 Planner 技能表 |
| `move_pose` | 同上 | 同时伺服 xyz、pitch、yaw。默认 `max_steps=150`。没有顺序执行的完成检查和取消接线 |
| `rotate_wrist` / `rotate_pitch` | 同上，未包装 | 不能代替开抽屉或拧旋钮 |

新技能进入 Planner 表之前，必须同时有参数契约、动作预算、完成条件、失败退出和取消处理。本轮没有实现这三项接触技能。
