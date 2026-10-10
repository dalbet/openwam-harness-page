# 英文邮件草稿：索取 BlockStack-v1 场景资产与安装说明

背景：Show-Harness 官方 `core/sim/maniskill_scenes.py` 通过 `RLINF_ROOT` 导入
`real_franka.real2sim_env`。BlockPAP-v1 已在 `AaronCaoZJ/RLinf` fork（main @ fd52554）
找到并成功复现；BlockStack-v1 的 `panda_v2_extended.urdf` 在所有公开位置均不存在
（检索证据见本文件末尾），故无法接入，需向作者索取。

收件人建议：Zhijun Cao <aaroncaozj@gmail.com>（`maniskill_scenes.py` 与 fork 作者），
抄送 Show-Harness 通讯作者 Prof. Mike Zheng Shou。

---

Subject: Show-Harness sim scenes: request for BlockStack-v1 robot asset (panda_v2_extended.urdf) and release notes

Dear Zhijun,

Thank you for open-sourcing Show-Harness — the harness, the released LoRA checkpoints and
the sim/real datasets have been extremely useful.

I am reproducing the released `qwen3_5_2b_showharness_sim` adapter in ManiSkill, following
`docs/simulators.md` and `configs/robot_maniskill.yaml`. I located the real2sim scene
modules in your RLinf fork (`AaronCaoZJ/RLinf`, branch `main`, commit `fd52554`):

- `real_franka/real2sim_env/pick_and_place.py` → `BlockPAP-v1` works end-to-end here:
  3/3 successful closed-loop episodes with the official runner
  (`--traj-id random --layout wide`, `table_tex 006`, seeds 0-2), matching the
  `blockpap_follow_wood` training shard. Thank you!
- `real_franka/real2sim_env/block_stack.py` → `BlockStack-v1` cannot be built: its
  `PandaExtendedGripper` agent loads `panda_v2_extended.urdf` from ManiSkill's
  `PACKAGE_ASSET_DIR/robots/panda/`, and that URDF does not exist in any public location
  I could check (see list below). Building the env fails with
  "Robot definition file not found: .../robots/panda/panda_v2_extended.urdf".

Could you please share:

1. **`panda_v2_extended.urdf`** (and its `.srdf` / any referenced meshes). The docstring in
   `block_stack.py` describes the modification (tcp joint +20 mm; extra 20 mm finger
   collision boxes), but I would rather run the exact official asset than re-derive it.
2. **The canonical source you recommend for `RLINF_ROOT`** when running the Show-Harness
   ManiSkill scenes: is your fork the intended public home of `real_franka.real2sim_env`,
   or is there a branch/PR against `RLinf/RLinf` I should track? Any plan to upstream it
   would help pinning a stable revision.
3. **Any other non-public assets** the two scenes need beyond the above (e.g. reference
   trajectory data under `real_franka/data_inspector/`, or the wood table textures — I
   currently use `RLinf/maniskill_assets` `carrot/more_table/textures/001-021.png`,
   ODC-BY, which renders correctly).
4. If available, the **installation/versions note** you run these scenes with
   (mani_skill / sapien versions), so the reproduction stays faithful.

For reference, my environment: ManiSkill 3.0.1, physx_cpu, RTX 5090, the released
`qwen3_5_2b_showharness_sim` adapter served with the v3 prompt contract.

Public locations already checked (all negative for `panda_v2_extended.urdf`):

- `RLinf/RLinf` main + all remote branches (`dev/dynarl`, `dev/geniesim`,
  `dev/physical-agent`, `release/v0.1`-`v0.4`)
- `AaronCaoZJ/RLinf` branches `main` and `sync/upstream-20260820` (full trees)
- `mani-skill/ManiSkill` upstream (full tree) and the `mani_skill` 3.0.1 wheel assets
- Hugging Face: `RLinf/maniskill_assets`, `RLinf/RLCo-maniskill-assets`,
  `haosulab/ManiSkill`, `ceilingFan456/ManiSkill` fork

Happy to send a PR upstreaming the scene modules or the asset packaging if that helps.

Best regards,
Qiming Huang
Show Lab, National University of Singapore

---

## 检索证据存档（中文）

| 位置 | 方法 | 结果 |
| --- | --- | --- |
| `RLinf/RLinf` main + 7 个远程分支 | `git ls-tree -r` 全树 grep | 无 `real_franka`、无 `panda_v2_extended` |
| RLinf PR/issue | GitHub search API `real_franka`、`real2sim in:title` | 0 命中 |
| `AaronCaoZJ/RLinf` main / sync 分支 | GitHub git trees API 全量 | 有 `real2sim_env/*.py`；无任何 URDF |
| `mani-skill/ManiSkill` main | GitHub git trees API 全量 | 无 extended URDF |
| `mani_skill` 3.0.1 本地包 | 文件系统 + `utils/assets/data.py` 注册表 | 无 |
| HF `RLinf/maniskill_assets` | tree API | 有木纹纹理（已用），无 URDF |
| HF `RLinf/RLCo-maniskill-assets` | API | 仅 `bridge_table_long.glb` |
| HF `haosulab/ManiSkill*` | tree API | 仅演示数据 |
| `ceilingFan456/ManiSkill` fork | git trees API | 无 |
