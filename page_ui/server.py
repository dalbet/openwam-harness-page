#!/usr/bin/env python3
"""Page — Show-Harness ManiSkill 闭环 Web UI 服务端。

真实闭环：本地 vLLM 服务 + 真实 ManiSkill 仿真 + 真实图像进入模型 + 真实动作执行。
- 单一 worker 线程持有 env（reset/step/render 不并发）。
- 观测 -> 编码 -> 网络发送解耦；观测队列 maxsize=1。
- 每个客户端有界发送队列，慢客户端丢旧帧，不反压仿真。
- Agent/Wrist 来自同一 obs，同一 episode_id/frame_id/仿真时间戳。
- 终止帧不被普通丢帧逻辑永久丢弃。
- 仅绑定 127.0.0.1，通过 SSH 隧道访问。
"""
from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import sys
import threading
import time
import traceback
import uuid
from queue import Empty, Full, Queue
from pathlib import Path

from aiohttp import web, WSMsgType

# ---- 把 Show-Harness 加入 import 路径 ----
REPO = Path(__file__).resolve().parent.parent / "Show-Harness"
sys.path.insert(0, str(REPO))

LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "server.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("page")

# ---- 仿真/模型配置 ----
VLM_BASE_URL = os.environ.get("VLM_BASE_URL", "http://127.0.0.1:8000/v1")
VLM_MODEL = os.environ.get("VLM_MODEL", "qwen3_5_2b_showharness_sim")
VLM_API_KEY = os.environ.get("VLM_API_KEY", "EMPTY")
VLM_TIMEOUT_S = float(os.environ.get("VLM_TIMEOUT_S", "120"))
PROMPT_PATH = REPO / "prompts" / "v3" / "mvtoken_generator_lite.txt"

# 仿真渲染/编码参数
# 模型输入保持官方约定：传感器 256，prepare_view 到 256，腕部裁成 4:3 再 letterbox。
# 给人看的是同姿态的独立相机，原生 512，不再把 256 放大当高清。
MODEL_CAM_RES = 256
DISPLAY_SIZE = 512
JPEG_QUALITY = 90
JPEG_SUBSAMPLING = 2       # 4:2:0
WRIST_CROP_ASPECT = 1.3333
MAX_STEPS = 80
SIM_STEPS_PER_DECISION = 2
NUM_STEPS_WAIT = 8
STEP_M = 0.026
DELTA_BOUND_M = 0.1

# 基坐标系移动向量（与 configs/robot_maniskill.yaml 一致：+X 前, -Y 左, +Z 上）
MOVE_VECTORS = {
    "MV_FWD":   [1.0, 0.0, 0.0],
    "MV_BACK":  [-1.0, 0.0, 0.0],
    "MV_LEFT":  [0.0, -1.0, 0.0],
    "MV_RIGHT": [0.0, 1.0, 0.0],
    "MV_UP":    [0.0, 0.0, 1.0],
    "MV_DOWN":  [0.0, 0.0, -1.0],
}

# 可加载的真实 ManiSkill 场景。
# 第一项是官方 real2sim 场景（模型训练分布）；其余是 stock 任务，视觉分布差距大，仅供对照。
SCENES = [
    {"id": "BlockPAP-v1",       "label": "BlockPAP（官方）— 橙块放上杯垫",
     "instruction": "pick up the orange block and place it on the coaster"},
    {"id": "PickCube-v1",       "label": "PickCube — 抓取红方块",
     "instruction": "pick up the red cube"},
    {"id": "StackCube-v1",      "label": "StackCube — 红方块叠到绿方块上",
     "instruction": "stack the red cube on top of the green cube"},
    {"id": "PushCube-v1",       "label": "PushCube — 推方块到目标点",
     "instruction": "push the cube to the goal marker"},
    {"id": "PullCube-v1",       "label": "PullCube — 拉方块到目标点",
     "instruction": "pull the cube to the goal marker"},
    {"id": "PokeCube-v1",       "label": "PokeCube — 戳方块到目标点",
     "instruction": "poke the cube to the goal marker"},
    {"id": "LiftPegUpright-v1", "label": "LiftPegUpright — 竖直抬起钉子",
     "instruction": "lift the peg upright"},
]
SCENE_BY_ID = {s["id"]: s for s in SCENES}

# 官方 real2sim 场景的构造契约（来源：Show-Harness core/sim/maniskill_scenes.py +
# configs/robot_maniskill.yaml + 已发布 sim 训练语料 shard=blockpap_follow_wood）。
# 需要 RLinf checkout（real_franka.real2sim_env 模块），由 RLINF_ROOT 指定，默认 ~/RLinf。
# table_tex 必须保持 "006"：训练集就是木纹桌面，白桌面的腕部视图在抓取前是空白灰场，
# 不在训练分布内（实测 white 1/3、006 3/3，见 README「官方场景」一节）。
OFFICIAL_SCENES = {
    "BlockPAP-v1": {
        "robot_uids": "panda_high_friction_wristcam",
        "agentview_cam": "external_cam",   # 标定 RealSense K，640x480，不可改方形
        "wrist_cam": "hand_camera",
        "camera_resolution": None,         # 保留标定内参
        "reset_qpos": None,                # env 按轨迹自带标定起始角
        "scene": {"table_tex": "006", "traj_id": "random", "cam_t": "og",
                  "wrist_resolution": 256, "wrist_mount": "centered"},
        "layout": "wide",                  # 评估协议：训练生成器的随机布局
    },
}

# stock 任务统一相机/机器人
STOCK_ROBOT_UIDS = "panda_wristcam"
STOCK_AGENTVIEW_CAM = "base_camera"
STOCK_WRIST_CAM = "hand_camera"
# ManiSkill 的桌面场景对 panda_wristcam 默认把 joint7 设成 -pi/4。
# 官方 configs/robot_maniskill.yaml 对方块任务给出的起始角是 +pi/4。
# 差 90 度时，腕部图像的左右轴变成世界系的前后轴，模型的 MV_LEFT/MV_RIGHT 不会朝目标走。
# 官方 real2sim 场景（OFFICIAL_SCENES）必须保持 null，不能套这个覆盖。
STOCK_RESET_QPOS = [0.0, 0.3927, 0.0, -1.9635, 0.0, 2.3562, 0.7854]

# 与官方 configs/robot_maniskill.yaml 一致的 auto_release 阈值（plugins.auto_release）。
EMPTY_GRASP_WIDTH_M = 0.005

# ---- JPEG 编码 ----
from PIL import Image
import numpy as np

def encode_jpeg(arr, quality=JPEG_QUALITY, subsampling=JPEG_SUBSAMPLING):
    """按图像自身像素编码，不再放大。"""
    img = Image.fromarray(arr if arr.dtype == np.uint8 else np.clip(arr, 0, 255).astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, subsampling=subsampling)
    return buf.getvalue()

def build_frame_bytes(hdr: dict, agent_jpg: bytes, wrist_jpg: bytes) -> bytes:
    hdr = dict(hdr)
    hdr["a_len"] = len(agent_jpg)
    hdr["w_len"] = len(wrist_jpg)
    hdr_bytes = json.dumps(hdr, ensure_ascii=False).encode("utf-8")
    out = bytearray()
    out += len(hdr_bytes).to_bytes(4, "big")
    out += hdr_bytes
    out += agent_jpg
    out += wrist_jpg
    return bytes(out)


# ---- Hub：客户端管理与跨线程广播 ----
class Hub:
    def __init__(self, loop):
        self.loop = loop
        self.clients = set()          # set of Client
        self._lock = threading.Lock()

    def add(self, c): 
        with self._lock: self.clients.add(c)
    def remove(self, c):
        with self._lock: self.clients.discard(c)

    def broadcast_text(self, obj):
        data = json.dumps(obj, ensure_ascii=False)
        with self._lock:
            clients = list(self.clients)
        def _do():
            for c in clients:
                c.enqueue_text(data)
        self.loop.call_soon_threadsafe(_do)

    def broadcast_frame(self, frame_bytes, eid):
        with self._lock:
            clients = list(self.clients)
        def _do():
            for c in clients:
                c.enqueue_frame(frame_bytes, eid)
        self.loop.call_soon_threadsafe(_do)


class Client:
    """一个 WebSocket 客户端 + 有界发送队列（慢客户端丢旧帧）。"""
    MAX_Q = 2

    def __init__(self, ws):
        self.ws = ws
        self.q = asyncio.Queue(maxsize=self.MAX_Q)
        self.sender_task = None
        self.dropped = 0

    async def sender(self):
        try:
            while True:
                item = await self.q.get()
                if item is None:
                    return
                kind, data = item
                try:
                    if kind == "text":
                        await self.ws.send_str(data)
                    else:
                        await self.ws.send_bytes(data)
                except Exception:
                    return
        except asyncio.CancelledError:
            return

    def enqueue_text(self, data):
        try:
            self.q.put_nowait(data if False else ("text", data))
        except asyncio.QueueFull:
            pass

    def enqueue_frame(self, frame_bytes, eid):
        # 满则丢最旧，发送不反压环境线程。
        while self.q.full():
            try:
                self.q.get_nowait()
                self.dropped += 1
            except asyncio.QueueEmpty:
                break
        try:
            self.q.put_nowait(("bin", frame_bytes))
        except asyncio.QueueFull:
            self.dropped += 1
        if self.dropped and self.dropped % 5 == 0:
            log.info("perf client_drop=%d", self.dropped)


# ---- SimWorker：单一线程持有 env，跑闭环 ----
class SimWorker(threading.Thread):
    def __init__(self, hub: Hub):
        super().__init__(daemon=True, name="sim-worker")
        self.hub = hub
        self.cmd_q: Queue = Queue()
        self.obs_q: Queue = Queue(maxsize=1)   # 只保留最新观测
        self._lock = threading.Lock()
        self._interrupt_eid = None             # 设置则中断当前 episode
        self._running_eid = None
        self.env = None
        self.handle = None
        self.controller = None
        self.agent = None                     # MvTokenController
        self.client = None                    # VLMClient
        self.deps = None
        self.cur_scene_id = None
        self.cur_instruction = None
        self.cur_spec = None                # OFFICIAL_SCENES 条目；stock 场景为 None
        self._episode_no = 0                # 官方场景的 episode 计数，seed 随它递增
        self._stop = False

    # ---- 外部接口（任意线程调用）----
    def post(self, cmd):
        self.cmd_q.put(cmd)

    def interrupt(self, new_eid):
        with self._lock:
            self._interrupt_eid = new_eid

    def shutdown(self):
        self._stop = True
        self.interrupt("shutdown")
        self.cmd_q.put(("stop",))

    def _is_cancelled(self, eid):
        with self._lock:
            return self._interrupt_eid is not None and self._interrupt_eid != eid

    # ---- 线程主循环 ----
    def run(self):
        while not self._stop:
            try:
                cmd = self.cmd_q.get(timeout=1.0)
            except Empty:
                continue
            kind = cmd[0]
            try:
                if kind == "stop":
                    self._cleanup_env()
                    break
                elif kind == "load":
                    self._do_load(cmd)
                elif kind in ("start", "restart"):
                    self._run_episode(cmd)
            except Exception:
                log.error("worker cmd %s failed: %s", kind, traceback.format_exc())
                self.hub.broadcast_text({"type": "state", "phase": "错误",
                                         "detail": "服务器内部错误，见日志", "cls": "bad"})

    def _ensure_deps(self):
        if self.deps is not None:
            return
        from core.sim.maniskill_task import (
            make_maniskill_task, maybe_set_mujoco_gl, reset_maniskill,
            step_maniskill, ms_rgb, ms_success, ms_tcp, ms_gripper_width,
        )
        from interpreters.maniskill_atomic_controller import ManiskillAtomicController
        from core.record.images import prepare_view
        from core.vlm.vlm_client import VLMClient
        from core.vlm.mvtoken_roles import MvTokenController
        maybe_set_mujoco_gl()
        self.deps = dict(
            make_maniskill_task=make_maniskill_task, reset_maniskill=reset_maniskill,
            step_maniskill=step_maniskill, ms_rgb=ms_rgb, ms_success=ms_success,
            ms_tcp=ms_tcp, ms_gripper_width=ms_gripper_width,
            prepare_view=prepare_view, ManiskillAtomicController=ManiskillAtomicController,
        )
        prompt = PROMPT_PATH.read_text(encoding="utf-8").strip()
        self.client = VLMClient(
            base_url=VLM_BASE_URL, model=VLM_MODEL, api_key=VLM_API_KEY,
            timeout_s=VLM_TIMEOUT_S, max_tokens=256, temperature=0.0,
            chat_template_kwargs={"enable_thinking": False},
            provider="vllm",
        )
        self.agent = MvTokenController(client=self.client, prompt_template=prompt)

    def _cleanup_env(self):
        if self.env is not None:
            try: self.env.close()
            except Exception: pass
            self.env = None
            self.handle = None
            self.controller = None
            self.cur_spec = None

    def _do_load(self, cmd):
        _, scene_id, reply = cmd
        self._cleanup_env()
        self.cur_scene_id = scene_id
        spec = SCENE_BY_ID.get(scene_id)
        if spec is None:
            self.hub.broadcast_text({"type": "load_failed", "detail": f"未知场景 {scene_id}"})
            return
        official = OFFICIAL_SCENES.get(scene_id)
        self.cur_spec = official
        try:
            self._ensure_deps()
            d = self.deps
            max_ep = MAX_STEPS * SIM_STEPS_PER_DECISION + NUM_STEPS_WAIT + SIM_STEPS_PER_DECISION + 8
            if official is not None:
                kw = dict(
                    robot_uids=official["robot_uids"],
                    camera_resolution=official["camera_resolution"],
                    scene=dict(official["scene"]),
                )
            else:
                kw = dict(
                    robot_uids=STOCK_ROBOT_UIDS,
                    camera_resolution=MODEL_CAM_RES,
                )
            self.handle = d["make_maniskill_task"](
                env_id=scene_id, control_mode="pd_ee_delta_pos",
                sim_backend="physx_cpu", max_episode_steps=max_ep,
                task_description=spec["instruction"], **kw,
            )
            self.env = self.handle.env
            self.controller = d["ManiskillAtomicController"](
                move_vectors=MOVE_VECTORS, step_m=STEP_M, delta_bound_m=DELTA_BOUND_M,
            )
            self.cur_instruction = spec["instruction"]
            # reset 并推一帧静态观测（官方场景同时按训练生成器随机化布局）
            obs, info = self._reset_for_episode(seed=0)
            self._push_display(eid=0, fid=0, phase="场景已加载", detail=scene_id,
                               term=False, success=False, reason="")
            self.hub.broadcast_text({"type": "loaded", "env_id": scene_id,
                                     "instruction": spec["instruction"]})
            log.info("loaded scene %s", scene_id)
        except Exception as e:
            log.error("load %s failed: %s", scene_id, traceback.format_exc())
            self._cleanup_env()
            self.hub.broadcast_text({"type": "load_failed", "detail": f"{type(e).__name__}: {e}"})

    def _reset_for_episode(self, seed: int):
        """官方 runner 的 reset 流程（core/sim/mvtoken_maniskill_runner.py::run）。

        stock 场景：seed=0 + STOCK_RESET_QPOS 覆盖（既有行为）。
        官方场景：env 自带标定 qpos（reset_qpos=None），随后用训练数据生成器同一个
        randomize_layout 采样 wide 布局，再 hold 几步让物体静止。
        """
        d = self.deps
        official = self.cur_spec
        reset_qpos = STOCK_RESET_QPOS if official is None else official["reset_qpos"]
        obs, info = d["reset_maniskill"](
            self.env, seed=seed, settle_steps=NUM_STEPS_WAIT,
            hold_action=self.controller.open_gripper(),
            reset_qpos=reset_qpos)
        if official is not None and official.get("layout") == "wide":
            from scripts.trajectory.real2sim.maniskill.tasks import randomize_layout
            task_key = "blockpap" if hasattr(self.env.unwrapped, "target") else "blockstack"
            randomize_layout(
                self.env, task_key, np.random.default_rng(seed),
                d["ms_tcp"](self.env)[:2], self.controller.step_m, snap=False,
            )
            for _ in range(6):
                obs, _t, _tr, info = d["step_maniskill"](
                    self.env, self.controller.hold_action())
        return obs, info

    def _cam_names(self):
        """(agentview, wrist) 传感器名。官方场景用标定 external_cam；stock 用 base_camera。"""
        if self.cur_spec is not None:
            return self.cur_spec["agentview_cam"], self.cur_spec["wrist_cam"]
        return STOCK_AGENTVIEW_CAM, STOCK_WRIST_CAM

    def _ensure_display(self):
        """同姿态的独立展示相机。模型仍读场景自己的传感器（见 _cam_names）。"""
        from mani_skill.sensors.camera import Camera, CameraConfig
        uw = self.env.unwrapped
        scene = uw.scene
        cams = getattr(uw, "_human_render_cameras", None) or {}
        if cams.get("page_agent") is not None and cams.get("page_wrist") is not None:
            return
        agent_cam, wrist_cam = self._cam_names()
        for src, dst in ((agent_cam, "page_agent"), (wrist_cam, "page_wrist")):
            cfg = uw._sensors[src].config
            kw = dict(uid=dst, pose=cfg.pose, near=cfg.near, far=cfg.far,
                      mount=cfg.mount)
            intrinsic = getattr(cfg, "intrinsic", None)
            if intrinsic is not None:
                # 标定内参与分辨率绑定（external_cam 是 640x480 RealSense K），
                # 展示相机保持原生分辨率，不能套方形 512。
                kw.update(width=cfg.width, height=cfg.height, intrinsic=intrinsic)
            else:
                kw.update(width=DISPLAY_SIZE, height=DISPLAY_SIZE, fov=cfg.fov)
            new = CameraConfig(**kw)
            uw._human_render_cameras[dst] = Camera(new, scene)
        scene.human_render_cameras = uw._human_render_cameras

    def _model_views(self, obs):
        """官方训练约定：翻转、腕部 4:3 裁剪、letterbox 到 256。顺序 agent 然后 wrist。"""
        d = self.deps
        agent_cam, wrist_cam = self._cam_names()
        t0 = time.monotonic()
        ag = d["prepare_view"](
            d["ms_rgb"](obs, agent_cam),
            rotation_degrees=0, flip="none", square_size=MODEL_CAM_RES,
        )
        wr = d["prepare_view"](
            d["ms_rgb"](obs, wrist_cam),
            rotation_degrees=0, flip="both", crop_aspect=WRIST_CROP_ASPECT,
            square_size=MODEL_CAM_RES,
        )
        return ag, wr, (time.monotonic() - t0) * 1000

    def _capture_display(self):
        """在环境线程里渲染展示相机，并复制像素。编码线程不再碰 SAPIEN。"""
        self._ensure_display()
        uw = self.env.unwrapped
        t0 = time.monotonic()
        uw.scene.update_render(update_sensors=False, update_human_render_cameras=True)
        images = uw.scene.get_human_render_camera_images()
        render_ms = (time.monotonic() - t0) * 1000
        t1 = time.monotonic()
        ag = self._as_rgb(images["page_agent"])
        wr = self._as_rgb(images["page_wrist"])
        d = self.deps
        ag = d["prepare_view"](ag, rotation_degrees=0, flip="none", square_size=DISPLAY_SIZE)
        wr = d["prepare_view"](
            wr, rotation_degrees=0, flip="both", crop_aspect=WRIST_CROP_ASPECT,
            square_size=DISPLAY_SIZE,
        )
        prep_ms = (time.monotonic() - t1) * 1000
        return np.ascontiguousarray(ag), np.ascontiguousarray(wr), render_ms, prep_ms

    @staticmethod
    def _as_rgb(arr):
        try:
            import torch
            if isinstance(arr, torch.Tensor):
                arr = arr.detach().cpu().numpy()
        except Exception:
            pass
        arr = np.asarray(arr)
        if arr.ndim == 4:
            arr = arr[0]
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        return arr

    def _push_display(self, eid, fid, phase, detail, term, success, reason, timing=None):
        """展示队列 maxsize=1。丢弃尚未编码的旧帧，不阻塞 env.step。"""
        try:
            ag, wr, render_ms, prep_ms = self._capture_display()
        except Exception:
            log.error("display render failed: %s", traceback.format_exc())
            return
        item = dict(
            eid=eid, fid=fid, mono_ms=time.monotonic() * 1000,
            agent=ag, wrist=wr, phase=phase, detail=detail,
            term=term, success=success, reason=reason,
            render_ms=render_ms, prep_ms=prep_ms, timing=timing or {},
            aw=int(ag.shape[1]), ah=int(ag.shape[0]),
        )
        dropped = 0
        try:
            self.obs_q.put_nowait(item)
        except Full:
            try:
                old = self.obs_q.get_nowait()
                dropped = 0 if old is None or old.get("term") else 1
                if old is not None and old.get("term") and not term:
                    # 终止帧还在队列里时，不拿普通帧替换它
                    try:
                        self.obs_q.put_nowait(old)
                    except Full:
                        pass
                    return
            except Empty:
                pass
            try:
                self.obs_q.put_nowait(item)
            except Full:
                dropped += 1
        if dropped:
            log.info("perf drop_unencoded=%d fid=%s q=%d", dropped, fid, self.obs_q.qsize())

    def _run_episode(self, cmd):
        kind, task, eid = cmd[0], cmd[1], cmd[2]
        if self.env is None:
            self.hub.broadcast_text({"type": "state", "phase": "失败",
                                    "detail": "请先加载场景", "cls": "bad"})
            return
        with self._lock:
            self._interrupt_eid = None
            self._running_eid = eid
        d = self.deps
        # 官方场景：seed 随 episode 递增（对应官方 runner 的 episode_index），
        # 每个 episode 一个新的 wide 布局；stock 场景保持既有 seed=0。
        if self.cur_spec is not None:
            seed = self._episode_no
            self._episode_no += 1
        else:
            seed = 0
        try:
            obs, info = self._reset_for_episode(seed=seed)
        except Exception:
            log.error("reset failed: %s", traceback.format_exc())
            self.hub.broadcast_text({"type": "ended", "episode_id": eid,
                                     "success": False, "reason": "环境 reset 失败"})
            return
        self._ensure_display()
        fid = 0
        self._push_display(eid=eid, fid=fid, phase="准备中", detail="等待模型首次推理…",
                           term=False, success=False, reason="")
        fid += 1
        self.hub.broadcast_text({"type": "state", "phase": "准备中",
                                  "detail": "等待模型首次推理…", "cls": "warn"})
        recent_moves = []
        success = False
        end_reason = "max_steps_exceeded"
        steps = 0
        n_new_frames = 1
        for step_idx in range(MAX_STEPS):
            if self._is_cancelled(eid):
                end_reason = "cancelled"
                log.info("episode %s cancelled at step %d", eid, step_idx)
                return
            steps = step_idx + 1
            ag, wr, prep_ms = self._model_views(obs)
            gripper_state = "closed" if self.controller.state.gripper_name == "CLOSE" else "open"
            t1 = time.monotonic()
            try:
                resp = self.agent.decide(
                    task=task, gripper_state=gripper_state,
                    recent_moves=", ".join(recent_moves) if recent_moves else "none",
                    agentview_image=ag, wrist_image=wr, debug=False)
                token = resp.token
            except Exception as e:
                log.error("vlm decide failed step %d: %s", step_idx, traceback.format_exc())
                self.hub.broadcast_text({"type": "state", "phase": "推理失败",
                                         "detail": f"{type(e).__name__}: {e}", "cls": "bad"})
                token = "MV_DOWN"
                resp = None
            t_vlm = (time.monotonic() - t1) * 1000
            if self._is_cancelled(eid):
                end_reason = "cancelled"
                log.info("episode %s cancelled after vlm step %d", eid, step_idx)
                return
            if token == "DONE":
                end_reason = "done"
                log.info("perf step=%d token=DONE model_prep=%.0fms vlm=%.0fms", step_idx, prep_ms, t_vlm)
                break
            action = self._action_for_token(token)
            step_ms = []
            ctrl_n = 0
            for _ in range(SIM_STEPS_PER_DECISION):
                t2 = time.monotonic()
                obs, terminated, truncated, info = d["step_maniskill"](self.env, action)
                step_ms.append((time.monotonic() - t2) * 1000)
                ctrl_n += 1
                success = bool(d["ms_success"](info))
                # 每个真实控制步一帧。不额外 step，不重复同一位移。
                self._push_display(
                    eid=eid, fid=fid, phase="运行中",
                    detail=f"step {step_idx} → {token}",
                    term=False, success=success, reason="",
                )
                fid += 1
                n_new_frames += 1
                if terminated or truncated or success:
                    break
            # 与官方 runner 一致的 auto_release：夹爪闭合但宽度为空抓时重新张开
            # （configs/robot_maniskill.yaml plugins.auto_release, empty_width_m=0.005）。
            if (not success and self.controller.state.gripper_name == "CLOSE"
                    and d["ms_gripper_width"](self.env) < EMPTY_GRASP_WIDTH_M):
                log.info("step %d: auto-release (empty grasp); reopening", step_idx)
                open_action = self.controller.open_gripper()
                for _ in range(SIM_STEPS_PER_DECISION):
                    obs, terminated, truncated, info = d["step_maniskill"](self.env, open_action)
                    success = bool(d["ms_success"](info))
                    self._push_display(
                        eid=eid, fid=fid, phase="运行中",
                        detail=f"step {step_idx} → RELEASE (空抓自动张开)",
                        term=False, success=success, reason="",
                    )
                    fid += 1
                    if terminated or truncated or success:
                        break
            if token in MOVE_VECTORS:
                recent_moves.insert(0, token)
                del recent_moves[5:]
            tcp = d["ms_tcp"](self.env)
            log.info(
                "perf step=%d token=%s model_hw=%s model_prep=%.0fms vlm=%.0fms ctrl=%d step_ms=%s q=%d tcp=%s",
                step_idx, token, tuple(ag.shape), prep_ms, t_vlm, ctrl_n,
                ",".join(f"{x:.0f}" for x in step_ms), self.obs_q.qsize(),
                ",".join(f"{float(x):.3f}" for x in tcp),
            )
            if success:
                end_reason = "success"
                break
        reason = {"success": "任务完成", "done": "模型输出 DONE 结束",
                  "max_steps_exceeded": "达到最大步数未完成",
                  "cancelled": "已取消"}.get(end_reason, end_reason)
        self._push_display(eid=eid, fid=99999, phase="完成" if success else "失败",
                           detail=reason, term=True, success=success, reason=reason)
        log.info("perf frames=%d decisions=%d", n_new_frames + 1, steps)
        self.hub.broadcast_text({"type": "ended", "episode_id": eid,
                                 "success": success, "reason": reason})
        log.info("episode %s ended: success=%s reason=%s steps=%d", eid, success, end_reason, steps)

    def _action_for_token(self, token):
        from core.action_units import MOVE_ATOMS
        if token in MOVE_ATOMS:
            return self.controller.action_for_atomic(token)
        if token == "GRASP":
            return self.controller.close_gripper()
        if token == "RELEASE":
            return self.controller.open_gripper()
        return self.controller.hold_action()


# ---- Encoder 线程：观测 -> JPEG -> 广播 ----
class Encoder(threading.Thread):
    def __init__(self, worker: SimWorker, hub: Hub):
        super().__init__(daemon=True, name="encoder")
        self.worker = worker
        self.hub = hub
        self._stop = False

    def shutdown(self):
        self._stop = True
        # 投一个哨兵让 obs_q 解阻塞
        try: self.worker.obs_q.put_nowait(None)
        except Full:
            try: self.worker.obs_q.get_nowait(); self.worker.obs_q.put_nowait(None)
            except Exception: pass

    def run(self):
        log.info("encoder started")
        while not self._stop:
            try:
                obs = self.worker.obs_q.get(timeout=1.0)
            except Empty:
                continue
            if obs is None:
                break
            try:
                t0 = time.monotonic()
                ag_jpg = encode_jpeg(obs["agent"])
                wr_jpg = encode_jpeg(obs["wrist"])
                enc_ms = (time.monotonic() - t0) * 1000
                hdr = {
                    "eid": obs["eid"], "fid": obs["fid"],
                    "phase": obs["phase"], "detail": obs["detail"],
                    "term": obs["term"], "success": obs["success"], "reason": obs["reason"],
                    "aw": obs.get("aw"), "ah": obs.get("ah"),
                    "ww": int(obs["wrist"].shape[1]), "wh": int(obs["wrist"].shape[0]),
                    "enc_ms": round(enc_ms, 1),
                    "render_ms": round(obs.get("render_ms") or 0, 1),
                    "q": JPEG_QUALITY,
                }
                frame = build_frame_bytes(hdr, ag_jpg, wr_jpg)
                self.hub.broadcast_frame(frame, obs["eid"])
                log.info(
                    "perf encode fid=%s bytes=%d enc=%.0fms render=%.0fms q=%d %dx%d",
                    obs["fid"], len(frame), enc_ms, obs.get("render_ms") or 0,
                    JPEG_QUALITY, obs.get("aw"), obs.get("ah"),
                )
            except Exception:
                log.error("encode failed: %s", traceback.format_exc())
        log.info("encoder stopped")


# ---- aiohttp 应用 ----
worker: SimWorker = None
encoder: Encoder = None
hub: Hub = None


async def index(request):
    return web.FileResponse(Path(__file__).resolve().parent / "static" / "index.html")


async def ws_handler(request):
    ws = web.WebSocketResponse(max_msg_size=0)
    await ws.prepare(request)
    client = Client(ws)
    client.sender_task = asyncio.create_task(client.sender())
    hub.add(client)
    log.info("ws client connected")
    try:
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                try:
                    m = json.loads(msg.data)
                except json.JSONDecodeError:
                    continue
                await handle_msg(m)
            elif msg.type == WSMsgType.ERROR:
                log.warning("ws error: %s", ws.exception())
    finally:
        if client.sender_task:
            client.sender_task.cancel()
        hub.remove(client)
        log.info("ws client disconnected")
    return ws


async def handle_msg(m):
    t = m.get("type")
    if t == "get_scenes":
        hub.broadcast_text({"type": "scenes", "scenes": SCENES})
    elif t == "ping":
        pass
    elif t == "client_perf":
        log.info(
            "perf client decode=%.0fms draw=%.0fms gap=%.0fms",
            float(m.get("decode_ms") or 0), float(m.get("draw_ms") or 0),
            float(m.get("gap_ms") or 0),
        )
    elif t == "load":
        worker.post(("load", m.get("scene"), None))
    elif t == "start":
        eid = uuid.uuid4().hex
        hub.broadcast_text({"type": "started", "episode_id": eid})
        worker.post(("start", m.get("task", ""), eid))
    elif t == "restart":
        eid = uuid.uuid4().hex
        worker.interrupt(eid)          # 中断旧 episode
        hub.broadcast_text({"type": "started", "episode_id": eid})
        worker.post(("restart", m.get("task", ""), eid))


async def on_startup(app):
    global worker, encoder, hub
    hub = Hub(asyncio.get_running_loop())
    worker = SimWorker(hub)
    worker.start()
    encoder = Encoder(worker, hub)
    encoder.start()
    app["worker"] = worker
    app["encoder"] = encoder


async def on_cleanup(app):
    w = app.get("worker"); e = app.get("encoder")
    if w: w.shutdown()
    if e: e.shutdown()
    if w: w.join(timeout=5)


def make_app():
    app = web.Application(client_max_size=0)
    app.router.add_get("/", index)
    app.router.add_static("/static", Path(__file__).resolve().parent / "static")
    app.router.add_get("/ws", ws_handler)
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    return app


def main():
    port = int(os.environ.get("PAGE_PORT", "8765"))
    host = os.environ.get("PAGE_HOST", "127.0.0.1")
    log.info("starting Page on %s:%d (VLM=%s model=%s)", host, port, VLM_BASE_URL, VLM_MODEL)
    web.run_app(make_app(), host=host, port=port, access_log=None)


if __name__ == "__main__":
    main()
