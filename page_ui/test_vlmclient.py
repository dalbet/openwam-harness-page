#!/usr/bin/env python3
"""用真实图像测试 VLMClient + MvTokenController -> 本地 transformers VLM 服务。"""
import sys, os
sys.path.insert(0, "/root/Page_dmh/Show-Harness")
import numpy as np
from core.vlm.vlm_client import VLMClient
from core.vlm.mvtoken_roles import MvTokenController

client = VLMClient(
    base_url="http://127.0.0.1:8000/v1", model="qwen3_5_2b_showharness_sim",
    api_key="EMPTY", timeout_s=120, max_tokens=256, temperature=0.0,
    chat_template_kwargs={"enable_thinking": False}, provider="vllm",
)
prompt = open("/root/Page_dmh/Show-Harness/prompts/v3/mvtoken_generator_lite.txt").read().strip()
agent = MvTokenController(client=client, prompt_template=prompt)

# 两张真实风格随机图像（模拟 agentview + wrist）
ag = (np.random.rand(256,256,3)*255).astype(np.uint8)
wr = (np.random.rand(256,256,3)*255).astype(np.uint8)
resp = agent.decide(task="pick up the red cube", gripper_state="open",
                    recent_moves="none", agentview_image=ag, wrist_image=wr, debug=False)
print("TOKEN:", repr(resp.token))
print("RAW:", repr(resp.raw_text)[:80])
print("LATENCY_s:", resp.payload.get("latency_s"))
