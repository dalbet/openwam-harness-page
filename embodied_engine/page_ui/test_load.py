#!/usr/bin/env python3
"""快速测试：transformers + peft 加载 Qwen3.5-2B + sim adapter，带图像生成一个 token。"""
import os, sys, io, base64, time
os.environ.setdefault("HF_HOME", "/root/Page_dmh/Show-Harness/models/huggingface")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
import torch
import numpy as np
from PIL import Image

BASE = "/root/Page_dmh/Show-Harness/models/huggingface/hub/Qwen/Qwen3.5-2B"
ADAPTER = "/root/Page_dmh/Show-Harness/models/Show-Harness-VLMs/qwen3_5_2b_sim"
TPL = "/root/Page_dmh/Show-Harness/models/chat_templates/qwen3_5_nothink.jinja"

from transformers import AutoProcessor, AutoModelForImageTextToText

print("loading processor...", flush=True)
proc = AutoProcessor.from_pretrained(BASE)
# 挂载对齐的 chat template
with open(TPL) as f:
    proc.chat_template = f.read()

print("loading model bf16 on cuda...", flush=True)
t0 = time.time()
model = AutoModelForImageTextToText.from_pretrained(BASE, dtype=torch.bfloat16, device_map="cuda:0")
print("base loaded %.1fs" % (time.time()-t0), flush=True)

print("loading peft adapter...", flush=True)
from peft import PeftModel
model = PeftModel.from_pretrained(model, ADAPTER)
model.eval()
print("adapter loaded", flush=True)

# 构造两张随机图像 + 一个 prompt
img1 = Image.fromarray((np.random.rand(256,256,3)*255).astype(np.uint8))
img2 = Image.fromarray((np.random.rand(256,256,3)*255).astype(np.uint8))
prompt = ("You are controlling a robot arm with two cameras:\nTask: pick up the red cube\n"
          "Recent moves, newest first: none\n\nOutput exactly one action token:\n"
          "MV_FWD, MV_BACK, MV_LEFT, MV_RIGHT, MV_UP, MV_DOWN, GRASP, RELEASE, DONE\n"
          "Return the single token only, no punctuation, no explanation:")

messages = [{"role":"user","content":[{"type":"image","image":img1},{"type":"image","image":img2},{"type":"text","text":prompt}]}]
text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
print("template head:", text[:120].replace("\n","\\n"), flush=True)
inputs = proc(text=text, images=[img1, img2], return_tensors="pt").to("cuda:0")
print("input_ids shape:", inputs["input_ids"].shape, flush=True)

t0 = time.time()
with torch.inference_mode():
    out = model.generate(**inputs, max_new_tokens=8, do_sample=False, temperature=0.0)
gen = out[0][inputs["input_ids"].shape[-1]:]
print("generate %.1fs" % (time.time()-t0), flush=True)
print("RAW GEN:", repr(proc.decode(gen, skip_special_tokens=True)), flush=True)
print("OK", flush=True)
