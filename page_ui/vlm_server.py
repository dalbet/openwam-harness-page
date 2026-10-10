#!/usr/bin/env python3
"""Page — OpenAI 兼容的本地 VLM 服务（transformers + peft）。

RTX 5090 (SM_120) 上 vLLM 0.24 依赖的 FlashInfer 不支持该架构，故改用 transformers
直接加载基础模型 + LoRA adapter，在 GPU 上推理，暴露与 VLMClient 兼容的
/v1/models 与 /v1/chat/completions。

仅绑定 127.0.0.1:8000，供同机的 page_ui/server.py 调用。
"""
from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
import threading
import time
import uuid
from typing import Any

os.environ.setdefault("HF_HOME", "/root/Page_dmh/Show-Harness/models/huggingface")

import torch
from PIL import Image
from aiohttp import web
from transformers import StoppingCriteria, StoppingCriteriaList

BASE = os.environ.get("PAGE_VLM_BASE", "/root/Page_dmh/Show-Harness/models/huggingface/hub/Qwen/Qwen3.5-2B")
ADAPTER = os.environ.get("PAGE_VLM_ADAPTER", "/root/Page_dmh/Show-Harness/models/Show-Harness-VLMs/qwen3_5_2b_sim")
TEMPLATE = os.environ.get("PAGE_VLM_TEMPLATE", "/root/Page_dmh/Show-Harness/models/chat_templates/qwen3_5_nothink.jinja")
SERVED_NAME = os.environ.get("VLM_MODEL", "qwen3_5_2b_showharness_sim")
PORT = int(os.environ.get("VLM_PORT", "8000"))
HOST = os.environ.get("VLM_HOST", "127.0.0.1")
DEVICE = os.environ.get("PAGE_VLM_DEVICE", "cuda:0")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("/root/Page_dmh/page_ui/logs/vlm_server.log")],
)
log = logging.getLogger("vlm")

_model = None
_proc = None
_gen_lock = threading.Lock()


def load_model():
    global _model, _proc
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from peft import PeftModel
    log.info("loading processor from %s", BASE)
    _proc = AutoProcessor.from_pretrained(BASE)
    with open(TEMPLATE) as f:
        _proc.chat_template = f.read()
    log.info("loading base model bf16 on %s", DEVICE)
    t0 = time.time()
    _model = AutoModelForImageTextToText.from_pretrained(BASE, dtype=torch.bfloat16, device_map=DEVICE)
    log.info("base loaded %.1fs", time.time() - t0)
    log.info("loading peft adapter %s", ADAPTER)
    _model = PeftModel.from_pretrained(_model, ADAPTER)
    _model.eval()
    devices = sorted({str(p.device) for p in _model.parameters()})
    dtypes = sorted({str(p.dtype) for p in _model.parameters()})
    text_cfg = getattr(_model.config, "text_config", None)
    vis_cfg = getattr(_model.config, "vision_config", None)
    log.info(
        "model ready devices=%s dtypes=%s text_attn=%s vision_attn=%s",
        devices, dtypes,
        getattr(text_cfg, "_attn_implementation", None),
        getattr(vis_cfg, "_attn_implementation", None),
    )


_DATA_URL = re.compile(r"^data:image/[a-zA-Z]+;base64,")


def decode_image(url: str) -> Image.Image:
    # data:image/png;base64,.... 或 http url（不支持远程，仅本地）
    if url.startswith("data:"):
        b64 = url.split(",", 1)[1]
        return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    raise ValueError(f"unsupported image url: {url[:40]}")


def extract_parts(messages: list[dict]) -> tuple[list[Image.Image], str]:
    """从 OpenAI chat messages 中提取图像与文本（按出现顺序：图像在前，文本在后）。"""
    images: list[Image.Image] = []
    texts: list[str] = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str):
            texts.append(content)
            continue
        if isinstance(content, list):
            for part in content:
                t = part.get("type")
                if t == "text":
                    texts.append(part.get("text", ""))
                elif t in ("image_url", "image"):
                    iu = part.get("image_url") or {}
                    url = iu.get("url") if isinstance(iu, dict) else part.get("image")
                    if t == "image" and not url:
                        # transformers 风格：{"type":"image","image": <PIL|np|path>}
                        img = part.get("image")
                        if isinstance(img, Image.Image):
                            images.append(img)
                        elif isinstance(img, str):
                            images.append(Image.open(img).convert("RGB"))
                    else:
                        images.append(decode_image(url))
    return images, "\n".join(texts)


async def handle_models(request):
    return web.json_response({
        "object": "list",
        "data": [{"id": SERVED_NAME, "object": "model", "owned_by": "page"}],
    })


# 官方动作词。MV_* / GRASP 通常是两个 token，DONE 是一个；未形成完整词之前不停止。
_ACTION_TOKENS = (
    "MV_FWD", "MV_BACK", "MV_LEFT", "MV_RIGHT", "MV_UP", "MV_DOWN",
    "GRASP", "RELEASE", "DONE",
)


class FirstActionToken(StoppingCriteria):
    """生成出第一个完整动作词（或词后换行）即停。每次请求新建，不复用上一张图的 KV。"""

    def __init__(self, tokenizer, start_len: int):
        self.tokenizer = tokenizer
        self.start_len = start_len

    def __call__(self, input_ids, scores, **kwargs):
        hit = False
        gen = input_ids[0, self.start_len:]
        if gen.numel() >= 1:
            text = self.tokenizer.decode(gen, skip_special_tokens=True).strip()
            for tok in _ACTION_TOKENS:
                if text == tok or (
                    text.startswith(tok) and len(text) > len(tok) and text[len(tok)].isspace()
                ):
                    hit = True
                    break
        return torch.full((input_ids.shape[0],), hit, device=input_ids.device, dtype=torch.bool)


async def handle_chat(request):
    body = await request.json()
    messages = body.get("messages", [])
    max_tokens = int(body.get("max_tokens", 24) or 24)
    temperature = float(body.get("temperature", 0.0) or 0.0)
    chat_kwargs = body.get("chat_template_kwargs") or {}
    enable_thinking = bool(chat_kwargs.get("enable_thinking", False))

    images, prompt_text = extract_parts(messages)
    if not prompt_text:
        return web.json_response({"error": {"message": "no text prompt"}}, status=400)

    # 构造 user message：图像 + 文本（与训练 wire order 一致：图像在前）
    t_prep = time.monotonic()
    content = [{"type": "image", "image": im} for im in images] + [{"type": "text", "text": prompt_text}]
    rendered = _proc.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking,
    )
    inputs = _proc(text=rendered, images=images or None, return_tensors="pt").to(DEVICE)
    prep_ms = (time.monotonic() - t_prep) * 1000
    start_len = int(inputs["input_ids"].shape[-1])
    # 上限仍是官方非托管预算（调用方传来的 max_tokens，客户端封顶 24）。
    # 停在第一个完整动作词，避免把同一词重复生成到上限。
    stopper = StoppingCriteriaList([FirstActionToken(_proc.tokenizer, start_len)])

    started = time.monotonic()
    with _gen_lock, torch.inference_mode():
        out = _model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else 1.0,
            use_cache=True,
            stopping_criteria=stopper,
        )
    gen_ids = out[0][start_len:]
    text = _proc.decode(gen_ids, skip_special_tokens=True)
    latency = time.monotonic() - started
    log.info(
        "chat prep=%.0fms gen=%.0fms n_img=%d max_tok=%d new_tok=%d -> %r",
        prep_ms, latency * 1000, len(images), max_tokens, int(gen_ids.shape[-1]), text[:40],
    )
    return web.json_response({
        "id": "chatcmpl-" + uuid.uuid4().hex[:12],
        "object": "chat.completion",
        "model": SERVED_NAME,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": int(inputs["input_ids"].shape[-1]), "completion_tokens": int(gen_ids.shape[-1])},
    })


async def handle_root(request):
    return web.Response(text="page-vlm", content_type="text/plain")


def make_app():
    app = web.Application(client_max_size=64 * 1024 * 1024)
    app.router.add_get("/", handle_root)
    app.router.add_get("/v1/models", handle_models)
    app.router.add_post("/v1/chat/completions", handle_chat)
    return app


def main():
    load_model()
    log.info("serving %s on %s:%d", SERVED_NAME, HOST, PORT)
    web.run_app(make_app(), host=HOST, port=PORT, access_log=None)


if __name__ == "__main__":
    main()
