#!/usr/bin/env python
"""Probe the OpenWAM-Alpha-Sim-LIBERO checkpoint without running inference.

Loads the checkpoint's own ``config.yaml``, resolves the architecture, builds
the full model from the config-embedded component specs, restores the
safetensors weights and reports the key-level load result plus the
normalization statistics. Used to prove the load contract before any GPU
inference: a file on disk is not a loadable checkpoint.

Usage:
    .venv-openwam/bin/python scripts/openwam_probe.py --device cpu
    .venv-openwam/bin/python scripts/openwam_probe.py --device cuda:2
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

CKPT_DIR = Path("/home/host/page-dmh/openwam_harness/checkpoints/OpenWAM-Alpha-Sim-LIBERO")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", default=str(CKPT_DIR))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip-load", action="store_true")
    args = parser.parse_args()

    import numpy as np
    import torch
    from omegaconf import OmegaConf

    ckpt_dir = Path(args.ckpt_dir)
    cfg = OmegaConf.load(ckpt_dir / "config.yaml")

    report: dict = {"ckpt_dir": str(ckpt_dir), "ckpt_file": None}

    files = sorted(ckpt_dir.glob("checkpoint_step_*.safetensors"))
    report["ckpt_files"] = [{"name": f.name, "bytes": f.stat().st_size} for f in files]

    from openwam.model import resolve_architecture_config

    resolved = resolve_architecture_config(cfg.model)
    report["architecture"] = {
        "registry_name": resolved.registry_name,
        "framework": resolved.canonical.framework,
        "variant": resolved.canonical.variant,
        "params_keys": sorted(str(k) for k in dict(resolved.params).keys()),
    }
    report["data_contract"] = {
        "action_mode": OmegaConf.select(cfg, "dataloader.action_mode", default=None),
        "unify_action": OmegaConf.select(cfg, "dataloader.unify_action", default=None),
        "normalize_mode": OmegaConf.select(cfg, "dataloader.normalize_mode", default=None),
        "num_frames": OmegaConf.select(cfg, "dataloader.num_frames", default=None),
        "video_stride": OmegaConf.select(cfg, "dataloader.video_stride", default=None),
        "height": OmegaConf.select(cfg, "dataloader.height", default=None),
        "width": OmegaConf.select(cfg, "dataloader.width", default=None),
        "multiview": OmegaConf.select(cfg, "dataloader.multiview", default=None),
        "camera_layout": OmegaConf.select(cfg, "dataloader.camera_layout", default=None),
        "target_camera": OmegaConf.select(cfg, "dataloader.target_camera", default=None),
        "split": OmegaConf.select(cfg, "dataloader.split", default=None),
        "dataset_dir": OmegaConf.select(cfg, "dataloader.dataset_dir", default=None),
        "binary_action_dims": OmegaConf.select(cfg, "dataloader.binary_action_dims", default=None),
    }
    report["architecture_cfg"] = {
        "action_dim": OmegaConf.select(cfg, "model.architecture.action_dim", default=None),
        "state_dim": OmegaConf.select(cfg, "model.architecture.state_dim", default=None),
        "use_proprioception": OmegaConf.select(cfg, "model.architecture.use_proprioception", default=None),
        "attention_mask_mode": OmegaConf.select(cfg, "model.architecture.attention_mask_mode", default=None),
    }
    report["video_backbone"] = {
        "name": OmegaConf.select(cfg, "model.video_backbone.name", default=None),
        "model_path": OmegaConf.select(cfg, "model.video_backbone.model_path", default=None),
        "model_path_exists": Path(str(OmegaConf.select(cfg, "model.video_backbone.model_path", default=""))).is_dir(),
        "has_components": OmegaConf.select(cfg, "model.video_backbone.components", default=None) is not None,
        "component_attrs": [
            c.get("attr") for c in (OmegaConf.select(cfg, "model.video_backbone.components", default=[]) or [])
        ],
        "tokenizer": OmegaConf.select(cfg, "model.video_backbone.tokenizer", default=None),
    }
    tok_dir = ckpt_dir / "tokenizer" / "google" / "umt5-xxl"
    report["tokenizer_dir"] = {"path": str(tok_dir), "exists": tok_dir.is_dir(),
                               "files": sorted(p.name for p in tok_dir.glob("*")) if tok_dir.is_dir() else []}

    stats_path = ckpt_dir / "normalization_stats.npy"
    stats = np.load(stats_path, allow_pickle=True)
    stats = stats.item() if stats.dtype == object and stats.shape == () else stats
    if isinstance(stats, dict):
        report["normalization_stats"] = {
            "file_bytes": stats_path.stat().st_size,
            "keys": sorted(str(k) for k in stats.keys()),
            "blocks": {},
        }
        for key, block in stats.items():
            if isinstance(block, dict):
                report["normalization_stats"]["blocks"][str(key)] = {
                    str(sub): (list(np.asarray(block[sub]).shape), str(np.asarray(block[sub]).dtype))
                    for sub in block
                }
            else:
                arr = np.asarray(block)
                report["normalization_stats"]["blocks"][str(key)] = {"shape": list(arr.shape), "dtype": str(arr.dtype)}
    else:
        report["normalization_stats"] = {"type": str(type(stats)), "shape": list(np.asarray(stats).shape)}

    if args.skip_load:
        print(json.dumps(report, indent=2, default=str))
        return 0

    from openwam.deploy.model_loader import _find_latest_checkpoint

    ckpt_file = _find_latest_checkpoint(str(ckpt_dir))
    report["ckpt_file"] = Path(ckpt_file).name

    started = time.time()
    from openwam.deploy.model_loader import load_from_checkpoint_dir

    cfg_loaded, arch = load_from_checkpoint_dir(str(ckpt_dir), device=args.device)
    report["load_s"] = round(time.time() - started, 2)

    # Per-module parameter counts prove which components actually materialized.
    counts = {}
    for name, module in arch.named_children():
        sub = sum(p.numel() for p in module.parameters())
        counts[name] = {"params_b": round(sub / 1e9, 4)}
    report["top_level_modules"] = counts
    total = sum(p.numel() for p in arch.parameters())
    report["total_params_b"] = round(total / 1e9, 4)
    report["device"] = str(next(arch.parameters()).device)
    report["dtype"] = str(next(arch.parameters()).dtype)
    report["repr_contract"] = dict(getattr(arch, "repr_contract", {}) or {})
    report["uses_proprioception"] = bool(getattr(arch, "uses_proprioception", False))
    report["state_dim"] = getattr(arch, "state_dim", None)
    report["action_dim"] = getattr(arch, "action_dim", None)
    report["normalizer"] = {
        "attached": getattr(arch, "normalizer", None) is not None,
        "type": type(getattr(arch, "normalizer", None)).__name__,
    }
    if torch.cuda.is_available():
        report["peak_gpu_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
