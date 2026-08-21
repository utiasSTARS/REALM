# Copyright (c) 2025 Space and Terrestrial Autonomous Robotic Systems (STARS) Lab,
# University of Toronto Institute for Aerospace Studies (UTIAS).
# All rights reserved.
#
# This software is provided for research and educational purposes only.
# Redistribution and use, with or without modification, are permitted provided
# that this copyright notice and attribution are retained.
#
# Maintainer: Vincenzo Polizzi <polivicio@gmail.com>

"""
model_factory.py — REALM model construction from a YAML config or model name.

Typical usage
-------------
    from realm.model_factory import REALM_creator

    # Pass a specific model name to auto-resolve configs and checkpoints from HF:
    model = REALM_creator("depth").to(device)
    model.eval()
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import yaml
from peft import LoraConfig, get_peft_model
from huggingface_hub import hf_hub_download

from realm.embedding import Vox2PatchEmbed
from realm.heads.head_factory import head_factory
from realm.model import REALM
from realm.dune.dune import load_dune_from_checkpoint
from realm.utils.log import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Global Configuration
# ---------------------------------------------------------------------------

# Hardcode your Hugging Face repository here
HF_REPO = "viciopoli/REALM"

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_path(path: str | Path) -> Path:
    """
    Check if a path exists locally. If not, download it from the hardcoded HF Hub.
    """
    local_path = Path(path)
    if local_path.is_file():
        return local_path

    logger.info("Local file '%s' not found. Fetching from Hugging Face Hub (%s)...", path, HF_REPO)
    try:
        # hf_hub_download mirrors the exact path structure of the repo
        cached_path = hf_hub_download(repo_id=HF_REPO, filename=str(path))
        return Path(cached_path) 
    except Exception as exc:
        raise FileNotFoundError(
            f"Could not find '{path}' locally, and failed to download from "
            f"Hugging Face repo '{HF_REPO}': {exc}"
        ) from exc


class _EncoderWrapper(nn.Module):
    def __init__(self, encoder: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder


def _load_checkpoint(path: str | Path, component: str) -> dict[str, Any]:
    try:
        ckpt_path = _resolve_path(path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Checkpoint for '{component}' not found: {exc}") from exc

    try:
        return torch.load(ckpt_path, map_location="cpu", weights_only=False)  # noqa: S614
    except Exception as exc:
        raise RuntimeError(f"Failed to load checkpoint for '{component}' from {ckpt_path}: {exc}") from exc


def _build_head(config: dict) -> nn.Module:
    if "head" not in config:
        return None
    head = head_factory(config["head"])
    ckpt = _load_checkpoint(config["pretrained_head"], "head")
    head.load_state_dict(ckpt["model_state_dict"], strict=True)
    logger.info("Head loaded from: %s", config["pretrained_head"])
    return head


def _build_rgb_encoder(config: dict) -> nn.Module | None:
    if "dune_checkpoint" not in config:
        return None
    ckpt_path = _resolve_path(config["dune_checkpoint"])
    encoder = load_dune_from_checkpoint(str(ckpt_path))[0].encoder
    logger.info("RGB encoder loaded from: %s", ckpt_path)
    return encoder


def _build_projector(config: dict) -> nn.Module | None:
    if "projector" not in config:
        return None
    ckpt_path = _resolve_path(config["base_checkpoint"])
    projector = load_dune_from_checkpoint(str(ckpt_path))[0].projectors[config["projector"]]
    logger.info("Projector '%s' loaded from: %s", config["projector"], ckpt_path)
    return projector


def _build_event_encoder(config: dict) -> nn.Module | None:
    if "realm_checkpoint" not in config:
        return None

    base_ckpt_path = _resolve_path(config["base_checkpoint"])
    encoder = load_dune_from_checkpoint(str(base_ckpt_path))[0].encoder
    encoder.patch_embed = Vox2PatchEmbed(**config["embedding"])

    ckpt = _load_checkpoint(config["realm_checkpoint"], "event encoder (REALM)")
    
    model_config = ckpt.get("config", {}).get("model", {})
    lora_cfg = model_config.get("lora")
    if lora_cfg is None:
        raise ValueError("The REALM checkpoint does not contain a LoRA configuration.")

    peft_config = LoraConfig(
        r=lora_cfg.get("rank", 16),
        lora_alpha=lora_cfg.get("lora_alpha", 32),
        target_modules=lora_cfg.get("target_modules", ["qkv"]),
        lora_dropout=lora_cfg.get("lora_dropout", 0.05),
        bias=lora_cfg.get("bias", "none"),
        modules_to_save=lora_cfg.get("modules_to_save", None),
    )

    logger.info(f"Model name: {HF_REPO}")
    logger.info(f"LoRA rank: {lora_cfg.get('rank', 16)}")
    logger.info(f"LoRA target_modules: {lora_cfg.get('target_modules', ['qkv'])}")

    wrapped = get_peft_model(_EncoderWrapper(encoder), peft_config)
    trainable_params = sum(p.numel() for p in wrapped.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in wrapped.parameters())
    logger.info(
        "LoRA params: %s | Total params: %s | Trainable: %.2f%%",
        f"{trainable_params:,}",
        f"{all_params:,}",
        100 * trainable_params / all_params,
    )
    wrapped.load_state_dict(ckpt["model_state_dict"], strict=True)
    merged = wrapped.merge_and_unload()

    logger.info("Event encoder loaded and LoRA weights merged from: %s", config["realm_checkpoint"])
    return merged.encoder


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def REALM_creator(config_or_name: dict | str | Path) -> REALM:
    """
    Build and return a :class:`~realm.model.REALM` model.
    Accepts a raw config dict, a path to a YAML file, or a model name (e.g., "depth").
    """
    if isinstance(config_or_name, dict):
        config = config_or_name
    elif isinstance(config_or_name, (str, Path)):
        path_str = str(config_or_name)
        
        # If the user passed a model name instead of a yaml file path
        if not path_str.endswith((".yaml", ".yml")):
            # Construct the default config path relative to the repo root
            target_path = f"realm/configs/{path_str}.yaml"
            logger.info("Interpreted '%s' as a model name. Targeting config: %s", path_str, target_path)
        else:
            target_path = path_str

        # Resolve the YAML config (locally or from HF)
        resolved_config_path = _resolve_path(target_path)
        
        with resolved_config_path.open("r") as fh:
            config = yaml.safe_load(fh)
        logger.debug("Config loaded from: %s", resolved_config_path)
    else:
        raise TypeError("config_or_name must be a dict, string, or Path object.")

    logger.info("Building REALM model...")

    model_args: dict[str, Any] = {
        "head":        _build_head(config),
        "encoder_rgb": _build_rgb_encoder(config),
        "projector":   _build_projector(config),
        "encoder_ev":  _build_event_encoder(config),
    }

    try:
        model = REALM(**model_args)
    except Exception as exc:
        raise RuntimeError(f"REALM model initialisation failed: {exc}") from exc

    logger.success("REALM model built successfully.")  # type: ignore[attr-defined]

    # ------------------------------------------------------------------ cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return model


# ---------------------------------------------------------------------------
# Quick smoke-test (python -m realm.model_factory)
# ---------------------------------------------------------------------------

def _parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="REALM model factory smoke-test")
    parser.add_argument(
        "--config",
        type=str,
        default="realm/realm/configs/mast3r.yaml",
        help="Path to YAML config.",
    )
    parser.add_argument(
        "--image", type=str, default="test/00326_r_3d.jpg", help="Test RGB image."
    )
    parser.add_argument(
        "--voxel", type=str, default="test/ev_l_17421491445.npy", help="Test voxel .npy."
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


if __name__ == "__main__":
    import cv2
    import numpy as np

    from realm.utils.vis import VisMast3r, image_to_normalized_tensor, voxel_to_rgb_image
    from realm.utils.transforms import Resize

    args = _parse_args()
    device = torch.device(args.device)

    model = REALM_creator(args.config).to(device)
    model.eval()

    logger.info("Running smoke-test inference on device: %s", device)

    with torch.inference_mode():
        # --- RGB input ---
        img_bgr = cv2.imread(args.image)
        if img_bgr is None:
            raise FileNotFoundError(f"Test image not found: {args.image}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_rgb = cv2.resize(img_rgb, (448, 448))
        img_tensor = image_to_normalized_tensor(img_rgb).unsqueeze(0).to(device)

        # --- Event voxel input ---
        voxel_path = Path(args.voxel)
        if not voxel_path.is_file():
            raise FileNotFoundError(f"Test voxel not found: {voxel_path}")
        ev = torch.from_numpy(np.load(voxel_path)).unsqueeze(0).to(device)
        ev = Resize((448, 448), keep_aspect_ratio=True)(ev)
        ev_vis = voxel_to_rgb_image(ev.squeeze(0)) * 255.0
        ev_vis_bgr = cv2.cvtColor(ev_vis.astype("uint8"), cv2.COLOR_RGB2BGR)

        # --- Forward pass ---
        out1, out2 = model({"view1": ev, "view2": img_tensor}, {"H": 448, "W": 448})

        # --- Visualise ---
        vis_data = {
            "view1": ev_vis_bgr,
            "view2": img_rgb,
            "pred1": out1,
            "pred2": out2,
        }
        result = VisMast3r(vis_data, n_viz=10)
        out_path = Path(args.image).parent / "smoke_test_result.png"
        cv2.imwrite(str(out_path), result)

    logger.success("Smoke-test complete. Result saved to: %s", out_path)  # type: ignore[attr-defined]