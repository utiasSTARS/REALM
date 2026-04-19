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
model_factory.py — REALM model construction from a YAML config.

The public entry point is :func:`REALM_creator`, which assembles a
:class:`~realm.model.REALM` model by loading and wiring:

* a task-specific **head** (depth, matching, …)
* an optional **RGB encoder** (from a DUNE checkpoint)
* an optional **projector** (from a DUNE checkpoint)
* an optional **event encoder** (from a REALM checkpoint, with LoRA weights
  merged and unloaded before returning)

Typical usage
-------------
    from realm.model_factory import REALM_creator

    model = REALM_creator("realm/configs/depth.yaml").to(device)
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

from realm.embedding import Vox2PatchEmbed
from realm.heads.head_factory import head_factory
from realm.model import REALM
from realm.dune.dune import load_dune_from_checkpoint
from realm.utils.log import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

class _EncoderWrapper(nn.Module):
    """
    Minimal ``nn.Module`` wrapper used to apply PEFT/LoRA to a bare encoder.

    The wrapper is intentionally thin: after ``merge_and_unload()`` the inner
    encoder is extracted and the wrapper is discarded.
    """

    def __init__(self, encoder: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder


def _load_checkpoint(path: str | Path, component: str) -> dict[str, Any]:
    """
    Load a PyTorch checkpoint from *path* with a clear error on failure.

    Args:
        path:      Filesystem path to the ``.pt`` / ``.pth`` checkpoint.
        component: Human-readable label used in error messages.

    Returns:
        The raw checkpoint dictionary.

    Raises:
        FileNotFoundError: If *path* does not exist.
        RuntimeError:      If the file cannot be loaded by PyTorch.
    """
    ckpt_path = Path(path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint for '{component}' not found: {ckpt_path}"
        )
    try:
        return torch.load(ckpt_path, map_location="cpu", weights_only=False)  # noqa: S614
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load checkpoint for '{component}' from {ckpt_path}: {exc}"
        ) from exc


def _build_head(config: dict) -> nn.Module:
    """Instantiate and load the task head from config."""
    head = head_factory(config["head"])
    ckpt = _load_checkpoint(config["pretrained_head"], "head")
    head.load_state_dict(ckpt["model_state_dict"], strict=True)
    logger.info("Head loaded from: %s", config["pretrained_head"])
    return head


def _build_rgb_encoder(config: dict) -> nn.Module | None:
    """Load the RGB encoder from a DUNE checkpoint, or return None."""
    if "dune_checkpoint" not in config:
        logger.warning(
            "No 'dune_checkpoint' key in config — RGB encoder will be None."
        )
        return None

    encoder = load_dune_from_checkpoint(config["dune_checkpoint"])[0].encoder
    logger.info("RGB encoder loaded from: %s", config["dune_checkpoint"])
    return encoder


def _build_projector(config: dict) -> nn.Module | None:
    """Load the projector from a DUNE base checkpoint, or return None."""
    if "projector" not in config:
        logger.warning(
            "No 'projector' key in config — projector will be None."
        )
        return None

    projector = (
        load_dune_from_checkpoint(config["base_checkpoint"])[0]
        .projectors[config["projector"]]
    )
    logger.info(
        "Projector '%s' loaded from: %s",
        config["projector"],
        config["base_checkpoint"],
    )
    return projector


def _build_event_encoder(config: dict) -> nn.Module | None:
    """
    Load the event encoder from a REALM checkpoint with LoRA weights merged.

    Steps
    -----
    1. Load the base DUNE encoder from the DUNE base checkpoint.
    2. Swap in the voxel patch embedding.
    3. Wrap in :class:`_EncoderWrapper` so PEFT can attach LoRA adapters.
    4. Load the REALM checkpoint state dict (which includes LoRA deltas).
    5. Merge LoRA weights back into the base weights and discard the adapter.

    Returns:
        The merged encoder module, or ``None`` if no REALM checkpoint is given.

    Raises:
        ValueError: If the checkpoint does not contain a LoRA configuration.
        RuntimeError: If state-dict loading fails.
    """
    if "realm_checkpoint" not in config:
        logger.warning(
            "No 'realm_checkpoint' key in config — event encoder will be None."
        )
        return None

    # 1. Base encoder
    encoder = load_dune_from_checkpoint(config["base_checkpoint"])[0].encoder

    # 2. Swap patch embedding
    encoder.patch_embed = Vox2PatchEmbed(**config["embedding"])

    # 3. Load REALM checkpoint
    ckpt = _load_checkpoint(config["realm_checkpoint"], "event encoder (REALM)")
    raw_state_dict = ckpt["model_state_dict"]

    # 4. Reconstruct LoRA config from the saved training config
    model_config = ckpt.get("config", {}).get("model", {})
    lora_cfg = model_config.get("lora")
    if lora_cfg is None:
        raise ValueError(
            "The REALM checkpoint does not contain a LoRA configuration. "
            "Cannot reconstruct the event encoder."
        )

    peft_config = LoraConfig(
        r=lora_cfg.get("rank", 16),
        lora_alpha=lora_cfg.get("lora_alpha", 32),
        target_modules=lora_cfg.get("target_modules", ["qkv"]),
        lora_dropout=lora_cfg.get("lora_dropout", 0.05),
        bias=lora_cfg.get("bias", "none"),
        modules_to_save=lora_cfg.get("modules_to_save", None),
    )

    # 5. Apply LoRA, load weights, then merge & unload
    wrapped = get_peft_model(_EncoderWrapper(encoder), peft_config)
    wrapped.load_state_dict(raw_state_dict, strict=True)
    merged = wrapped.merge_and_unload()

    logger.info(
        "Event encoder loaded and LoRA weights merged from: %s",
        config["realm_checkpoint"],
    )
    return merged.encoder


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def REALM_creator(config: dict | str | Path) -> REALM:
    """
    Build and return a :class:`~realm.model.REALM` model from *config*.

    Args:
        config: Either a pre-parsed config dictionary or a path to a YAML file.

    Returns:
        An assembled :class:`~realm.model.REALM` instance (on CPU, not yet
        moved to a device). Call ``.to(device)`` and ``.eval()`` on the result.

    Raises:
        FileNotFoundError: If any checkpoint path does not exist.
        ValueError:        If a required config key is missing or invalid.
        RuntimeError:      If model initialisation fails.
    """
    # ------------------------------------------------------------------ config
    if isinstance(config, (str, Path)):
        config_path = Path(config)
        if not config_path.is_file():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with config_path.open("r") as fh:
            config = yaml.safe_load(fh)
        logger.debug("Config loaded from: %s", config_path)

    # ------------------------------------------------------------------ parts
    logger.info("Building REALM model...")

    model_args: dict[str, Any] = {
        "head":        _build_head(config),
        "encoder_rgb": _build_rgb_encoder(config),
        "projector":   _build_projector(config),
        "encoder_ev":  _build_event_encoder(config),
    }

    # ------------------------------------------------------------------ assemble
    try:
        model = REALM(**model_args)
    except Exception as exc:
        raise RuntimeError(
            f"REALM model initialisation failed: {exc}"
        ) from exc

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
    from realm.utils.transform import Resize

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