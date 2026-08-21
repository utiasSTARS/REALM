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
visualize_embeddings.py — Qualitative check of event/image token alignment.

Runs a Hybrid REALM model (event encoder + RGB encoder) on a paired
event-voxel / image sample, and visualises the patch embeddings of both
modalities with a *shared* PCA projection to 3 channels (displayed as RGB).

Two embedding stages are compared:

* "raw tokenizer" — output of ``patch_embed`` alone (Vox2PatchEmbed for
  events, the plain conv PatchEmbed for RGB), i.e. before any transformer
  block has mixed information across patches.
* "encoded"        — output of the full ViT encoder (``x_norm_patchtokens``).

If the event tokenizer has learned to map events into the same embedding
space as images, the PCA-coloured maps for both modalities should highlight
matching structures with similar colours. A per-patch cosine-similarity map
between the two modalities is also shown as a second diagnostic (meaningful
only when the event/image pair is spatially aligned, e.g. via calibration
warping).

Usage
-----
    python -m evaluation.visualize_embeddings \\
        --config realm/realm/configs/mast3r.yaml \\
        --image test/00326_r_3d.jpg \\
        --voxel test/ev_l_17421491445.npy \\
        --out embedding_pca.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

from realm.model_factory import REALM_creator
from realm.model import ModelType
from realm.utils.log import get_logger
from realm.utils.transforms import Resize
from realm.utils.vis import image_to_normalized_tensor, voxel_to_rgb_image

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# PCA-to-RGB projection
# ---------------------------------------------------------------------------

def shared_pca_rgb(token_sets: list[Tensor], grid: tuple[int, int]) -> list[np.ndarray]:
    """
    Project multiple ``(N, D)`` token sets onto a *shared* 3-component PCA
    basis and normalise them onto a common ``[0, 1]`` range, so the
    resulting RGB maps are directly colour-comparable across modalities.

    Args:
        token_sets: List of ``(N, D)`` token tensors (batch dim removed).
        grid:       ``(Ph, Pw)`` patch grid to reshape each map to.

    Returns:
        List of ``(Ph, Pw, 3)`` float32 NumPy arrays in ``[0, 1]``, one per
        input token set.
    """
    sizes = [t.shape[0] for t in token_sets]
    combined = torch.cat(token_sets, dim=0).float()  # (sum_N, D)

    mean = combined.mean(dim=0, keepdim=True)
    centered = combined - mean

    _, _, V = torch.pca_lowrank(centered, q=3)
    proj = centered @ V[:, :3]  # (sum_N, 3)

    proj_min = proj.min(dim=0, keepdim=True).values
    proj_max = proj.max(dim=0, keepdim=True).values
    proj = (proj - proj_min) / (proj_max - proj_min + 1e-8)

    out = []
    start = 0
    for n in sizes:
        chunk = proj[start : start + n].reshape(grid[0], grid[1], 3)
        out.append(chunk.cpu().numpy())
        start += n
    return out


def resize_for_display(arr: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """
    Resize a small ``(h, w, 3)`` map up to display *size*.

    Cubic interpolation can overshoot the source range at sharp edges, so
    the result is clipped back to ``[0, 1]`` for safe display with imshow.
    """
    resized = cv2.resize(arr, size, interpolation=cv2.INTER_CUBIC)
    return np.clip(resized, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualise event/image patch-embedding alignment.")
    parser.add_argument("--config", type=str, default="realm/realm/configs/mast3r.yaml")
    parser.add_argument("--image", type=str, default="test/00326_r_3d.jpg")
    parser.add_argument("--voxel", type=str, default="test/ev_l_17421491445.npy")
    parser.add_argument("--size", type=int, default=448, help="Square side length models are fed.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=str, default="embedding_pca.png")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    device = torch.device(args.device)

    model = REALM_creator(args.config).to(device)
    model.eval()

    if model.model_type != ModelType.Hybrid:
        raise ValueError(
            f"This script compares event and RGB embeddings and needs a Hybrid model "
            f"(both encoders). Got model_type={model.model_type.value}. "
            f"Use a config with both 'dune_checkpoint' and 'realm_checkpoint' set."
        )

    size = (args.size, args.size)

    with torch.inference_mode():
        # --- RGB input ---
        img_bgr = cv2.imread(args.image)
        if img_bgr is None:
            raise FileNotFoundError(f"Test image not found: {args.image}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_rgb = cv2.resize(img_rgb, size)
        img_tensor = image_to_normalized_tensor(img_rgb).unsqueeze(0).to(device)

        # --- Event voxel input ---
        voxel_path = Path(args.voxel)
        if not voxel_path.is_file():
            raise FileNotFoundError(f"Test voxel not found: {voxel_path}")
        ev = torch.from_numpy(np.load(voxel_path)).unsqueeze(0).to(device)
        ev = Resize(size, keep_aspect_ratio=True)(ev)
        ev_vis = voxel_to_rgb_image(ev.squeeze(0)).numpy()

        # --- Raw tokenizer output (pre-transformer) ---
        raw_ev = model.encoder_ev.patch_embed(ev, aux=True)["patch_tokens"].squeeze(0)
        raw_rgb = model.encoder_rgb.patch_embed(img_tensor).squeeze(0)

        # --- Full-encoder patch tokens (post-transformer) ---
        enc_ev = model.encoder_ev(ev)["x_norm_patchtokens"].squeeze(0)
        enc_rgb = model.encoder_rgb(img_tensor)["x_norm_patchtokens"].squeeze(0)

    grid_side = int(round(raw_ev.shape[0] ** 0.5))
    grid = (grid_side, grid_side)
    logger.info("Patch grid: %s (N=%d tokens)", grid, raw_ev.shape[0])

    # --- PCA-to-RGB, shared basis per stage so colours are comparable ---
    raw_ev_rgb, raw_img_rgb = shared_pca_rgb([raw_ev, raw_rgb], grid)
    enc_ev_rgb, enc_img_rgb = shared_pca_rgb([enc_ev, enc_rgb], grid)

    raw_ev_rgb = resize_for_display(raw_ev_rgb, size)
    raw_img_rgb = resize_for_display(raw_img_rgb, size)
    enc_ev_rgb = resize_for_display(enc_ev_rgb, size)
    enc_img_rgb = resize_for_display(enc_img_rgb, size)

    # --- Per-patch cosine similarity between the two modalities ---
    raw_sim = torch.nn.functional.cosine_similarity(raw_ev, raw_rgb, dim=-1).reshape(grid).cpu().numpy()
    enc_sim = torch.nn.functional.cosine_similarity(enc_ev, enc_rgb, dim=-1).reshape(grid).cpu().numpy()
    logger.info("Mean cosine similarity — raw tokenizer: %.3f | encoded: %.3f", raw_sim.mean(), enc_sim.mean())

    # --- Figure ---
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 4)

    def _panel(ax, img, title):
        ax.imshow(img)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    _panel(fig.add_subplot(gs[0, 0]), ev_vis, "event input")
    _panel(fig.add_subplot(gs[0, 1]), raw_ev_rgb, "event PCA (raw tokenizer)")
    _panel(fig.add_subplot(gs[0, 2]), raw_img_rgb, "image PCA (raw tokenizer)")
    _panel(fig.add_subplot(gs[0, 3]), img_rgb, "image input")

    _panel(fig.add_subplot(gs[1, 0]), ev_vis, "event input")
    _panel(fig.add_subplot(gs[1, 1]), enc_ev_rgb, "event PCA (encoded)")
    _panel(fig.add_subplot(gs[1, 2]), enc_img_rgb, "image PCA (encoded)")
    _panel(fig.add_subplot(gs[1, 3]), img_rgb, "image input")

    ax_raw_sim = fig.add_subplot(gs[2, 0:2])
    im0 = ax_raw_sim.imshow(raw_sim, cmap="coolwarm", vmin=-1, vmax=1)
    ax_raw_sim.set_title(f"cosine sim (raw), mean={raw_sim.mean():.3f}", fontsize=10)
    ax_raw_sim.axis("off")
    fig.colorbar(im0, ax=ax_raw_sim, fraction=0.046)

    ax_enc_sim = fig.add_subplot(gs[2, 2:4])
    im1 = ax_enc_sim.imshow(enc_sim, cmap="coolwarm", vmin=-1, vmax=1)
    ax_enc_sim.set_title(f"cosine sim (encoded), mean={enc_sim.mean():.3f}", fontsize=10)
    ax_enc_sim.axis("off")
    fig.colorbar(im1, ax=ax_enc_sim, fraction=0.046)

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    logger.success(f"Saved embedding visualisation to: {args.out}")  # type: ignore[attr-defined]


if __name__ == "__main__":
    main()
