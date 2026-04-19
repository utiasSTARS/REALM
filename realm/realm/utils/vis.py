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
vis.py — Visualisation and image-conversion utilities for REALM.

Functions
---------
    matches                  — compute reciprocal nearest-neighbour matches
    vis_matches              — draw match lines on a side-by-side image pair
    VisMast3r                — end-to-end Mast3r output visualiser
    voxel_to_rgb_image       — convert an event voxel grid to an RGB image
    image_to_normalized_tensor — normalise an image to an ImageNet-normalised tensor
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image
from torch import Tensor

from realm.heads.mast3r.fast_nn import fast_reciprocal_NNs
from realm.utils.log import get_logger

__all__ = [
    "matches",
    "vis_matches",
    "VisMast3r",
    "voxel_to_rgb_image",
    "image_to_normalized_tensor",
]

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# ImageNet statistics (constant — never mutated)
# ---------------------------------------------------------------------------

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)

# Pre-built tensors (shape (3, 1, 1)) reused across calls on the same device.
# Built lazily and cached per device to avoid repeated allocation.
_mean_cache: dict[torch.device, Tensor] = {}
_std_cache:  dict[torch.device, Tensor] = {}


def _get_imagenet_stats(device: torch.device) -> tuple[Tensor, Tensor]:
    """Return cached (mean, std) tensors on *device* with shape (3, 1, 1)."""
    if device not in _mean_cache:
        _mean_cache[device] = torch.tensor(_IMAGENET_MEAN, dtype=torch.float32, device=device).view(3, 1, 1)
        _std_cache[device]  = torch.tensor(_IMAGENET_STD,  dtype=torch.float32, device=device).view(3, 1, 1)
    return _mean_cache[device], _std_cache[device]


# ---------------------------------------------------------------------------
# Match computation
# ---------------------------------------------------------------------------

def matches(
    desc1: Tensor,
    desc2: Tensor,
    H: int,
    W: int,
    border: int = 3,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute reciprocal nearest-neighbour feature matches and filter border hits.

    Args:
        desc1:  Descriptor tensor for image 1, shape ``(N, D)``.
        desc2:  Descriptor tensor for image 2, shape ``(M, D)``.
        H:      Image height (pixels) — used for border filtering.
        W:      Image width  (pixels) — used for border filtering.
        border: Minimum distance (pixels) from any image edge for a match
                to be considered valid. Default is 3.
        device: PyTorch device string for the NN search.

    Returns:
        ``(matches_im0, matches_im1)`` — two ``(K, 2)`` int arrays of
        ``[x, y]`` keypoint coordinates, one per image.
    """
    matches_im0, matches_im1 = fast_reciprocal_NNs(
        desc1, desc2,
        subsample_or_initxy1=8,
        device=device,
        dist="dot",
        block_size=2 ** 13,
    )

    def _in_bounds(m: np.ndarray) -> np.ndarray:
        return (
            (m[:, 0] >= border) & (m[:, 0] < W - border) &
            (m[:, 1] >= border) & (m[:, 1] < H - border)
        )

    valid = _in_bounds(matches_im0) & _in_bounds(matches_im1)
    return matches_im0[valid], matches_im1[valid]


# ---------------------------------------------------------------------------
# Match visualisation
# ---------------------------------------------------------------------------

def vis_matches(
    img1: np.ndarray,
    img2: np.ndarray,
    matches_im0: np.ndarray,
    matches_im1: np.ndarray,
    inliers_mask: Optional[np.ndarray] = None,
    n_viz: int = 20,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Draw match lines on a side-by-side visualisation of two images.

    Args:
        img1:          Left  image ``(H0, W0, 3)`` uint8 or float.
        img2:          Right image ``(H1, W1, 3)`` uint8 or float.
        matches_im0:   ``(N, 2)`` int array of ``[x, y]`` coords in *img1*.
        matches_im1:   ``(N, 2)`` int array of ``[x, y]`` coords in *img2*.
        inliers_mask:  Optional boolean array of length ``N``.  Inliers are
                       drawn in green, outliers in blue.  If ``None`` all
                       matches are drawn in green.
        n_viz:         Maximum number of matches to draw.
        rng:           NumPy random generator for reproducible subsampling.
                       Pass ``np.random.default_rng(42)`` for a fixed seed.

    Returns:
        ``(H_max, W0+W1, 3)`` uint8 visualisation image.
    """
    num_matches = len(matches_im0)
    if num_matches == 0:
        return np.concatenate((img1, img2), axis=1)

    _rng = rng or np.random.default_rng()
    indices = (
        _rng.choice(num_matches, size=n_viz, replace=False)
        if num_matches > n_viz
        else np.arange(num_matches)
    )

    viz0 = matches_im0[indices]
    viz1 = matches_im1[indices]
    mask = np.asarray(inliers_mask)[indices] if inliers_mask is not None else None

    # --- Canvas ---
    H0, W0 = img1.shape[:2]
    H1, W1 = img2.shape[:2]
    H_max = max(H0, H1)

    canvas = np.concatenate(
        (
            cv2.copyMakeBorder(img1, 0, H_max - H0, 0, 0, cv2.BORDER_CONSTANT),
            cv2.copyMakeBorder(img2, 0, H_max - H1, 0, 0, cv2.BORDER_CONSTANT),
        ),
        axis=1,
    )

    # --- Draw ---
    _COLOR_INLIER  = (0, 115, 54)   # dark green
    _COLOR_OUTLIER = (52, 128, 235)  # blue
    _COLOR_DEFAULT = (0, 255, 0)     # bright green

    for i in range(len(viz0)):
        pt1 = (int(viz0[i, 0]),        int(viz0[i, 1]))
        pt2 = (int(viz1[i, 0]) + W0,   int(viz1[i, 1]))

        if mask is not None:
            color = _COLOR_INLIER if mask[i] else _COLOR_OUTLIER
        else:
            color = _COLOR_DEFAULT

        cv2.line(canvas, pt1, pt2, color, 2, cv2.LINE_AA)
        cv2.circle(canvas, pt1, 3, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, pt2, 3, color, -1, cv2.LINE_AA)

    return canvas


# ---------------------------------------------------------------------------
# Mast3r end-to-end visualiser
# ---------------------------------------------------------------------------

def VisMast3r(
    output: dict,
    n_viz: int = 100,
    device: str = "cpu",
    seed: Optional[int] = 42,
) -> np.ndarray:
    """
    Visualise Mast3r model output as a match overlay on two images.

    Args:
        output: Dict with keys ``"view1"``, ``"view2"``, ``"pred1"``,
                ``"pred2"``.  Images are ``(H, W, 3)`` arrays; predictions
                are dicts containing ``"desc"`` tensors.
        n_viz:  Maximum number of matches to draw.
        device: Device for the NN descriptor search.
        seed:   Random seed for reproducible match subsampling.
                Pass ``None`` for non-deterministic selection.

    Returns:
        ``(H_max, 2*W, 3)`` uint8 match visualisation.

    Raises:
        KeyError: If required keys are missing from *output*.
    """
    for key in ("view1", "view2", "pred1", "pred2"):
        if key not in output:
            raise KeyError(
                f"Required key '{key}' not found in output dict. "
                f"Available keys: {list(output.keys())}"
            )

    img1, pred1 = output["view1"], output["pred1"]
    img2, pred2 = output["view2"], output["pred2"]

    desc1 = pred1["desc"].squeeze(0)
    desc2 = pred2["desc"].squeeze(0)

    H, W = img1.shape[:2]
    matches_im0, matches_im1 = matches(desc1, desc2, H, W, device=device)

    num_matches = len(matches_im0)
    if num_matches == 0:
        logger.warning("VisMast3r: no matches found — returning plain side-by-side.")
        return np.concatenate((img1, img2), axis=1)

    rng = np.random.default_rng(seed)
    indices = (
        np.round(np.linspace(0, num_matches - 1, min(n_viz, num_matches)))
        .astype(int)
    )
    viz0 = matches_im0[indices]
    viz1 = matches_im1[indices]

    H0, W0 = img1.shape[:2]
    H1, W1 = img2.shape[:2]
    H_max = max(H0, H1)

    canvas = np.concatenate(
        (
            np.pad(img1, ((0, H_max - H0), (0, 0), (0, 0))),
            np.pad(img2, ((0, H_max - H1), (0, 0), (0, 0))),
        ),
        axis=1,
    )

    for i in range(len(viz0)):
        pt1 = (int(viz0[i, 0]),       int(viz0[i, 1]))
        pt2 = (int(viz1[i, 0]) + W0,  int(viz1[i, 1]))
        color = tuple(int(c) for c in rng.integers(0, 255, 3))
        cv2.line(canvas, pt1, pt2, color, 2, cv2.LINE_AA)
        cv2.circle(canvas, pt1, 4, color, 2, cv2.LINE_AA)
        cv2.circle(canvas, pt2, 4, color, 2, cv2.LINE_AA)

    return canvas


# ---------------------------------------------------------------------------
# Voxel → RGB
# ---------------------------------------------------------------------------

def voxel_to_rgb_image(
    voxel_grid_tensor: Tensor,
    background_color: float = 0.0,
) -> np.ndarray:
    """
    Convert an event voxel grid tensor to a polarity-coloured RGB image.

    Positive events → Blue ``(0, 0, 1)``
    Negative events → Red  ``(1, 0, 0)``
    No events       → ``background_color`` (greyscale)

    Args:
        voxel_grid_tensor: ``(C, H, W)`` float tensor with ``C > 3`` temporal
                           bins.
        background_color:  Background pixel intensity in ``[0, 1]``.

    Returns:
        ``(H, W, 3)`` float32 NumPy array in ``[0, 1]``.

    Raises:
        TypeError:  If the input is not a :class:`torch.Tensor`.
        ValueError: If the tensor does not have exactly 3 dimensions, or if
                    ``C <= 3`` (not a multi-bin voxel grid).
    """
    if not isinstance(voxel_grid_tensor, Tensor):
        raise TypeError(
            f"Expected a torch.Tensor, got {type(voxel_grid_tensor).__name__}."
        )
    if voxel_grid_tensor.ndim != 3:
        raise ValueError(
            f"Expected a 3-D tensor (C, H, W), got shape {tuple(voxel_grid_tensor.shape)}."
        )
    if voxel_grid_tensor.shape[0] <= 3:
        raise ValueError(
            f"Expected a voxel grid with > 3 temporal bins (C > 3), "
            f"got C={voxel_grid_tensor.shape[0]}. "
            "If you intended to pass an RGB image, no conversion is needed."
        )

    flat = voxel_grid_tensor.detach().cpu().sum(dim=0)  # (H, W)
    H, W = flat.shape

    fill = float(background_color)
    rgb = torch.full((H, W, 3), fill, dtype=torch.float32)
    rgb[flat > 0] = torch.tensor([0.0, 0.0, 1.0])   # Blue  — positive
    rgb[flat < 0] = torch.tensor([1.0, 0.0, 0.0])   # Red   — negative

    return rgb.numpy()


# ---------------------------------------------------------------------------
# Image → normalised tensor
# ---------------------------------------------------------------------------

def image_to_normalized_tensor(img: Image.Image | np.ndarray | Tensor) -> Tensor:
    """
    Convert an image to an ImageNet-normalised ``(3, H, W)`` float32 tensor.

    Accepted input types
    --------------------
    * :class:`PIL.Image.Image`   — converted to ``(H, W, C)`` uint8 array first.
    * :class:`numpy.ndarray`     — ``(H, W)``, ``(H, W, C)``, values in
                                   ``[0, 255]`` or ``[0, 1]``.
    * :class:`torch.Tensor`      — ``(C, H, W)`` or ``(H, W, C)`` or their
                                   batched equivalents ``(B, C, H, W)`` /
                                   ``(B, H, W, C)``, values in ``[0, 255]``
                                   or ``[0, 1]``.

    Returns:
        ImageNet-normalised tensor of shape ``(3, H, W)`` (or ``(B, 3, H, W)``
        for batched tensor input), dtype ``float32``.

    Raises:
        TypeError:  If *img* is not a PIL image, NumPy array, or torch Tensor.
        ValueError: If the spatial/channel layout cannot be inferred.
    """
    # ------------------------------------------------------------------ PIL
    if isinstance(img, Image.Image):
        img = np.array(img)

    # ------------------------------------------------------------------ NumPy
    if isinstance(img, np.ndarray):
        arr = img.astype(np.float32)
        if arr.ndim == 2:                          # (H, W) → (H, W, 3)
            arr = np.stack([arr] * 3, axis=-1)
        if arr.ndim != 3 or arr.shape[2] not in (1, 3):
            raise ValueError(
                f"Unsupported numpy array shape: {arr.shape}. "
                "Expected (H, W) or (H, W, C) with C in {{1, 3}}."
            )
        tensor = torch.from_numpy(arr).permute(2, 0, 1).float()  # (C, H, W)
        if tensor.max() > 1.0:
            tensor = tensor / 255.0
        mean, std = _get_imagenet_stats(tensor.device)
        return (tensor - mean) / std

    # ------------------------------------------------------------------ Tensor
    if isinstance(img, Tensor):
        t = img.float()

        # Scale [0, 255] → [0, 1] if needed
        if t.max() > 1.0:
            t = t / 255.0

        # Normalise dimension layout to (..., C, H, W)
        if t.ndim == 3:
            # (C, H, W) or (H, W, C)
            if t.shape[0] in (1, 3):
                pass                              # already (C, H, W)
            elif t.shape[2] in (1, 3):
                t = t.permute(2, 0, 1)
            else:
                raise ValueError(
                    f"Cannot infer channel layout for 3-D tensor of shape {tuple(t.shape)}."
                )
        elif t.ndim == 4:
            # (B, C, H, W) or (B, H, W, C)
            if t.shape[1] in (1, 3):
                pass                              # already (B, C, H, W)
            elif t.shape[3] in (1, 3):
                t = t.permute(0, 3, 1, 2)
            else:
                raise ValueError(
                    f"Cannot infer channel layout for 4-D tensor of shape {tuple(t.shape)}."
                )
        else:
            raise ValueError(
                f"Unsupported tensor rank {t.ndim}. Expected 3-D or 4-D input."
            )

        mean, std = _get_imagenet_stats(t.device)
        return (t - mean) / std

    raise TypeError(
        f"Unsupported input type: {type(img).__name__}. "
        "Expected PIL.Image, numpy.ndarray, or torch.Tensor."
    )