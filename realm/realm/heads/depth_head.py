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
depth_head.py — Soft-classification depth head for REALM.

Decodes ViT backbone features (patch tokens + CLS token) into a dense metric
depth map via a soft-classification over linearly spaced depth bins.

Architecture
------------
::

    patch tokens (B, N, C)  →  1×1 conv  →  ×4 bilinear upsample
                                                      ↘
                                                        (+)  →  softmax  →  Σ wᵢ·bᵢ  →  depth (B, 1, H, W)
                                                      ↗
    CLS token    (B, C)     →  Linear    →  broadcast (B, num_bins, 1, 1)

Public API
----------
    LinearDepthHead — patch + CLS token → metric depth map
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

__all__ = ["LinearDepthHead"]


class LinearDepthHead(nn.Module):
    """
    Metric depth head using soft classification over linear depth bins.

    Two complementary feature streams are fused:

    * **Patch stream** — spatial detail from the patch token grid, processed
      with a 1×1 convolution and upsampled ×4.
    * **CLS stream** — global scene context from the CLS token, projected
      with a linear layer and broadcast over the spatial grid.

    The fused logits are converted to a depth map via softmax + weighted sum
    over ``num_bins`` linearly spaced depth values in ``[min_depth, max_depth]``.

    Args:
        embed_dim: Dimensionality of backbone patch/CLS tokens.
        min_depth: Minimum metric depth value (metres).
        max_depth: Maximum metric depth value (metres).
        num_bins:  Number of discrete depth bins for soft classification.

    Raises:
        ValueError: If depth range or bin count are invalid.

    Example
    -------
    ::

        head  = LinearDepthHead(embed_dim=768, min_depth=0.001, max_depth=80.0)
        depth = head(features, options={"H": 480, "W": 640, "upsample": True})
        # depth: (B, 1, 480, 640)
    """

    def __init__(
        self,
        embed_dim: int = 768,
        min_depth: float = 0.001,
        max_depth: float = 80.0,
        num_bins: int = 256,
    ) -> None:
        super().__init__()

        if min_depth <= 0:
            raise ValueError(f"min_depth must be positive, got {min_depth}.")
        if max_depth <= min_depth:
            raise ValueError(
                f"max_depth ({max_depth}) must be greater than min_depth ({min_depth})."
            )
        if num_bins < 2:
            raise ValueError(f"num_bins must be >= 2, got {num_bins}.")
        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {embed_dim}.")

        self.min_depth = min_depth
        self.max_depth = max_depth
        self.num_bins  = num_bins
        self.embed_dim = embed_dim

        # Linearly spaced bin centres — registered as a non-trainable buffer
        # so they are automatically moved with .to(device) and saved in state_dict.
        self.register_buffer(
            "bin_centers",
            torch.linspace(min_depth, max_depth, num_bins),
        )

        # Patch-token head: (B, C, Ph, Pw) → (B, num_bins, Ph, Pw)
        self.head_patch = nn.Conv2d(embed_dim, num_bins, kernel_size=1, bias=True)

        # CLS-token head:  (B, C) → (B, num_bins)
        self.head_cls   = nn.Linear(embed_dim, num_bins, bias=True)

        self._init_weights()

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        """Xavier-uniform for both heads; zero biases."""
        nn.init.xavier_uniform_(self.head_patch.weight)
        nn.init.zeros_(self.head_patch.bias)
        nn.init.xavier_uniform_(self.head_cls.weight)
        nn.init.zeros_(self.head_cls.bias)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        features: dict[str, Tensor],
        options: dict | None = None,
    ) -> Tensor:
        """
        Decode backbone features into a metric depth map.

        Args:
            features: Dictionary produced by the ViT backbone containing:

                      * ``"x_norm_patchtokens"`` — ``(B, N, C)`` patch tokens.
                      * ``"x_norm_clstoken"``    — ``(B, C)`` or ``(B, 1, C)``
                        CLS token.

            options:  Optional inference-time settings dict with keys:

                      * ``"upsample"`` *(bool, default True)* — bilinearly
                        upsample the depth map to ``(H, W)``.
                      * ``"H"`` *(int)* — target height (required when
                        ``upsample=True``).
                      * ``"W"`` *(int)* — target width  (required when
                        ``upsample=True``).

        Returns:
            Depth map of shape:

            * ``(B, 1, Ph*4, Pw*4)`` when ``upsample=False``
            * ``(B, 1, H, W)``       when ``upsample=True``

        Raises:
            KeyError:   If required keys are absent from *features*.
            ValueError: If the patch token count is not a perfect square, or if
                        ``upsample=True`` but ``H``/``W`` are not provided.
        """
        opts = options or {}

        # ---- Validate inputs ------------------------------------------------
        for key in ("x_norm_patchtokens", "x_norm_clstoken"):
            if key not in features:
                raise KeyError(
                    f"Required key '{key}' not found in features dict. "
                    f"Available keys: {list(features.keys())}"
                )

        patches   = features["x_norm_patchtokens"]  # (B, N, C)
        cls_token = features["x_norm_clstoken"]     # (B, C) or (B, 1, C)

        if patches.ndim != 3:
            raise ValueError(
                f"Expected patch_tokens of shape (B, N, C), got {tuple(patches.shape)}."
            )

        B, N, C = patches.shape

        # ---- Infer square patch grid, trim register tokens if present --------
        Ph = int(math.isqrt(N))
        if Ph * Ph != N:
            # Some backbones append register tokens after the patch tokens;
            # trim from the front, keeping the last Ph² tokens.
            patches = patches[:, -(Ph * Ph) :, :]
            N = Ph * Ph

        Pw = Ph  # square grid assumed

        # ---- Patch stream ---------------------------------------------------
        # (B, N, C) → (B, C, Ph, Pw) → (B, num_bins, Ph, Pw)
        x = patches.permute(0, 2, 1).reshape(B, C, Ph, Pw)
        patch_logits = self.head_patch(x)

        # ×4 upsample: (B, num_bins, Ph, Pw) → (B, num_bins, Ph*4, Pw*4)
        patch_logits = F.interpolate(
            patch_logits, scale_factor=4, mode="bilinear", align_corners=False
        )

        # ---- CLS stream -----------------------------------------------------
        if cls_token.ndim == 3:
            cls_token = cls_token.squeeze(1)   # (B, 1, C) → (B, C)

        cls_logits = self.head_cls(cls_token)                    # (B, num_bins)
        cls_logits = cls_logits.view(B, self.num_bins, 1, 1)     # broadcast-ready

        # ---- Fusion + soft classification -----------------------------------
        logits = patch_logits + cls_logits                        # (B, num_bins, H', W')
        probs  = F.softmax(logits, dim=1)

        bins  = self.bin_centers.view(1, self.num_bins, 1, 1)    # type: ignore[attr-defined]
        depth = torch.sum(probs * bins, dim=1, keepdim=True)     # (B, 1, H', W')

        # ---- Optional upsample to target resolution -------------------------
        if opts.get("upsample", True):
            H = opts.get("H")
            W = opts.get("W")
            if H is None or W is None:
                raise ValueError(
                    "'H' and 'W' must be provided in options when upsample=True."
                )
            target = (int(H), int(W))
            if depth.shape[-2:] != target:
                depth = F.interpolate(
                    depth, size=target, mode="bilinear", align_corners=False
                )

        return depth

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"embed_dim={self.embed_dim}, "
            f"num_bins={self.num_bins}, "
            f"depth_range=[{self.min_depth}, {self.max_depth}])"
        )